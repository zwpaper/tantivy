use std::io;

use columnar::{ColumnType, ColumnarWriter, NumericalValue};
use common::{DateTimePrecision, JsonPathWriter};
use stacker::Addr;
use tokenizer_api::Token;

use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::schema::document::{Document, ReferenceValue, ReferenceValueLeaf, Value};
use crate::schema::{value_type_to_column_type, Field, FieldType, Schema, Type};
use crate::tokenizer::{TextAnalyzer, TokenizerManager, RAW_TOKENIZER_NAME};
use crate::{DocId, TantivyError};

/// Only index JSON down to a depth of 20.
/// This is mostly to guard us from a stack overflow triggered by malicious input.
const JSON_DEPTH_LIMIT: usize = 20;

/// Where a fast-field leaf value should be recorded: at the cached column address (non-JSON fast
/// fields, the common case) or by column name (the fallback).
enum FastFieldTarget<'a> {
    Addr(Addr),
    Name(&'a str),
}

impl FastFieldTarget<'_> {
    #[inline]
    fn record_numerical(&self, writer: &mut ColumnarWriter, doc: DocId, value: NumericalValue) {
        match self {
            Self::Addr(addr) => writer.record_numerical_at(doc, *addr, value),
            Self::Name(name) => writer.record_numerical(doc, name, value),
        }
    }

    #[inline]
    fn record_str(&self, writer: &mut ColumnarWriter, doc: DocId, value: &str) {
        match self {
            Self::Addr(addr) => writer.record_str_at(doc, *addr, value),
            Self::Name(name) => writer.record_str(doc, name, value),
        }
    }

    #[inline]
    fn record_datetime(&self, writer: &mut ColumnarWriter, doc: DocId, value: common::DateTime) {
        match self {
            Self::Addr(addr) => writer.record_datetime_at(doc, *addr, value),
            Self::Name(name) => writer.record_datetime(doc, name, value),
        }
    }

    #[inline]
    fn record_bool(&self, writer: &mut ColumnarWriter, doc: DocId, value: bool) {
        match self {
            Self::Addr(addr) => writer.record_bool_at(doc, *addr, value),
            Self::Name(name) => writer.record_bool(doc, name, value),
        }
    }

    #[inline]
    fn record_ip_addr(&self, writer: &mut ColumnarWriter, doc: DocId, value: std::net::Ipv6Addr) {
        match self {
            Self::Addr(addr) => writer.record_ip_addr_at(doc, *addr, value),
            Self::Name(name) => writer.record_ip_addr(doc, name, value),
        }
    }

    #[inline]
    fn record_bytes(&self, writer: &mut ColumnarWriter, doc: DocId, value: &[u8]) {
        match self {
            Self::Addr(addr) => writer.record_bytes_at(doc, *addr, value),
            Self::Name(name) => writer.record_bytes(doc, name, value),
        }
    }
}

/// The `ColumnType` of the `ColumnarWriter` column a fast-field leaf value is recorded into.
///
/// A document may carry a value whose type does not match the field's declared type (tantivy
/// accepts it and records it in a different map, under the same name). `add_doc_value` uses this
/// to check that the column address cached for the field belongs to the map the leaf targets.
fn leaf_column_type(leaf: &ReferenceValueLeaf<'_>) -> Option<ColumnType> {
    match leaf {
        ReferenceValueLeaf::Null => None,
        ReferenceValueLeaf::Str(_)
        | ReferenceValueLeaf::Facet(_)
        | ReferenceValueLeaf::PreTokStr(_) => Some(ColumnType::Str),
        ReferenceValueLeaf::U64(_) => Some(ColumnType::U64),
        ReferenceValueLeaf::I64(_) => Some(ColumnType::I64),
        ReferenceValueLeaf::F64(_) => Some(ColumnType::F64),
        ReferenceValueLeaf::Date(_) => Some(ColumnType::DateTime),
        ReferenceValueLeaf::Bytes(_) => Some(ColumnType::Bytes),
        ReferenceValueLeaf::IpAddr(_) => Some(ColumnType::IpAddr),
        ReferenceValueLeaf::Bool(_) => Some(ColumnType::Bool),
        ReferenceValueLeaf::Custom(_) => None,
    }
}

/// The `FastFieldsWriter` groups all of the fast field writers.
pub struct FastFieldsWriter {
    columnar_writer: ColumnarWriter,
    fast_field_names: Vec<Option<String>>, //< TODO see if we can hash the field name hash too.
    // Address of the `ColumnWriter` registered in the `ColumnarWriter` for a non-JSON fast field.
    // The address is stable for the lifetime of the writer, so `add_doc_value` can update the
    // column in place instead of looking it up by name on every value. JSON fast fields and
    // tie-breaker columns have no entry here (`None`).
    fast_field_addrs: Vec<Option<Addr>>,
    // The `ColumnType` of the column `fast_field_addrs` points at. A document value whose type
    // does not match the field's declared type is recorded in a different `ColumnarWriter` map
    // (under the same name); `add_doc_value` falls back to the name-based path for those, so this
    // is used to confirm the cached address belongs to the map the value would be recorded in.
    fast_field_column_types: Vec<Option<ColumnType>>,
    per_field_tokenizer: Vec<Option<TextAnalyzer>>,
    date_precisions: Vec<DateTimePrecision>,
    expand_dots: Vec<bool>,
    num_docs: DocId,
    // Buffer that we recycle to avoid allocation.
    json_path_buffer: JsonPathWriter,
    // Test-only counters proving the mechanism: a non-JSON fast-field leaf takes the
    // cached-address path (which never hashes the column name) and the name-based fallback is not
    // used.
    #[cfg(test)]
    leaf_addr_path_count: usize,
    #[cfg(test)]
    leaf_name_path_count: usize,
}

impl FastFieldsWriter {
    /// Create all `FastFieldWriter` required by the schema.
    #[cfg(test)]
    pub fn from_schema(schema: &Schema) -> crate::Result<FastFieldsWriter> {
        FastFieldsWriter::from_schema_and_tokenizer_manager(schema, TokenizerManager::new())
    }

    /// Create all `FastFieldWriter` required by the schema.
    pub fn from_schema_and_tokenizer_manager(
        schema: &Schema,
        tokenizer_manager: TokenizerManager,
    ) -> crate::Result<FastFieldsWriter> {
        let mut columnar_writer = ColumnarWriter::default();

        let mut fast_field_names: Vec<Option<String>> = vec![None; schema.num_fields()];
        let mut fast_field_addrs: Vec<Option<Addr>> = vec![None; schema.num_fields()];
        let mut fast_field_column_types: Vec<Option<ColumnType>> = vec![None; schema.num_fields()];
        let mut date_precisions: Vec<DateTimePrecision> =
            std::iter::repeat_with(DateTimePrecision::default)
                .take(schema.num_fields())
                .collect();
        let mut expand_dots = vec![false; schema.num_fields()];
        let mut per_field_tokenizer: Vec<Option<TextAnalyzer>> = vec![None; schema.num_fields()];
        // TODO see other types
        for (field_id, field_entry) in schema.fields() {
            if !field_entry.field_type().is_fast() {
                continue;
            }
            if matches!(field_entry.field_type(), FieldType::TieBreaker) {
                columnar_writer.record_tie_breaker_column(field_entry.name());
                continue;
            }
            fast_field_names[field_id.field_id() as usize] = Some(field_entry.name().to_string());
            let value_type = field_entry.field_type().value_type();
            if let FieldType::Date(date_options) = field_entry.field_type() {
                date_precisions[field_id.field_id() as usize] = date_options.get_precision();
            }
            if let FieldType::JsonObject(json_object_options) = field_entry.field_type() {
                if let Some(json_text_fast_field_option) = &json_object_options.fast {
                    let tokenizer_name = json_text_fast_field_option.tokenizer.as_str();
                    if tokenizer_name != RAW_TOKENIZER_NAME {
                        let text_analyzer =
                            tokenizer_manager.get(tokenizer_name).ok_or_else(|| {
                                TantivyError::InvalidArgument(format!(
                                    "Tokenizer {tokenizer_name:?} not found"
                                ))
                            })?;
                        per_field_tokenizer[field_id.field_id() as usize] = Some(text_analyzer);
                    }
                }
                expand_dots[field_id.field_id() as usize] =
                    json_object_options.is_expand_dots_enabled();
            }
            if let FieldType::Str(text_options) = field_entry.field_type() {
                if let Some(fast_field_text_options) = &text_options.fast {
                    let tokenizer_name = fast_field_text_options.tokenizer.as_str();
                    if tokenizer_name != RAW_TOKENIZER_NAME {
                        let text_analyzer =
                            tokenizer_manager.get(tokenizer_name).ok_or_else(|| {
                                TantivyError::InvalidArgument(format!(
                                    "Tokenizer `{tokenizer_name}` not found"
                                ))
                            })?;
                        per_field_tokenizer[field_id.field_id() as usize] = Some(text_analyzer);
                    }
                }
            }

            let sort_values_within_row = value_type == Type::Facet;
            if let Some(column_type) = value_type_to_column_type(value_type) {
                let addr = columnar_writer.record_column_type(
                    field_entry.name(),
                    column_type,
                    sort_values_within_row,
                );
                fast_field_addrs[field_id.field_id() as usize] = Some(addr);
                fast_field_column_types[field_id.field_id() as usize] = Some(column_type);
            }
        }
        Ok(FastFieldsWriter {
            columnar_writer,
            fast_field_names,
            fast_field_addrs,
            fast_field_column_types,
            per_field_tokenizer,
            num_docs: 0u32,
            date_precisions,
            expand_dots,
            json_path_buffer: JsonPathWriter::default(),
            #[cfg(test)]
            leaf_addr_path_count: 0,
            #[cfg(test)]
            leaf_name_path_count: 0,
        })
    }

    /// The memory used (inclusive childs)
    pub fn mem_usage(&self) -> usize {
        self.columnar_writer.mem_usage()
    }

    pub(crate) fn sort_order(
        &self,
        sort_field: &str,
        num_docs: DocId,
        reversed: bool,
    ) -> Vec<DocId> {
        self.columnar_writer
            .sort_order(sort_field, num_docs, reversed)
    }

    /// Indexes all of the fastfields of a new document.
    pub fn add_document<D: Document>(&mut self, doc: &D) -> crate::Result<()> {
        let doc_id = self.num_docs;
        for (field, value) in doc.iter_fields_and_values() {
            let value_access = value as D::Value<'_>;

            self.add_doc_value(doc_id, field, value_access)?;
        }
        self.num_docs += 1;
        Ok(())
    }

    fn add_doc_value<'a, V: Value<'a>>(
        &mut self,
        doc_id: DocId,
        field: Field,
        value: V,
    ) -> crate::Result<()> {
        let field_name = match &self.fast_field_names[field.field_id() as usize] {
            None => return Ok(()),
            Some(name) => name,
        };

        match value.as_value() {
            ReferenceValue::Leaf(leaf) => {
                // A fast, non-JSON leaf field has its column registered in the `ColumnarWriter` at
                // schema setup; that address is stable, so the value can be recorded in place
                // without hashing/probing the column name on every value. A document value whose
                // type does not match the field's declared type (tantivy accepts it) is recorded
                // in a different `ColumnarWriter` map, so the cached address is only used when it
                // points at the map the leaf targets; otherwise fall back to the name-based path.
                // (`Custom` fields are never fast, so they never reach here; JSON fast fields take
                // the `Object` branch below.)
                let field_id = field.field_id() as usize;
                let addr_opt = match (
                    self.fast_field_addrs[field_id],
                    self.fast_field_column_types[field_id],
                ) {
                    (Some(addr), Some(column_type))
                        if leaf_column_type(&leaf) == Some(column_type) =>
                    {
                        Some(addr)
                    }
                    _ => None,
                };
                let target = match addr_opt {
                    Some(addr) => {
                        #[cfg(test)]
                        {
                            self.leaf_addr_path_count += 1;
                        }
                        FastFieldTarget::Addr(addr)
                    }
                    None => {
                        #[cfg(test)]
                        {
                            self.leaf_name_path_count += 1;
                        }
                        FastFieldTarget::Name(field_name)
                    }
                };
                match leaf {
                    ReferenceValueLeaf::Null => {}
                    ReferenceValueLeaf::Str(val) => {
                        if let Some(tokenizer) = &mut self.per_field_tokenizer[field_id] {
                            let mut token_stream = tokenizer.token_stream(val);
                            token_stream.process(&mut |token: &Token| {
                                target.record_str(&mut self.columnar_writer, doc_id, &token.text);
                            })
                        } else {
                            target.record_str(&mut self.columnar_writer, doc_id, val);
                        }
                    }
                    ReferenceValueLeaf::U64(val) => {
                        target.record_numerical(
                            &mut self.columnar_writer,
                            doc_id,
                            NumericalValue::from(val),
                        );
                    }
                    ReferenceValueLeaf::I64(val) => {
                        target.record_numerical(
                            &mut self.columnar_writer,
                            doc_id,
                            NumericalValue::from(val),
                        );
                    }
                    ReferenceValueLeaf::F64(val) => {
                        target.record_numerical(
                            &mut self.columnar_writer,
                            doc_id,
                            NumericalValue::from(val),
                        );
                    }
                    ReferenceValueLeaf::Date(val) => {
                        let date_precision = self.date_precisions[field_id];
                        let truncated_datetime = val.truncate(date_precision);
                        target.record_datetime(
                            &mut self.columnar_writer,
                            doc_id,
                            truncated_datetime,
                        );
                    }
                    ReferenceValueLeaf::Facet(val) => {
                        target.record_str(&mut self.columnar_writer, doc_id, val);
                    }
                    ReferenceValueLeaf::Bytes(val) => {
                        target.record_bytes(&mut self.columnar_writer, doc_id, val);
                    }
                    ReferenceValueLeaf::IpAddr(val) => {
                        target.record_ip_addr(&mut self.columnar_writer, doc_id, val);
                    }
                    ReferenceValueLeaf::Bool(val) => {
                        target.record_bool(&mut self.columnar_writer, doc_id, val);
                    }
                    ReferenceValueLeaf::PreTokStr(val) => {
                        for token in &val.tokens {
                            target.record_str(&mut self.columnar_writer, doc_id, &token.text);
                        }
                    }
                    // Custom fields are never fast (`is_fast() == false`), so they are not
                    // registered in `fast_field_names` and `add_doc_value` returns early above.
                    ReferenceValueLeaf::Custom(_) => {
                        unreachable!("the fast field writer does not support custom field types")
                    }
                }
            }
            ReferenceValue::Array(val) => {
                // TODO: Check this is the correct behaviour we want.
                for value in val {
                    self.add_doc_value(doc_id, field, value)?;
                }
            }
            ReferenceValue::Object(val) => {
                let expand_dots = self.expand_dots[field.field_id() as usize];
                self.json_path_buffer.clear();
                // First field should not be expanded.
                self.json_path_buffer.set_expand_dots(false);
                self.json_path_buffer.push(field_name);
                self.json_path_buffer.set_expand_dots(expand_dots);

                let text_analyzer = &mut self.per_field_tokenizer[field.field_id() as usize];

                record_json_obj_to_columnar_writer::<V>(
                    doc_id,
                    val,
                    JSON_DEPTH_LIMIT,
                    &mut self.json_path_buffer,
                    &mut self.columnar_writer,
                    text_analyzer,
                );
            }
        }

        Ok(())
    }

    /// Serializes all of the `FastFieldWriter`s by pushing them in
    /// order to the fast field serializer.
    pub fn serialize(
        &mut self,
        wrt: &mut dyn io::Write,
        doc_id_map_opt: Option<&DocIdMapping>,
    ) -> io::Result<()> {
        let num_docs = self.num_docs;
        let old_to_new_row_ids =
            doc_id_map_opt.map(|doc_id_mapping| doc_id_mapping.old_to_new_ids());
        self.columnar_writer
            .serialize(num_docs, old_to_new_row_ids, wrt)?;
        Ok(())
    }
}

fn record_json_obj_to_columnar_writer<'a, V: Value<'a>>(
    doc: DocId,
    json_visitor: V::ObjectIter,
    remaining_depth_limit: usize,
    json_path_buffer: &mut JsonPathWriter,
    columnar_writer: &mut columnar::ColumnarWriter,
    tokenizer: &mut Option<TextAnalyzer>,
) {
    for (key, child) in json_visitor {
        json_path_buffer.push(key);
        record_json_value_to_columnar_writer(
            doc,
            child,
            remaining_depth_limit,
            json_path_buffer,
            columnar_writer,
            tokenizer,
        );
        json_path_buffer.pop();
    }
}

fn record_json_value_to_columnar_writer<'a, V: Value<'a>>(
    doc: DocId,
    json_val: V,
    mut remaining_depth_limit: usize,
    json_path_writer: &mut JsonPathWriter,
    columnar_writer: &mut columnar::ColumnarWriter,
    tokenizer: &mut Option<TextAnalyzer>,
) {
    if remaining_depth_limit == 0 {
        return;
    }
    remaining_depth_limit -= 1;

    match json_val.as_value() {
        ReferenceValue::Leaf(leaf) => match leaf {
            ReferenceValueLeaf::Null => {} // TODO: Handle null
            ReferenceValueLeaf::Str(val) => {
                if let Some(text_analyzer) = tokenizer.as_mut() {
                    let mut token_stream = text_analyzer.token_stream(val);
                    token_stream.process(&mut |token| {
                        columnar_writer.record_str(doc, json_path_writer.as_str(), &token.text);
                    })
                } else {
                    columnar_writer.record_str(doc, json_path_writer.as_str(), val);
                }
            }
            ReferenceValueLeaf::U64(val) => {
                columnar_writer.record_numerical(
                    doc,
                    json_path_writer.as_str(),
                    NumericalValue::from(val),
                );
            }
            ReferenceValueLeaf::I64(val) => {
                columnar_writer.record_numerical(
                    doc,
                    json_path_writer.as_str(),
                    NumericalValue::from(val),
                );
            }
            ReferenceValueLeaf::F64(val) => {
                columnar_writer.record_numerical(
                    doc,
                    json_path_writer.as_str(),
                    NumericalValue::from(val),
                );
            }
            ReferenceValueLeaf::Bool(val) => {
                columnar_writer.record_bool(doc, json_path_writer.as_str(), val);
            }
            ReferenceValueLeaf::Date(val) => {
                columnar_writer.record_datetime(doc, json_path_writer.as_str(), val);
            }
            ReferenceValueLeaf::Facet(_) => {
                unimplemented!("Facet support in dynamic fields is not yet implemented")
            }
            ReferenceValueLeaf::Bytes(_) => {
                // TODO: This can be re added once it is added to the JSON Utils section as well.
                // columnar_writer.record_bytes(doc, json_path_writer.as_str(), val);
                unimplemented!("Bytes support in dynamic fields is not yet implemented")
            }
            ReferenceValueLeaf::IpAddr(_) => {
                unimplemented!("IP address support in dynamic fields is not yet implemented")
            }
            ReferenceValueLeaf::PreTokStr(_) => {
                unimplemented!(
                    "Pre-tokenized string support in dynamic fields is not yet implemented"
                )
            }
            ReferenceValueLeaf::Custom(_) => {
                unimplemented!("the JSON field does not support custom field types")
            }
        },
        ReferenceValue::Array(elements) => {
            for el in elements {
                record_json_value_to_columnar_writer(
                    doc,
                    el,
                    remaining_depth_limit,
                    json_path_writer,
                    columnar_writer,
                    tokenizer,
                );
            }
        }
        ReferenceValue::Object(object) => {
            record_json_obj_to_columnar_writer::<V>(
                doc,
                object,
                remaining_depth_limit,
                json_path_writer,
                columnar_writer,
                tokenizer,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use columnar::{Column, ColumnarReader, ColumnarWriter, StrColumn};
    use common::JsonPathWriter;

    use super::{record_json_value_to_columnar_writer, FastFieldsWriter};
    use crate::fastfield::writer::JSON_DEPTH_LIMIT;
    use crate::schema::{Schema, FAST};
    use crate::DocId;

    #[test]
    fn test_raw_fast_fields_bypass_tokenizer() {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", FAST);
        let json_field = schema_builder.add_json_field("json", FAST);
        let schema = schema_builder.build();

        // `from_schema` uses an empty tokenizer manager, so this also verifies that the raw
        // tokenizer is not looked up.
        let fast_fields_writer = FastFieldsWriter::from_schema(&schema).unwrap();
        assert!(fast_fields_writer.per_field_tokenizer[text_field.field_id() as usize].is_none());
        assert!(fast_fields_writer.per_field_tokenizer[json_field.field_id() as usize].is_none());
    }

    fn test_columnar_from_jsons_aux(
        json_docs: &[serde_json::Value],
        expand_dots: bool,
    ) -> ColumnarReader {
        let mut columnar_writer = ColumnarWriter::default();
        let mut json_path = JsonPathWriter::default();
        json_path.set_expand_dots(expand_dots);
        for (doc, json_doc) in json_docs.iter().enumerate() {
            record_json_value_to_columnar_writer(
                doc as u32,
                json_doc,
                JSON_DEPTH_LIMIT,
                &mut json_path,
                &mut columnar_writer,
                &mut None,
            );
        }
        let mut buffer = Vec::new();
        columnar_writer
            .serialize(json_docs.len() as DocId, None, &mut buffer)
            .unwrap();
        ColumnarReader::open(buffer).unwrap()
    }

    #[test]
    fn test_json_fastfield_record_simple() {
        let json_doc = serde_json::json!({
            "float": 1.02,
            "text": "hello happy tax payer",
            "nested": {"child": 3, "child2": 5},
            "arr": ["hello", "happy", "tax", "payer"]
        });
        let columnar_reader = test_columnar_from_jsons_aux(&[json_doc], false);
        let columns = columnar_reader.list_columns().unwrap();
        {
            assert_eq!(columns[0].0, "arr");
            let column_arr_opt: Option<StrColumn> = columns[0].1.open().unwrap().into();
            assert!(column_arr_opt
                .unwrap()
                .term_ords(0)
                .eq([1, 0, 3, 2].into_iter()));
        }
        {
            assert_eq!(columns[1].0, "float");
            let column_float_opt: Option<Column<f64>> = columns[1].1.open().unwrap().into();
            assert!(column_float_opt
                .unwrap()
                .values_for_doc(0)
                .eq([1.02f64].into_iter()));
        }
        {
            assert_eq!(columns[2].0, "nested\u{1}child");
            let column_nest_child_opt: Option<Column<i64>> = columns[2].1.open().unwrap().into();
            assert!(column_nest_child_opt
                .unwrap()
                .values_for_doc(0)
                .eq([3].into_iter()));
        }
        {
            assert_eq!(columns[3].0, "nested\u{1}child2");
            let column_nest_child2_opt: Option<Column<i64>> = columns[3].1.open().unwrap().into();
            assert!(column_nest_child2_opt
                .unwrap()
                .values_for_doc(0)
                .eq([5].into_iter()));
        }
        {
            assert_eq!(columns[4].0, "text");
            let column_text_opt: Option<StrColumn> = columns[4].1.open().unwrap().into();
            assert!(column_text_opt.unwrap().term_ords(0).eq([0].into_iter()));
        }
    }

    #[test]
    fn test_json_fastfield_deep_obj() {
        let json_doc = serde_json::json!(
            {"a": {"a": {"a": {"a": {"a":
            {"a": {"a": {"a": {"a": {"a":
            {"a": {"a": {"a": {"a": {"a":
            {"a": {"a": {"a": {"depth_accepted": 19, "a": {  "depth_truncated": 20}
        }}}}}}}}}}}}}}}}}}});
        let columnar_reader = test_columnar_from_jsons_aux(&[json_doc], false);
        let columns = columnar_reader.list_columns().unwrap();
        assert_eq!(columns.len(), 1);
        assert!(columns[0].0.ends_with("a\u{1}a\u{1}a\u{1}depth_accepted"));
    }

    #[test]
    fn test_json_fastfield_deep_arr() {
        let json_doc = json!(
        {"obj":
        [[[[[,
        [[[[[,
        [[[[[,
        [[18, [19, //< within limits
        [20]]]]]]]]]]]]]]]]]]]});
        let columnar_reader = test_columnar_from_jsons_aux(&[json_doc], false);
        let columns = columnar_reader.list_columns().unwrap();
        assert_eq!(columns.len(), 1);
        assert_eq!(columns[0].0, "obj");
        let dynamic_column = columns[0].1.open().unwrap();
        let col: Option<Column<i64>> = dynamic_column.into();
        let vals: Vec<i64> = col.unwrap().values_for_doc(0).collect();
        assert_eq!(&vals, &[18, 19])
    }

    #[test]
    fn test_json_fast_field_do_not_expand_dots() {
        let json_doc = json!({"field.with.dots": {"child.with.dot": "hello"}});
        let columnar_reader = test_columnar_from_jsons_aux(&[json_doc], false);
        let columns = columnar_reader.list_columns().unwrap();
        assert_eq!(columns.len(), 1);
        assert_eq!(columns[0].0, "field.with.dots\u{1}child.with.dot");
    }

    #[test]
    fn test_json_fast_field_expand_dots() {
        let json_doc = json!({"field.with.dots": {"child.with.dot": "hello"}});
        let columnar_reader = test_columnar_from_jsons_aux(&[json_doc], true);
        let columns = columnar_reader.list_columns().unwrap();
        assert_eq!(columns.len(), 1);
        assert_eq!(
            columns[0].0,
            "field\u{1}with\u{1}dots\u{1}child\u{1}with\u{1}dot"
        );
    }

    #[test]
    fn test_fast_field_addrs_for_non_json_fast_fields() {
        let mut schema_builder = Schema::builder();
        let num_field = schema_builder.add_u64_field("num", FAST);
        let text_field = schema_builder.add_text_field("text", FAST);
        let json_field = schema_builder.add_json_field("json", FAST);
        let schema = schema_builder.build();

        let mut writer = FastFieldsWriter::from_schema(&schema).unwrap();
        // Non-JSON fast fields cache the address of their column in the `ColumnarWriter`; JSON fast
        // fields do not, because their columns are created dynamically per JSON path.
        assert!(writer.fast_field_addrs[num_field.field_id() as usize].is_some());
        assert!(writer.fast_field_addrs[text_field.field_id() as usize].is_some());
        assert!(writer.fast_field_addrs[json_field.field_id() as usize].is_none());

        writer
            .add_document(&crate::doc!(num_field => 42u64, text_field => "hello"))
            .unwrap();
        writer
            .add_document(&crate::doc!(num_field => 7u64, text_field => "world"))
            .unwrap();

        let mut buffer = Vec::new();
        writer.serialize(&mut buffer, None).unwrap();
        let reader = ColumnarReader::open(buffer).unwrap();
        let columns = reader.list_columns().unwrap();
        let num_column: Option<Column<u64>> = columns
            .iter()
            .find(|(name, _)| name.as_str() == "num")
            .unwrap()
            .1
            .open()
            .unwrap()
            .into();
        let num_column = num_column.unwrap();
        assert_eq!(
            num_column.values_for_doc(0).collect::<Vec<_>>(),
            vec![42u64]
        );
        assert_eq!(num_column.values_for_doc(1).collect::<Vec<_>>(), vec![7u64]);
    }

    /// Mechanism count: over a non-JSON fast field, every leaf value takes the cached-address path
    /// (`FastFieldTarget::Addr`, which never calls `get_hash`/`get_value_addr_if_key_match`) and
    /// the name-based fallback is not used.
    #[test]
    fn test_add_doc_value_uses_cached_addr_for_non_json_fast_fields() {
        let mut schema_builder = Schema::builder();
        let num_field = schema_builder.add_u64_field("num", FAST);
        let text_field = schema_builder.add_text_field("text", FAST);
        let schema = schema_builder.build();

        let mut writer = FastFieldsWriter::from_schema(&schema).unwrap();
        assert_eq!(writer.leaf_addr_path_count, 0);
        assert_eq!(writer.leaf_name_path_count, 0);

        writer
            .add_document(&crate::doc!(num_field => 42u64, text_field => "hello"))
            .unwrap();
        writer
            .add_document(&crate::doc!(num_field => 7u64, text_field => "world"))
            .unwrap();

        // 2 numerical + 2 text leaf values, all via the cached address, zero name lookups.
        assert_eq!(writer.leaf_addr_path_count, 4);
        assert_eq!(writer.leaf_name_path_count, 0);
    }
}
