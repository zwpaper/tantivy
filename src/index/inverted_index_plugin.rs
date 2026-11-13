//! The inverted index as a [`SegmentPlugin`] implementation.
//!
//! Field norms, the term dictionary, postings, and positions form a single subsystem:
//! field norms are produced by the same tokenization pass that feeds the postings, and
//! postings serialization reads the field norms back. This plugin therefore owns all four
//! files (`fieldnorm`, `term`, `idx`, `pos`) and drives them together, both while indexing
//! (`add_document`) and while merging.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use columnar::MonotonicallyMappableToU64;
use common::JsonPathWriter;
use itertools::Itertools;
use measure_time::debug_time;
use tokenizer_api::BoxTokenStream;

use crate::directory::{CompositeFile, Directory};
use crate::docset::{DocSet, TERMINATED};
use crate::error::DataCorruption;
use crate::fieldnorm::{FieldNormReader, FieldNormReaders, FieldNormsSerializer, FieldNormsWriter};
use crate::index::{Segment, SegmentComponent, SegmentReader};
use crate::indexer::doc_id_mapping::{DocIdMapping, MappingType, SegmentDocIdMapping};
use crate::indexer::indexing_term::IndexingTerm;
use crate::json_utils::{index_json_value, IndexingPositionsPerPath};
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};
use crate::postings::{
    compute_table_memory_size, serialize_postings, IndexingContext, IndexingPosition,
    InvertedIndexSerializer, PerFieldPostingsWriter, Postings, PostingsMerger, PostingsWriter,
    SegmentPostings,
};
use crate::schema::document::{Document, Value};
use crate::schema::{Field, FieldType, Schema, DATE_TIME_PRECISION_INDEXED};
use crate::space_usage::{ComponentSpaceUsage, FIELDNORMS, POSITIONS, POSTINGS, TERMDICT};
use crate::termdict::{TermMerger, TermOrdinal};
use crate::tokenizer::{FacetTokenizer, PreTokenizedStream, TextAnalyzer, Tokenizer};
use crate::{DocId, InvertedIndexReader, TantivyError};

/// Built-in segment plugin that stores and merges the inverted index.
pub struct InvertedIndexPlugin;

/// Computes the initial size of the term hash table.
///
/// Returns the recommended initial table size as a power of 2.
///
/// Note this is a very dumb way to compute log2, but it is easier to proofread that way.
fn compute_initial_table_size(per_thread_memory_budget: usize) -> crate::Result<usize> {
    let table_memory_upper_bound = per_thread_memory_budget / 3;
    (10..20) // We cap it at 2^19 = 512K capacity.
        // TODO: There are cases where this limit causes a
        // reallocation in the hashmap. Check if this affects performance.
        .map(|power| 1 << power)
        .take_while(|capacity| compute_table_memory_size(*capacity) < table_memory_upper_bound)
        .last()
        .ok_or_else(|| {
            crate::TantivyError::InvalidArgument(format!(
                "per thread memory budget (={per_thread_memory_budget}) is too small. Raise the \
                 memory budget or lower the number of threads."
            ))
        })
}

impl SegmentPlugin for InvertedIndexPlugin {
    fn extensions(&self) -> &[&str] {
        &["fieldnorm", "term", "idx", "pos"]
    }

    fn create_writer(&self, _ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        unimplemented!("InvertedIndexPlugin is a built-in; use InvertedIndexPluginWriter::new")
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        // Field norms first: postings merge reads them back from the target segment.
        merge_fieldnorms(&ctx)?;

        debug_time!("write-postings");
        debug!("write-postings");
        let target_segment = ctx.target_segment;
        let mut serializer = InvertedIndexSerializer::open(target_segment)?;
        let fieldnorm_data = target_segment.open_read(SegmentComponent::FieldNorms)?;
        let fieldnorm_readers = FieldNormReaders::open(fieldnorm_data)?;
        write_postings_merge(
            ctx.readers,
            ctx.schema,
            &mut serializer,
            fieldnorm_readers,
            ctx.doc_id_mapping,
        )?;
        serializer.close()?;
        Ok(())
    }

    fn space_usage(
        &self,
        segment_reader: &SegmentReader,
    ) -> crate::Result<BTreeMap<String, ComponentSpaceUsage>> {
        let schema = segment_reader.schema();

        let fieldnorms =
            FieldNormReaders::open(segment_reader.open_read(SegmentComponent::FieldNorms)?)?
                .space_usage(schema);
        let termdict = CompositeFile::open(&segment_reader.open_read(SegmentComponent::Terms)?)?
            .space_usage(schema);
        let postings = CompositeFile::open(&segment_reader.open_read(SegmentComponent::Postings)?)?
            .space_usage(schema);
        let positions = match segment_reader.open_read(SegmentComponent::Positions) {
            Ok(file) => CompositeFile::open(&file)?.space_usage(schema),
            Err(_) => CompositeFile::empty().space_usage(schema),
        };
        Ok(BTreeMap::from([
            (
                FIELDNORMS.to_string(),
                ComponentSpaceUsage::PerField(fieldnorms),
            ),
            (
                TERMDICT.to_string(),
                ComponentSpaceUsage::PerField(termdict),
            ),
            (
                POSTINGS.to_string(),
                ComponentSpaceUsage::PerField(postings),
            ),
            (
                POSITIONS.to_string(),
                ComponentSpaceUsage::PerField(positions),
            ),
        ]))
    }
}

// --- Plugin writer ---

/// Accumulates and serializes inverted-index data for a segment.
pub struct InvertedIndexPluginWriter {
    schema: Schema,
    per_field_postings_writers: PerFieldPostingsWriter,
    per_field_text_analyzers: Vec<TextAnalyzer>,
    fieldnorms_writer: FieldNormsWriter,
    term_buffer: IndexingTerm,
    json_path_writer: JsonPathWriter,
    json_positions_per_path: IndexingPositionsPerPath,
    ctx: IndexingContext,
    postings_serializer: InvertedIndexSerializer,
    fieldnorm_serializer: FieldNormsSerializer,
    max_doc: DocId,
}

impl InvertedIndexPluginWriter {
    pub(crate) fn new(ctx: &PluginWriterContext) -> crate::Result<Self> {
        let segment = ctx.segment;
        let schema = segment.schema();
        let tokenizer_manager = segment.index().tokenizers().clone();

        let per_field_text_analyzers = schema
            .fields()
            .map(|(_, field_entry)| {
                let text_options = match field_entry.field_type() {
                    FieldType::Str(ref text_options) => text_options.get_indexing_options(),
                    FieldType::JsonObject(ref json_object_options) => {
                        json_object_options.get_text_indexing_options()
                    }
                    _ => None,
                };
                let tokenizer_name = text_options
                    .map(|text_index_option| text_index_option.tokenizer())
                    .unwrap_or("default");
                tokenizer_manager.get(tokenizer_name).ok_or_else(|| {
                    TantivyError::SchemaError(format!(
                        "Error getting tokenizer for field: {}",
                        field_entry.name()
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let fieldnorm_path = segment.relative_path(SegmentComponent::FieldNorms);
        let fieldnorm_write = segment.index().directory().open_write(&fieldnorm_path)?;

        let table_size = compute_initial_table_size(ctx.memory_budget_in_bytes)?;
        Ok(InvertedIndexPluginWriter {
            per_field_postings_writers: PerFieldPostingsWriter::for_schema(&schema),
            per_field_text_analyzers,
            fieldnorms_writer: FieldNormsWriter::for_schema(&schema),
            term_buffer: IndexingTerm::with_capacity(16),
            json_path_writer: JsonPathWriter::default(),
            json_positions_per_path: IndexingPositionsPerPath::default(),
            ctx: IndexingContext::new(table_size),
            postings_serializer: InvertedIndexSerializer::open(segment)?,
            fieldnorm_serializer: FieldNormsSerializer::from_write(fieldnorm_write)?,
            schema,
            max_doc: 0,
        })
    }

    /// Generic, zero-copy document ingestion. The `SegmentWriter` calls this directly on the
    /// built-in inverted-index writer, so a custom `Document` is tokenized and indexed in
    /// place — no `TantivyDocument` is materialized.
    pub(crate) fn index_document<D: Document>(
        &mut self,
        doc_id: DocId,
        doc: &D,
    ) -> crate::Result<()> {
        if let Some(groups) = doc.grouped_field_ranges() {
            // The document already stores each field's values as one contiguous run, so
            // there is nothing to collect, sort or group.
            for (field, range) in groups {
                let values = range.clone().filter_map(|index| doc.value_at(index));
                self.index_field_values(doc_id, *field, values)?;
            }
        } else {
            let vals_grouped_by_field = doc
                .iter_fields_and_values()
                .sorted_by_key(|(field, _)| *field)
                .chunk_by(|(field, _)| *field);

            for (field, field_values) in &vals_grouped_by_field {
                self.index_field_values(doc_id, field, field_values.map(|el| el.1))?;
            }
        }
        self.max_doc = doc_id + 1;
        Ok(())
    }

    /// Tokenizes and indexes one field's values for one document.
    ///
    /// Shared by the generic collect/sort/group path and the pre-grouped path; `values`
    /// yields the field's values in document order.
    fn index_field_values<'a, V: Value<'a>, I: Iterator<Item = V>>(
        &mut self,
        doc_id: DocId,
        field: Field,
        values: I,
    ) -> crate::Result<()> {
        let field_entry = self.schema.get_field_entry(field);
        let make_schema_error = || {
            TantivyError::SchemaError(format!(
                "Expected a {:?} for field {:?}",
                field_entry.field_type().value_type(),
                field_entry.name()
            ))
        };
        if !field_entry.is_indexed() {
            return Ok(());
        }

        let (term_buffer, ctx) = (&mut self.term_buffer, &mut self.ctx);
            let postings_writer: &mut dyn PostingsWriter =
                self.per_field_postings_writers.get_for_field_mut(field);
            term_buffer.clear_with_field(field);

            match field_entry.field_type() {
                FieldType::Facet(_) => {
                    let mut facet_tokenizer = FacetTokenizer::default(); // this can be global
                    for value in values {
                        let value = value.as_value();

                        let facet_str = value.as_facet().ok_or_else(make_schema_error)?;
                        let mut facet_tokenizer = facet_tokenizer.token_stream(facet_str);
                        let mut indexing_position = IndexingPosition::default();
                        postings_writer.index_text(
                            doc_id,
                            &mut facet_tokenizer,
                            term_buffer,
                            ctx,
                            &mut indexing_position,
                        );
                    }
                }
                FieldType::Str(_) => {
                    let mut indexing_position = IndexingPosition::default();
                    for value in values {
                        let value = value.as_value();

                        let mut token_stream = if let Some(text) = value.as_str() {
                            let text_analyzer =
                                &mut self.per_field_text_analyzers[field.field_id() as usize];
                            text_analyzer.token_stream(text)
                        } else if let Some(tok_str) = value.into_pre_tokenized_text() {
                            BoxTokenStream::new(PreTokenizedStream::from(*tok_str.clone()))
                        } else {
                            continue;
                        };

                        assert!(term_buffer.is_empty());
                        postings_writer.index_text(
                            doc_id,
                            &mut *token_stream,
                            term_buffer,
                            ctx,
                            &mut indexing_position,
                        );
                    }
                    if field_entry.has_fieldnorms() {
                        self.fieldnorms_writer
                            .record(doc_id, field, indexing_position.num_tokens);
                    }
                }
                FieldType::U64(_) => {
                    let mut num_vals = 0;
                    for value in values {
                        let value = value.as_value();

                        num_vals += 1;
                        let u64_val = value.as_u64().ok_or_else(make_schema_error)?;
                        term_buffer.set_u64(u64_val);
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                FieldType::Date(_) => {
                    let mut num_vals = 0;
                    for value in values {
                        let value = value.as_value();

                        num_vals += 1;
                        let date_val = value.as_datetime().ok_or_else(make_schema_error)?;
                        term_buffer
                            .set_u64(date_val.truncate(DATE_TIME_PRECISION_INDEXED).to_u64());
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                FieldType::I64(_) => {
                    let mut num_vals = 0;
                    for value in values {
                        let value = value.as_value();

                        num_vals += 1;
                        let i64_val = value.as_i64().ok_or_else(make_schema_error)?;
                        term_buffer.set_i64(i64_val);
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                FieldType::F64(_) => {
                    let mut num_vals = 0;
                    for value in values {
                        let value = value.as_value();
                        num_vals += 1;
                        let f64_val = value.as_f64().ok_or_else(make_schema_error)?;
                        term_buffer.set_f64(f64_val);
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                FieldType::Bool(_) => {
                    let mut num_vals = 0;
                    for value in values {
                        let value = value.as_value();
                        num_vals += 1;
                        let bool_val = value.as_bool().ok_or_else(make_schema_error)?;
                        term_buffer.set_bool(bool_val);
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                FieldType::Bytes(_) => {
                    let mut num_vals = 0;
                    for value in values {
                        let value = value.as_value();
                        num_vals += 1;
                        let bytes = value.as_bytes().ok_or_else(make_schema_error)?;
                        term_buffer.set_bytes(bytes);
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                FieldType::JsonObject(json_options) => {
                    let text_analyzer =
                        &mut self.per_field_text_analyzers[field.field_id() as usize];

                    self.json_positions_per_path.clear();
                    self.json_path_writer
                        .set_expand_dots(json_options.is_expand_dots_enabled());
                    for json_value in values {
                        self.json_path_writer.clear();

                        index_json_value(
                            doc_id,
                            json_value,
                            text_analyzer,
                            term_buffer,
                            &mut self.json_path_writer,
                            postings_writer,
                            ctx,
                            &mut self.json_positions_per_path,
                        );
                    }
                }
                FieldType::IpAddr(_) => {
                    let mut num_vals = 0;
                    for value in values {
                        let value = value.as_value();

                        num_vals += 1;
                        let ip_addr = value.as_ip_addr().ok_or_else(make_schema_error)?;
                        term_buffer.set_ip_addr(ip_addr);
                        postings_writer.subscribe(doc_id, 0u32, term_buffer, ctx);
                    }
                    if field_entry.has_fieldnorms() {
                        self.fieldnorms_writer.record(doc_id, field, num_vals);
                    }
                }
                // These fields are not indexed; the `is_indexed()` guard above skips them.
                FieldType::TieBreaker | FieldType::Custom(_) => {
                    unreachable!("the inverted index does not support this field type")
                }
            }
        Ok(())
    }
}

impl PluginWriter for InvertedIndexPluginWriter {
    fn serialize(
        mut self: Box<Self>,
        segment: &Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        // Field norms first: postings serialization reads them back below.
        self.fieldnorms_writer.fill_up_to_max_doc(self.max_doc);
        self.fieldnorms_writer
            .serialize(self.fieldnorm_serializer, doc_id_map)
            .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;

        let fieldnorm_data = segment.open_read(SegmentComponent::FieldNorms)?;
        let fieldnorm_readers = FieldNormReaders::open(fieldnorm_data)?;
        serialize_postings(
            self.ctx,
            self.schema,
            &self.per_field_postings_writers,
            fieldnorm_readers,
            doc_id_map,
            &mut self.postings_serializer,
        )?;
        self.postings_serializer.close()?;
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.ctx.mem_usage() + self.fieldnorms_writer.mem_usage()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// --- Field norm merge ---

fn merge_fieldnorms(ctx: &PluginMergeContext) -> crate::Result<()> {
    let path = ctx
        .target_segment
        .relative_path(SegmentComponent::FieldNorms);
    let write = ctx.target_segment.index().directory().open_write(&path)?;
    let mut serializer = FieldNormsSerializer::from_write(write)?;

    let schema = ctx.schema;
    let fields = FieldNormsWriter::fields_with_fieldnorm(schema);
    let max_doc: usize = ctx
        .readers
        .iter()
        .map(|reader| reader.num_docs() as usize)
        .sum();
    let mut fieldnorms_data = Vec::with_capacity(max_doc);

    for field in fields {
        fieldnorms_data.clear();
        let fieldnorms_readers: Vec<FieldNormReader> = ctx
            .readers
            .iter()
            .map(|reader| reader.get_fieldnorms_reader(field))
            .collect::<Result<_, _>>()?;
        for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
            let reader = &fieldnorms_readers[old_doc_addr.segment_ord as usize];
            let fieldnorm_id = reader.fieldnorm_id(old_doc_addr.doc_id);
            fieldnorms_data.push(fieldnorm_id);
        }
        serializer.serialize_field(field, &fieldnorms_data)?;
    }
    serializer.close()?;
    Ok(())
}

// --- Postings merge helpers (moved from IndexMerger) ---

struct DeltaComputer {
    buffer: Vec<u32>,
}

impl DeltaComputer {
    fn new() -> DeltaComputer {
        DeltaComputer {
            buffer: vec![0u32; 512],
        }
    }

    fn compute_delta(&mut self, positions: &[u32]) -> &[u32] {
        if positions.len() > self.buffer.len() {
            self.buffer.resize(positions.len(), 0u32);
        }
        let mut last_pos = 0u32;
        for (cur_pos, dest) in positions.iter().cloned().zip(self.buffer.iter_mut()) {
            *dest = cur_pos - last_pos;
            last_pos = cur_pos;
        }
        &self.buffer[..positions.len()]
    }
}

fn estimate_total_num_tokens_in_single_segment(
    reader: &SegmentReader,
    field: Field,
) -> crate::Result<u64> {
    if !reader.has_deletes() {
        return Ok(reader.inverted_index(field)?.total_num_tokens());
    }
    if let Some(fieldnorm_reader) = reader.fieldnorms_readers().get_field(field)? {
        let mut count: [usize; 256] = [0; 256];
        for doc in reader.doc_ids_alive() {
            let fieldnorm_id = fieldnorm_reader.fieldnorm_id(doc);
            count[fieldnorm_id as usize] += 1;
        }
        let total_num_tokens = count
            .iter()
            .cloned()
            .enumerate()
            .map(|(fieldnorm_ord, count)| {
                count as u64 * u64::from(FieldNormReader::id_to_fieldnorm(fieldnorm_ord as u8))
            })
            .sum::<u64>();
        return Ok(total_num_tokens);
    }
    let segment_num_tokens = reader.inverted_index(field)?.total_num_tokens();
    if reader.max_doc() == 0 {
        return Ok(0u64);
    }
    let ratio = reader.num_docs() as f64 / reader.max_doc() as f64;
    Ok((segment_num_tokens as f64 * ratio) as u64)
}

fn estimate_total_num_tokens(readers: &[SegmentReader], field: Field) -> crate::Result<u64> {
    let mut total_num_tokens: u64 = 0;
    for reader in readers {
        total_num_tokens += estimate_total_num_tokens_in_single_segment(reader, field)?;
    }
    Ok(total_num_tokens)
}

/// Whether the merge needs a materialized `merged_doc_id_map`.
///
/// A delete-free `Stacked` mapping is affine (`mapped = segment_base_doc_id[segment_ord] + local`),
/// so its table is never built; `StackedWithDeletes` and `Shuffled` still need it.
fn needs_merged_doc_id_map(mapping_type: MappingType) -> bool {
    mapping_type != MappingType::Stacked
}

/// Whether the merge needs a [`PostingsMerger`]. Only `Shuffled` reorders documents across
/// segments; the two stacked mappings walk each segment's postings in order.
fn needs_postings_merger(mapping_type: MappingType) -> bool {
    mapping_type == MappingType::Shuffled
}

/// The mapped doc id of local doc `d` in segment `s` is `segment_base_doc_id[s] + d` for a
/// delete-free `Stacked` merge: the running sum of the segments' `max_doc()` (which equals
/// `num_docs()` when there are no deletes). It is field-independent, so it is computed once per
/// merge.
fn compute_segment_base_doc_id(readers: &[SegmentReader]) -> Vec<DocId> {
    let mut base = Vec::with_capacity(readers.len());
    let mut next_base: DocId = 0;
    for reader in readers {
        base.push(next_base);
        next_base += reader.max_doc();
    }
    base
}

/// Builds `merged_doc_id_map[segment_ord][local_doc] = Some(mapped_doc)` for the mapping types that
/// need it. Returns an empty vector for a delete-free `Stacked` mapping (see
/// [`needs_merged_doc_id_map`]).
fn build_merged_doc_id_map(
    readers: &[SegmentReader],
    doc_id_mapping: &SegmentDocIdMapping,
) -> Vec<Vec<Option<DocId>>> {
    if !needs_merged_doc_id_map(doc_id_mapping.mapping_type()) {
        return Vec::new();
    }
    let mut merged_doc_id_map: Vec<Vec<Option<DocId>>> = readers
        .iter()
        .map(|reader| vec![None; reader.max_doc() as usize])
        .collect();
    for (new_doc_id, old_doc_addr) in doc_id_mapping.iter_old_doc_addrs().enumerate() {
        let segment_map = &mut merged_doc_id_map[old_doc_addr.segment_ord as usize];
        segment_map[old_doc_addr.doc_id as usize] = Some(new_doc_id as DocId);
    }
    merged_doc_id_map
}

fn write_postings_for_field(
    readers: &[SegmentReader],
    schema: &Schema,
    indexed_field: Field,
    serializer: &mut InvertedIndexSerializer,
    fieldnorm_reader: Option<FieldNormReader>,
    doc_id_mapping: &SegmentDocIdMapping,
    segment_base_doc_id: &[DocId],
) -> crate::Result<()> {
    debug_time!("write-postings-for-field");
    let mut positions_buffer: Vec<u32> = Vec::with_capacity(1_000);
    let mut delta_computer = DeltaComputer::new();

    let mut max_term_ords: Vec<TermOrdinal> = Vec::new();

    let field_readers: Vec<Arc<InvertedIndexReader>> = readers
        .iter()
        .map(|reader| reader.inverted_index(indexed_field))
        .collect::<crate::Result<Vec<_>>>()?;

    let mut field_term_streams = Vec::new();
    for field_reader in &field_readers {
        let terms = field_reader.terms();
        field_term_streams.push(terms.stream()?);
        max_term_ords.push(terms.num_terms() as u64);
    }

    let mut merged_terms = TermMerger::new(field_term_streams);

    let mapping_type = doc_id_mapping.mapping_type();
    let merged_doc_id_map = build_merged_doc_id_map(readers, doc_id_mapping);

    let total_num_tokens: u64 = estimate_total_num_tokens(readers, indexed_field)?;

    let mut field_serializer =
        serializer.new_field(indexed_field, total_num_tokens, fieldnorm_reader)?;

    let field_entry = schema.get_field_entry(indexed_field);

    let segment_postings_option = field_entry.field_type().get_index_record_option().expect(
        "Encountered a field that is not supposed to be
                     indexed. Have you modified the schema?",
    );

    // One reusable cursor per source segment, allocated once per field. A term resets the
    // cursors of its active segments in place instead of building a fresh `SegmentPostings`
    // per (term, segment) and moving it in and out of a per-term `Vec`.
    let mut segment_cursors: Vec<SegmentPostings> = (0..readers.len())
        .map(|_| SegmentPostings::empty())
        .collect();
    let mut active_segment_ords: Vec<usize> = Vec::new();
    let mut merger =
        needs_postings_merger(mapping_type).then(|| PostingsMerger::new(&merged_doc_id_map));

    while merged_terms.advance() {
        active_segment_ords.clear();
        let term_bytes: &[u8] = merged_terms.key();

        let mut total_doc_freq = 0;

        for (segment_ord, term_info) in merged_terms.current_segment_ords_and_term_infos() {
            let segment_reader = &readers[segment_ord];
            let inverted_index: &InvertedIndexReader = &field_readers[segment_ord];
            let segment_postings = &mut segment_cursors[segment_ord];
            inverted_index.read_postings_from_terminfo_into(
                &term_info,
                segment_postings_option,
                segment_postings,
            )?;
            let alive_bitset_opt = segment_reader.alive_bitset();
            let doc_freq = if let Some(alive_bitset) = alive_bitset_opt {
                segment_postings.doc_freq_given_deletes(alive_bitset)
            } else {
                segment_postings.doc_freq()
            };
            if doc_freq > 0u32 {
                total_doc_freq += doc_freq;
                active_segment_ords.push(segment_ord);
            }
        }

        if total_doc_freq == 0u32 {
            continue;
        }

        assert!(!active_segment_ords.is_empty());

        let has_term_freq = {
            let has_term_freq = !segment_cursors[active_segment_ords[0]]
                .block_cursor
                .freqs()
                .is_empty();
            for &segment_ord in &active_segment_ords[1..] {
                if has_term_freq
                    == segment_cursors[segment_ord]
                        .block_cursor
                        .freqs()
                        .is_empty()
                {
                    return Err(DataCorruption::comment_only(
                        "Term freqs are inconsistent across segments",
                    )
                    .into());
                }
            }
            has_term_freq
        };

        field_serializer.new_term(term_bytes, total_doc_freq, has_term_freq)?;

        match mapping_type {
            MappingType::Stacked => {
                // Delete-free stacked merge: the mapping is affine (`base[s] + local`), so remap
                // arithmetically instead of loading an `Option<DocId>` table per posting.
                for &segment_ord in &active_segment_ords {
                    let postings = &mut segment_cursors[segment_ord];
                    let base = segment_base_doc_id[segment_ord];
                    if has_term_freq {
                        while postings.doc() != TERMINATED {
                            let doc = base + postings.doc();
                            postings.positions(&mut positions_buffer);
                            let term_freq = postings.term_freq();
                            let delta_positions = delta_computer.compute_delta(&positions_buffer);
                            field_serializer.write_doc(doc, term_freq, delta_positions);
                            postings.advance();
                        }
                    } else {
                        // `Basic` postings carry no per-doc body, so append each already-decoded
                        // doc-id block in bulk instead of calling `write_doc` once per doc.
                        loop {
                            let docs = postings.block_cursor.docs();
                            if docs.is_empty() {
                                break;
                            }
                            field_serializer.write_doc_block(docs, base);
                            postings.block_cursor.advance();
                        }
                    }
                }
            }
            MappingType::StackedWithDeletes => {
                // Deleted docs must still be skipped, so this keeps the mapped table.
                for &segment_ord in &active_segment_ords {
                    let postings = &mut segment_cursors[segment_ord];
                    let mapping = &merged_doc_id_map[segment_ord];
                    while let Some(doc) = crate::postings::next_mapped_doc(postings, mapping) {
                        let term_freq = if has_term_freq {
                            postings.positions(&mut positions_buffer);
                            postings.term_freq()
                        } else {
                            positions_buffer.clear();
                            0
                        };
                        let delta_positions = delta_computer.compute_delta(&positions_buffer);
                        field_serializer.write_doc(doc, term_freq, delta_positions);
                        postings.advance();
                    }
                }
            }
            MappingType::Shuffled => {
                let merger = merger
                    .as_mut()
                    .expect("Shuffled doc-id mapping needs a PostingsMerger");
                // Not the merge hot path: `PostingsMerger` takes its cursors by value, so hand it
                // copies rather than the reusable per-segment cursors.
                merger.reset(
                    active_segment_ords
                        .iter()
                        .map(|&segment_ord| (segment_ord, segment_cursors[segment_ord].clone())),
                );
                while merger.advance() {
                    let term_freq = if has_term_freq {
                        merger.positions(&mut positions_buffer);
                        merger.term_freq()
                    } else {
                        positions_buffer.clear();
                        0
                    };
                    let delta_positions = delta_computer.compute_delta(&positions_buffer);
                    field_serializer.write_doc(merger.doc(), term_freq, delta_positions);
                }
            }
        }
        field_serializer.close_term()?;
    }
    field_serializer.close()?;
    Ok(())
}

fn write_postings_merge(
    readers: &[SegmentReader],
    schema: &Schema,
    serializer: &mut InvertedIndexSerializer,
    fieldnorm_readers: FieldNormReaders,
    doc_id_mapping: &SegmentDocIdMapping,
) -> crate::Result<()> {
    // Field-independent: the per-segment base doc id of a delete-free `Stacked` merge. Computed
    // once per merge instead of rebuilding a `Vec<Vec<Option<DocId>>>` per indexed field.
    let segment_base_doc_id = compute_segment_base_doc_id(readers);
    for (field, field_entry) in schema.fields() {
        let fieldnorm_reader = fieldnorm_readers.get_field(field)?;
        if field_entry.is_indexed() {
            write_postings_for_field(
                readers,
                schema,
                field,
                serializer,
                fieldnorm_reader,
                doc_id_mapping,
                &segment_base_doc_id,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        build_merged_doc_id_map, compute_segment_base_doc_id, needs_merged_doc_id_map,
        needs_postings_merger,
    };
    use crate::directory::RamDirectory;
    use crate::docset::{DocSet, TERMINATED};
    use crate::index::SegmentReader;
    use crate::indexer::doc_id_mapping::{MappingType, SegmentDocIdMapping};
    use crate::indexer::{merge_filtered_segments, NoMergePolicy};
    use crate::postings::SegmentPostings;
    use crate::schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions};
    use crate::{DocAddress, DocId, Index, Segment, Term};

    #[cfg(not(feature = "compare_hash_only"))]
    #[test]
    fn test_hashmap_size() {
        use super::compute_initial_table_size;

        assert_eq!(compute_initial_table_size(100_000).unwrap(), 1 << 12);
        assert_eq!(compute_initial_table_size(1_000_000).unwrap(), 1 << 15);
        assert_eq!(compute_initial_table_size(15_000_000).unwrap(), 1 << 19);
        assert_eq!(compute_initial_table_size(1_000_000_000).unwrap(), 1 << 19);
        assert_eq!(compute_initial_table_size(4_000_000_000).unwrap(), 1 << 19);
    }

    /// The old unconditional map build (before the `Stacked` arithmetic specialization), kept here
    /// as the reference the new path must match.
    fn reference_merged_doc_id_map(
        readers: &[SegmentReader],
        doc_id_mapping: &SegmentDocIdMapping,
    ) -> Vec<Vec<Option<DocId>>> {
        let mut map: Vec<Vec<Option<DocId>>> = readers
            .iter()
            .map(|reader| vec![None; reader.max_doc() as usize])
            .collect();
        for (new_doc_id, old_doc_addr) in doc_id_mapping.iter_old_doc_addrs().enumerate() {
            map[old_doc_addr.segment_ord as usize][old_doc_addr.doc_id as usize] =
                Some(new_doc_id as DocId);
        }
        map
    }

    /// The single raw term of every local doc of a segment (empty for a doc with no posting).
    fn doc_terms(reader: &SegmentReader, field: Field) -> Vec<String> {
        let inverted_index = reader.inverted_index(field).unwrap();
        let mut out = vec![String::new(); reader.max_doc() as usize];
        let mut term_stream = inverted_index.terms().stream().unwrap();
        while term_stream.advance() {
            let term = String::from_utf8(term_stream.key().to_vec()).unwrap();
            let mut postings = inverted_index
                .read_postings(
                    &Term::from_field_text(field, &term),
                    IndexRecordOption::Basic,
                )
                .unwrap()
                .unwrap();
            while postings.doc() != TERMINATED {
                out[postings.doc() as usize] = term.clone();
                postings.advance();
            }
        }
        out
    }

    /// The merged `term -> doc ids` implied by `doc_id_mapping`, independent of the (arbitrary)
    /// order in which `searchable_segments()` returns the source segments.
    fn expected_postings(
        readers: &[SegmentReader],
        field: Field,
        doc_id_mapping: &SegmentDocIdMapping,
    ) -> BTreeMap<String, Vec<DocId>> {
        let reference = reference_merged_doc_id_map(readers, doc_id_mapping);
        let mut expected: BTreeMap<String, Vec<DocId>> = BTreeMap::new();
        for (segment_ord, reader) in readers.iter().enumerate() {
            for (local, term) in doc_terms(reader, field).iter().enumerate() {
                if let Some(new_doc_id) = reference[segment_ord][local] {
                    expected.entry(term.clone()).or_default().push(new_doc_id);
                }
            }
        }
        expected
    }

    #[test]
    fn test_grouped_document_indexes_identically() {
        use crate::schema::{document::TantivyDocument, FAST, INDEXED};

        // Builds a one-segment index of the same documents, optionally installing the
        // grouped sidecar before each `add_document`. The values are added in an order that
        // is neither field-id nor lexical, and `body` is multivalued.
        fn build(seal: bool) -> (Index, Field, Field) {
            let mut schema_builder = Schema::builder();
            let body = schema_builder.add_text_field(
                "body",
                TextOptions::default().set_indexing_options(
                    TextFieldIndexing::default()
                        .set_tokenizer("default")
                        .set_index_option(IndexRecordOption::WithFreqsAndPositions),
                ),
            );
            let num = schema_builder.add_u64_field("num", INDEXED | FAST);
            let index = Index::create_in_ram(schema_builder.build());
            let mut writer = index.writer_with_num_threads(1, 15_000_000).unwrap();
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for i in 0..3u64 {
                let mut doc = TantivyDocument::new();
                doc.add_u64(num, i);
                doc.add_text(body, format!("alpha beta{i}"));
                doc.add_text(body, "gamma");
                if seal {
                    assert!(doc.seal_groups());
                }
                writer.add_document(doc).unwrap();
            }
            writer.commit().unwrap();
            (index, body, num)
        }

        let (grouped, grouped_body, grouped_num) = build(true);
        let (generic, generic_body, generic_num) = build(false);
        assert_eq!(grouped_body, generic_body);
        assert_eq!(grouped_num, generic_num);

        let grouped_segments = grouped.searchable_segments().unwrap();
        let generic_segments = generic.searchable_segments().unwrap();
        assert_eq!(grouped_segments.len(), 1);
        assert_eq!(generic_segments.len(), 1);
        let grouped_reader = SegmentReader::open(&grouped_segments[0]).unwrap();
        let generic_reader = SegmentReader::open(&generic_segments[0]).unwrap();
        assert_eq!(grouped_reader.num_docs(), generic_reader.num_docs());

        // Term dictionaries and postings are identical for both the text and the numeric field.
        for (field, reader_a, reader_b) in [
            (grouped_body, &grouped_reader, &generic_reader),
            (grouped_num, &grouped_reader, &generic_reader),
        ] {
            assert_eq!(
                postings_by_term(reader_a, field),
                postings_by_term(reader_b, field)
            );
        }

        // Field norms (token counts) are identical.
        let grouped_norms = grouped_reader.get_fieldnorms_reader(grouped_body).unwrap();
        let generic_norms = generic_reader.get_fieldnorms_reader(generic_body).unwrap();
        for doc in 0..grouped_reader.num_docs() {
            assert_eq!(
                grouped_norms.fieldnorm_id(doc),
                generic_norms.fieldnorm_id(doc)
            );
        }
    }

    /// Micro-bench for the pre-check: the grouping pass the plugin runs per document,
    /// legacy (collect + sort + lazy group) versus the pre-grouped sidecar. Run with
    /// `cargo test --release --lib test_grouped_grouping_microbench -- --nocapture`.
    #[test]
    fn test_grouped_grouping_microbench() {
        use std::time::Instant;

        use itertools::Itertools;

        use crate::schema::document::{Document, Value};
        use crate::schema::{document::TantivyDocument, FAST, INDEXED, TEXT};

        // ~100 fields, like the ClickBench schema.
        let mut schema_builder = Schema::builder();
        let text_fields: Vec<Field> = (0..50)
            .map(|i| schema_builder.add_text_field(&format!("t{i:03}"), TEXT))
            .collect();
        let num_fields: Vec<Field> = (0..50)
            .map(|i| schema_builder.add_u64_field(&format!("n{i:03}"), INDEXED | FAST))
            .collect();

        let make_doc = |seal: bool| {
            let mut doc = TantivyDocument::new();
            // Deliberately added in an order that is neither field-id nor lexical.
            for (i, field) in num_fields.iter().enumerate().rev() {
                doc.add_u64(*field, i as u64);
            }
            for (i, field) in text_fields.iter().enumerate().rev() {
                doc.add_text(*field, format!("value{i}"));
            }
            if seal {
                assert!(doc.seal_groups());
            }
            doc
        };
        let grouped_doc = make_doc(true);
        let generic_doc = make_doc(false);

        let iters = 200_000u64;
        let time = |doc: &TantivyDocument, grouped: bool| {
            let start = Instant::now();
            let mut sink = 0usize;
            for _ in 0..iters {
                if grouped {
                    for (field, range) in doc.grouped_field_ranges().unwrap() {
                        for index in range.clone() {
                            let value = doc.value_at(index).unwrap();
                            sink += field.field_id() as usize
                                + value.as_u64().unwrap_or(0) as usize
                                + value.as_str().map_or(0, str::len);
                        }
                    }
                } else {
                    let groups = doc
                        .iter_fields_and_values()
                        .sorted_by_key(|(field, _)| *field)
                        .chunk_by(|(field, _)| *field);
                    for (field, values) in &groups {
                        for value in values.map(|el| el.1) {
                            sink += field.field_id() as usize
                                + value.as_u64().unwrap_or(0) as usize
                                + value.as_str().map_or(0, str::len);
                        }
                    }
                }
            }
            std::hint::black_box(sink);
            start.elapsed().as_secs_f64() / iters as f64
        };
        let generic_s = time(&generic_doc, false);
        let grouped_s = time(&grouped_doc, true);
        println!(
            "grouping microbench: legacy {:.1} ns/doc, grouped {:.1} ns/doc, dI {:.1} s/100M",
            generic_s * 1e9,
            grouped_s * 1e9,
            (generic_s - grouped_s) * 1e8
        );
        assert!(grouped_s < generic_s);
    }

    /// One in-RAM segment per entry of `docs_per_segment`, one raw indexed text field.
    fn build_raw_segments(docs_per_segment: &[&[&str]]) -> (Index, Vec<Segment>) {
        let mut schema_builder = Schema::builder();
        let body = schema_builder.add_text_field(
            "body",
            TextOptions::default().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_index_option(IndexRecordOption::Basic),
            ),
        );
        let index = Index::create_in_ram(schema_builder.build());
        {
            let mut writer = index.writer_with_num_threads(1, 15_000_000).unwrap();
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for docs in docs_per_segment {
                for value in *docs {
                    writer.add_document(doc!(body => *value)).unwrap();
                }
                writer.commit().unwrap();
            }
        }
        let segments = index.searchable_segments().unwrap();
        (index, segments)
    }

    fn open_readers(segments: &[Segment]) -> Vec<SegmentReader> {
        segments
            .iter()
            .map(SegmentReader::open)
            .collect::<crate::Result<Vec<_>>>()
            .unwrap()
    }

    /// The old doc addresses in merged order: segments stacked, alive docs in order.
    fn concatenated_doc_addrs(readers: &[SegmentReader]) -> Vec<DocAddress> {
        readers
            .iter()
            .enumerate()
            .flat_map(|(segment_ord, reader)| {
                reader.doc_ids_alive().map(move |doc_id| DocAddress {
                    segment_ord: segment_ord as u32,
                    doc_id,
                })
            })
            .collect()
    }

    /// `term -> doc ids` of a single-segment index.
    fn postings_by_term(reader: &SegmentReader, field: Field) -> BTreeMap<String, Vec<DocId>> {
        let inverted_index = reader.inverted_index(field).unwrap();
        let mut out = BTreeMap::new();
        let mut term_stream = inverted_index.terms().stream().unwrap();
        while term_stream.advance() {
            let term = String::from_utf8(term_stream.key().to_vec()).unwrap();
            let mut postings = inverted_index
                .read_postings(
                    &Term::from_field_text(field, &term),
                    IndexRecordOption::Basic,
                )
                .unwrap()
                .unwrap();
            let mut docs = Vec::new();
            while postings.doc() != TERMINATED {
                docs.push(postings.doc());
                postings.advance();
            }
            out.insert(term, docs);
        }
        out
    }

    fn merge_all(index: &Index, segments: &[Segment]) -> Index {
        merge_filtered_segments(
            segments,
            index.settings().clone(),
            vec![None; segments.len()],
            RamDirectory::create(),
        )
        .unwrap()
    }

    #[test]
    fn test_merge_stacked_matches_reference() {
        let docs_per_segment: Vec<&[&str]> = vec![&["a", "b", "a"], &["a", "c"], &["b", "a"]];
        let (index, segments) = build_raw_segments(&docs_per_segment);
        let readers = open_readers(&segments);
        let body = index.schema().get_field("body").unwrap();

        let stacked = SegmentDocIdMapping::new(
            concatenated_doc_addrs(&readers),
            MappingType::Stacked,
            vec![None; readers.len()],
        );

        // The `Stacked` path builds neither the per-segment map nor a `PostingsMerger`.
        assert!(!needs_merged_doc_id_map(MappingType::Stacked));
        assert!(!needs_postings_merger(MappingType::Stacked));
        assert!(build_merged_doc_id_map(&readers, &stacked).is_empty());

        // `base[s] + local` equals the old per-field map for every alive doc.
        let base = compute_segment_base_doc_id(&readers);
        let reference = reference_merged_doc_id_map(&readers, &stacked);
        for (new_doc_id, addr) in stacked.iter_old_doc_addrs().enumerate() {
            let segment_ord = addr.segment_ord as usize;
            assert_eq!(
                reference[segment_ord][addr.doc_id as usize],
                Some(new_doc_id as DocId)
            );
            assert_eq!(base[segment_ord] + addr.doc_id, new_doc_id as DocId);
        }

        // A forced `Shuffled` control still needs the map, and there `base[s] + local` is wrong.
        let mut shuffled_addrs = concatenated_doc_addrs(&readers);
        shuffled_addrs.reverse();
        let shuffled = SegmentDocIdMapping::new(
            shuffled_addrs,
            MappingType::Shuffled,
            vec![None; readers.len()],
        );
        assert!(needs_merged_doc_id_map(MappingType::Shuffled));
        assert!(needs_postings_merger(MappingType::Shuffled));
        assert!(!build_merged_doc_id_map(&readers, &shuffled).is_empty());
        assert!(shuffled
            .iter_old_doc_addrs()
            .enumerate()
            .any(
                |(new_doc_id, addr)| base[addr.segment_ord as usize] + addr.doc_id
                    != new_doc_id as DocId
            ));

        // The merged postings equal the plain concatenation of the source segments.
        let merged = merge_all(&index, &segments);
        let merged_segments = merged.searchable_segments().unwrap();
        assert_eq!(merged_segments.len(), 1);
        let merged_reader = SegmentReader::open(&merged_segments[0]).unwrap();
        assert_eq!(
            postings_by_term(&merged_reader, body),
            expected_postings(&readers, body, &stacked)
        );
    }

    #[test]
    fn test_merge_stacked_with_deletes_skips_deleted_docs() {
        let docs_per_segment: Vec<&[&str]> = vec![&["x0", "x1", "x2"], &["y0", "y1"], &["z0"]];
        let (index, _) = build_raw_segments(&docs_per_segment);
        let body = index.schema().get_field("body").unwrap();
        {
            let mut writer: crate::IndexWriter =
                index.writer_with_num_threads(1, 15_000_000).unwrap();
            writer.set_merge_policy(Box::new(NoMergePolicy));
            writer.delete_term(Term::from_field_text(body, "x1"));
            writer.commit().unwrap();
        }
        let segments = index.searchable_segments().unwrap();
        let readers = open_readers(&segments);
        assert!(readers.iter().any(SegmentReader::has_deletes));

        // A delete forces `StackedWithDeletes`, which keeps the mapped table: the naive `max_doc`
        // base would leave a hole where the deleted doc was, so the arithmetic path must not be
        // used here.
        let mapping = SegmentDocIdMapping::new(
            concatenated_doc_addrs(&readers),
            MappingType::StackedWithDeletes,
            vec![None; readers.len()],
        );
        assert!(needs_merged_doc_id_map(MappingType::StackedWithDeletes));
        assert!(!needs_postings_merger(MappingType::StackedWithDeletes));
        assert!(!build_merged_doc_id_map(&readers, &mapping).is_empty());

        let merged = merge_all(&index, &segments);
        let merged_segments = merged.searchable_segments().unwrap();
        assert_eq!(merged_segments.len(), 1);
        let merged_reader = SegmentReader::open(&merged_segments[0]).unwrap();
        assert_eq!(merged_reader.num_docs(), 5);
        let got = postings_by_term(&merged_reader, body);

        // The deleted doc is skipped and the survivors are packed with no hole.
        assert_eq!(got.get("x1"), None);
        assert_eq!(got.len(), 5);
        let mut all_docs: Vec<DocId> = got.values().flatten().copied().collect();
        all_docs.sort_unstable();
        assert_eq!(all_docs, vec![0, 1, 2, 3, 4]);
        assert_eq!(got, expected_postings(&readers, body, &mapping));
    }

    /// `term -> [(doc_id, term_freq, positions)]` of a single-segment index, read with
    /// frequencies and positions (only valid for a `WithFreqsAndPositions` field).
    fn postings_with_positions(
        reader: &SegmentReader,
        field: Field,
    ) -> BTreeMap<String, Vec<(DocId, u32, Vec<u32>)>> {
        use crate::postings::Postings;
        let inverted_index = reader.inverted_index(field).unwrap();
        let mut out = BTreeMap::new();
        let mut term_stream = inverted_index.terms().stream().unwrap();
        while term_stream.advance() {
            let term = String::from_utf8(term_stream.key().to_vec()).unwrap();
            let mut postings = inverted_index
                .read_postings(
                    &Term::from_field_text(field, &term),
                    IndexRecordOption::WithFreqsAndPositions,
                )
                .unwrap()
                .unwrap();
            let mut rows = Vec::new();
            while postings.doc() != TERMINATED {
                let mut positions = Vec::new();
                postings.positions(&mut positions);
                rows.push((postings.doc(), postings.term_freq(), positions));
                postings.advance();
            }
            out.insert(term, rows);
        }
        out
    }

    /// The source postings (with frequencies and positions) remapped into merged order by the
    /// `Stacked` affine mapping `base[s] + local`.
    fn expected_postings_with_positions(
        readers: &[SegmentReader],
        field: Field,
    ) -> BTreeMap<String, Vec<(DocId, u32, Vec<u32>)>> {
        let base = compute_segment_base_doc_id(readers);
        let mut out: BTreeMap<String, Vec<(DocId, u32, Vec<u32>)>> = BTreeMap::new();
        for (segment_ord, reader) in readers.iter().enumerate() {
            for (term, rows) in postings_with_positions(reader, field) {
                for (local_doc, term_freq, positions) in rows {
                    out.entry(term.clone())
                        .or_default()
                        .push((base[segment_ord] + local_doc, term_freq, positions));
                }
            }
        }
        for rows in out.values_mut() {
            rows.sort_by_key(|(doc, _, _)| *doc);
        }
        out
    }

    /// The source postings remapped into merged order by the `Stacked` affine mapping
    /// `base[s] + local`.
    fn expected_postings_by_term(
        readers: &[SegmentReader],
        field: Field,
    ) -> BTreeMap<String, Vec<DocId>> {
        let base = compute_segment_base_doc_id(readers);
        let mut out: BTreeMap<String, Vec<DocId>> = BTreeMap::new();
        for (segment_ord, reader) in readers.iter().enumerate() {
            for (term, docs) in postings_by_term(reader, field) {
                for local_doc in docs {
                    out.entry(term.clone())
                        .or_default()
                        .push(base[segment_ord] + local_doc);
                }
            }
        }
        for docs in out.values_mut() {
            docs.sort_unstable();
        }
        out
    }

    /// A `Basic` field (bulk `write_doc_block` path) and a `WithFreqsAndPositions` field (per-doc
    /// path) merged together, with a common term of doc frequency 180 so the 128-doc compression
    /// block is flushed and a skip list is written. The merged postings must match the sources
    /// exactly, positions included.
    #[test]
    fn test_merge_stacked_bulk_and_positions_fields() {
        use crate::schema::document::TantivyDocument;

        let mut schema_builder = Schema::builder();
        let basic = schema_builder.add_text_field(
            "basic",
            TextOptions::default().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_index_option(IndexRecordOption::Basic),
            ),
        );
        let body = schema_builder.add_text_field(
            "body",
            TextOptions::default().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            ),
        );
        let index = Index::create_in_ram(schema_builder.build());
        {
            let mut writer = index.writer_with_num_threads(1, 15_000_000).unwrap();
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for segment in 0..3u32 {
                for doc in 0..60u32 {
                    let mut document = TantivyDocument::new();
                    document.add_text(basic, "common");
                    document.add_text(basic, format!("unique_{segment}_{doc}"));
                    document.add_text(body, "alpha beta gamma");
                    document.add_text(body, format!("word_{segment}_{doc}"));
                    writer.add_document(document).unwrap();
                }
                writer.commit().unwrap();
            }
        }
        let segments = index.searchable_segments().unwrap();
        assert_eq!(segments.len(), 3);
        let readers = open_readers(&segments);

        let merged = merge_all(&index, &segments);
        let merged_segments = merged.searchable_segments().unwrap();
        assert_eq!(merged_segments.len(), 1);
        let merged_reader = SegmentReader::open(&merged_segments[0]).unwrap();

        // `common` has doc frequency 180 > 128: the bulk path flushed a full block and wrote the
        // skip list. `postings_by_term` reads with `Basic`, so it compares the doc ids only.
        assert_eq!(
            postings_by_term(&merged_reader, basic),
            expected_postings_by_term(&readers, basic)
        );
        // The positions field keeps the per-doc path: same docs, frequencies and positions.
        assert_eq!(
            postings_by_term(&merged_reader, body),
            expected_postings_by_term(&readers, body)
        );
        assert_eq!(
            postings_with_positions(&merged_reader, body),
            expected_postings_with_positions(&readers, body)
        );
    }

    /// Compares two cursors doc by doc: doc ids, term frequencies and positions.
    ///
    /// Positions are only compared when the term carries frequencies: a JSON numeric term has
    /// none (the reader downgrades it to `Basic`), and `Postings::positions` must not be called
    /// on such a term.
    fn assert_same_postings(mut left: SegmentPostings, mut right: SegmentPostings) {
        use crate::postings::Postings;
        loop {
            assert_eq!(left.doc(), right.doc());
            if left.doc() == TERMINATED {
                break;
            }
            assert_eq!(left.term_freq(), right.term_freq());
            assert_eq!(
                left.block_cursor.freqs().is_empty(),
                right.block_cursor.freqs().is_empty()
            );
            if !left.block_cursor.freqs().is_empty() {
                let mut left_positions = Vec::new();
                let mut right_positions = Vec::new();
                left.positions(&mut left_positions);
                right.positions(&mut right_positions);
                assert_eq!(left_positions, right_positions);
            }
            left.advance();
            right.advance();
        }
    }

    /// The in-place `read_postings_from_terminfo_into` must produce exactly the cursor the owned
    /// `read_postings_from_terminfo` builds, for every term of a text field (positions), a numeric
    /// field (no frequencies) and a JSON field (mixed text/numeric terms, where `open` downgrades
    /// `record_option` to `Basic` per term). One cursor is reused across all terms, as the merge
    /// does per segment.
    #[test]
    fn test_read_postings_from_terminfo_into_matches_owned() -> crate::Result<()> {
        use crate::schema::document::TantivyDocument;
        use crate::schema::{OwnedValue, INDEXED, TEXT};

        let mut schema_builder = Schema::builder();
        let body = schema_builder.add_text_field(
            "body",
            TextOptions::default().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            ),
        );
        let num = schema_builder.add_u64_field("num", INDEXED);
        let json = schema_builder.add_json_field("json", TEXT);
        let index = Index::create_in_ram(schema_builder.build());

        let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
        writer.set_merge_policy(Box::new(NoMergePolicy));
        // More than one compression block for the common terms, so the skip list and the
        // bitpacked frequency/position decoders are exercised, plus short lists for the rest.
        for i in 0..300u64 {
            let mut doc = TantivyDocument::new();
            doc.add_text(body, "alpha beta gamma");
            doc.add_text(body, format!("word{i}"));
            doc.add_u64(num, i % 7);
            let object: BTreeMap<String, OwnedValue> =
                serde_json::from_str(r#"{"text":"delta","number":1,"nested":{"s":"epsilon"}}"#)
                    .unwrap();
            doc.add_object(json, object);
            writer.add_document(doc)?;
        }
        writer.commit()?;

        let searcher = index.reader()?.searcher();
        let segment_reader = searcher.segment_reader(0);
        let schema = index.schema();
        for field in [body, num, json] {
            let inverted_index = segment_reader.inverted_index(field)?;
            let option = schema
                .get_field_entry(field)
                .field_type()
                .get_index_record_option()
                .expect("indexed field");
            let mut cursor = SegmentPostings::empty();
            let mut term_stream = inverted_index.terms().stream()?;
            let mut num_terms = 0usize;
            while term_stream.advance() {
                let term_info = term_stream.value().clone();
                let owned = inverted_index.read_postings_from_terminfo(&term_info, option)?;
                inverted_index.read_postings_from_terminfo_into(&term_info, option, &mut cursor)?;
                assert_same_postings(owned, cursor.clone());
                num_terms += 1;
            }
            assert!(num_terms > 0, "field {field:?} has no indexed terms");
        }
        Ok(())
    }
}
