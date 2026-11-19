use std::cmp::Ordering;

use stacker::{ExpUnrolledLinkedList, MemoryArena};

use crate::columnar::writer::column_operation::{ColumnOperation, SymbolValue};
use crate::dictionary::{DictionaryBuilder, UnorderedId};
use crate::{Cardinality, NumericalType, NumericalValue, RowId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
enum DocumentStep {
    Same = 0,
    Next = 1,
    Skipped = 2,
}

#[inline(always)]
fn delta_with_last_doc(last_doc_opt: Option<u32>, doc: u32) -> DocumentStep {
    let expected_next_doc = last_doc_opt.map(|last_doc| last_doc + 1).unwrap_or(0u32);
    match doc.cmp(&expected_next_doc) {
        Ordering::Less => DocumentStep::Same,
        Ordering::Equal => DocumentStep::Next,
        Ordering::Greater => DocumentStep::Skipped,
    }
}

#[derive(Copy, Clone, Default)]
pub struct ColumnWriter {
    // Detected cardinality of the column so far.
    cardinality: Cardinality,
    // Last document inserted.
    // None if no doc has been added yet.
    last_doc_opt: Option<u32>,
    // Buffer containing the serialized values.
    values: ExpUnrolledLinkedList,
}

/// Marks a new doc id that has no value in the column being reordered.
const NO_VALUE_RANGE: (u32, u32) = (u32::MAX, 0);

/// Scratch space reused by [`ColumnWriter::operation_iterator`] to reorder a column's
/// operations by new doc id.
///
/// A single instance is kept for the whole `ColumnarWriter::serialize` call, so the
/// `O(num_docs)` index and the reordered byte buffer are allocated once per segment
/// finalize rather than once per column.
#[derive(Default)]
pub(super) struct ColumnReorderBuffers {
    /// The column operations, serialized in increasing new doc id order.
    reordered: Vec<u8>,
    /// Byte range of the serialized values of each new doc id, indexed by new doc id.
    /// `NO_VALUE_RANGE` marks a doc without a value in the column being reordered.
    value_ranges: Vec<(u32, u32)>,
}

impl ColumnReorderBuffers {
    /// Rewrites `column_ops` (a column's operations, serialized, grouped by increasing old
    /// doc id and each group starting with a `NewDoc` op) so that the groups appear in
    /// increasing new doc id order, and returns the rewritten bytes.
    ///
    /// `old_to_new_ids` is a permutation of `0..num_docs` (`doc_id_mapping.rs`), so each
    /// group can be copied verbatim into its new position. Copying whole groups keeps the
    /// stable order of the values inside a row, and docs without a value in this column
    /// emit nothing, so the emitted `NewDoc` ids stay strictly increasing.
    fn remap<'a, V: SymbolValue>(
        &'a mut self,
        column_ops: &[u8],
        old_to_new_ids: &[RowId],
    ) -> &'a [u8] {
        let ColumnReorderBuffers {
            reordered,
            value_ranges,
        } = self;
        let num_docs = old_to_new_ids.len();
        if value_ranges.len() < num_docs {
            value_ranges.resize(num_docs, NO_VALUE_RANGE);
        }
        value_ranges[..num_docs].fill(NO_VALUE_RANGE);

        // Pass 1: remap the `NewDoc` ids and record the byte range of each group's values.
        let mut cursor: &[u8] = column_ops;
        let mut offset = 0usize;
        let mut previous_new_doc: Option<RowId> = None;
        while let Some(op) = ColumnOperation::<V>::deserialize(&mut cursor) {
            let op_start = offset;
            offset = column_ops.len() - cursor.len();
            if let ColumnOperation::NewDoc(old_doc) = op {
                if let Some(previous_new_doc) = previous_new_doc {
                    value_ranges[previous_new_doc as usize].1 = op_start as u32;
                }
                let new_doc = old_to_new_ids[old_doc as usize];
                value_ranges[new_doc as usize] = (offset as u32, 0);
                previous_new_doc = Some(new_doc);
            }
        }
        if let Some(previous_new_doc) = previous_new_doc {
            value_ranges[previous_new_doc as usize].1 = offset as u32;
        }

        // Pass 2: emit the groups in increasing new doc id order.
        reordered.clear();
        for (new_doc, &(values_start, values_end)) in value_ranges[..num_docs].iter().enumerate() {
            if values_start == NO_VALUE_RANGE.0 {
                continue;
            }
            let new_doc_op = ColumnOperation::<V>::NewDoc(new_doc as RowId);
            reordered.extend_from_slice(new_doc_op.serialize().as_ref());
            reordered.extend_from_slice(&column_ops[values_start as usize..values_end as usize]);
        }
        reordered
    }
}

impl ColumnWriter {
    /// Returns an iterator over the Symbol that have been recorded
    /// for the given column.
    ///
    /// When `old_to_new_ids_opt` is set, the operations are reordered by new doc id
    /// using `reorder_buffers` (see [`ColumnReorderBuffers::remap`]).
    pub(super) fn operation_iterator<'a, V: SymbolValue>(
        &self,
        arena: &MemoryArena,
        old_to_new_ids_opt: Option<&[RowId]>,
        buffer: &'a mut Vec<u8>,
        reorder_buffers: &'a mut ColumnReorderBuffers,
    ) -> impl Iterator<Item = ColumnOperation<V>> + 'a + use<'a, V> {
        buffer.clear();
        self.values.read_to_end(arena, buffer);
        let column_ops: &[u8] = match old_to_new_ids_opt {
            Some(old_to_new_ids) => reorder_buffers.remap::<V>(buffer, old_to_new_ids),
            None => buffer,
        };
        let mut cursor: &[u8] = column_ops;
        std::iter::from_fn(move || ColumnOperation::deserialize(&mut cursor))
    }

    /// Records a change of the document being recorded.
    ///
    /// This function will also update the cardinality of the column
    /// if necessary.
    pub(super) fn record<S: SymbolValue>(&mut self, doc: RowId, value: S, arena: &mut MemoryArena) {
        // Difference between `doc` and the last doc.
        match delta_with_last_doc(self.last_doc_opt, doc) {
            DocumentStep::Same => {
                // This is the last encounterred document.
                self.cardinality = Cardinality::Multivalued;
            }
            DocumentStep::Next => {
                self.last_doc_opt = Some(doc);
                self.write_symbol::<S>(ColumnOperation::NewDoc(doc), arena);
            }
            DocumentStep::Skipped => {
                self.cardinality = self.cardinality.max(Cardinality::Optional);
                self.last_doc_opt = Some(doc);
                self.write_symbol::<S>(ColumnOperation::NewDoc(doc), arena);
            }
        }
        self.write_symbol(ColumnOperation::Value(value), arena);
    }

    // Get the cardinality.
    // The overall number of docs in the column is necessary to
    // deal with the case where the all docs contain 1 value, except some documents
    // at the end of the column.
    pub(crate) fn get_cardinality(&self, num_docs: RowId) -> Cardinality {
        match delta_with_last_doc(self.last_doc_opt, num_docs) {
            DocumentStep::Same | DocumentStep::Next => self.cardinality,
            DocumentStep::Skipped => self.cardinality.max(Cardinality::Optional),
        }
    }

    /// Appends a new symbol to the `ColumnWriter`.
    fn write_symbol<V: SymbolValue>(
        &mut self,
        column_operation: ColumnOperation<V>,
        arena: &mut MemoryArena,
    ) {
        self.values
            .writer(arena)
            .extend_from_slice(column_operation.serialize().as_ref());
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct NumericalColumnWriter {
    compatible_numerical_types: CompatibleNumericalTypes,
    column_writer: ColumnWriter,
}

impl NumericalColumnWriter {
    pub fn force_numerical_type(&mut self, numerical_type: NumericalType) {
        assert!(
            self.compatible_numerical_types
                .is_type_accepted(numerical_type)
        );
        self.compatible_numerical_types = CompatibleNumericalTypes::StaticType(numerical_type);
    }
}

/// State used to store what types are still acceptable
/// after having seen a set of numerical values.
#[derive(Clone, Copy)]
pub(crate) enum CompatibleNumericalTypes {
    Dynamic {
        all_values_within_i64_range: bool,
        all_values_within_u64_range: bool,
    },
    StaticType(NumericalType),
}

impl Default for CompatibleNumericalTypes {
    fn default() -> CompatibleNumericalTypes {
        CompatibleNumericalTypes::Dynamic {
            all_values_within_i64_range: true,
            all_values_within_u64_range: true,
        }
    }
}

impl CompatibleNumericalTypes {
    pub fn is_type_accepted(&self, numerical_type: NumericalType) -> bool {
        match self {
            CompatibleNumericalTypes::Dynamic {
                all_values_within_i64_range,
                all_values_within_u64_range,
            } => match numerical_type {
                NumericalType::I64 => *all_values_within_i64_range,
                NumericalType::U64 => *all_values_within_u64_range,
                NumericalType::F64 => true,
            },
            CompatibleNumericalTypes::StaticType(static_numerical_type) => {
                *static_numerical_type == numerical_type
            }
        }
    }

    pub fn accept_value(&mut self, numerical_value: NumericalValue) {
        match self {
            CompatibleNumericalTypes::Dynamic {
                all_values_within_i64_range,
                all_values_within_u64_range,
            } => match numerical_value {
                NumericalValue::I64(val_i64) => {
                    let value_within_u64_range = val_i64 >= 0i64;
                    *all_values_within_u64_range &= value_within_u64_range;
                }
                NumericalValue::U64(val_u64) => {
                    let value_within_i64_range = val_u64 < i64::MAX as u64;
                    *all_values_within_i64_range &= value_within_i64_range;
                }
                NumericalValue::F64(_) => {
                    *all_values_within_i64_range = false;
                    *all_values_within_u64_range = false;
                }
            },
            CompatibleNumericalTypes::StaticType(typ) => {
                assert_eq!(
                    numerical_value.numerical_type(),
                    *typ,
                    "Input type forbidden. This column has been forced to type {typ:?}, received \
                     {numerical_value:?}"
                );
            }
        }
    }

    pub fn to_numerical_type(self) -> NumericalType {
        for numerical_type in [NumericalType::I64, NumericalType::U64] {
            if self.is_type_accepted(numerical_type) {
                return numerical_type;
            }
        }
        NumericalType::F64
    }
}

impl NumericalColumnWriter {
    pub fn numerical_type(&self) -> NumericalType {
        self.compatible_numerical_types.to_numerical_type()
    }

    pub fn cardinality(&self, num_docs: RowId) -> Cardinality {
        self.column_writer.get_cardinality(num_docs)
    }

    pub fn record_numerical_value(
        &mut self,
        doc: RowId,
        value: NumericalValue,
        arena: &mut MemoryArena,
    ) {
        self.compatible_numerical_types.accept_value(value);
        self.column_writer.record(doc, value, arena);
    }

    pub(super) fn operation_iterator<'a>(
        self,
        arena: &MemoryArena,
        old_to_new_ids: Option<&[RowId]>,
        buffer: &'a mut Vec<u8>,
        reorder_buffers: &'a mut ColumnReorderBuffers,
    ) -> impl Iterator<Item = ColumnOperation<NumericalValue>> + 'a + use<'a> {
        self.column_writer
            .operation_iterator(arena, old_to_new_ids, buffer, reorder_buffers)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct StrOrBytesColumnWriter {
    pub(crate) dictionary_id: u32,
    pub(crate) column_writer: ColumnWriter,
    // If true, when facing a multivalued cardinality,
    // values associated to a given document will be sorted.
    //
    // This is useful for facets.
    //
    // If false, the order of appearance in the document will be
    // observed.
    pub(crate) sort_values_within_row: bool,
}

impl StrOrBytesColumnWriter {
    pub(crate) fn with_dictionary_id(dictionary_id: u32) -> StrOrBytesColumnWriter {
        StrOrBytesColumnWriter {
            dictionary_id,
            column_writer: Default::default(),
            sort_values_within_row: false,
        }
    }

    pub(crate) fn record_bytes(
        &mut self,
        doc: RowId,
        bytes: &[u8],
        dictionaries: &mut [DictionaryBuilder],
        arena: &mut MemoryArena,
    ) {
        let unordered_id =
            dictionaries[self.dictionary_id as usize].get_or_allocate_id(bytes, arena);
        self.column_writer.record(doc, unordered_id, arena);
    }

    pub(super) fn operation_iterator<'a>(
        &self,
        arena: &MemoryArena,
        old_to_new_ids: Option<&[RowId]>,
        byte_buffer: &'a mut Vec<u8>,
        reorder_buffers: &'a mut ColumnReorderBuffers,
    ) -> impl Iterator<Item = ColumnOperation<UnorderedId>> + 'a + use<'a> {
        self.column_writer
            .operation_iterator(arena, old_to_new_ids, byte_buffer, reorder_buffers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_with_last_doc() {
        assert_eq!(delta_with_last_doc(None, 0u32), DocumentStep::Next);
        assert_eq!(delta_with_last_doc(None, 1u32), DocumentStep::Skipped);
        assert_eq!(delta_with_last_doc(None, 2u32), DocumentStep::Skipped);
        assert_eq!(delta_with_last_doc(Some(0u32), 0u32), DocumentStep::Same);
        assert_eq!(delta_with_last_doc(Some(1u32), 1u32), DocumentStep::Same);
        assert_eq!(delta_with_last_doc(Some(1u32), 2u32), DocumentStep::Next);
        assert_eq!(delta_with_last_doc(Some(1u32), 3u32), DocumentStep::Skipped);
        assert_eq!(delta_with_last_doc(Some(1u32), 4u32), DocumentStep::Skipped);
    }

    #[track_caller]
    fn test_column_writer_coercion_iter_aux(
        values: impl Iterator<Item = NumericalValue>,
        expected_numerical_type: NumericalType,
    ) {
        let mut compatible_numerical_types = CompatibleNumericalTypes::default();
        for value in values {
            compatible_numerical_types.accept_value(value);
        }
        assert_eq!(
            compatible_numerical_types.to_numerical_type(),
            expected_numerical_type
        );
    }

    #[track_caller]
    fn test_column_writer_coercion_aux(
        values: &[NumericalValue],
        expected_numerical_type: NumericalType,
    ) {
        test_column_writer_coercion_iter_aux(values.iter().copied(), expected_numerical_type);
        test_column_writer_coercion_iter_aux(values.iter().rev().copied(), expected_numerical_type);
    }

    #[test]
    fn test_column_writer_coercion() {
        test_column_writer_coercion_aux(&[], NumericalType::I64);
        test_column_writer_coercion_aux(&[1i64.into()], NumericalType::I64);
        test_column_writer_coercion_aux(&[1u64.into()], NumericalType::I64);
        // We don't detect exact integer at the moment. We could!
        test_column_writer_coercion_aux(&[1f64.into()], NumericalType::F64);
        test_column_writer_coercion_aux(&[u64::MAX.into()], NumericalType::U64);
        test_column_writer_coercion_aux(&[(i64::MAX as u64).into()], NumericalType::U64);
        test_column_writer_coercion_aux(&[(1u64 << 63).into()], NumericalType::U64);
        test_column_writer_coercion_aux(&[1i64.into(), 1u64.into()], NumericalType::I64);
        test_column_writer_coercion_aux(&[u64::MAX.into(), (-1i64).into()], NumericalType::F64);
    }

    #[test]
    #[should_panic]
    fn test_compatible_numerical_types_static_incompatible_type() {
        let mut compatible_numerical_types =
            CompatibleNumericalTypes::StaticType(NumericalType::U64);
        compatible_numerical_types.accept_value(NumericalValue::I64(1i64));
    }

    #[test]
    fn test_compatible_numerical_types_static_different_type_forbidden() {
        let mut compatible_numerical_types =
            CompatibleNumericalTypes::StaticType(NumericalType::U64);
        compatible_numerical_types.accept_value(NumericalValue::U64(u64::MAX));
    }

    #[test]
    fn test_compatible_numerical_types_static() {
        for typ in [NumericalType::I64, NumericalType::I64, NumericalType::F64] {
            let compatible_numerical_types = CompatibleNumericalTypes::StaticType(typ);
            assert_eq!(compatible_numerical_types.to_numerical_type(), typ);
        }
    }
}
