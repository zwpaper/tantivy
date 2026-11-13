use std::fmt::Debug;
use std::sync::Arc;

use crate::column_values::MonotonicallyMappableToU64;
use crate::iterable::{Iterable, append_column_values_u64};
use crate::{ColumnIndex, ColumnValues, MergeRowOrder};

pub(crate) struct MergedColumnValues<'a, T> {
    pub(crate) column_indexes: &'a [ColumnIndex],
    pub(crate) column_values: &'a [Option<Arc<dyn ColumnValues<T>>>],
    pub(crate) merge_row_order: &'a MergeRowOrder,
}

impl<T: Copy + PartialOrd + Debug + 'static> Iterable<T> for MergedColumnValues<'_, T> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = T> + '_> {
        match self.merge_row_order {
            MergeRowOrder::Stack(_) => Box::new(
                self.column_values
                    .iter()
                    .flatten()
                    .flat_map(|column_value| column_value.iter()),
            ),
            MergeRowOrder::Shuffled(shuffle_merge_order) => Box::new(
                shuffle_merge_order
                    .iter_new_to_old_row_addrs()
                    .flat_map(|row_addr| {
                        let column_index = &self.column_indexes[row_addr.segment_ord as usize];
                        let column_values =
                            self.column_values[row_addr.segment_ord as usize].as_ref()?;
                        let value_range = column_index.value_row_ids(row_addr.row_id);
                        Some((value_range, column_values))
                    })
                    .flat_map(|(value_range, column_values)| {
                        value_range
                            .into_iter()
                            .map(|val| column_values.get_val(val))
                    }),
            ),
        }
    }

    fn num_vals_hint(&self) -> Option<usize> {
        match self.merge_row_order {
            // Stacking yields every value of every present column, so the count is a sum.
            MergeRowOrder::Stack(_) => Some(
                self.column_values
                    .iter()
                    .flatten()
                    .map(|column_values| column_values.num_vals() as usize)
                    .sum(),
            ),
            // Shuffled can drop rows, so the count is not known without walking the order.
            MergeRowOrder::Shuffled(_) => None,
        }
    }

    fn extend_u64(&self, out: &mut Vec<u64>)
    where T: MonotonicallyMappableToU64 {
        match self.merge_row_order {
            // Stacking yields every value of every present column, in column order: decode each
            // present column in bulk instead of one virtual `get_val` per value.
            MergeRowOrder::Stack(_) => {
                for column_values in self.column_values.iter().flatten() {
                    append_column_values_u64(column_values.as_ref(), out);
                }
            }
            // Shuffled reorders (and can drop) rows; keep the boxed walk.
            MergeRowOrder::Shuffled(_) => {
                out.extend(self.boxed_iter().map(|value| value.to_u64()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use common::OwnedBytes;

    use super::*;
    use crate::column_values::{
        CodecType, load_u64_based_column_values, serialize_u64_based_column_values,
    };
    use crate::columnar::{ColumnarReader, ColumnarWriter, MergeRowOrder, StackMergeOrder};
    use crate::{Column, ColumnType, DynamicColumn, RowId};

    const CODECS: &[CodecType] = &[CodecType::Bitpacked, CodecType::BlockwiseLinear];

    fn make_full_columnar(column_name: &str, vals: &[u64]) -> ColumnarReader {
        let mut writer = ColumnarWriter::default();
        writer.record_column_type(column_name, ColumnType::U64, false);
        for (row_id, val) in vals.iter().copied().enumerate() {
            writer.record_numerical(row_id as RowId, column_name, val);
        }
        let mut buffer = Vec::new();
        writer
            .serialize(vals.len() as RowId, None, &mut buffer)
            .unwrap();
        ColumnarReader::open(buffer).unwrap()
    }

    fn open_full_column(columnar: &ColumnarReader, column_name: &str) -> Column<u64> {
        let handles = columnar.read_columns(column_name).unwrap();
        let handle = handles.into_iter().next().expect("column is present");
        match handle.open().unwrap() {
            DynamicColumn::U64(column) => column,
            _ => panic!("expected a U64 column"),
        }
    }

    /// The merged walk and the equivalent contiguous slice must serialize to the same bytes: the
    /// second pass must not depend on how the values are produced, only on the value sequence.
    #[test]
    fn test_serialize_merged_column_values_matches_slice() {
        let vals_a: Vec<u64> = (0..1_000u64).map(|i| i * 7 + 3).collect();
        let vals_b: Vec<u64> = (0..1_000u64).map(|i| i * 11 + 1).collect();
        let columnar_a = make_full_columnar("price", &vals_a);
        let columnar_b = make_full_columnar("price", &vals_b);
        let column_a = open_full_column(&columnar_a, "price");
        let column_b = open_full_column(&columnar_b, "price");

        let columnar_readers = [&columnar_a, &columnar_b];
        let merge_row_order: MergeRowOrder = StackMergeOrder::stack(&columnar_readers[..]).into();
        let column_indexes = [column_a.index, column_b.index];
        let column_values = [Some(column_a.values), Some(column_b.values)];
        let merged_column_values = MergedColumnValues {
            column_indexes: &column_indexes,
            column_values: &column_values,
            merge_row_order: &merge_row_order,
        };
        let concatenated: Vec<u64> = vals_a.iter().chain(vals_b.iter()).copied().collect();

        let mut merged_bytes = Vec::new();
        serialize_u64_based_column_values(&merged_column_values, CODECS, &mut merged_bytes)
            .unwrap();
        let mut slice_bytes = Vec::new();
        serialize_u64_based_column_values(&&concatenated[..], CODECS, &mut slice_bytes).unwrap();

        assert_eq!(merged_bytes, slice_bytes);
        let column = load_u64_based_column_values::<u64>(OwnedBytes::new(merged_bytes)).unwrap();
        assert_eq!(column.num_vals(), concatenated.len() as u32);
        let read_back: Vec<u64> = (0..column.num_vals()).map(|i| column.get_val(i)).collect();
        assert_eq!(read_back, concatenated);
    }
}
