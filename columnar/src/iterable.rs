use std::ops::Range;
use std::sync::Arc;

use crate::column_values::MonotonicallyMappableToU64;
use crate::{ColumnValues, RowId};

/// Number of values decoded per `ColumnValues::get_range` call by
/// [`append_column_values_u64`].
const BULK_DECODE_CHUNK: usize = 1024;

pub trait Iterable<T = u64> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = T> + '_>;

    /// Returns the number of items [`Iterable::boxed_iter`] will yield, when that count is known
    /// without walking the values. Callers use it to pre-size buffers; `None` means unknown.
    fn num_vals_hint(&self) -> Option<usize> {
        None
    }

    /// Appends every value, converted to u64, to `out` in iteration order.
    ///
    /// The default walks [`Iterable::boxed_iter`] one value at a time. An implementation backed by
    /// `ColumnValues` can override this to decode in bulk with [`append_column_values_u64`] instead
    /// of paying one virtual `get_val` call per value.
    fn extend_u64(&self, out: &mut Vec<u64>)
    where T: MonotonicallyMappableToU64 {
        out.extend(self.boxed_iter().map(|value| value.to_u64()));
    }
}

/// Appends the values of a `ColumnValues` column to `out` in row order, decoding fixed-size chunks
/// with `ColumnValues::get_range` (SIMD in the bitpacked codec) instead of one virtual `get_val`
/// per value.
///
/// The caller must append whole columns in iteration order; this helper only decodes one column.
pub(crate) fn append_column_values_u64<T: MonotonicallyMappableToU64>(
    column_values: &dyn ColumnValues<T>,
    out: &mut Vec<u64>,
) {
    let num_vals = column_values.num_vals() as usize;
    out.reserve(num_vals);
    let mut buffer = [T::from_u64(0); BULK_DECODE_CHUNK];
    let mut start = 0;
    while start < num_vals {
        let len = BULK_DECODE_CHUNK.min(num_vals - start);
        let chunk = &mut buffer[..len];
        column_values.get_range(start as u64, chunk);
        out.extend(chunk.iter().map(|value| value.to_u64()));
        start += len;
    }
}

impl<T: Copy> Iterable<T> for &[T] {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = T> + '_> {
        Box::new(self.iter().copied())
    }

    fn num_vals_hint(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<T: Copy> Iterable<T> for Range<T>
where Range<T>: Iterator<Item = T>
{
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = T> + '_> {
        Box::new(self.clone())
    }

    fn num_vals_hint(&self) -> Option<usize> {
        self.clone().size_hint().1
    }
}

impl Iterable for Arc<dyn crate::ColumnValues<RowId>> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        Box::new(self.iter().map(|row_id| row_id as u64))
    }

    fn num_vals_hint(&self) -> Option<usize> {
        Some(self.num_vals() as usize)
    }
}
