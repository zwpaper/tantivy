use std::ops::Range;
use std::sync::Arc;

use crate::{ColumnValues, RowId};

pub trait Iterable<T = u64> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = T> + '_>;

    /// Returns the number of items [`Iterable::boxed_iter`] will yield, when that count is known
    /// without walking the values. Callers use it to pre-size buffers; `None` means unknown.
    fn num_vals_hint(&self) -> Option<usize> {
        None
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
