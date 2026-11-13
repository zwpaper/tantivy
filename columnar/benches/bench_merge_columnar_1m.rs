//! E033 pre-check: `merge_columnar` over two 1M-row full numerical columnars.
//!
//! `serialize_u64_based_column_values` used to walk its source iterator twice (stats/estimator
//! pass, then the codec serialization pass); on the merge path that second walk re-decodes the
//! source column values. This bench measures the end-to-end `merge_columnar` cost of that path so
//! the pre-change and post-change numbers are directly comparable.

pub mod common;

use binggan::BenchRunner;
use common::{Card, generate_columnar_with_name};
use tantivy_columnar::*;

const NUM_DOCS: u32 = 1_000_000;

fn main() {
    let columnar_readers = vec![
        generate_columnar_with_name(Card::Full, NUM_DOCS, "price"),
        generate_columnar_with_name(Card::Full, NUM_DOCS, "price"),
    ];

    let mut runner: BenchRunner = BenchRunner::new();
    let mut group = runner.new_group();
    group.register_with_input(
        "merge_columnar_2x1m_full",
        &columnar_readers,
        move |columnar_readers: &Vec<ColumnarReader>| {
            let mut out = Vec::new();
            let columnar_readers = columnar_readers.iter().collect::<Vec<_>>();
            let merge_row_order = StackMergeOrder::stack(&columnar_readers[..]);
            merge_columnar(&columnar_readers, &[], merge_row_order.into(), &mut out).unwrap();
            Some(out.len() as u64)
        },
    );
    group.run();
}
