//! E007 pre-check: byte-identity and serialize time of the columnar doc-id remap.
//!
//! This file is deliberately self-contained and uses only the public `tantivy-columnar` API, so
//! the exact same source compiles against the pinned baseline (`tantivy e1acb3e0`) and against the
//! candidate. The two builds are run with the same env vars and their dumps compared byte for byte:
//!
//! ```text
//! # baseline worktree
//! E007_DUMP=/tmp/e007-base.fast E007_TIME=1 cargo test --release -p tantivy-columnar \
//!     --test e007_reorder_precheck -- --nocapture
//! # candidate worktree
//! E007_DUMP=/tmp/e007-cand.fast E007_BASELINE_DUMP=/tmp/e007-base.fast E007_TIME=1 \
//!     cargo test --release -p tantivy-columnar --test e007_reorder_precheck -- --nocapture
//! ```

use std::io::Write;
use std::net::Ipv6Addr;

use tantivy_columnar::{ColumnType, ColumnarWriter, DateTime, RowId};

const NUM_DOCS: RowId = 1_000_000;
const NUM_TIMED_ROUNDS: usize = 7;

/// Deterministic xorshift64* generator, so both builds record the same values.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: u32) -> u32 {
        (self.next_u64() % u64::from(bound)) as u32
    }
}

/// Returns a shuffled permutation, `old_to_new_ids[old_doc] = new_doc`.
fn shuffled_old_to_new_ids(num_docs: RowId) -> Vec<RowId> {
    let mut old_to_new_ids: Vec<RowId> = (0..num_docs).collect();
    let mut rng = Rng(0x0DDB_1A5E_5BAD_5EED);
    for i in (1..num_docs as usize).rev() {
        old_to_new_ids.swap(i, rng.below(i as u32 + 1) as usize);
    }
    old_to_new_ids
}

/// Records dense numerical, dense str, multivalued (plain and sorted-within-row), sparse/optional,
/// bool, ip and datetime columns for `NUM_DOCS` docs.
fn record_docs(writer: &mut ColumnarWriter) {
    // A str column whose multivalues are sorted within each row (`sort_values_within_row`).
    writer.record_column_type("sorted_str", ColumnType::Str, true);
    let mut rng = Rng(0xE007_1234_5678_9ABC);
    for doc in 0..NUM_DOCS {
        writer.record_numerical(doc, "dense_u64", u64::from(doc) * 7 + 3);
        writer.record_numerical(doc, "dense_i64", i64::from(doc) - 500_000);
        writer.record_numerical(doc, "dense_f64", f64::from(doc) * 0.5);
        writer.record_str(doc, "dense_str", &format!("service-{:04}", doc % 5000));
        if doc % 2 == 0 {
            for _ in 0..1 + rng.below(3) {
                writer.record_numerical(doc, "multivalued", u64::from(rng.below(1_000_000)));
            }
        }
        if doc % 5 == 0 {
            writer.record_str(doc, "sorted_str", "zeta");
            writer.record_str(doc, "sorted_str", "alpha");
        }
        // Sparse / optional columns: present for ~1 doc out of 97.
        if doc % 97 == 0 {
            writer.record_str(doc, "sparse_str", "present");
            writer.record_numerical(doc, "sparse_u64", u64::from(doc));
        }
        writer.record_bool(doc, "flag", doc % 3 == 0);
        if doc % 1000 == 0 {
            writer.record_ip_addr(doc, "ip", Ipv6Addr::LOCALHOST);
        }
        if doc % 11 == 0 {
            writer.record_datetime(
                doc,
                "ts",
                DateTime::from_timestamp_nanos(i64::from(doc) * 1_000_000),
            );
        }
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
fn e007_remap_byte_identity() {
    let old_to_new_ids = shuffled_old_to_new_ids(NUM_DOCS);
    let mut writer = ColumnarWriter::default();
    record_docs(&mut writer);
    let mut bytes = Vec::new();
    writer
        .serialize(NUM_DOCS, Some(&old_to_new_ids), &mut bytes)
        .unwrap();
    println!("remapped columnar output: {} bytes", bytes.len());
    if let Ok(dump_path) = std::env::var("E007_DUMP") {
        std::fs::File::create(&dump_path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        println!("E007_DUMP: wrote {} bytes to {dump_path}", bytes.len());
    }
    if let Ok(baseline_path) = std::env::var("E007_BASELINE_DUMP") {
        let baseline = std::fs::read(&baseline_path).unwrap();
        assert_eq!(
            baseline.len(),
            bytes.len(),
            "remapped columnar output length differs from {baseline_path}"
        );
        assert!(
            baseline == bytes,
            "remapped columnar output differs from {baseline_path}"
        );
        println!("E007_BASELINE_DUMP: {baseline_path} is byte-identical");
    }
}

#[test]
fn e007_serialize_timing() {
    if std::env::var("E007_TIME").is_err() {
        return;
    }
    let old_to_new_ids = shuffled_old_to_new_ids(NUM_DOCS);
    let mut writer = ColumnarWriter::default();
    record_docs(&mut writer);
    let mut bytes = Vec::new();

    let mut remapped = Vec::new();
    let mut unmapped = Vec::new();
    for round in 0..NUM_TIMED_ROUNDS {
        let start = std::time::Instant::now();
        writer
            .serialize(NUM_DOCS, Some(&old_to_new_ids), &mut bytes)
            .unwrap();
        remapped.push(start.elapsed().as_secs_f64());
        let start = std::time::Instant::now();
        writer.serialize(NUM_DOCS, None, &mut bytes).unwrap();
        unmapped.push(start.elapsed().as_secs_f64());
        println!(
            "round {round}: remapped {:.4} s, unmapped {:.4} s",
            remapped[round], unmapped[round]
        );
    }
    println!("remapped serialize: median {:.4} s", median(remapped));
    println!("unmapped serialize: median {:.4} s", median(unmapped));
}
