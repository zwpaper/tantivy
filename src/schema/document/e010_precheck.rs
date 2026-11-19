//! E010 pre-check harness (verification agent).
//!
//! Cheap differential mechanism check for removing the per-object/array heap `collect()` in
//! [`BinaryValueSerializer::serialize_value`]. The source compiles unchanged against the pinned
//! baseline (`e1acb3e0`) and the candidate, so the same file is dropped into both worktrees.
//!
//! Run (release):
//! ```text
//! E010_DOCS=<ndjson> E010_OUT=<dir> \
//!   cargo test --release -p tantivy --lib schema::document::e010_precheck -- --nocapture
//! ```
//!
//! `E010_DOCS` holds real `otel_logs_10m` `_dynamic` objects (the unmapped-key subset of the
//! corpus rows, one JSON object per line), so the harness exercises the exact production store
//! path (`StoreWriter::store` -> `BinaryDocumentSerializer::serialize_doc`).
//!
//! `e010_byte_identity` writes `docstore_direct.bin` (per-document serializer bytes),
//! `store_file.bin` (a real store file) and asserts a `StoreReader::get_document_bytes`
//! round-trip. `e010_differential_precheck` times 100k whole-document serializations over
//! several rounds and counts allocations under the serializer.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use crate::directory::{Directory, RamDirectory};
use crate::schema::document::{BinaryDocumentSerializer, OwnedValue};
use crate::schema::{Field, JsonObjectOptions, Schema, TantivyDocument};
use crate::store::{Compressor, StoreReader, StoreWriter};

// ---- allocation counter (inert unless `COUNTING` is set) --------------------

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static E010_ALLOC: CountingAlloc = CountingAlloc;

// ---- corpus + doc construction ----------------------------------------------

fn docs_path() -> String {
    std::env::var("E010_DOCS").expect("set E010_DOCS to the NDJSON corpus path")
}

fn out_dir() -> String {
    std::env::var("E010_OUT").expect("set E010_OUT to the output directory")
}

fn schema_and_field() -> (Schema, Field) {
    let mut builder = Schema::builder();
    let field = builder.add_json_field("_dynamic", JsonObjectOptions::default().set_stored());
    (builder.build(), field)
}

/// Builds the `_dynamic` `CompactDoc` exactly as the doc mapper does: the unmapped keys as a
/// `BTreeMap<String, OwnedValue>` handed to `add_object`.
fn build_doc(field: Field, line: &str) -> TantivyDocument {
    let obj: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(line).expect("valid NDJSON object");
    let dynamic: BTreeMap<String, OwnedValue> =
        obj.into_iter().map(|(k, v)| (k, OwnedValue::from(v))).collect();
    let mut doc = TantivyDocument::default();
    doc.add_object(field, dynamic);
    doc
}

fn read_lines(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

fn serialize_doc(schema: &Schema, doc: &TantivyDocument, out: &mut Vec<u8>) {
    let mut serializer = BinaryDocumentSerializer::new(out, schema);
    serializer.serialize_doc(doc).unwrap();
}

#[test]
#[ignore = "E010 pre-check; requires E010_DOCS/E010_OUT (see module docs)"]
fn e010_byte_identity() {
    let (schema, field) = schema_and_field();
    let lines = read_lines(&docs_path());
    let out = out_dir();
    std::fs::create_dir_all(&out).unwrap();

    let mut direct = Vec::new();
    let mut buf = Vec::with_capacity(4096);

    let dir = RamDirectory::create();
    let path = Path::new("store");
    let mut store_writer = StoreWriter::new(
        dir.open_write(path).unwrap(),
        Compressor::None,
        1 << 24,
        false,
    )
    .unwrap();

    for line in &lines {
        let doc = build_doc(field, line);
        buf.clear();
        serialize_doc(&schema, &doc, &mut buf);
        direct.extend_from_slice(&(buf.len() as u32).to_le_bytes());
        direct.extend_from_slice(&buf);
        store_writer.store(&doc, &schema).unwrap();
    }
    store_writer.close().unwrap();

    let file = dir.open_read(path).unwrap();
    let reader = StoreReader::open(file.clone(), 10).unwrap();
    for (i, line) in lines.iter().enumerate() {
        let doc = build_doc(field, line);
        buf.clear();
        serialize_doc(&schema, &doc, &mut buf);
        let got = reader.get_document_bytes(i as u32).unwrap();
        assert_eq!(&got[..], &buf[..], "StoreReader round-trip mismatch doc {i}");
    }

    std::fs::write(format!("{out}/docstore_direct.bin"), &direct).unwrap();
    let store_bytes = file.read_bytes().unwrap();
    std::fs::write(format!("{out}/store_file.bin"), store_bytes.as_slice()).unwrap();
    std::fs::write(
        format!("{out}/roundtrip_ok.txt"),
        format!("OK docs={}\n", lines.len()),
    )
    .unwrap();
    eprintln!(
        "byte-identity: docs={} direct_bytes={} store_bytes={}",
        lines.len(),
        direct.len(),
        store_bytes.len(),
    );
}

#[test]
#[ignore = "E010 pre-check; requires E010_DOCS/E010_OUT (see module docs)"]
fn e010_differential_precheck() {
    const N: usize = 100_000;
    const ROUNDS: usize = 7;

    let (schema, field) = schema_and_field();
    let lines = read_lines(&docs_path());

    // Representative `_dynamic`: the median-by-bytes real document.
    let mut sized: Vec<usize> = lines.iter().map(|l| l.len()).collect();
    sized.sort_unstable();
    let median_len = sized[sized.len() / 2];
    let rep_line = lines
        .iter()
        .min_by_key(|l| l.len().abs_diff(median_len))
        .unwrap();
    let doc = build_doc(field, rep_line);
    eprintln!(
        "representative _dynamic: json_bytes={} top_level_keys={}",
        rep_line.len(),
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(rep_line)
            .unwrap()
            .len(),
    );

    let mut buf = Vec::with_capacity(4096);
    let mut times = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS {
        let start = Instant::now();
        for _ in 0..N {
            buf.clear();
            serialize_doc(&schema, &doc, &mut buf);
            black_box(&buf);
        }
        let secs = start.elapsed().as_secs_f64();
        times.push(secs);
        eprintln!(
            "round {round}: {secs:.6} s  ({:.3} us/doc)",
            secs / N as f64 * 1e6
        );
    }
    let mut steady = times[1..].to_vec();
    steady.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = steady[steady.len() / 2];
    eprintln!(
        "median (rounds 1..): {median:.6} s  ({:.3} us/doc)",
        median / N as f64 * 1e6
    );

    buf.clear();
    COUNTING.store(true, Ordering::SeqCst);
    ALLOCS.store(0, Ordering::SeqCst);
    for _ in 0..N {
        buf.clear();
        serialize_doc(&schema, &doc, &mut buf);
        black_box(&buf);
    }
    COUNTING.store(false, Ordering::SeqCst);
    let allocs = ALLOCS.load(Ordering::SeqCst);
    eprintln!(
        "allocations over {N} docs = {allocs} ({:.3}/doc)",
        allocs as f64 / N as f64
    );

    let summary = format!(
        "{{\"representative_json_bytes\": {rep_len}, \"median_seconds\": {median}, \
         \"median_us_per_doc\": {us}, \"allocations\": {allocs}, \"allocs_per_doc\": {apd}, \
         \"rounds\": {times:?}}}\n",
        rep_len = rep_line.len(),
        us = median / N as f64 * 1e6,
        apd = allocs as f64 / N as f64,
    );
    let out = out_dir();
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(format!("{out}/precheck_summary.json"), summary).unwrap();
}
