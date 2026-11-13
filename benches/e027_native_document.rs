//! E027 pre-check: cost of the `CompactDoc` byte representation vs the native owned
//! representation, measured on a ClickBench-shaped document.
//!
//! The producer side builds the document from the owned values the mapper already produces;
//! the consumer side iterates it the way the built-in writers do (borrow each value, decode its
//! leaf, then drop the document). Both sides include destruction, so the measured delta is the
//! net transport cost that E027 replaces.
//!
//! Run in release on a node: `cargo run --release --bench e027_native_document`.
//!
//! Output is a per-document nanosecond table; the projection at the bottom scales it to the
//! 10M-document rig and to the 100M folded baseline budget (producer encode 214.343 CPU s,
//! consumer decode 199.182 CPU s).

use std::hint::black_box;
use std::time::Instant;

use tantivy::schema::document::{NativeDocument, ReferenceValue, ReferenceValueLeaf, Value};
use tantivy::schema::{Field, OwnedValue};
use tantivy::{DateTime, TantivyDocument};

/// ClickBench `hits` doc_mapping: 73 i64, 28 text, 4 datetime, `store_source: false`.
const NUM_I64: u32 = 73;
const NUM_TEXT: u32 = 28;
const NUM_DATE: u32 = 4;
const NUM_FIELDS: u32 = NUM_I64 + NUM_TEXT + NUM_DATE;

const NUM_DOCS: usize = 20_000;

fn sample_values() -> Vec<(Field, OwnedValue)> {
    let mut values = Vec::with_capacity(NUM_FIELDS as usize);
    let mut field_id = 0u32;
    for i in 0..NUM_I64 {
        values.push((
            Field::from_field_id(field_id),
            OwnedValue::I64(1_000_000 + i as i64),
        ));
        field_id += 1;
    }
    for i in 0..NUM_TEXT {
        let text = match i % 4 {
            0 => "https://example.com/path/to/some/resource?query=value&x=1",
            1 => "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            2 => "short",
            _ => "some-medium-length-raw-text-field-value",
        };
        values.push((Field::from_field_id(field_id), OwnedValue::Str(text.to_string())));
        field_id += 1;
    }
    for i in 0..NUM_DATE {
        values.push((
            Field::from_field_id(field_id),
            OwnedValue::Date(DateTime::from_timestamp_nanos(1_700_000_000_000_000_000 + i as i64)),
        ));
        field_id += 1;
    }
    values
}

fn build_compact(values: &[(Field, OwnedValue)]) -> TantivyDocument {
    let mut doc = TantivyDocument::new();
    for (field, value) in values {
        doc.add_field_value(*field, value);
    }
    doc
}

fn build_native(values: Vec<(Field, OwnedValue)>) -> NativeDocument {
    let mut doc = NativeDocument::with_capacity(values.len());
    for (field, value) in values {
        doc.add_owned_value(field, value);
    }
    doc
}

#[inline]
fn decode_leaf<'a, V: Value<'a>>(value: ReferenceValue<'a, V>) -> u64 {
    match value {
        ReferenceValue::Leaf(leaf) => match leaf {
            ReferenceValueLeaf::Str(text) => text.len() as u64,
            ReferenceValueLeaf::U64(num) => num,
            ReferenceValueLeaf::I64(num) => num as u64,
            ReferenceValueLeaf::F64(num) => num as u64,
            ReferenceValueLeaf::Bool(flag) => flag as u64,
            ReferenceValueLeaf::Date(date) => date.into_timestamp_nanos() as u64,
            _ => 0,
        },
        ReferenceValue::Array(_) | ReferenceValue::Object(_) => 0,
    }
}

fn consume_compact(doc: TantivyDocument) -> u64 {
    let mut acc = 0u64;
    for (_, value) in doc.iter_fields_and_values() {
        acc = acc.wrapping_add(decode_leaf(value.as_value()));
    }
    acc
}

fn consume_native(doc: NativeDocument) -> u64 {
    let mut acc = 0u64;
    for (_, value) in doc.iter_fields_and_values() {
        acc = acc.wrapping_add(decode_leaf(value.as_value()));
    }
    acc
}

fn bench_build(values: &[(Field, OwnedValue)]) -> (f64, f64) {
    let compact_batches: Vec<Vec<(Field, OwnedValue)>> =
        (0..NUM_DOCS).map(|_| values.to_vec()).collect();
    let start = Instant::now();
    for batch in &compact_batches {
        black_box(build_compact(batch));
    }
    let compact_ns = start.elapsed().as_nanos() as f64 / NUM_DOCS as f64;
    drop(compact_batches);

    let native_batches: Vec<Vec<(Field, OwnedValue)>> =
        (0..NUM_DOCS).map(|_| values.to_vec()).collect();
    let start = Instant::now();
    for batch in native_batches {
        black_box(build_native(batch));
    }
    let native_ns = start.elapsed().as_nanos() as f64 / NUM_DOCS as f64;
    (compact_ns, native_ns)
}

fn bench_consume(values: &[(Field, OwnedValue)]) -> (f64, f64) {
    let compact_docs: Vec<TantivyDocument> =
        (0..NUM_DOCS).map(|_| build_compact(values)).collect();
    let start = Instant::now();
    for doc in compact_docs {
        black_box(consume_compact(doc));
    }
    let compact_ns = start.elapsed().as_nanos() as f64 / NUM_DOCS as f64;

    let native_docs: Vec<NativeDocument> = (0..NUM_DOCS)
        .map(|_| build_native(values.to_vec()))
        .collect();
    let start = Instant::now();
    for doc in native_docs {
        black_box(consume_native(doc));
    }
    let native_ns = start.elapsed().as_nanos() as f64 / NUM_DOCS as f64;
    (compact_ns, native_ns)
}

fn main() {
    let values = sample_values();

    // Warm up both code paths and the allocator.
    for _ in 0..3 {
        black_box(build_compact(&values));
        black_box(build_native(values.clone()));
        black_box(consume_compact(build_compact(&values)));
        black_box(consume_native(build_native(values.clone())));
    }

    let (compact_build, native_build) = bench_build(&values);
    let (compact_consume, native_consume) = bench_consume(&values);

    let build_delta = compact_build - native_build;
    let consume_delta = compact_consume - native_consume;

    println!("E027 native-document-transport pre-check (ClickBench-shaped, {NUM_FIELDS} fields/doc)");
    println!("{NUM_DOCS} docs per measurement, release build, ns/doc");
    println!();
    println!("phase                     compact      native      delta   removed");
    println!(
        "producer build         {compact_build:10.1} {native_build:10.1} {build_delta:10.1}  {:6.1}%",
        100.0 * build_delta / compact_build
    );
    println!(
        "consumer consume+drop  {compact_consume:10.1} {native_consume:10.1} {consume_delta:10.1}  {:6.1}%",
        100.0 * consume_delta / compact_consume
    );

    let dp_10m = build_delta * 10_000_000.0 / 1e9;
    let di_10m = consume_delta * 10_000_000.0 / 1e9;
    let dp_100m = build_delta * 100_000_000.0 / 1e9;
    let di_100m = consume_delta * 100_000_000.0 / 1e9;
    println!();
    println!("projected net savings (s):");
    println!("  10M  producer dP={dp_10m:7.2}  consumer dI={di_10m:7.2}");
    println!("  100M producer dP={dp_100m:7.2}  consumer dI={di_100m:7.2}");
    println!();
    println!("folded baseline budget (100M): producer encode 214.343 s, consumer decode 199.182 s");
    println!(
        "  100M removal of the folded subtrees: producer {:5.1}%  consumer {:5.1}%",
        100.0 * dp_100m / 214.343,
        100.0 * di_100m / 199.182
    );
}
