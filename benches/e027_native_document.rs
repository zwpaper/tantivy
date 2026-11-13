//! E027 pre-check: cost of the `CompactDoc` byte representation vs the native owned
//! representation, measured on a ClickBench-shaped document.
//!
//! The measurement mirrors where each stage pays destruction in the real pipeline:
//!
//! * the compact producer encodes each value into `CompactDoc` bytes and then drops the owned
//!   temporary (the mapper's `let value = ...; document.add_field_value(field, &value)`);
//! * the native producer only moves the owned value into the document;
//! * the compact consumer decodes each value back out and drops the single `node_data` buffer;
//! * the native consumer matches the enum and drops the owned values (the transferred frees).
//!
//! So `producer_compact - producer_native` is the producer saving `dP`, and
//! `consumer_compact - consumer_native` is the consumer saving `dI`. A negative `dI` means the
//! change moves more destruction into the already-binding consumer than it removes.
//!
//! Run in release on a node: `cargo bench --bench e027_native_document`.
//! `E027_DOCS` overrides the per-measurement document count (default 20000).

use std::hint::black_box;
use std::sync::OnceLock;
use std::time::Instant;

use tantivy::schema::document::{NativeDocument, ReferenceValue, ReferenceValueLeaf, Value};
use tantivy::schema::{Field, OwnedValue};
use tantivy::{DateTime, Document, TantivyDocument};

/// ClickBench `hits` doc_mapping: 73 i64, 28 text, 4 datetime, `store_source: false`.
const NUM_I64: u32 = 73;
const NUM_TEXT: u32 = 28;
const NUM_DATE: u32 = 4;
const NUM_FIELDS: u32 = NUM_I64 + NUM_TEXT + NUM_DATE;

fn num_docs() -> usize {
    static NUM_DOCS: OnceLock<usize> = OnceLock::new();
    *NUM_DOCS.get_or_init(|| {
        std::env::var("E027_DOCS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(20_000)
    })
}

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
        values.push((
            Field::from_field_id(field_id),
            OwnedValue::Str(text.to_string()),
        ));
        field_id += 1;
    }
    for i in 0..NUM_DATE {
        values.push((
            Field::from_field_id(field_id),
            OwnedValue::Date(DateTime::from_timestamp_nanos(
                1_700_000_000_000_000_000 + i as i64,
            )),
        ));
        field_id += 1;
    }
    values
}

/// Encode only (the input stays owned by the caller): the compact producer's encode subtree.
fn encode_compact(values: &[(Field, OwnedValue)]) -> TantivyDocument {
    let mut doc = TantivyDocument::new();
    for (field, value) in values {
        doc.add_field_value(*field, value);
    }
    doc
}

/// Compact producer: encode, then drop the owned temporaries, as the mapper does.
fn producer_compact(values: Vec<(Field, OwnedValue)>) -> TantivyDocument {
    let mut doc = TantivyDocument::new();
    for (field, value) in &values {
        doc.add_field_value(*field, value);
    }
    drop(values);
    doc
}

/// Native producer: move the owned values in; the caller keeps the document.
fn producer_native(values: Vec<(Field, OwnedValue)>) -> NativeDocument {
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

/// Compact consumer: decode each value, then drop the single `node_data` buffer.
fn consume_compact(doc: TantivyDocument) -> u64 {
    let mut acc = 0u64;
    for (_, value) in doc.iter_fields_and_values() {
        acc = acc.wrapping_add(decode_leaf(value.as_value()));
    }
    acc
}

/// Native consumer: match each value, then drop the owned values (transferred frees).
fn consume_native(doc: NativeDocument) -> u64 {
    let mut acc = 0u64;
    for (_, value) in doc.iter_fields_and_values() {
        acc = acc.wrapping_add(decode_leaf(value.as_value()));
    }
    acc
}

fn per_doc_ns(run: impl FnOnce()) -> f64 {
    let start = Instant::now();
    run();
    start.elapsed().as_nanos() as f64 / num_docs() as f64
}

fn batches(values: &[(Field, OwnedValue)]) -> Vec<Vec<(Field, OwnedValue)>> {
    (0..num_docs()).map(|_| values.to_vec()).collect()
}

fn bench_encode(values: &[(Field, OwnedValue)]) -> f64 {
    let batches = batches(values);
    let mut docs = Vec::with_capacity(num_docs());
    per_doc_ns(|| {
        for batch in &batches {
            docs.push(black_box(encode_compact(batch)));
        }
    })
}

fn bench_drop_only(values: &[(Field, OwnedValue)]) -> f64 {
    let batches = batches(values);
    per_doc_ns(|| {
        for batch in batches {
            black_box(batch);
        }
    })
}

fn bench_producer(values: &[(Field, OwnedValue)]) -> (f64, f64) {
    let compact_batches = batches(values);
    let mut docs = Vec::with_capacity(num_docs());
    let compact = per_doc_ns(|| {
        for batch in compact_batches {
            docs.push(black_box(producer_compact(batch)));
        }
    });
    drop(docs);

    let native_batches = batches(values);
    let mut docs = Vec::with_capacity(num_docs());
    let native = per_doc_ns(|| {
        for batch in native_batches {
            docs.push(black_box(producer_native(batch)));
        }
    });
    drop(docs);
    (compact, native)
}

fn bench_consumer(values: &[(Field, OwnedValue)]) -> (f64, f64) {
    let docs: Vec<TantivyDocument> = (0..num_docs()).map(|_| encode_compact(values)).collect();
    let compact = per_doc_ns(|| {
        for doc in docs {
            black_box(consume_compact(doc));
        }
    });

    let docs: Vec<NativeDocument> = (0..num_docs())
        .map(|_| producer_native(values.to_vec()))
        .collect();
    let native = per_doc_ns(|| {
        for doc in docs {
            black_box(consume_native(doc));
        }
    });
    (compact, native)
}

fn main() {
    let values = sample_values();

    // Warm up both code paths and the allocator.
    for _ in 0..3 {
        black_box(producer_compact(values.clone()));
        black_box(producer_native(values.clone()));
        black_box(consume_compact(encode_compact(&values)));
        black_box(consume_native(producer_native(values.clone())));
    }

    println!("E027 native-document-transport pre-check (ClickBench-shaped, {NUM_FIELDS} fields/doc)");
    println!(
        "{} docs per measurement, release build, ns/doc",
        num_docs()
    );
    println!(
        "layout: size_of OwnedValue={} (Field,OwnedValue)={}",
        std::mem::size_of::<OwnedValue>(),
        std::mem::size_of::<(Field, OwnedValue)>()
    );
    println!();

    let rounds = 3;
    let mut encode = 0.0;
    let mut drop_only = 0.0;
    let mut producer_compact = 0.0;
    let mut producer_native = 0.0;
    let mut consumer_compact = 0.0;
    let mut consumer_native = 0.0;
    for _ in 0..rounds {
        encode += bench_encode(&values);
        drop_only += bench_drop_only(&values);
        let (pc, pn) = bench_producer(&values);
        producer_compact += pc;
        producer_native += pn;
        let (cc, cn) = bench_consumer(&values);
        consumer_compact += cc;
        consumer_native += cn;
    }
    let rounds = rounds as f64;
    let encode = encode / rounds;
    let drop_only = drop_only / rounds;
    let producer_compact = producer_compact / rounds;
    let producer_native = producer_native / rounds;
    let consumer_compact = consumer_compact / rounds;
    let consumer_native = consumer_native / rounds;

    let dp = producer_compact - producer_native;
    let di = consumer_compact - consumer_native;

    println!("component                     ns/doc");
    println!("encode (compact only)        {encode:9.1}");
    println!("drop owned values only       {drop_only:9.1}");
    println!();
    println!("phase                     compact      native      delta   removed");
    println!(
        "producer (dP)          {producer_compact:10.1} {producer_native:10.1} {dp:10.1}  {:6.1}%",
        100.0 * dp / producer_compact
    );
    println!(
        "consumer (dI)          {consumer_compact:10.1} {consumer_native:10.1} {di:10.1}  {:6.1}%",
        100.0 * di / consumer_compact
    );

    let scale = |ns: f64, docs: f64| ns * docs / 1e9;
    let dp_10m = scale(dp, 1e7);
    let di_10m = scale(di, 1e7);
    let dp_100m = scale(dp, 1e8);
    let di_100m = scale(di, 1e8);
    println!();
    println!("projected net savings (s):");
    println!("  10M  producer dP={dp_10m:8.2}  consumer dI={di_10m:8.2}");
    println!("  100M producer dP={dp_100m:8.2}  consumer dI={di_100m:8.2}");
    println!();
    println!("folded baseline budget (100M): producer encode 214.343 s, consumer decode 199.182 s");
    println!(
        "  100M removal of the folded subtrees: producer {:6.1}%  consumer {:6.1}%",
        100.0 * dp_100m / 214.343,
        100.0 * di_100m / 199.182
    );
}
