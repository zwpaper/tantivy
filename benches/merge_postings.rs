//! Microbenchmark for the doc-id remap in the postings merge (E002 pre-check).
//!
//! A delete-free merge uses `MappingType::Stacked`, whose doc-id mapping is affine:
//! `mapped = segment_base_doc_id[segment_ord] + local`. `write_postings_for_field` therefore remaps
//! arithmetically instead of building a `Vec<Vec<Option<DocId>>>` per field and calling
//! `next_mapped_doc` per posting. This bench walks the same real `SegmentPostings` of 9 synthetic
//! segments (one raw indexed field, no deletes) with both loops:
//!
//! - `vec_mapping`: the old path - materialize the per-segment map, then one table load + `Option`
//!   match per posting (in a non-inlined frame, like `tantivy::postings::next_mapped_doc`);
//! - `stacked_arith`: the new path - one add per posting.
//!
//! The postings are read once and cloned in the untimed setup (`iter_batched`), so the timed region
//! is the map build + remap only. Run: `cargo bench --bench merge_postings`. The pre-check keeps
//! the change only if `stacked_arith` is at least 1.3x the throughput of `vec_mapping` (ns/posting
//! ratio <= 0.77).

use std::hint::black_box;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use tantivy::indexer::NoMergePolicy;
use tantivy::postings::SegmentPostings;
use tantivy::schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions};
use tantivy::{doc, DocId, DocSet, Index, InvertedIndexReader, SegmentReader, Term, TERMINATED};

const NUM_SEGMENTS: usize = 9;
const DOCS_PER_SEGMENT: usize = 111_112; // ~1M docs total
const VOCAB: usize = 10_000; // every term appears in every segment, so each term has many postings

/// The old per-posting remap: a table load + `Option` match, kept in its own (non-inlined) frame
/// like `tantivy::postings::next_mapped_doc`.
#[inline(never)]
fn next_mapped_doc(postings: &mut SegmentPostings, mapping: &[Option<DocId>]) -> Option<DocId> {
    while postings.doc() != TERMINATED {
        if let Some(doc) = mapping[postings.doc() as usize] {
            return Some(doc);
        }
        postings.advance();
    }
    None
}

struct Scenario {
    max_docs: Vec<DocId>,
    segment_base_doc_id: Vec<DocId>,
    /// `terms[term] = [(segment_ord, pristine postings)]`, in merged term order.
    terms: Vec<Vec<(usize, SegmentPostings)>>,
    total_postings: usize,
}

impl Scenario {
    fn build() -> Self {
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
            let mut writer = index.writer_with_num_threads(1, 256_000_000).unwrap();
            writer.set_merge_policy(Box::new(NoMergePolicy));
            for _ in 0..NUM_SEGMENTS {
                for doc_ord in 0..DOCS_PER_SEGMENT {
                    let value = format!("term_{}", doc_ord % VOCAB);
                    writer.add_document(doc!(body => value)).unwrap();
                }
                writer.commit().unwrap();
            }
        }

        let segments = index.searchable_segments().unwrap();
        assert_eq!(segments.len(), NUM_SEGMENTS);
        let readers: Vec<SegmentReader> = segments
            .iter()
            .map(SegmentReader::open)
            .collect::<tantivy::Result<Vec<_>>>()
            .unwrap();

        let max_docs: Vec<DocId> = readers.iter().map(SegmentReader::max_doc).collect();
        let mut segment_base_doc_id = Vec::with_capacity(max_docs.len());
        let mut next_base: DocId = 0;
        for &max_doc in &max_docs {
            segment_base_doc_id.push(next_base);
            next_base += max_doc;
        }

        let inverted: Vec<Arc<InvertedIndexReader>> = readers
            .iter()
            .map(|reader| reader.inverted_index(body).unwrap())
            .collect();

        let mut term_keys = Vec::new();
        let mut stream = inverted[0].terms().stream().unwrap();
        while stream.advance() {
            term_keys.push(stream.key().to_vec());
        }
        assert_eq!(term_keys.len(), VOCAB);

        let mut terms = Vec::with_capacity(term_keys.len());
        for key in &term_keys {
            let term = Term::from_field_bytes(body, key);
            let mut per_segment = Vec::with_capacity(inverted.len());
            for (segment_ord, inverted) in inverted.iter().enumerate() {
                if let Some(postings) = inverted
                    .read_postings(&term, IndexRecordOption::Basic)
                    .unwrap()
                {
                    per_segment.push((segment_ord, postings));
                }
            }
            terms.push(per_segment);
        }

        // Each doc carries exactly one raw term, so there is one posting per doc.
        let total_postings = NUM_SEGMENTS * DOCS_PER_SEGMENT;

        Scenario {
            max_docs,
            segment_base_doc_id,
            terms,
            total_postings,
        }
    }

    /// Fresh, unadvanced copies of every term's postings (untimed setup).
    fn clone_postings(&self) -> Vec<Vec<(usize, SegmentPostings)>> {
        self.terms
            .iter()
            .map(|term| term.iter().map(|(ord, p)| (*ord, p.clone())).collect())
            .collect()
    }

    /// Old path: build the per-segment map once (as the old code did per field), then remap every
    /// posting through it.
    fn vec_mapping(&self, mut terms: Vec<Vec<(usize, SegmentPostings)>>) -> usize {
        let mut map: Vec<Vec<Option<DocId>>> = self
            .max_docs
            .iter()
            .map(|&max_doc| vec![None; max_doc as usize])
            .collect();
        let mut next_doc: DocId = 0;
        for (segment_ord, &max_doc) in self.max_docs.iter().enumerate() {
            for doc_id in 0..max_doc {
                map[segment_ord][doc_id as usize] = Some(next_doc);
                next_doc += 1;
            }
        }

        let mut visited = 0usize;
        for term in &mut terms {
            for (segment_ord, mut postings) in term.drain(..) {
                let mapping = &map[segment_ord];
                while let Some(doc) = next_mapped_doc(&mut postings, mapping) {
                    black_box(doc);
                    visited += 1;
                    postings.advance();
                }
            }
        }
        visited
    }

    /// New path: affine remap, no table and no `next_mapped_doc` call.
    fn stacked_arith(&self, mut terms: Vec<Vec<(usize, SegmentPostings)>>) -> usize {
        let mut visited = 0usize;
        for term in &mut terms {
            for (segment_ord, mut postings) in term.drain(..) {
                let base = self.segment_base_doc_id[segment_ord];
                while postings.doc() != TERMINATED {
                    black_box(base + postings.doc());
                    visited += 1;
                    postings.advance();
                }
            }
        }
        visited
    }
}

fn bench_merge_stacked(c: &mut Criterion) {
    let scenario = Scenario::build();
    let mut group = c.benchmark_group("merge_stacked");
    group.throughput(Throughput::Elements(scenario.total_postings as u64));
    group.bench_function("vec_mapping", |b| {
        b.iter_batched(
            || scenario.clone_postings(),
            |terms| scenario.vec_mapping(terms),
            BatchSize::LargeInput,
        )
    });
    group.bench_function("stacked_arith", |b| {
        b.iter_batched(
            || scenario.clone_postings(),
            |terms| scenario.stacked_arith(terms),
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_merge_stacked);
criterion_main!(benches);
