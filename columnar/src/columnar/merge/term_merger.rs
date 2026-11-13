use std::cmp::Ordering;
use std::collections::BinaryHeap;

use sstable::TermOrdinal;

use crate::Streamer;

/// The terms of a column with the ordinal of the segment.
pub struct TermsWithSegmentOrd<'a> {
    /// Boxed so the `BinaryHeap` moves a small pointer, not the whole `Streamer`
    /// (`size_of::<TermsWithSegmentOrd>()` 312 B -> 16 B).
    pub terms: Box<Streamer<'a>>,
    pub segment_ord: usize,
}

impl PartialEq for TermsWithSegmentOrd<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.segment_ord == other.segment_ord
    }
}

impl Eq for TermsWithSegmentOrd<'_> {}

impl<'a> PartialOrd for TermsWithSegmentOrd<'a> {
    fn partial_cmp(&self, other: &TermsWithSegmentOrd<'a>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for TermsWithSegmentOrd<'a> {
    fn cmp(&self, other: &TermsWithSegmentOrd<'a>) -> Ordering {
        (&other.terms.key(), &other.segment_ord).cmp(&(&self.terms.key(), &self.segment_ord))
    }
}

/// Given a list of sorted term streams,
/// returns an iterator over sorted unique terms.
///
/// The item yield is actually a pair with
/// - the term
/// - a slice with the ordinal of the segments containing the terms.
pub struct TermMerger<'a> {
    heap: BinaryHeap<TermsWithSegmentOrd<'a>>,
    term_streams_with_segment: Vec<TermsWithSegmentOrd<'a>>,
}

impl<'a> TermMerger<'a> {
    /// Stream of merged term dictionary
    pub fn new(term_streams_with_segment: Vec<TermsWithSegmentOrd<'a>>) -> TermMerger<'a> {
        TermMerger {
            heap: BinaryHeap::new(),
            term_streams_with_segment,
        }
    }

    pub(crate) fn matching_segments<'b: 'a>(
        &'b self,
    ) -> impl 'b + Iterator<Item = (usize, TermOrdinal)> {
        self.term_streams_with_segment
            .iter()
            .map(|heap_item| (heap_item.segment_ord, heap_item.terms.term_ord()))
    }

    fn advance_segments(&mut self) {
        let streamers = &mut self.term_streams_with_segment;
        let heap = &mut self.heap;
        for mut heap_item in streamers.drain(..) {
            if heap_item.terms.advance() {
                heap.push(heap_item);
            }
        }
    }

    /// Advance the term iterator to the next term.
    /// Returns true if there is indeed another term
    /// False if there is none.
    pub fn advance(&mut self) -> bool {
        self.advance_segments();
        match self.heap.pop() {
            Some(head) => {
                self.term_streams_with_segment.push(head);
                while let Some(next_streamer) = self.heap.peek() {
                    if self.term_streams_with_segment[0].terms.key() != next_streamer.terms.key() {
                        break;
                    }
                    let next_heap_it = self.heap.pop().unwrap(); // safe : we peeked beforehand
                    self.term_streams_with_segment.push(next_heap_it);
                }
                true
            }
            _ => false,
        }
    }

    /// Returns the current term.
    ///
    /// This method may be called
    /// if and only if advance() has been called before
    /// and "true" was returned.
    pub fn key(&self) -> &[u8] {
        self.term_streams_with_segment[0].terms.key()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::hint::black_box;
    use std::time::Instant;

    use common::OwnedBytes;
    use sstable::{Dictionary, VoidSSTable};

    use super::*;

    /// `streams` interleaved streams of `terms_per_stream` terms each: stream `s` holds
    /// `term{i * streams + s}`. Returns one encoded sstable per stream.
    fn build_dicts(terms_per_stream: usize, streams: usize) -> Vec<Vec<u8>> {
        (0..streams)
            .map(|s| {
                let mut builder = Dictionary::<VoidSSTable>::builder(Vec::new()).unwrap();
                for i in 0..terms_per_stream {
                    let ord = (i * streams + s) as u64;
                    let term = format!("term{ord:010}");
                    builder.insert(term.as_bytes(), &()).unwrap();
                }
                builder.finish().unwrap()
            })
            .collect()
    }

    fn open_dicts(dicts: &[Vec<u8>]) -> Vec<Dictionary<VoidSSTable>> {
        dicts
            .iter()
            .map(|bytes| {
                Dictionary::<VoidSSTable>::from_bytes(OwnedBytes::new(bytes.clone())).unwrap()
            })
            .collect()
    }

    fn terms_with_segment_ords<'a>(
        dictionaries: &'a [Dictionary<VoidSSTable>],
    ) -> Vec<TermsWithSegmentOrd<'a>> {
        dictionaries
            .iter()
            .enumerate()
            .map(|(segment_ord, dictionary)| TermsWithSegmentOrd {
                terms: Box::new(dictionary.stream().unwrap()),
                segment_ord,
            })
            .collect()
    }

    /// Full merge of `dicts`; the merged `(term, [(segment_ord, term_ord)])` sequence.
    fn merge_all(dicts: &[Vec<u8>]) -> Vec<(Vec<u8>, Vec<(usize, TermOrdinal)>)> {
        let dictionaries = open_dicts(dicts);
        let mut merger = TermMerger::new(terms_with_segment_ords(&dictionaries));
        let mut merged = Vec::new();
        while merger.advance() {
            let segments = merger.matching_segments().collect();
            merged.push((merger.key().to_vec(), segments));
        }
        merged
    }

    /// Independent reference merge, one stream per dict.
    fn reference_merge(dicts: &[Vec<u8>]) -> BTreeMap<Vec<u8>, Vec<(usize, TermOrdinal)>> {
        let dictionaries = open_dicts(dicts);
        let mut reference: BTreeMap<Vec<u8>, Vec<(usize, TermOrdinal)>> = BTreeMap::new();
        for (segment_ord, dictionary) in dictionaries.iter().enumerate() {
            let mut stream = dictionary.stream().unwrap();
            while stream.advance() {
                reference
                    .entry(stream.key().to_vec())
                    .or_default()
                    .push((segment_ord, stream.term_ord()));
            }
        }
        reference
    }

    /// FNV-1a over the merged term sequence + `(segment_ord, term_ord)`; the digest is
    /// compared across the pre-change and post-change builds (byte-identity evidence).
    fn digest(merged: &[(Vec<u8>, Vec<(usize, TermOrdinal)>)]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |bytes: &[u8]| {
            for &byte in bytes {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for (term, segments) in merged {
            feed(term);
            feed(&(segments.len() as u64).to_le_bytes());
            for (segment_ord, term_ord) in segments {
                feed(&(*segment_ord as u64).to_le_bytes());
                feed(&term_ord.to_le_bytes());
            }
        }
        hash
    }

    #[test]
    fn test_terms_with_segment_ord_is_lightweight() {
        let size = std::mem::size_of::<TermsWithSegmentOrd<'static>>();
        println!("precheck size_of::<TermsWithSegmentOrd>() = {size} B");
        assert!(
            size <= 16,
            "size_of::<TermsWithSegmentOrd>() = {size} B; the Streamer must stay out of the heap \
             entry"
        );
    }

    #[test]
    fn test_merged_terms_match_reference_and_digest() {
        let dicts = build_dicts(2_000, 8);
        let merged = merge_all(&dicts);
        let reference = reference_merge(&dicts);
        assert_eq!(merged.len(), reference.len());
        for (term, segments) in &merged {
            assert_eq!(reference.get(term), Some(segments), "term {term:?}");
        }
        println!(
            "precheck merged_terms={} digest={:016x}",
            merged.len(),
            digest(&merged)
        );
    }

    #[test]
    fn test_term_merger_microbench() {
        const TERMS_PER_STREAM: usize = 200_000;
        const STREAMS: usize = 8;
        const ITERS: usize = 5;
        let dicts = build_dicts(TERMS_PER_STREAM, STREAMS);
        let dictionaries = open_dicts(&dicts);
        let mut best = f64::MAX;
        let mut merged_terms = 0usize;
        for _ in 0..ITERS {
            let start = Instant::now();
            let mut merger = TermMerger::new(terms_with_segment_ords(&dictionaries));
            let mut count = 0usize;
            while merger.advance() {
                count += 1;
            }
            let elapsed = start.elapsed().as_secs_f64();
            black_box(count);
            merged_terms = count;
            best = best.min(elapsed);
        }
        let merged = merge_all(&dicts);
        println!(
            "precheck microbench best={best:.4}s terms={merged_terms} ns_per_term={:.1} \
             digest={:016x}",
            best * 1e9 / merged_terms as f64,
            digest(&merged)
        );
    }
}
