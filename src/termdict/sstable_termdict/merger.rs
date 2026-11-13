use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::postings::TermInfo;
use crate::termdict::TermStreamer;

pub struct HeapItem<'a> {
    /// Boxed so that the `BinaryHeap` moves a small pointer + ordinal, not the whole
    /// `TermStreamer` (`size_of::<HeapItem>()` 336 B -> 16 B).
    pub streamer: Box<TermStreamer<'a>>,
    pub segment_ord: usize,
}

impl<'a> PartialEq for HeapItem<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.segment_ord == other.segment_ord
    }
}

impl<'a> Eq for HeapItem<'a> {}

impl<'a> PartialOrd for HeapItem<'a> {
    fn partial_cmp(&self, other: &HeapItem<'a>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for HeapItem<'a> {
    fn cmp(&self, other: &HeapItem<'a>) -> Ordering {
        (&other.streamer.key(), &other.segment_ord).cmp(&(&self.streamer.key(), &self.segment_ord))
    }
}

/// Given a list of sorted term streams,
/// returns an iterator over sorted unique terms.
///
/// The item yield is actually a pair with
/// - the term
/// - a slice with the ordinal of the segments containing the terms.
pub struct TermMerger<'a> {
    heap: BinaryHeap<HeapItem<'a>>,
    current_streamers: Vec<HeapItem<'a>>,
}

impl<'a> TermMerger<'a> {
    /// Stream of merged term dictionary
    pub fn new(streams: Vec<TermStreamer<'a>>) -> TermMerger<'a> {
        TermMerger {
            heap: BinaryHeap::new(),
            current_streamers: streams
                .into_iter()
                .enumerate()
                .map(|(ord, streamer)| HeapItem {
                    streamer: Box::new(streamer),
                    segment_ord: ord,
                })
                .collect(),
        }
    }

    fn advance_segments(&mut self) {
        let streamers = &mut self.current_streamers;
        let heap = &mut self.heap;
        for mut heap_item in streamers.drain(..) {
            if heap_item.streamer.advance() {
                heap.push(heap_item);
            }
        }
    }

    /// Advance the term iterator to the next term.
    /// Returns true if there is indeed another term
    /// False if there is none.
    pub fn advance(&mut self) -> bool {
        self.advance_segments();
        let Some(head) = self.heap.pop() else {
            return false;
        };
        self.current_streamers.push(head);
        while let Some(next_streamer) = self.heap.peek() {
            if self.current_streamers[0].streamer.key() != next_streamer.streamer.key() {
                break;
            }
            let next_heap_it = self.heap.pop().unwrap(); // safe : we peeked beforehand
            self.current_streamers.push(next_heap_it);
        }
        true
    }

    /// Returns the current term.
    ///
    /// This method may be called
    /// if and only if advance() has been called before
    /// and "true" was returned.
    pub fn key(&self) -> &[u8] {
        self.current_streamers[0].streamer.key()
    }

    /// Returns the sorted list of segment ordinals
    /// that include the current term.
    ///
    /// This method may be called
    /// if and only if advance() has been called before
    /// and "true" was returned.
    pub fn current_segment_ords_and_term_infos<'b: 'a>(
        &'b self,
    ) -> impl 'b + Iterator<Item = (usize, TermInfo)> {
        self.current_streamers
            .iter()
            .map(|heap_item| (heap_item.segment_ord, heap_item.streamer.value().clone()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::hint::black_box;
    use std::time::Instant;

    use common::OwnedBytes;
    use sstable::Dictionary;

    use super::{HeapItem, TermMerger};
    use crate::postings::TermInfo;
    use crate::termdict::sstable_termdict::TermSSTable;
    use crate::termdict::TermStreamer;

    fn make_term_info(term_ord: u64) -> TermInfo {
        let offset = |ord: u64| (ord * 100 + ord * ord) as usize;
        TermInfo {
            doc_freq: term_ord as u32 + 1,
            postings_range: offset(term_ord)..offset(term_ord + 1),
            positions_range: offset(term_ord)..offset(term_ord + 1),
        }
    }

    /// `streams` interleaved streams of `terms_per_stream` terms each: stream `s` holds
    /// `term{i * streams + s}`. Returns one encoded sstable per stream.
    fn build_dicts(terms_per_stream: usize, streams: usize) -> Vec<Vec<u8>> {
        (0..streams)
            .map(|s| {
                let mut builder = Dictionary::<TermSSTable>::builder(Vec::new()).unwrap();
                for i in 0..terms_per_stream {
                    let ord = (i * streams + s) as u64;
                    let term = format!("term{ord:010}");
                    builder
                        .insert(term.as_bytes(), &make_term_info(ord))
                        .unwrap();
                }
                builder.finish().unwrap()
            })
            .collect()
    }

    fn open_dicts(dicts: &[Vec<u8>]) -> Vec<Dictionary<TermSSTable>> {
        dicts
            .iter()
            .map(|bytes| {
                Dictionary::<TermSSTable>::from_bytes(OwnedBytes::new(bytes.clone())).unwrap()
            })
            .collect()
    }

    /// Full merge of `dicts`; the merged `(term, [(segment_ord, TermInfo)])` sequence.
    fn merge_all(dicts: &[Vec<u8>]) -> Vec<(Vec<u8>, Vec<(usize, TermInfo)>)> {
        let dictionaries = open_dicts(dicts);
        let streams: Vec<TermStreamer<'_>> =
            dictionaries.iter().map(|d| d.stream().unwrap()).collect();
        let mut merger = TermMerger::new(streams);
        let mut merged = Vec::new();
        while merger.advance() {
            let infos = merger.current_segment_ords_and_term_infos().collect();
            merged.push((merger.key().to_vec(), infos));
        }
        merged
    }

    /// Independent reference merge, one stream per dict.
    fn reference_merge(dicts: &[Vec<u8>]) -> BTreeMap<Vec<u8>, Vec<(usize, TermInfo)>> {
        let dictionaries = open_dicts(dicts);
        let mut reference: BTreeMap<Vec<u8>, Vec<(usize, TermInfo)>> = BTreeMap::new();
        for (segment_ord, dictionary) in dictionaries.iter().enumerate() {
            let mut stream = dictionary.stream().unwrap();
            while stream.advance() {
                reference
                    .entry(stream.key().to_vec())
                    .or_default()
                    .push((segment_ord, stream.value().clone()));
            }
        }
        reference
    }

    /// FNV-1a over the merged term sequence + `(segment_ord, TermInfo)`; the digest is
    /// compared across the pre-change and post-change builds (byte-identity evidence).
    fn digest(merged: &[(Vec<u8>, Vec<(usize, TermInfo)>)]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |bytes: &[u8]| {
            for &byte in bytes {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for (term, infos) in merged {
            feed(term);
            feed(&(infos.len() as u64).to_le_bytes());
            for (segment_ord, term_info) in infos {
                feed(&(*segment_ord as u64).to_le_bytes());
                feed(&term_info.doc_freq.to_le_bytes());
                feed(&(term_info.postings_range.start as u64).to_le_bytes());
                feed(&(term_info.postings_range.end as u64).to_le_bytes());
                feed(&(term_info.positions_range.start as u64).to_le_bytes());
                feed(&(term_info.positions_range.end as u64).to_le_bytes());
            }
        }
        hash
    }

    #[test]
    fn test_heap_item_is_lightweight() {
        let size = std::mem::size_of::<HeapItem<'static>>();
        println!("precheck size_of::<HeapItem>() = {size} B");
        assert!(
            size <= 16,
            "size_of::<HeapItem>() = {size} B; the TermStreamer must stay out of the heap entry"
        );
    }

    #[test]
    fn test_merged_terms_match_reference_and_digest() {
        let dicts = build_dicts(2_000, 8);
        let merged = merge_all(&dicts);
        let reference = reference_merge(&dicts);
        assert_eq!(merged.len(), reference.len());
        for (term, infos) in &merged {
            assert_eq!(reference.get(term), Some(infos), "term {term:?}");
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
            let streams: Vec<TermStreamer<'_>> =
                dictionaries.iter().map(|d| d.stream().unwrap()).collect();
            let start = Instant::now();
            let mut merger = TermMerger::new(streams);
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
