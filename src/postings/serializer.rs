use std::cmp::Ordering;
use std::io::{self, Write};

use common::{BinarySerializable, CountingWriter, VInt};

use super::TermInfo;
use crate::directory::{CompositeWrite, WritePtr};
use crate::fieldnorm::FieldNormReader;
use crate::index::Segment;
use crate::positions::PositionSerializer;
use crate::postings::compression::{BlockEncoder, VIntEncoder, COMPRESSION_BLOCK_SIZE};
use crate::postings::skip::SkipSerializer;
use crate::query::Bm25Weight;
use crate::schema::{Field, FieldEntry, IndexRecordOption, Schema};
use crate::termdict::TermDictionaryBuilder;
use crate::{DocId, Score};

/// `InvertedIndexSerializer` is in charge of serializing
/// postings on disk, in the
/// * `.idx` (inverted index)
/// * `.pos` (positions file)
/// * `.term` (term dictionary)
///
/// `PostingsWriter` are in charge of pushing the data to the
/// serializer.
///
/// The serializer expects to receive the following calls
/// in this order :
/// * `set_field(...)`
/// * `new_term(...)`
/// * `write_doc(...)`
/// * `write_doc(...)`
/// * `write_doc(...)`
/// * ...
/// * `close_term()`
/// * `new_term(...)`
/// * `write_doc(...)`
/// * ...
/// * `close_term()`
/// * `set_field(...)`
/// * ...
/// * `close()`
///
/// Terms have to be pushed in a lexicographically-sorted order.
/// Within a term, documents have to be pushed in increasing order.
///
/// A description of the serialization format is
/// [available here](https://fulmicoton.gitbooks.io/tantivy-doc/content/inverted-index.html).
pub struct InvertedIndexSerializer {
    terms_write: CompositeWrite<WritePtr>,
    postings_write: CompositeWrite<WritePtr>,
    positions_write: CompositeWrite<WritePtr>,
    schema: Schema,
}

impl InvertedIndexSerializer {
    /// Open a new `InvertedIndexSerializer` for the given segment
    pub fn open(segment: &Segment) -> crate::Result<InvertedIndexSerializer> {
        use crate::index::SegmentComponent::{Positions, Postings, Terms};
        let inv_index_serializer = InvertedIndexSerializer {
            terms_write: CompositeWrite::wrap(segment.open_write(Terms)?),
            postings_write: CompositeWrite::wrap(segment.open_write(Postings)?),
            positions_write: CompositeWrite::wrap(segment.open_write(Positions)?),
            schema: segment.schema(),
        };
        Ok(inv_index_serializer)
    }

    /// Must be called before starting pushing terms of
    /// a given field.
    ///
    /// Loads the indexing options for the given field.
    pub fn new_field(
        &mut self,
        field: Field,
        total_num_tokens: u64,
        fieldnorm_reader: Option<FieldNormReader>,
    ) -> io::Result<FieldSerializer<'_>> {
        let field_entry: &FieldEntry = self.schema.get_field_entry(field);
        let term_dictionary_write = self.terms_write.for_field(field);
        let postings_write = self.postings_write.for_field(field);
        let positions_write = self.positions_write.for_field(field);
        let index_record_option = field_entry
            .field_type()
            .index_record_option()
            .unwrap_or(IndexRecordOption::Basic);
        FieldSerializer::create(
            index_record_option,
            total_num_tokens,
            term_dictionary_write,
            postings_write,
            positions_write,
            fieldnorm_reader,
        )
    }

    /// Closes the serializer.
    pub fn close(self) -> io::Result<()> {
        self.terms_write.close()?;
        self.postings_write.close()?;
        self.positions_write.close()?;
        Ok(())
    }
}

/// The field serializer is in charge of
/// the serialization of a specific field.
pub struct FieldSerializer<'a, W: Write = WritePtr> {
    term_dictionary_builder: TermDictionaryBuilder<&'a mut CountingWriter<W>>,
    postings_serializer: PostingsSerializer,
    positions_serializer_opt: Option<PositionSerializer<&'a mut CountingWriter<W>>>,
    current_term_info: TermInfo,
    term_open: bool,
    postings_write: &'a mut CountingWriter<W>,
    postings_start_offset: u64,
}

impl<'a, W: Write> FieldSerializer<'a, W> {
    /// Creates a new `FieldSerializer` for the given field type.
    pub fn create(
        index_record_option: IndexRecordOption,
        total_num_tokens: u64,
        term_dictionary_write: &'a mut CountingWriter<W>,
        postings_write: &'a mut CountingWriter<W>,
        positions_write: &'a mut CountingWriter<W>,
        fieldnorm_reader: Option<FieldNormReader>,
    ) -> io::Result<FieldSerializer<'a, W>> {
        total_num_tokens.serialize(postings_write)?;
        let term_dictionary_builder = TermDictionaryBuilder::create(term_dictionary_write)?;
        let average_fieldnorm = fieldnorm_reader
            .as_ref()
            .map(|ff_reader| total_num_tokens as Score / ff_reader.num_docs() as Score)
            .unwrap_or(0.0);
        let postings_serializer =
            PostingsSerializer::new(average_fieldnorm, index_record_option, fieldnorm_reader);
        let positions_serializer_opt = if index_record_option.has_positions() {
            Some(PositionSerializer::new(positions_write))
        } else {
            None
        };

        let postings_start_offset = postings_write.written_bytes();
        Ok(FieldSerializer {
            term_dictionary_builder,
            postings_serializer,
            positions_serializer_opt,
            current_term_info: TermInfo::default(),
            term_open: false,
            postings_write,
            postings_start_offset,
        })
    }

    fn postings_offset(&self) -> usize {
        (self.postings_write.written_bytes() - self.postings_start_offset) as usize
    }

    fn current_term_info(&self) -> TermInfo {
        let positions_start =
            if let Some(positions_serializer) = self.positions_serializer_opt.as_ref() {
                positions_serializer.written_bytes()
            } else {
                0u64
            } as usize;
        let addr = self.postings_offset();
        TermInfo {
            doc_freq: 0,
            postings_range: addr..addr,
            positions_range: positions_start..positions_start,
        }
    }

    /// Starts the postings for a new term.
    /// * term - the term. It needs to come after the previous term according to the lexicographical
    ///   order.
    /// * term_doc_freq - return the number of document containing the term.
    pub fn new_term(
        &mut self,
        term: &[u8],
        term_doc_freq: u32,
        record_term_freq: bool,
    ) -> io::Result<()> {
        assert!(
            !self.term_open,
            "Called new_term, while the previous term was not closed."
        );
        self.term_open = true;
        self.postings_serializer.clear();
        self.current_term_info = self.current_term_info();
        self.term_dictionary_builder.insert_key(term)?;
        self.postings_serializer
            .new_term(term_doc_freq, record_term_freq);
        Ok(())
    }

    /// Starts the postings for a new term without recording term frequencies.
    pub fn new_term_without_freq(&mut self, term: &[u8]) -> io::Result<()> {
        self.new_term(term, 0, false)
    }

    /// Serialize the information that a document contains for the current term:
    /// its term frequency, and the position deltas.
    ///
    /// At this point, the positions are already `delta-encoded`.
    /// For instance, if the positions are `2, 3, 17`,
    /// `position_deltas` is `2, 1, 14`
    ///
    /// Term frequencies and positions may be ignored by the serializer depending
    /// on the configuration of the field in the `Schema`.
    pub fn write_doc(&mut self, doc_id: DocId, term_freq: u32, position_deltas: &[u32]) {
        self.current_term_info.doc_freq += 1;
        self.postings_serializer.write_doc(doc_id, term_freq);
        if let Some(ref mut positions_serializer) = self.positions_serializer_opt.as_mut() {
            assert_eq!(term_freq as usize, position_deltas.len());
            positions_serializer.write_positions_delta(position_deltas);
        }
    }

    /// Serializes a whole block of documents containing the current term, in bulk.
    ///
    /// `docs` are local document ids, shifted by `doc_offset`. This is the bulk equivalent
    /// of calling [`Self::write_doc`] once per document with a term frequency and positions
    /// that are ignored; it is only valid for fields that record neither term frequencies
    /// nor positions (the `Basic` record option, the ClickBench case), where the per-posting
    /// body is empty.
    pub fn write_doc_block(&mut self, docs: &[DocId], doc_offset: DocId) {
        debug_assert!(self.positions_serializer_opt.is_none());
        self.current_term_info.doc_freq += docs.len() as u32;
        self.postings_serializer.write_doc_block(docs, doc_offset);
    }

    /// Finish the serialization for this term postings.
    ///
    /// If the current block is incomplete, it needs to be encoded
    /// using `VInt` encoding.
    pub fn close_term(&mut self) -> io::Result<()> {
        crate::fail_point!("FieldSerializer::close_term", |msg: Option<String>| {
            Err(io::Error::other(format!("{msg:?}")))
        });

        if !self.term_open {
            return Ok(());
        };

        self.postings_serializer
            .close_term(self.current_term_info.doc_freq, self.postings_write)?;
        self.current_term_info.postings_range.end = self.postings_offset();
        if let Some(positions_serializer) = self.positions_serializer_opt.as_mut() {
            positions_serializer.close_term()?;
            self.current_term_info.positions_range.end =
                positions_serializer.written_bytes() as usize;
        }
        self.term_dictionary_builder
            .insert_value(&self.current_term_info)?;
        self.term_open = false;
        Ok(())
    }

    /// Closes the current field.
    pub fn close(mut self) -> io::Result<()> {
        self.close_term()?;
        if let Some(positions_serializer) = self.positions_serializer_opt {
            positions_serializer.close()?;
        }
        self.term_dictionary_builder.finish()?;
        Ok(())
    }
}

struct Block {
    doc_ids: [DocId; COMPRESSION_BLOCK_SIZE],
    term_freqs: [u32; COMPRESSION_BLOCK_SIZE],
    len: usize,
}

impl Block {
    fn new() -> Self {
        Block {
            doc_ids: [0u32; COMPRESSION_BLOCK_SIZE],
            term_freqs: [0u32; COMPRESSION_BLOCK_SIZE],
            len: 0,
        }
    }

    fn doc_ids(&self) -> &[DocId] {
        &self.doc_ids[..self.len]
    }

    fn term_freqs(&self) -> &[u32] {
        &self.term_freqs[..self.len]
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn append_doc(&mut self, doc: DocId, term_freq: u32) {
        let len = self.len;
        self.doc_ids[len] = doc;
        self.term_freqs[len] = term_freq;
        self.len = len + 1;
    }

    fn is_full(&self) -> bool {
        self.len == COMPRESSION_BLOCK_SIZE
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn last_doc(&self) -> DocId {
        assert_eq!(self.len, COMPRESSION_BLOCK_SIZE);
        self.doc_ids[COMPRESSION_BLOCK_SIZE - 1]
    }
}

/// Serializer for postings lists.
pub struct PostingsSerializer {
    last_doc_id_encoded: u32,

    block_encoder: BlockEncoder,
    block: Box<Block>,

    postings_write: Vec<u8>,
    skip_write: SkipSerializer,

    mode: IndexRecordOption,
    fieldnorm_reader: Option<FieldNormReader>,

    bm25_weight: Option<Bm25Weight>,
    avg_fieldnorm: Score, /* Average number of term in the field for that segment.
                           * this value is used to compute the block wand information. */
    term_has_freq: bool,
}

impl PostingsSerializer {
    /// Creates a new `PostingsSerializer`.
    /// * avg_fieldnorm - average field norm for the field being serialized.
    /// * mode - indexing options for the field being serialized.
    pub fn new(
        avg_fieldnorm: Score,
        mode: IndexRecordOption,
        fieldnorm_reader: Option<FieldNormReader>,
    ) -> PostingsSerializer {
        PostingsSerializer {
            block_encoder: BlockEncoder::new(),
            block: Box::new(Block::new()),

            postings_write: Vec::new(),
            skip_write: SkipSerializer::new(),

            last_doc_id_encoded: 0u32,
            mode,

            fieldnorm_reader,
            bm25_weight: None,
            avg_fieldnorm,
            term_has_freq: false,
        }
    }

    /// Starts the serialization for a new term.
    /// * term_doc_freq - the number of documents containing the term.
    pub fn new_term(&mut self, term_doc_freq: u32, record_term_freq: bool) {
        self.bm25_weight = None;

        self.term_has_freq = self.mode.has_freq() && record_term_freq;
        if !self.term_has_freq {
            return;
        }

        let num_docs_in_segment: u64 =
            if let Some(fieldnorm_reader) = self.fieldnorm_reader.as_ref() {
                fieldnorm_reader.num_docs() as u64
            } else {
                return;
            };

        if num_docs_in_segment == 0 {
            return;
        }

        self.bm25_weight = Some(Bm25Weight::for_one_term_without_explain(
            term_doc_freq as u64,
            num_docs_in_segment,
            self.avg_fieldnorm,
        ));
    }

    fn write_block(&mut self) {
        {
            // encode the doc ids
            let (num_bits, block_encoded): (u8, &[u8]) = self
                .block_encoder
                .compress_block_sorted(self.block.doc_ids(), self.last_doc_id_encoded);
            self.last_doc_id_encoded = self.block.last_doc();
            self.skip_write
                .write_doc(self.last_doc_id_encoded, num_bits);
            // last el block 0, offset block 1,
            self.postings_write.extend(block_encoded);
        }
        if self.term_has_freq {
            // encode the term frequencies
            let (num_bits, block_encoded): (u8, &[u8]) = self
                .block_encoder
                .compress_block_unsorted(self.block.term_freqs(), true);
            self.postings_write.extend(block_encoded);
            self.skip_write.write_term_freq(num_bits);
            if self.mode.has_positions() {
                // We serialize the sum of term freqs within the skip information
                // in order to navigate through positions.
                let sum_freq = self.block.term_freqs().iter().cloned().sum();
                self.skip_write.write_total_term_freq(sum_freq);
            }
            let mut blockwand_params = (0u8, 0u32);
            if let Some(bm25_weight) = self.bm25_weight.as_ref() {
                if let Some(fieldnorm_reader) = self.fieldnorm_reader.as_ref() {
                    let docs = self.block.doc_ids().iter().cloned();
                    let term_freqs = self.block.term_freqs().iter().cloned();
                    let fieldnorms = docs.map(|doc| fieldnorm_reader.fieldnorm_id(doc));
                    blockwand_params = fieldnorms
                        .zip(term_freqs)
                        .max_by(
                            |(left_fieldnorm_id, left_term_freq),
                             (right_fieldnorm_id, right_term_freq)| {
                                let left_score =
                                    bm25_weight.tf_factor(*left_fieldnorm_id, *left_term_freq);
                                let right_score =
                                    bm25_weight.tf_factor(*right_fieldnorm_id, *right_term_freq);
                                left_score
                                    .partial_cmp(&right_score)
                                    .unwrap_or(Ordering::Equal)
                            },
                        )
                        .unwrap();
                }
            }
            let (fieldnorm_id, term_freq) = blockwand_params;
            self.skip_write.write_blockwand_max(fieldnorm_id, term_freq);
        }
        self.block.clear();
    }

    /// Register that the given document contains the current term.
    /// * doc_id - the document id.
    /// * term_freq - the term frequency within the document.
    pub fn write_doc(&mut self, doc_id: DocId, term_freq: u32) {
        self.block.append_doc(doc_id, term_freq);
        if self.block.is_full() {
            self.write_block();
        }
    }

    /// Register that each document of `docs` contains the current term, in bulk.
    ///
    /// The local document ids are shifted by `doc_offset` (the affine `Stacked` merge mapping)
    /// before being appended. This is the bulk equivalent of calling [`Self::write_doc`] once
    /// per document with an (ignored) term frequency of `0`: it fills and flushes the same
    /// `COMPRESSION_BLOCK_SIZE`-sized blocks, so the serialized postings are byte-identical.
    pub fn write_doc_block(&mut self, docs: &[DocId], doc_offset: DocId) {
        let mut remaining = docs;
        while !remaining.is_empty() {
            let free = COMPRESSION_BLOCK_SIZE - self.block.len;
            let take = free.min(remaining.len());
            let dst = self.block.len..self.block.len + take;
            for (out, &doc) in self.block.doc_ids[dst.clone()].iter_mut().zip(&remaining[..take]) {
                *out = doc_offset + doc;
            }
            self.block.term_freqs[dst].fill(0);
            self.block.len += take;
            if self.block.is_full() {
                self.write_block();
            }
            remaining = &remaining[take..];
        }
    }

    /// Finish the serialization for this term.
    pub fn close_term(
        &mut self,
        doc_freq: u32,
        output_write: &mut impl std::io::Write,
    ) -> io::Result<()> {
        if !self.block.is_empty() {
            // we have doc ids waiting to be written
            // this happens when the number of doc ids is
            // not a perfect multiple of our block size.
            //
            // In that case, the remaining part is encoded
            // using variable int encoding.
            {
                let block_encoded = self
                    .block_encoder
                    .compress_vint_sorted(self.block.doc_ids(), self.last_doc_id_encoded);
                self.postings_write.write_all(block_encoded)?;
            }
            // ... Idem for term frequencies
            if self.term_has_freq {
                let block_encoded = self
                    .block_encoder
                    .compress_vint_unsorted(self.block.term_freqs());
                self.postings_write.write_all(block_encoded)?;
            }
            self.block.clear();
        }
        if doc_freq >= COMPRESSION_BLOCK_SIZE as u32 {
            let skip_data = self.skip_write.data();
            VInt(skip_data.len() as u64).serialize(output_write)?;
            output_write.write_all(skip_data)?;
        }
        output_write.write_all(&self.postings_write[..])?;
        self.skip_write.clear();
        self.postings_write.clear();
        self.bm25_weight = None;
        Ok(())
    }

    fn clear(&mut self) {
        self.block.clear();
        self.last_doc_id_encoded = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::PostingsSerializer;
    use crate::postings::compression::COMPRESSION_BLOCK_SIZE;
    use crate::schema::IndexRecordOption;
    use crate::DocId;

    /// Serializes one `Basic` term by calling [`PostingsSerializer::write_doc`] once per doc,
    /// the reference the bulk path must reproduce.
    fn serialize_per_doc(docs: &[DocId], doc_offset: DocId) -> Vec<u8> {
        let mut serializer = PostingsSerializer::new(0.0, IndexRecordOption::Basic, None);
        serializer.new_term(docs.len() as u32, false);
        for &doc in docs {
            serializer.write_doc(doc_offset + doc, 0);
        }
        let mut out = Vec::new();
        serializer.close_term(docs.len() as u32, &mut out).unwrap();
        out
    }

    /// Serializes the same term through [`PostingsSerializer::write_doc_block`], handing it the
    /// docs in the `COMPRESSION_BLOCK_SIZE`-sized chunks a decoded source block would provide.
    fn serialize_blocks(docs: &[DocId], doc_offset: DocId) -> Vec<u8> {
        let mut serializer = PostingsSerializer::new(0.0, IndexRecordOption::Basic, None);
        serializer.new_term(docs.len() as u32, false);
        for block in docs.chunks(COMPRESSION_BLOCK_SIZE) {
            serializer.write_doc_block(block, doc_offset);
        }
        let mut out = Vec::new();
        serializer.close_term(docs.len() as u32, &mut out).unwrap();
        out
    }

    /// The bulk path must serialize exactly the bytes the per-doc path does, around the
    /// `COMPRESSION_BLOCK_SIZE` boundary (the skip list appears at `doc_freq >= 128`) and with
    /// the affine `Stacked` doc offset applied.
    #[test]
    fn test_write_doc_block_matches_write_doc() {
        for num_docs in [0usize, 1, 127, 128, 129, 256, 300, 1025] {
            let docs: Vec<DocId> = (0..num_docs as DocId).map(|i| i * 3).collect();
            for doc_offset in [0, 1, 1000] {
                assert_eq!(
                    serialize_blocks(&docs, doc_offset),
                    serialize_per_doc(&docs, doc_offset),
                    "num_docs={num_docs} doc_offset={doc_offset}"
                );
            }
        }
    }
}
