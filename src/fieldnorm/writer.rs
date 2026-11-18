use std::cmp::Ordering;
use std::sync::OnceLock;
use std::{io, iter};

use super::{fieldnorm_to_id, FieldNormsSerializer};
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::schema::{Field, Schema};
use crate::DocId;

/// The `FieldNormsWriter` is in charge of tracking the fieldnorm byte
/// of each document for each field with field norms.
///
/// `FieldNormsWriter` stores a `Vec<u8>` for each tracked field, using a
/// byte per document per field.
pub struct FieldNormsWriter {
    fieldnorms_buffers: Vec<Option<Vec<u8>>>,
    // Exact sum of the same capacities observed by the original mem_usage, never lengths.
    cached_capacity_bytes_opt: Option<usize>,
}

impl FieldNormsWriter {
    /// Returns the fields that should have field norms computed
    /// according to the given schema.
    pub(crate) fn fields_with_fieldnorm(schema: &Schema) -> Vec<Field> {
        schema
            .fields()
            .filter_map(|(field, field_entry)| {
                if field_entry.is_indexed() && field_entry.has_fieldnorms() {
                    Some(field)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }

    /// Initialize with state for tracking the field norm fields
    /// specified in the schema.
    pub fn for_schema(schema: &Schema) -> FieldNormsWriter {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        let cache = *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("TANTIVY_E054_CACHED_FIELDNORM_MEMORY").as_deref(),
                Ok("1")
            )
        });
        log::info!("e054-fieldnorm-capacity-cache enabled={cache}");
        Self::for_schema_with_cache(schema, cache)
    }

    fn for_schema_with_cache(schema: &Schema, cache: bool) -> FieldNormsWriter {
        let mut fieldnorms_buffers: Vec<Option<Vec<u8>>> = iter::repeat_with(|| None)
            .take(schema.num_fields())
            .collect();
        for field in FieldNormsWriter::fields_with_fieldnorm(schema) {
            fieldnorms_buffers[field.field_id() as usize] = Some(Vec::with_capacity(1_000));
        }
        let cached_capacity_bytes_opt = cache.then(|| {
            fieldnorms_buffers
                .iter()
                .flatten()
                .map(|buffer| buffer.capacity())
                .sum()
        });
        FieldNormsWriter {
            fieldnorms_buffers,
            cached_capacity_bytes_opt,
        }
    }

    /// The memory used inclusive childs. Both paths return the same capacity sum.
    pub fn mem_usage(&self) -> usize {
        if let Some(bytes) = self.cached_capacity_bytes_opt {
            debug_assert_eq!(bytes, self.recomputed_mem_usage());
            bytes
        } else {
            self.recomputed_mem_usage()
        }
    }

    fn recomputed_mem_usage(&self) -> usize {
        self.fieldnorms_buffers
            .iter()
            .flatten()
            .map(|buf| buf.capacity())
            .sum()
    }
    /// Ensure that all documents in 0..max_doc have a byte associated with them
    /// in each of the fieldnorm vectors.
    ///
    /// Will extend with 0-bytes for documents that have not been seen.
    pub fn fill_up_to_max_doc(&mut self, max_doc: DocId) {
        for fieldnorms_buffer_opt in self.fieldnorms_buffers.iter_mut() {
            if let Some(fieldnorms_buffer) = fieldnorms_buffer_opt.as_mut() {
                let capacity_before = fieldnorms_buffer.capacity();
                fieldnorms_buffer.resize(max_doc as usize, 0u8);
                if let Some(bytes) = self.cached_capacity_bytes_opt.as_mut() {
                    *bytes += fieldnorms_buffer.capacity() - capacity_before;
                }
            }
        }
    }

    /// Set the fieldnorm byte for the given document for the given field.
    ///
    /// Will internally convert the u32 `fieldnorm` value to the appropriate byte
    /// to approximate the field norm in less space.
    ///
    /// * doc       - the document id
    /// * field     - the field being set
    /// * fieldnorm - the number of terms present in document `doc` in field `field`
    pub fn record(&mut self, doc: DocId, field: Field, fieldnorm: u32) {
        if let Some(fieldnorm_buffer) = self
            .fieldnorms_buffers
            .get_mut(field.field_id() as usize)
            .and_then(Option::as_mut)
        {
            let capacity_before = fieldnorm_buffer.capacity();
            match fieldnorm_buffer.len().cmp(&(doc as usize)) {
                Ordering::Less => {
                    // we fill intermediary `DocId` as  having a fieldnorm of 0.
                    fieldnorm_buffer.resize(doc as usize, 0u8);
                }
                Ordering::Equal => {}
                Ordering::Greater => {
                    panic!("Cannot register a given fieldnorm twice")
                }
            }
            fieldnorm_buffer.push(fieldnorm_to_id(fieldnorm));
            if let Some(bytes) = self.cached_capacity_bytes_opt.as_mut() {
                *bytes += fieldnorm_buffer.capacity() - capacity_before;
            }
        }
    }

    /// Serialize the seen fieldnorm values to the serializer for all fields.
    pub fn serialize(
        &self,
        mut fieldnorms_serializer: FieldNormsSerializer,
        doc_id_map: Option<&DocIdMapping>,
    ) -> io::Result<()> {
        for (field, fieldnorms_buffer) in self.fieldnorms_buffers.iter().enumerate().filter_map(
            |(field_id, fieldnorms_buffer_opt)| {
                fieldnorms_buffer_opt.as_ref().map(|fieldnorms_buffer| {
                    (Field::from_field_id(field_id as u32), fieldnorms_buffer)
                })
            },
        ) {
            if let Some(doc_id_map) = doc_id_map {
                let remapped_fieldnorm_buffer = doc_id_map.remap(fieldnorms_buffer);
                fieldnorms_serializer.serialize_field(field, &remapped_fieldnorm_buffer)?;
            } else {
                fieldnorms_serializer.serialize_field(field, fieldnorms_buffer)?;
            }
        }
        fieldnorms_serializer.close()?;
        Ok(())
    }
}

#[cfg(test)]
mod e054_tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::path::Path;

    use super::*;
    use crate::directory::{Directory, RamDirectory};
    use crate::schema::{STORED, TEXT};

    fn assert_oracle(cached: &FieldNormsWriter, original: &FieldNormsWriter) {
        assert_eq!(cached.mem_usage(), cached.recomputed_mem_usage());
        assert_eq!(cached.mem_usage(), original.mem_usage());
        assert_eq!(cached.fieldnorms_buffers, original.fieldnorms_buffers);
        for limit in [0, 1000, 64000, 2_000_000_000] {
            assert_eq!(cached.mem_usage() >= limit, original.mem_usage() >= limit);
        }
    }

    fn serialize(writer: &FieldNormsWriter) -> Vec<u8> {
        let directory = RamDirectory::create();
        let path = Path::new("fieldnorms");
        let serializer =
            FieldNormsSerializer::from_write(directory.open_write(path).unwrap()).unwrap();
        writer.serialize(serializer, None).unwrap();
        directory
            .open_read(path)
            .unwrap()
            .read_bytes()
            .unwrap()
            .to_vec()
    }

    #[test]
    fn test_e054_capacity_oracle_every_mutation_and_serialized_bytes() {
        for width in [0, 1, 64, 105] {
            let mut builder = Schema::builder();
            let mut fields = Vec::new();
            let mut ignored = Vec::new();
            for i in 0..width {
                fields.push(builder.add_text_field(&format!("tracked{i}"), TEXT));
                ignored.push(builder.add_text_field(&format!("stored{i}"), STORED));
            }
            let schema = builder.build();
            let mut cached = FieldNormsWriter::for_schema_with_cache(&schema, true);
            let mut original = FieldNormsWriter::for_schema_with_cache(&schema, false);
            assert_oracle(&cached, &original);
            for doc in [0, 1, 7, 999, 1000, 1001, 2048, 4003, 8192] {
                for (i, field) in fields.iter().enumerate() {
                    if i % 3 == 0 && doc % 2 == 0 {
                        continue;
                    }
                    cached.record(doc, *field, doc + i as u32);
                    original.record(doc, *field, doc + i as u32);
                    assert_oracle(&cached, &original);
                }
                for field in ignored.iter().copied().chain([Field::from_field_id(9999)]) {
                    cached.record(doc, field, 1);
                    original.record(doc, field, 1);
                    assert_oracle(&cached, &original);
                }
                cached.fill_up_to_max_doc(doc + 1);
                original.fill_up_to_max_doc(doc + 1);
                assert_oracle(&cached, &original);
            }
            for max_doc in [4096, 16385] {
                cached.fill_up_to_max_doc(max_doc);
                original.fill_up_to_max_doc(max_doc);
                assert_oracle(&cached, &original);
            }
            assert_eq!(serialize(&cached), serialize(&original));
        }
    }

    #[test]
    fn test_e054_duplicate_record_unwind_keeps_exact_accounting() {
        let mut builder = Schema::builder();
        let field = builder.add_text_field("tracked", TEXT);
        let schema = builder.build();
        let mut writer = FieldNormsWriter::for_schema_with_cache(&schema, true);
        writer.record(2001, field, 1);
        let before = writer.mem_usage();
        assert!(catch_unwind(AssertUnwindSafe(|| writer.record(2001, field, 2))).is_err());
        assert_eq!(writer.mem_usage(), before);
        assert_eq!(writer.mem_usage(), writer.recomputed_mem_usage());
    }
}
