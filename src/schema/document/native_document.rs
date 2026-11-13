//! An insertion-ordered, ownership-taking document backed by native [`OwnedValue`]s.
//!
//! [`TantivyDocument`] (a.k.a. `CompactDoc`) serializes every value into a byte buffer on
//! insertion and binary-decodes it again on read. `NativeDocument` instead keeps the owned
//! values as-is, in insertion order, so a producer that already holds an [`OwnedValue`] can
//! move it in and a consumer can borrow it back without an encode/decode round trip.
//!
//! It is a general [`Document`]: it preserves repeated root fields and their insertion order
//! and reuses the existing [`OwnedValue`] type rather than inventing new ones.

use std::collections::BTreeMap;
use std::net::Ipv6Addr;

use common::DateTime;

use super::{Document, OwnedValue, Value};
use crate::schema::{Facet, Field, TantivyDocument};
use crate::tokenizer::PreTokenizedString;

/// An insertion-ordered document that owns its native values.
///
/// Unlike [`TantivyDocument`], which stores encoded bytes, this keeps the
/// [`OwnedValue`]s themselves. The representation is the general "native transport":
/// any caller that produces owned values can move them in, and any [`Document`]
/// consumer can borrow them back.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct NativeDocument {
    field_values: Vec<(Field, OwnedValue)>,
}

impl NativeDocument {
    /// Creates a new, empty document.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new, empty document with room for `capacity` field values.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            field_values: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of field values in the document.
    pub fn len(&self) -> usize {
        self.field_values.len()
    }

    /// Returns `true` if the document has no field value.
    pub fn is_empty(&self) -> bool {
        self.field_values.is_empty()
    }

    /// Shrinks the capacity of the document to fit the data.
    pub fn shrink_to_fit(&mut self) {
        self.field_values.shrink_to_fit();
    }

    /// Moves an owned value into the document, preserving insertion order and repeated roots.
    ///
    /// This is the ownership-taking counterpart of
    /// [`add_field_value`](Self::add_field_value): the value is not encoded or copied.
    pub fn add_owned_value(&mut self, field: Field, value: OwnedValue) {
        self.field_values.push((field, value));
    }

    /// Adds a value by borrowing it: the value is materialized into an [`OwnedValue`].
    pub fn add_field_value<'a, V: Value<'a>>(&mut self, field: Field, value: V) {
        self.field_values
            .push((field, OwnedValue::from(value.as_value())));
    }

    /// Add a text field.
    pub fn add_text<S: AsRef<str>>(&mut self, field: Field, text: S) {
        self.add_owned_value(field, OwnedValue::Str(text.as_ref().to_string()));
    }

    /// Add a u64 field.
    pub fn add_u64(&mut self, field: Field, value: u64) {
        self.add_owned_value(field, OwnedValue::U64(value));
    }

    /// Add an i64 field.
    pub fn add_i64(&mut self, field: Field, value: i64) {
        self.add_owned_value(field, OwnedValue::I64(value));
    }

    /// Add an f64 field.
    pub fn add_f64(&mut self, field: Field, value: f64) {
        self.add_owned_value(field, OwnedValue::F64(value));
    }

    /// Add a bool field.
    pub fn add_bool(&mut self, field: Field, value: bool) {
        self.add_owned_value(field, OwnedValue::Bool(value));
    }

    /// Add a date field.
    pub fn add_date(&mut self, field: Field, value: DateTime) {
        self.add_owned_value(field, OwnedValue::Date(value));
    }

    /// Add a bytes field.
    pub fn add_bytes(&mut self, field: Field, value: &[u8]) {
        self.add_owned_value(field, OwnedValue::Bytes(value.to_vec()));
    }

    /// Add an IP address field. Internally only `Ipv6Addr` is used.
    pub fn add_ip_addr(&mut self, field: Field, value: Ipv6Addr) {
        self.add_owned_value(field, OwnedValue::IpAddr(value));
    }

    /// Add a facet field.
    pub fn add_facet<F>(&mut self, field: Field, path: F)
    where Facet: From<F> {
        let facet = Facet::from(path);
        self.add_owned_value(field, OwnedValue::Facet(facet));
    }

    /// Add a pre-tokenized text field.
    pub fn add_pre_tokenized_text(&mut self, field: Field, value: PreTokenizedString) {
        self.add_owned_value(field, OwnedValue::PreTokStr(value));
    }

    /// Add a dynamic object field.
    pub fn add_object(&mut self, field: Field, object: BTreeMap<String, OwnedValue>) {
        self.add_owned_value(field, OwnedValue::from(object));
    }

    /// Add a plugin-defined custom field value.
    pub fn add_custom(&mut self, field: Field, value: &[u8]) {
        self.add_owned_value(field, OwnedValue::Custom(value.to_vec()));
    }

    /// Returns all of the values associated with the given field, in insertion order.
    pub fn get_all(&self, field: Field) -> impl Iterator<Item = &OwnedValue> + '_ {
        self.field_values
            .iter()
            .filter(move |(field_value_field, _)| *field_value_field == field)
            .map(|(_, value)| value)
    }

    /// Returns the first value associated with the given field.
    pub fn get_first(&self, field: Field) -> Option<&OwnedValue> {
        self.get_all(field).next()
    }

    /// Iterates over the `(field, value)` pairs in insertion order.
    pub fn field_values(&self) -> impl Iterator<Item = (Field, &OwnedValue)> {
        self.field_values
            .iter()
            .map(|(field, value)| (*field, value))
    }
}

/// The iterator produced by [`NativeDocument::iter_fields_and_values`].
pub struct NativeDocumentIter<'a>(std::slice::Iter<'a, (Field, OwnedValue)>);

impl<'a> Iterator for NativeDocumentIter<'a> {
    type Item = (Field, &'a OwnedValue);

    fn next(&mut self) -> Option<Self::Item> {
        let (field, value) = self.0.next()?;
        Some((*field, value))
    }
}

impl Document for NativeDocument {
    type Value<'a> = &'a OwnedValue;
    type FieldsValuesIter<'a> = NativeDocumentIter<'a>;

    fn iter_fields_and_values(&self) -> Self::FieldsValuesIter<'_> {
        NativeDocumentIter(self.field_values.iter())
    }
}

impl From<NativeDocument> for TantivyDocument {
    fn from(doc: NativeDocument) -> Self {
        doc.to_tantivy_document()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Schema, TEXT};

    #[test]
    fn test_insertion_order_and_repeated_roots() {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let u64_field = schema_builder.add_u64_field("num", crate::schema::INDEXED);
        let schema = schema_builder.build();

        let mut doc = NativeDocument::new();
        doc.add_text(text_field, "first");
        doc.add_u64(u64_field, 1);
        doc.add_text(text_field, "second");

        let fields: Vec<_> = doc.iter_fields_and_values().collect();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].0, text_field);
        assert_eq!(fields[1].0, u64_field);
        assert_eq!(fields[2].0, text_field);
        assert_eq!(doc.get_first(text_field).unwrap().as_str(), Some("first"));
        let all: Vec<_> = doc.get_all(text_field).collect();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].as_str(), Some("first"));
        assert_eq!(all[1].as_str(), Some("second"));

        // `to_named_doc` keeps both repeated values in order.
        let named = doc.to_named_doc(&schema);
        let values = named.0.get("text").unwrap();
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_into_tantivy_document_matches_values() {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let u64_field = schema_builder.add_u64_field("num", crate::schema::INDEXED);
        schema_builder.build();

        let mut native = NativeDocument::new();
        native.add_text(text_field, "hello");
        native.add_u64(u64_field, 42);
        native.add_text(text_field, "world");

        let compact: TantivyDocument = native.into_tantivy_document();
        assert_eq!(compact.get_all(text_field).count(), 2);
        assert_eq!(compact.get_first(text_field).unwrap().as_str(), Some("hello"));
        assert_eq!(compact.get_first(u64_field).unwrap().as_u64(), Some(42));
    }

    #[test]
    fn test_owned_value_moved_in_place() {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        schema_builder.build();

        let value = OwnedValue::Str("owned".to_string());
        let mut doc = NativeDocument::with_capacity(1);
        doc.add_owned_value(text_field, value);
        assert_eq!(doc.len(), 1);
        assert_eq!(doc.get_first(text_field).unwrap().as_str(), Some("owned"));
    }
}
