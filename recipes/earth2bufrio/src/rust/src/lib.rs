use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ffi::FFI_ArrowArray;
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3_arrow::PyArrowType;
use rayon::prelude::*;
use serde::Deserialize;

// ── Table types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
struct TableBEntryJson {
    name: String,
    units: String,
    scale: i32,
    reference_value: i64,
    bit_width: u32,
}

#[derive(Debug, Clone)]
struct TableBEntry {
    name: String,
    units: String,
    scale: i32,
    reference_value: i64,
    bit_width: u32,
}

#[derive(Debug, Clone)]
struct TableDEntry {
    descriptors: Vec<u32>,
}

struct TableSet {
    table_b: HashMap<u32, TableBEntry>,
    table_d: HashMap<u32, TableDEntry>,
}

// ── BUFR message types ───────────────────────────────────────────────

struct BufrMessage {
    data: Vec<u8>,
    offset: usize,
    index: usize,
}

#[derive(Debug, Clone)]
struct IndicatorSection {
    length: u32,
    edition: u8,
}

#[derive(Debug, Clone)]
struct IdentificationSection {
    originating_center: u16,
    data_category: u8,
    year: u16,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
    num_subsets: u16,
    observed: bool,
    compressed: bool,
}

struct ParsedMessage {
    indicator: IndicatorSection,
    identification: IdentificationSection,
    descriptors: Vec<u32>,
    data_bytes: Vec<u8>,
}

// ── Expanded descriptor types ────────────────────────────────────────

#[derive(Debug, Clone)]
struct ExpandedDescriptor {
    fxy: u32,
    entry: TableBEntry,
}

#[derive(Debug, Clone)]
enum ExpandedItem {
    Descriptor(ExpandedDescriptor),
    DelayedReplication {
        factor: ExpandedDescriptor,
        group: Vec<ExpandedItem>,
    },
}

#[derive(Debug, Clone)]
enum DecodedValue {
    Float(f64),
    Str(String),
    Missing,
}

// ── Row type for Arrow conversion ────────────────────────────────────

type DecodedRow = HashMap<String, RowValue>;

#[derive(Debug, Clone)]
enum RowValue {
    Int(i32),
    Float(f64),
    Str(String),
    FloatList(Vec<f64>),
    Null,
}

// ── PyO3 entry point ─────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (file_path, table_b_json, table_d_json, mnemonics=None, data_category_filter=None))]
fn read_bufr_rust(
    py: Python<'_>,
    file_path: &str,
    table_b_json: &str,
    table_d_json: &str,
    mnemonics: Option<Vec<String>>,
    data_category_filter: Option<i32>,
) -> PyResult<PyArrowType<RecordBatch>> {
    // TODO: implement full pipeline — for now return empty batch
    let schema = Arc::new(Schema::new(vec![
        Field::new("message_type", DataType::Utf8, true),
        Field::new("message_index", DataType::Int32, true),
        Field::new("subset_index", DataType::Int32, true),
        Field::new("YEAR", DataType::Int32, true),
        Field::new("MNTH", DataType::Int32, true),
        Field::new("DAYS", DataType::Int32, true),
        Field::new("HOUR", DataType::Int32, true),
        Field::new("MINU", DataType::Int32, true),
        Field::new("SECO", DataType::Int32, true),
    ]));

    let batch = RecordBatch::new_empty(schema);
    Ok(PyArrowType(batch))
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_bufr_rust, m)?)?;
    Ok(())
}
