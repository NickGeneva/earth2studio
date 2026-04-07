use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, Int32Builder, ListBuilder, ArrayBuilder, StringBuilder};
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

impl TableSet {
    fn from_json(table_b_json: &str, table_d_json: &str) -> Result<Self, String> {
        let raw_b: HashMap<String, TableBEntryJson> =
            serde_json::from_str(table_b_json).map_err(|e| format!("Table B parse error: {e}"))?;
        let raw_d: HashMap<String, Vec<u32>> =
            serde_json::from_str(table_d_json).map_err(|e| format!("Table D parse error: {e}"))?;

        let mut table_b = HashMap::with_capacity(raw_b.len());
        for (key, entry) in raw_b {
            let fxy: u32 = key.parse().map_err(|e| format!("Bad Table B key {key}: {e}"))?;
            table_b.insert(fxy, TableBEntry {
                name: entry.name,
                units: entry.units,
                scale: entry.scale,
                reference_value: entry.reference_value,
                bit_width: entry.bit_width,
            });
        }

        let mut table_d = HashMap::with_capacity(raw_d.len());
        for (key, members) in raw_d {
            let fxy: u32 = key.parse().map_err(|e| format!("Bad Table D key {key}: {e}"))?;
            table_d.insert(fxy, TableDEntry { descriptors: members });
        }

        Ok(TableSet { table_b, table_d })
    }

    fn lookup_b(&self, fxy: u32) -> Option<&TableBEntry> {
        self.table_b.get(&fxy)
    }

    fn lookup_d(&self, fxy: u32) -> Option<&TableDEntry> {
        self.table_d.get(&fxy)
    }
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

// ── Message reader ───────────────────────────────────────────────────

const BUFR_MARKER: &[u8] = b"BUFR";
const END_MARKER: &[u8] = b"7777";

fn read_messages(data: &[u8]) -> Result<Vec<BufrMessage>, String> {
    let mut messages = Vec::new();
    let mut pos = 0;
    let mut counter = 0usize;
    let len = data.len();

    while pos < len {
        let start = match find_marker(data, pos) {
            Some(s) => s,
            None => break,
        };

        if start + 7 >= len {
            return Err(format!(
                "Truncated BUFR header at offset {start}: need 8 bytes but only {} available",
                len - start
            ));
        }

        let msg_len = ((data[start + 4] as u32) << 16)
            | ((data[start + 5] as u32) << 8)
            | (data[start + 6] as u32);
        let msg_len = msg_len as usize;

        if start + msg_len > len {
            return Err(format!(
                "Truncated BUFR message at offset {start}: declared length {msg_len} but only {} bytes available",
                len - start
            ));
        }

        let end_slice = &data[start + msg_len - 4..start + msg_len];
        if end_slice != END_MARKER {
            return Err(format!(
                "Bad end marker at offset {start}: expected b\"7777\", got {:?}",
                end_slice
            ));
        }

        messages.push(BufrMessage {
            data: data[start..start + msg_len].to_vec(),
            offset: start,
            index: counter,
        });

        counter += 1;
        pos = start + msg_len;
    }

    Ok(messages)
}

fn find_marker(data: &[u8], from: usize) -> Option<usize> {
    data[from..].windows(4).position(|w| w == BUFR_MARKER).map(|p| from + p)
}

// ── Section parsing ──────────────────────────────────────────────────

fn parse_message(msg: &BufrMessage) -> Result<ParsedMessage, String> {
    let data = &msg.data;

    if data.len() < 8 || &data[0..4] != BUFR_MARKER {
        return Err("Missing BUFR magic bytes".into());
    }
    let length = u32::from_be_bytes([0, data[4], data[5], data[6]]);
    let edition = data[7];

    let indicator = IndicatorSection { length, edition };
    let mut offset = 8usize;

    let identification = match edition {
        4 => parse_identification_ed4(data, &mut offset)?,
        3 => parse_identification_ed3(data, &mut offset)?,
        _ => return Err(format!("Unsupported BUFR edition {edition}")),
    };

    let has_optional = has_optional_section(data, edition);
    if has_optional {
        let sec2_len = u32::from_be_bytes([0, data[offset], data[offset + 1], data[offset + 2]]) as usize;
        offset += sec2_len;
    }

    let (descriptors, num_subsets, observed, compressed, sec3_end) =
        parse_data_description(data, offset)?;
    offset = sec3_end;

    let sec4_len = u32::from_be_bytes([0, data[offset], data[offset + 1], data[offset + 2]]) as usize;
    let data_bytes = data[offset + 4..offset + sec4_len].to_vec();

    let ident = IdentificationSection {
        num_subsets,
        observed,
        compressed,
        ..identification
    };

    Ok(ParsedMessage {
        indicator,
        identification: ident,
        descriptors,
        data_bytes,
    })
}

fn parse_identification_ed4(data: &[u8], offset: &mut usize) -> Result<IdentificationSection, String> {
    let sec_len = u32::from_be_bytes([0, data[*offset], data[*offset + 1], data[*offset + 2]]) as usize;
    let base = *offset + 3;

    let center = u16::from_be_bytes([data[base + 1], data[base + 2]]);
    let data_cat = data[base + 7];
    let year = u16::from_be_bytes([data[base + 12], data[base + 13]]);
    let month = data[base + 14];
    let day = data[base + 15];
    let hour = data[base + 16];
    let minute = data[base + 17];
    let second = data[base + 18];

    *offset += sec_len;
    Ok(IdentificationSection {
        originating_center: center,
        data_category: data_cat,
        year, month, day, hour, minute, second,
        num_subsets: 0, observed: false, compressed: false,
    })
}

fn parse_identification_ed3(data: &[u8], offset: &mut usize) -> Result<IdentificationSection, String> {
    let sec_len = u32::from_be_bytes([0, data[*offset], data[*offset + 1], data[*offset + 2]]) as usize;
    let base = *offset + 3;

    let center = data[base + 2] as u16;
    let data_cat = data[base + 5];
    let yoc = data[base + 9];
    let year = if yoc >= 70 { 1900 + yoc as u16 } else { 2000 + yoc as u16 };
    let month = data[base + 10];
    let day = data[base + 11];
    let hour = data[base + 12];
    let minute = data[base + 13];

    *offset += sec_len;
    Ok(IdentificationSection {
        originating_center: center,
        data_category: data_cat,
        year, month, day, hour, minute, second: 0,
        num_subsets: 0, observed: false, compressed: false,
    })
}

fn parse_data_description(data: &[u8], offset: usize) -> Result<(Vec<u32>, u16, bool, bool, usize), String> {
    let sec_len = u32::from_be_bytes([0, data[offset], data[offset + 1], data[offset + 2]]) as usize;
    let base = offset + 3;

    let num_subsets = u16::from_be_bytes([data[base + 1], data[base + 2]]);
    let flags = data[base + 3];
    let observed = (flags & 0x80) != 0;
    let compressed = (flags & 0x40) != 0;

    let num_desc = (sec_len - 7) / 2;
    let mut descriptors = Vec::with_capacity(num_desc);
    let mut desc_offset = base + 4;
    for _ in 0..num_desc {
        let packed = u16::from_be_bytes([data[desc_offset], data[desc_offset + 1]]);
        let fxy = packed_to_fxy(packed);
        descriptors.push(fxy);
        desc_offset += 2;
    }

    Ok((descriptors, num_subsets, observed, compressed, offset + sec_len))
}

fn packed_to_fxy(packed: u16) -> u32 {
    let f = ((packed >> 14) & 0x3) as u32;
    let x = ((packed >> 8) & 0x3F) as u32;
    let y = (packed & 0xFF) as u32;
    f * 100000 + x * 1000 + y
}

fn has_optional_section(data: &[u8], edition: u8) -> bool {
    match edition {
        4 => (data[8 + 3 + 6] & 0x80) != 0,
        3 => (data[8 + 3 + 4] & 0x80) != 0,
        _ => false,
    }
}

// ── Descriptor expansion ─────────────────────────────────────────────

const MAX_DEPTH: usize = 50;

struct OperatorState {
    width_delta: i32,
    scale_delta: i32,
    assoc_field_width: u32,
}

impl Default for OperatorState {
    fn default() -> Self {
        Self { width_delta: 0, scale_delta: 0, assoc_field_width: 0 }
    }
}

fn expand_descriptors(raw_ids: &[u32], tables: &TableSet) -> Result<Vec<ExpandedItem>, String> {
    let mut state = OperatorState::default();
    expand_inner(raw_ids, tables, &mut state, 0)
}

fn expand_inner(
    ids: &[u32],
    tables: &TableSet,
    state: &mut OperatorState,
    depth: usize,
) -> Result<Vec<ExpandedItem>, String> {
    if depth > MAX_DEPTH {
        return Err(format!("Descriptor expansion exceeded max depth ({MAX_DEPTH})"));
    }

    let mut result = Vec::new();
    let mut idx = 0;

    while idx < ids.len() {
        let fxy = ids[idx];
        let f = fxy / 100000;
        let x = (fxy % 100000) / 1000;
        let y = fxy % 1000;

        match f {
            0 => {
                let entry = tables.lookup_b(fxy)
                    .ok_or_else(|| format!("Unknown Table B descriptor: {fxy:06}"))?;

                let mut entry = entry.clone();
                if state.width_delta != 0 || state.scale_delta != 0 {
                    entry.scale += state.scale_delta;
                    entry.bit_width = (entry.bit_width as i32 + state.width_delta) as u32;
                }

                if state.assoc_field_width > 0 && x != 31 {
                    let assoc = TableBEntry {
                        name: "ASSOCIATED FIELD".into(),
                        units: "CODE TABLE".into(),
                        scale: 0,
                        reference_value: 0,
                        bit_width: state.assoc_field_width,
                    };
                    result.push(ExpandedItem::Descriptor(ExpandedDescriptor {
                        fxy: 999999,
                        entry: assoc,
                    }));
                }

                result.push(ExpandedItem::Descriptor(ExpandedDescriptor { fxy, entry }));
                idx += 1;
            }
            1 => {
                let num_desc = x as usize;
                let rep_count = y;

                if rep_count == 0 {
                    idx += 1;
                    if idx >= ids.len() {
                        return Err("Delayed replication missing factor descriptor".into());
                    }
                    let factor_fxy = ids[idx];
                    let factor_list = expand_inner(&[factor_fxy], tables, state, depth + 1)?;
                    let factor = match factor_list.into_iter().next() {
                        Some(ExpandedItem::Descriptor(d)) => d,
                        _ => return Err("Delayed replication factor must be Table B element".into()),
                    };
                    idx += 1;

                    let group_ids: Vec<u32> = ids[idx..idx + num_desc].to_vec();
                    let group = expand_inner(&group_ids, tables, state, depth + 1)?;
                    idx += num_desc;

                    result.push(ExpandedItem::DelayedReplication { factor, group });
                } else {
                    idx += 1;
                    let group_ids: Vec<u32> = ids[idx..idx + num_desc].to_vec();
                    let group = expand_inner(&group_ids, tables, state, depth + 1)?;
                    idx += num_desc;

                    for _ in 0..rep_count {
                        result.extend(group.clone());
                    }
                }
            }
            2 => {
                match x {
                    1 => {
                        if y == 0 { state.width_delta = 0; }
                        else { state.width_delta = y as i32 - 128; }
                    }
                    2 => {
                        if y == 0 { state.scale_delta = 0; }
                        else { state.scale_delta = y as i32 - 128; }
                    }
                    4 => {
                        state.assoc_field_width = y;
                    }
                    7 => {
                        if y == 0 {
                            state.width_delta = 0;
                            state.scale_delta = 0;
                        } else {
                            state.scale_delta = y as i32;
                            state.width_delta = ((10 * y + 2) / 3) as i32;
                        }
                    }
                    _ => {}
                }
                idx += 1;
            }
            3 => {
                let d_entry = tables.lookup_d(fxy)
                    .ok_or_else(|| format!("Unknown Table D descriptor: {fxy:06}"))?;
                let members = d_entry.descriptors.clone();
                let expanded = expand_inner(&members, tables, state, depth + 1)?;
                result.extend(expanded);
                idx += 1;
            }
            _ => {
                return Err(format!("Unknown descriptor class F={f} in {fxy:06}"));
            }
        }
    }

    Ok(result)
}

// ── Bit-level decoder ────────────────────────────────────────────────

fn read_bits(data: &[u8], bit_offset: usize, num_bits: u32) -> u64 {
    let mut result: u64 = 0;
    for i in 0..num_bits as usize {
        let byte_idx = (bit_offset + i) / 8;
        let bit_idx = (bit_offset + i) % 8;
        result = (result << 1) | ((data[byte_idx] >> (7 - bit_idx)) & 1) as u64;
    }
    result
}

fn is_missing(raw: u64, num_bits: u32) -> bool {
    raw == (1u64 << num_bits) - 1
}

fn decode_value(raw: u64, entry: &TableBEntry) -> Option<f64> {
    if is_missing(raw, entry.bit_width) {
        return None;
    }
    Some((raw as f64 + entry.reference_value as f64) / 10f64.powi(entry.scale))
}

fn decode_string(data: &[u8], bit_offset: usize, num_bytes: usize) -> Option<String> {
    let mut bytes = vec![0u8; num_bytes];
    let mut all_ones = true;
    for i in 0..num_bytes {
        let b = read_bits(data, bit_offset + i * 8, 8) as u8;
        bytes[i] = b;
        if b != 0xFF { all_ones = false; }
    }
    if all_ones { return None; }
    let s = String::from_utf8_lossy(&bytes);
    Some(s.trim_end_matches(|c: char| c == ' ' || c == '\0').to_string())
}

type SubsetValues = Vec<(ExpandedDescriptor, DecodedValue)>;

fn decode(
    expanded: &[ExpandedItem],
    data_bytes: &[u8],
    num_subsets: u16,
    compressed: bool,
) -> Result<Vec<SubsetValues>, String> {
    if compressed {
        decode_compressed(expanded, data_bytes, num_subsets)
    } else {
        decode_uncompressed(expanded, data_bytes, num_subsets)
    }
}

fn decode_uncompressed(
    expanded: &[ExpandedItem],
    data_bytes: &[u8],
    num_subsets: u16,
) -> Result<Vec<SubsetValues>, String> {
    let mut subsets = Vec::with_capacity(num_subsets as usize);
    let mut bit_offset = 0usize;

    for _ in 0..num_subsets {
        let mut values = Vec::new();
        bit_offset = decode_items_uncompressed(expanded, data_bytes, bit_offset, &mut values)?;
        subsets.push(values);
    }

    Ok(subsets)
}

fn decode_items_uncompressed(
    items: &[ExpandedItem],
    data: &[u8],
    mut bit_offset: usize,
    values: &mut SubsetValues,
) -> Result<usize, String> {
    for item in items {
        match item {
            ExpandedItem::Descriptor(desc) => {
                let entry = &desc.entry;
                if entry.bit_width == 0 { continue; }

                if entry.units == "CCITT IA5" {
                    let num_bytes = entry.bit_width as usize / 8;
                    let val = decode_string(data, bit_offset, num_bytes);
                    let dv = match val {
                        Some(s) => DecodedValue::Str(s),
                        None => DecodedValue::Missing,
                    };
                    values.push((desc.clone(), dv));
                    bit_offset += entry.bit_width as usize;
                } else {
                    let raw = read_bits(data, bit_offset, entry.bit_width);
                    bit_offset += entry.bit_width as usize;
                    let dv = match decode_value(raw, entry) {
                        Some(v) => DecodedValue::Float(v),
                        None => DecodedValue::Missing,
                    };
                    values.push((desc.clone(), dv));
                }
            }
            ExpandedItem::DelayedReplication { factor, group } => {
                let entry = &factor.entry;
                let raw = read_bits(data, bit_offset, entry.bit_width);
                bit_offset += entry.bit_width as usize;
                let factor_val = match decode_value(raw, entry) {
                    Some(v) => DecodedValue::Float(v),
                    None => DecodedValue::Missing,
                };
                values.push((factor.clone(), factor_val));

                let rep_count = (raw as i64 + entry.reference_value) as usize;
                for _ in 0..rep_count {
                    bit_offset = decode_items_uncompressed(group, data, bit_offset, values)?;
                }
            }
        }
    }
    Ok(bit_offset)
}

fn decode_compressed(
    expanded: &[ExpandedItem],
    data_bytes: &[u8],
    num_subsets: u16,
) -> Result<Vec<SubsetValues>, String> {
    let ns = num_subsets as usize;
    let mut subset_values: Vec<SubsetValues> = (0..ns).map(|_| Vec::new()).collect();
    decode_items_compressed(expanded, data_bytes, 0, ns, &mut subset_values)?;
    Ok(subset_values)
}

fn decode_items_compressed(
    items: &[ExpandedItem],
    data: &[u8],
    mut bit_offset: usize,
    num_subsets: usize,
    subset_values: &mut [SubsetValues],
) -> Result<usize, String> {
    for item in items {
        match item {
            ExpandedItem::Descriptor(desc) => {
                let entry = &desc.entry;
                if entry.bit_width == 0 { continue; }

                if entry.units == "CCITT IA5" {
                    let num_bytes = entry.bit_width as usize / 8;
                    let r0 = decode_string(data, bit_offset, num_bytes);
                    bit_offset += entry.bit_width as usize;

                    let nbinc = read_bits(data, bit_offset, 6) as usize;
                    bit_offset += 6;

                    if nbinc == 0 {
                        let dv = match &r0 {
                            Some(s) => DecodedValue::Str(s.clone()),
                            None => DecodedValue::Missing,
                        };
                        for sv in subset_values.iter_mut() {
                            sv.push((desc.clone(), dv.clone()));
                        }
                    } else {
                        for sv in subset_values.iter_mut() {
                            let val = decode_string(data, bit_offset, nbinc);
                            bit_offset += nbinc * 8;
                            let dv = match val {
                                Some(s) => DecodedValue::Str(s),
                                None => DecodedValue::Missing,
                            };
                            sv.push((desc.clone(), dv));
                        }
                    }
                } else {
                    let r0 = read_bits(data, bit_offset, entry.bit_width);
                    bit_offset += entry.bit_width as usize;

                    let nbinc = read_bits(data, bit_offset, 6) as u32;
                    bit_offset += 6;

                    if nbinc == 0 {
                        let dv = match decode_value(r0, entry) {
                            Some(v) => DecodedValue::Float(v),
                            None => DecodedValue::Missing,
                        };
                        for sv in subset_values.iter_mut() {
                            sv.push((desc.clone(), dv.clone()));
                        }
                    } else {
                        for sv in subset_values.iter_mut() {
                            let increment = read_bits(data, bit_offset, nbinc);
                            bit_offset += nbinc as usize;
                            let combined = r0 + increment;
                            let dv = match decode_value(combined, entry) {
                                Some(v) => DecodedValue::Float(v),
                                None => DecodedValue::Missing,
                            };
                            sv.push((desc.clone(), dv));
                        }
                    }
                }
            }
            ExpandedItem::DelayedReplication { factor, group } => {
                let entry = &factor.entry;
                let r0 = read_bits(data, bit_offset, entry.bit_width);
                bit_offset += entry.bit_width as usize;

                let nbinc = read_bits(data, bit_offset, 6) as u32;
                bit_offset += 6;

                let factor_val = match decode_value(r0, entry) {
                    Some(v) => DecodedValue::Float(v),
                    None => DecodedValue::Missing,
                };

                let rep_count = (r0 as i64 + entry.reference_value) as usize;

                if nbinc == 0 {
                    for sv in subset_values.iter_mut() {
                        sv.push((factor.clone(), factor_val.clone()));
                    }
                } else {
                    for sv in subset_values.iter_mut() {
                        let inc = read_bits(data, bit_offset, nbinc);
                        bit_offset += nbinc as usize;
                        let combined = r0 + inc;
                        let dv = match decode_value(combined, entry) {
                            Some(v) => DecodedValue::Float(v),
                            None => DecodedValue::Missing,
                        };
                        sv.push((factor.clone(), dv));
                    }
                }

                for _ in 0..rep_count {
                    bit_offset = decode_items_compressed(group, data, bit_offset, subset_values.len(), subset_values)?;
                }
            }
        }
    }
    Ok(bit_offset)
}

// ── Arrow table construction ─────────────────────────────────────────

const FIXED_KEYS: &[&str] = &[
    "message_type", "message_index", "subset_index",
    "YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO",
];

fn subsets_to_rows(
    msg_index: usize,
    ident: &IdentificationSection,
    subsets: &[SubsetValues],
) -> Vec<DecodedRow> {
    let mut rows = Vec::with_capacity(subsets.len());
    for (subset_idx, values) in subsets.iter().enumerate() {
        let mut row = DecodedRow::new();
        row.insert("message_type".into(), RowValue::Str(ident.data_category.to_string()));
        row.insert("message_index".into(), RowValue::Int(msg_index as i32));
        row.insert("subset_index".into(), RowValue::Int(subset_idx as i32));
        row.insert("YEAR".into(), RowValue::Int(ident.year as i32));
        row.insert("MNTH".into(), RowValue::Int(ident.month as i32));
        row.insert("DAYS".into(), RowValue::Int(ident.day as i32));
        row.insert("HOUR".into(), RowValue::Int(ident.hour as i32));
        row.insert("MINU".into(), RowValue::Int(ident.minute as i32));
        row.insert("SECO".into(), RowValue::Int(ident.second as i32));

        for (desc, val) in values {
            let name = &desc.entry.name;
            match val {
                DecodedValue::Float(v) => {
                    if let Some(existing) = row.get_mut(name) {
                        match existing {
                            RowValue::FloatList(list) => list.push(*v),
                            RowValue::Float(prev) => {
                                let prev_v = *prev;
                                *existing = RowValue::FloatList(vec![prev_v, *v]);
                            }
                            _ => { row.insert(name.clone(), RowValue::Float(*v)); }
                        }
                    } else {
                        row.insert(name.clone(), RowValue::Float(*v));
                    }
                }
                DecodedValue::Str(s) => {
                    row.insert(name.clone(), RowValue::Str(s.clone()));
                }
                DecodedValue::Missing => {
                    if !row.contains_key(name) {
                        row.insert(name.clone(), RowValue::Null);
                    }
                }
            }
        }
        rows.push(row);
    }
    rows
}

fn rows_to_record_batch(
    rows: &[DecodedRow],
    mnemonics_filter: &Option<Vec<String>>,
) -> Result<RecordBatch, String> {
    if rows.is_empty() {
        let schema = Arc::new(Schema::new(
            FIXED_KEYS.iter().map(|k| {
                if *k == "message_type" {
                    Field::new(*k, DataType::Utf8, true)
                } else {
                    Field::new(*k, DataType::Int32, true)
                }
            }).collect::<Vec<_>>()
        ));
        return RecordBatch::try_new(schema, FIXED_KEYS.iter().map(|k| -> ArrayRef {
            if *k == "message_type" {
                Arc::new(arrow::array::StringArray::from(Vec::<Option<&str>>::new()))
            } else {
                Arc::new(arrow::array::Int32Array::from(Vec::<Option<i32>>::new()))
            }
        }).collect()).map_err(|e| format!("Arrow error: {e}"));
    }

    let mut mnemonic_keys: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for row in rows {
        for key in row.keys() {
            if !FIXED_KEYS.contains(&key.as_str()) && seen.insert(key.clone()) {
                mnemonic_keys.push(key.clone());
            }
        }
    }

    if let Some(filter) = mnemonics_filter {
        let allowed: std::collections::HashSet<&str> = filter.iter().map(|s| s.as_str()).collect();
        mnemonic_keys.retain(|k| allowed.contains(k.as_str()));
    }

    let mut col_types: HashMap<String, &str> = HashMap::new();
    for key in &mnemonic_keys {
        for row in rows {
            if let Some(val) = row.get(key) {
                match val {
                    RowValue::Float(_) => { col_types.insert(key.clone(), "float64"); break; }
                    RowValue::Str(_) => { col_types.insert(key.clone(), "string"); break; }
                    RowValue::FloatList(_) => { col_types.insert(key.clone(), "list_float64"); break; }
                    RowValue::Null | RowValue::Int(_) => {}
                }
            }
        }
        col_types.entry(key.clone()).or_insert("float64");
    }

    let n = rows.len();

    let mut msg_type_builder = StringBuilder::with_capacity(n, n * 8);
    let mut msg_idx_builder = Int32Builder::with_capacity(n);
    let mut sub_idx_builder = Int32Builder::with_capacity(n);
    let mut year_builder = Int32Builder::with_capacity(n);
    let mut mnth_builder = Int32Builder::with_capacity(n);
    let mut days_builder = Int32Builder::with_capacity(n);
    let mut hour_builder = Int32Builder::with_capacity(n);
    let mut minu_builder = Int32Builder::with_capacity(n);
    let mut seco_builder = Int32Builder::with_capacity(n);

    for row in rows {
        match row.get("message_type") {
            Some(RowValue::Str(s)) => msg_type_builder.append_value(s),
            _ => msg_type_builder.append_null(),
        }
        match row.get("message_index") {
            Some(RowValue::Int(v)) => msg_idx_builder.append_value(*v),
            _ => msg_idx_builder.append_null(),
        }
        match row.get("subset_index") {
            Some(RowValue::Int(v)) => sub_idx_builder.append_value(*v),
            _ => sub_idx_builder.append_null(),
        }
        match row.get("YEAR") {
            Some(RowValue::Int(v)) => year_builder.append_value(*v),
            _ => year_builder.append_null(),
        }
        match row.get("MNTH") {
            Some(RowValue::Int(v)) => mnth_builder.append_value(*v),
            _ => mnth_builder.append_null(),
        }
        match row.get("DAYS") {
            Some(RowValue::Int(v)) => days_builder.append_value(*v),
            _ => days_builder.append_null(),
        }
        match row.get("HOUR") {
            Some(RowValue::Int(v)) => hour_builder.append_value(*v),
            _ => hour_builder.append_null(),
        }
        match row.get("MINU") {
            Some(RowValue::Int(v)) => minu_builder.append_value(*v),
            _ => minu_builder.append_null(),
        }
        match row.get("SECO") {
            Some(RowValue::Int(v)) => seco_builder.append_value(*v),
            _ => seco_builder.append_null(),
        }
    }

    let mut fields: Vec<Field> = vec![
        Field::new("message_type", DataType::Utf8, true),
        Field::new("message_index", DataType::Int32, true),
        Field::new("subset_index", DataType::Int32, true),
        Field::new("YEAR", DataType::Int32, true),
        Field::new("MNTH", DataType::Int32, true),
        Field::new("DAYS", DataType::Int32, true),
        Field::new("HOUR", DataType::Int32, true),
        Field::new("MINU", DataType::Int32, true),
        Field::new("SECO", DataType::Int32, true),
    ];
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(msg_type_builder.finish()),
        Arc::new(msg_idx_builder.finish()),
        Arc::new(sub_idx_builder.finish()),
        Arc::new(year_builder.finish()),
        Arc::new(mnth_builder.finish()),
        Arc::new(days_builder.finish()),
        Arc::new(hour_builder.finish()),
        Arc::new(minu_builder.finish()),
        Arc::new(seco_builder.finish()),
    ];

    for key in &mnemonic_keys {
        let col_type = col_types.get(key).map(|s| *s).unwrap_or("float64");
        match col_type {
            "float64" => {
                let mut builder = Float64Builder::with_capacity(n);
                for row in rows {
                    match row.get(key) {
                        Some(RowValue::Float(v)) => builder.append_value(*v),
                        _ => builder.append_null(),
                    }
                }
                fields.push(Field::new(key, DataType::Float64, true));
                arrays.push(Arc::new(builder.finish()));
            }
            "string" => {
                let mut builder = StringBuilder::with_capacity(n, n * 16);
                for row in rows {
                    match row.get(key) {
                        Some(RowValue::Str(s)) => builder.append_value(s),
                        _ => builder.append_null(),
                    }
                }
                fields.push(Field::new(key, DataType::Utf8, true));
                arrays.push(Arc::new(builder.finish()));
            }
            "list_float64" => {
                let mut builder = ListBuilder::new(Float64Builder::new());
                for row in rows {
                    match row.get(key) {
                        Some(RowValue::FloatList(list)) => {
                            let vals = builder.values();
                            for v in list { vals.append_value(*v); }
                            builder.append(true);
                        }
                        Some(RowValue::Float(v)) => {
                            builder.values().append_value(*v);
                            builder.append(true);
                        }
                        _ => builder.append_null(),
                    }
                }
                fields.push(Field::new(key, DataType::List(Arc::new(Field::new("item", DataType::Float64, true))), true));
                arrays.push(Arc::new(builder.finish()));
            }
            _ => {}
        }
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).map_err(|e| format!("Arrow error: {e}"))
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
    let tables = TableSet::from_json(table_b_json, table_d_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let raw_data = std::fs::read(file_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot read {file_path}: {e}")))?;

    if raw_data.is_empty() {
        let batch = rows_to_record_batch(&[], &mnemonics)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        return Ok(PyArrowType(batch));
    }

    let messages = read_messages(&raw_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let result = py.allow_threads(|| {
        let parsed: Vec<(usize, ParsedMessage)> = messages.iter()
            .filter_map(|msg| {
                parse_message(msg).ok().map(|p| (msg.index, p))
            })
            .collect();

        let filtered: Vec<&(usize, ParsedMessage)> = parsed.iter()
            .filter(|(_, p)| p.identification.data_category != 11)
            .filter(|(_, p)| {
                match data_category_filter {
                    Some(cat) => p.identification.data_category as i32 == cat,
                    None => true,
                }
            })
            .collect();

        let all_rows: Vec<DecodedRow> = filtered
            .par_iter()
            .flat_map(|(msg_idx, parsed)| {
                let expanded = match expand_descriptors(&parsed.descriptors, &tables) {
                    Ok(e) => e,
                    Err(_) => return Vec::new(),
                };
                let subsets = match decode(
                    &expanded,
                    &parsed.data_bytes,
                    parsed.identification.num_subsets,
                    parsed.identification.compressed,
                ) {
                    Ok(s) => s,
                    Err(_) => return Vec::new(),
                };
                subsets_to_rows(*msg_idx, &parsed.identification, &subsets)
            })
            .collect();

        rows_to_record_batch(&all_rows, &mnemonics)
    }).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    Ok(PyArrowType(result))
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_bufr_rust, m)?)?;
    Ok(())
}
