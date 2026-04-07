<!-- markdownlint-disable MD013 MD032 -->

# earth2bufrio Rust Backend Design

**Date:** 2026-04-07

**Status:** Approved

**Scope:** Add a Rust backend to earth2bufrio that reimplements the full
BUFR decode pipeline in Rust with Rayon message-level parallelism and
zero-copy Arrow FFI transfer to Python.

---

## 1. Overview

The Rust backend provides a high-performance alternative to the pure-Python
decoder. It reimplements the entire BUFR pipeline (message splitting,
section parsing, descriptor expansion, bit-level decoding, Arrow table
construction) as a single Rust crate compiled via maturin into a PyO3
extension module (`earth2bufrio._lib`).

Key properties:

- **Full pipeline in Rust** — no Python calls during decode.
- **Rayon thread pool** — message-level parallelism with no GIL involvement.
- **Zero-copy Arrow FFI** — `RecordBatch` transferred to PyArrow via
  `pyo3-arrow` without serialization.
- **Same output format** — identical wide-format table as Python and Fortran
  backends (fixed columns + dynamic mnemonic columns).

## 2. Architecture

### 2.1 Crate Location

```text
recipes/earth2bufrio/src/rust/
├── Cargo.toml
└── src/
    └── lib.rs
```

The existing placeholder `Cargo.toml` and `lib.rs` are replaced with the
full implementation. All Rust code lives in a single `lib.rs` file
organized into logical sections (tables, reader, section, descriptor,
decoder, arrow, pymodule). This can be split into separate modules later
if it grows beyond ~2000 lines.

### 2.2 Cargo Dependencies

```toml
[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
arrow = "54"
pyo3-arrow = "0.7"
rayon = "1.10"
serde_json = "1"
serde = { version = "1", features = ["derive"] }
```

### 2.3 Rust Types

Mirror the Python `_types.py` dataclasses:

```rust
struct TableBEntry {
    name: String,
    units: String,
    scale: i32,
    reference_value: i64,
    bit_width: u32,
}

struct TableDEntry {
    descriptors: Vec<u32>,
}

struct TableSet {
    table_b: HashMap<u32, TableBEntry>,
    table_d: HashMap<u32, TableDEntry>,
}

struct BufrMessage {
    data: Vec<u8>,
    offset: usize,
    index: usize,
}

struct IndicatorSection { length: u32, edition: u8 }
struct IdentificationSection {
    originating_center: u16,
    data_category: u8,
    year: u16, month: u8, day: u8,
    hour: u8, minute: u8, second: u8,
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

enum ExpandedItem {
    Descriptor(ExpandedDescriptor),
    DelayedReplication {
        factor: ExpandedDescriptor,
        group: Vec<ExpandedItem>,
    },
}

struct ExpandedDescriptor {
    fxy: u32,
    entry: TableBEntry,
}

enum DecodedValue {
    Float(f64),
    String(String),
    Missing,
}
```

### 2.4 Pipeline

```text
read_bufr_rust(path, table_b_json, table_d_json, mnemonics, filter)
  1. Parse TableSet from JSON strings (serde_json)
  2. Read file bytes (std::fs::read)
  3. Split into Vec<BufrMessage> (scan for b"BUFR" markers)
  4. Filter by data_category if filter is Some
  5. Rayon par_iter over messages:
     a. parse_message() -> ParsedMessage
     b. expand_descriptors() -> Vec<ExpandedItem>
     c. decode() -> Vec<Vec<(ExpandedDescriptor, DecodedValue)>>
     d. Convert subsets to row HashMap<String, Value>
  6. Collect all rows (flatten)
  7. Build arrow RecordBatch from rows
  8. Return via pyo3-arrow FFI zero-copy
```

### 2.5 PyO3 Entry Point

```rust
#[pyfunction]
fn read_bufr_rust(
    py: Python<'_>,
    file_path: &str,
    table_b_json: &str,
    table_d_json: &str,
    mnemonics: Option<Vec<String>>,
    data_category_filter: Option<i32>,
) -> PyResult<PyArrowType<RecordBatch>> {
    // ... full pipeline ...
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_bufr_rust, m)?)?;
    Ok(())
}
```

## 3. Python Integration

### 3.1 `_api.py` Changes

Add `backend="rust"` branch to `read_bufr()`:

```python
if backend == "rust":
    from earth2bufrio._lib import read_bufr_rust
    table_b_str = _load_table_json("table_b.json")
    table_d_str = _load_table_json("table_d.json")
    cat_filter = filters.get("data_category") if filters else None
    batch = read_bufr_rust(
        str(file_path), table_b_str, table_d_str, mnemonics, cat_filter
    )
    return pa.Table.from_batches([batch])
```

Add helper `_load_table_json(filename: str) -> str` that reads the
bundled JSON table file and returns its content as a string. Cache the
result at module level to avoid repeated I/O.

### 3.2 `__init__.py`

No changes. `read_bufr` and `BufrDecodeError` remain the only exports.

### 3.3 `pyproject.toml`

- Add `rust` pytest marker.
- No new Python dependencies (maturin handles the Rust build).

### 3.4 `Makefile`

Add `rust` target:

```text
rust:
    cd src/rust && maturin develop --release
```

### 3.5 Backend Dispatch

The `backend` parameter in `read_bufr()` accepts `"python"`, `"fortran"`,
or `"rust"`. The `workers` parameter is ignored for the Rust backend
because Rayon manages its own thread pool internally.

If `backend="rust"` but `_lib` is not importable (maturin not built),
`ImportError` propagates to the caller.

## 4. Parallelism

- **Level:** Message-level. Each BUFR message is parsed, expanded, and
  decoded independently.
- **Engine:** Rayon `par_iter` with default thread pool (number of CPUs).
- **Thread safety:** All Rust types are `Send + Sync`. No shared mutable
  state between messages. Table lookups are read-only after initial parse.
- **GIL:** Released during the entire Rust pipeline. Only acquired at the
  end to construct the PyArrow return object.
- **Comparison to Python backend:** Python uses `ProcessPoolExecutor`
  (serialization overhead). Rust uses threads (zero serialization, shared
  memory).

## 5. Arrow Integration

- **Library:** `arrow` crate (Apache Arrow Rust implementation).
- **Transfer:** `pyo3-arrow` crate provides `PyArrowType<RecordBatch>`
  which implements the Arrow C Data Interface for zero-copy FFI.
- **Schema construction:** Dynamic based on discovered mnemonic columns
  (same logic as Python `_arrow.py` — fixed columns + one column per
  mnemonic).
- **Type inference:** Same rules as Python — first non-null value
  determines column type (`Float64`, `Utf8`, `List<Float64>`).

## 6. BUFR Table Handling

- Tables are loaded from the same bundled JSON files
  (`tables/table_b.json`, `tables/table_d.json`).
- Python reads the JSON content and passes it as strings to Rust.
- Rust deserializes with `serde_json` into `HashMap<u32, TableBEntry>`
  and `HashMap<u32, TableDEntry>`.
- DX table (PrepBUFR data_category=11) messages are skipped (same as
  Python backend).

## 7. Testing

### 7.1 Unit Tests (`test_rust_backend.py`)

- Test `read_bufr_rust()` on hand-crafted minimal BUFR bytes.
- Verify column names, row count, and decoded values.
- Marked `@pytest.mark.rust` — skipped if `_lib` not importable.

### 7.2 Cross-Validation

- Run both Python and Rust backends on the same real BUFR fixture files.
- Compare row counts (must match exactly).
- Compare numeric values within tolerance (`abs(python - rust) < 1e-6`).
- Compare string values exactly.

### 7.3 Integration

- `read_bufr(path, backend="rust")` end-to-end.
- Verify returned `pa.Table` has expected schema and data.

### 7.4 Pytest Marker

```ini
[tool.pytest.ini_options]
markers = [
    ...
    "rust: tests requiring the Rust backend (maturin develop)",
]
```

## 8. Documentation

### 8.1 `backends.md`

Add Rust backend section:

- Dependencies (Rust toolchain, maturin).
- Build instructions (`make rust` or `maturin develop`).
- Performance characteristics (Rayon parallelism, zero-copy Arrow).
- Comparison table updated with Rust column.

### 8.2 `api.md`

- Add `backend="rust"` to parameter documentation.
- Note that `workers` is ignored for Rust backend.

## 9. Files Changed

| File | Action | Description |
| --- | --- | --- |
| `src/rust/Cargo.toml` | Modify | Add arrow, pyo3-arrow, rayon, serde, serde_json deps |
| `src/rust/src/lib.rs` | Rewrite | Full Rust BUFR pipeline (~1500-2000 lines) |
| `src/earth2bufrio/_api.py` | Modify | Add `backend="rust"` branch + `_load_table_json()` |
| `pyproject.toml` | Modify | Add `rust` pytest marker |
| `Makefile` | Modify | Add `rust` target |
| `tests/test_rust_backend.py` | Create | Unit + cross-val + integration tests |
| `docs/api.md` | Modify | Document `backend="rust"` |
| `docs/backends.md` | Modify | Add Rust backend section |

## 10. Build Requirements

- **Rust toolchain:** stable (edition 2021)
- **maturin:** `>=1.0,<2.0` (already in dev deps)
- **No runtime Python deps added** — PyArrow already required

## 11. Scope Exclusions

- **PrepBUFR DX table parsing in Rust** — DX messages (data_category=11)
  are skipped, same as Python backend.
- **Operators 222-237** — Not implemented (same xfail as Python).
- **Streaming/incremental decode** — Full file read into memory.
- **Custom Rayon thread count** — Uses Rayon default (num CPUs). A future
  enhancement could expose this via an environment variable.
