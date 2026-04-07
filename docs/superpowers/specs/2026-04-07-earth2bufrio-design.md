# earth2bufrio v0.1.0 — Design Specification

**Date:** 2026-04-07
**Status:** Approved
**Location:** `recipes/earth2bufrio/`

## 1. Overview

**earth2bufrio** is a standalone installable Python package for reading WMO BUFR
(Binary Universal Form for the Representation of meteorological data) files. It
lives at `recipes/earth2bufrio/` within the earth2studio repository but is a
proper installable package (intentionally breaking the "recipes are not
installable" convention).

**Goals:**

- Read BUFR Edition 3 and Edition 4 files, including NCEP PrepBUFR
- Return decoded data as a PyArrow Table
- Pure Python custom binary parser (no pybufrkit/eccodes runtime dependency)
- Maturin build with empty Rust skeleton for future acceleration
- Single public function: `read_bufr(path) -> pa.Table`

**Non-goals for v0.1.0:**

- Rust or Fortran decode backends (skeleton only)
- BUFR writing/encoding
- Streaming or memory-mapped decode
- Network data fetching (paths only)

## 2. Public API

```python
import os
from typing import Any
import pyarrow as pa

def read_bufr(
    path: str | os.PathLike[str],
    *,
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    workers: int = 1,
) -> pa.Table:
    """Read a BUFR file and return a PyArrow Table.

    Parameters
    ----------
    path : str or PathLike
        Path to the BUFR file (Edition 3 or 4, including PrepBUFR).
    columns : list of str, optional
        Subset of columns to return. If None, all decoded columns returned.
    filters : dict, optional
        Key-value filters applied at the message level before full decode
        (e.g., {"data_category": 0}). Reduces memory and compute.
    workers : int, default 1
        Number of parallel workers for message-level decoding.

    Returns
    -------
    pyarrow.Table
        Decoded observations in long format (one row per observation
        element per subset per message).

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    BufrDecodeError
        If the file is not valid BUFR or contains unsupported features.
    """
```

### Output Schema

Long format — one row per descriptor value per subset:

| Column | Type | Description |
|--------|------|-------------|
| `message_index` | `int32` | BUFR message index in the file |
| `subset_index` | `int32` | Subset index within the message |
| `data_category` | `int32` | WMO data category (or NCEP type) |
| `latitude` | `float64` | Latitude (degrees), nullable |
| `longitude` | `float64` | Longitude (degrees), nullable |
| `time` | `timestamp[us]` | Observation time |
| `station_id` | `string` | Station identifier, nullable |
| `pressure` | `float64` | Pressure level (Pa), nullable |
| `elevation` | `float64` | Station elevation (m), nullable |
| `descriptor_id` | `string` | F-X-Y descriptor (e.g., "012001") |
| `descriptor_name` | `string` | Human-readable name from Table B |
| `value` | `float64` | Decoded observation value, nullable |
| `units` | `string` | Units from Table B |
| `quality_mark` | `int32` | Quality flag, nullable |

## 3. Directory Structure

```text
recipes/earth2bufrio/
├── pyproject.toml
├── README.md
├── Makefile
├── src/
│   ├── earth2bufrio/
│   │   ├── __init__.py          # re-exports read_bufr, BufrDecodeError
│   │   ├── _api.py              # read_bufr() implementation
│   │   ├── _reader.py           # Binary message splitting
│   │   ├── _section.py          # Section parsing (Indicator→End)
│   │   ├── _descriptors.py      # F-X-Y expansion & Table B/D lookup
│   │   ├── _decoder.py          # Bit-level data decoding
│   │   ├── _tables.py           # Table management (bundled + DX)
│   │   ├── _arrow.py            # PyArrow Table construction
│   │   ├── _types.py            # Dataclasses and exceptions
│   │   └── tables/
│   │       ├── table_b.json     # Bundled WMO Table B
│   │       └── table_d.json     # Bundled WMO Table D
│   └── rust/
│       ├── Cargo.toml           # Empty Rust/PyO3 skeleton
│       └── src/
│           └── lib.rs           # Placeholder module
├── tests/
│   ├── conftest.py
│   ├── test_reader.py
│   ├── test_section.py
│   ├── test_descriptors.py
│   ├── test_decoder.py
│   ├── test_tables.py
│   ├── test_arrow.py
│   ├── test_api.py
│   ├── test_crossval.py         # Cross-validation vs pybufrkit refs
│   ├── generate_references.py   # Script to regen .ref.json
│   └── data/
│       ├── profiler_european.bufr       # Ed3 uncompressed (426 B)
│       ├── profiler_european.bufr.ref.json
│       ├── 207003.bufr                  # Ed3 compressed (244 B)
│       ├── 207003.bufr.ref.json
│       ├── uegabe.bufr                  # Ed4 uncompressed (494 B)
│       ├── uegabe.bufr.ref.json
│       ├── g2nd_208.bufr                # Ed4 compressed+strings (928 B)
│       ├── g2nd_208.bufr.ref.json
│       ├── b005_89.bufr                 # Ed3 operators (3.98 KB)
│       └── b005_89.bufr.ref.json
├── docs/
│   ├── conf.py
│   ├── index.md
│   ├── api.md
│   ├── format.md
│   └── backends.md
```

## 4. Internal Architecture

Five-stage linear pipeline:

```text
bytes → _reader → _section → _descriptors → _decoder → _arrow → pa.Table
```

### 4.1 `_reader.py` — Message Splitting

- Scan raw bytes for 4-byte `BUFR` magic markers
- Read 3-byte message length from indicator section (bytes 4-6)
- Validate end section ends with `7777`
- Yield `BufrMessage(raw_bytes, offset, index)` dataclasses

### 4.2 `_section.py` — Section Parsing

- Parse 6 sections from message bytes
- Edition detection from indicator byte 7
- Edition 3: 18-byte identification, 1-byte year
- Edition 4: 22-byte identification, 2-byte year
- Extract: originating center, data category, time, compression flag,
  number of subsets
- Extract data description section: list of raw F-X-Y descriptor integers
- Return `ParsedMessage` with typed section dataclasses

### 4.3 `_descriptors.py` — Descriptor Expansion

- Expand Table D sequences recursively into Table B elements
- Handle replication descriptors (F=1): regular and delayed replication
- Handle operator descriptors (F=2): change data width, change scale,
  associated fields
- Return flat list of `ExpandedDescriptor` objects

### 4.4 `_decoder.py` — Bit-Level Decoding

- Walk expanded descriptors against data section bits
- Uncompressed mode: read N subsets sequentially
- Compressed mode: read min values + increments per subset
- Apply Table B transform: `value = (raw_bits + reference_value) / 10^scale`
- Handle missing values (all bits set to 1 → None)
- Yield `DecodedSubset` objects (list of descriptor-value pairs)

### 4.5 `_arrow.py` — PyArrow Table Construction

- Collect decoded subsets into columnar arrays
- Build schema with appropriate types and metadata
- Apply column filtering and message-level filters
- Identify well-known descriptors (lat, lon, time, station_id, pressure,
  elevation) and promote to named columns
- Return `pa.Table`

### 4.6 `_tables.py` — Table Management

- Load bundled WMO Table B/D from `tables/*.json` at import time
- Parse DX table messages from PrepBUFR (data_category=11) at runtime
- Merge NCEP-local descriptors with WMO base tables
- Thread-safe scoped table context for parallel decode

### 4.7 `_types.py` — Data Types

- `BufrMessage`: raw bytes, offset, index
- `IndicatorSection`, `IdentificationSection`, `DataDescriptionSection`,
  `DataSection`, `EndSection`
- `ParsedMessage`: all sections combined
- `TableBEntry`: scale, reference_value, bit_width, units, name
- `TableDEntry`: sequence of descriptor IDs
- `ExpandedDescriptor`: resolved Table B entry
- `DecodedSubset`: list of (descriptor, value) pairs
- `BufrDecodeError`: custom exception class

## 5. Testing Strategy

### Tier 1: Unit Tests

Each internal module tested in isolation with hand-crafted binary data:

- `test_reader.py` — message splitting edge cases
- `test_section.py` — Ed3/Ed4 section parsing
- `test_descriptors.py` — Table D expansion, replication, operators
- `test_decoder.py` — uncompressed/compressed bit-level decode
- `test_tables.py` — WMO table loading, DX table parsing
- `test_arrow.py` — schema construction, column filtering

### Tier 2: Cross-Validation Tests

Compare `read_bufr()` output against pre-computed reference values from
pybufrkit. Reference values stored as `.ref.json` files committed to repo.

| Fixture | Size | Edition | Features |
|---------|------|---------|----------|
| `profiler_european.bufr` | 426 B | Ed3 | Uncompressed, basic meteorological |
| `207003.bufr` | 244 B | Ed3 | Compressed, delayed replication |
| `uegabe.bufr` | 494 B | Ed4 | Uncompressed, Edition 4 baseline |
| `g2nd_208.bufr` | 928 B | Ed4 | Compressed, string data |
| `b005_89.bufr` | 3.98 KB | Ed3 | Statistics operators |

Tests validate:

- Message count
- Subset count per message
- Numeric values within 1e-6 tolerance
- String values (exact match)
- Missing value handling (None)
- Message metadata (data_category, originating_center, year, month)

pybufrkit is only needed to regenerate reference files
(`tests/generate_references.py`), not at test runtime.

### Tier 3: Integration Test

`test_api.py` — End-to-end `read_bufr()` with all fixtures, testing column
filtering, message filters, and multi-worker decode.

### Markers

- `unit` (default) — Tier 1
- `crossval` — Tier 2
- `integration` — Tier 3
- `slow` — Future large-file tests

## 6. Tooling & Dependencies

### Core Dependencies

- `pyarrow >= 14.0` (only runtime dependency)

### Build System

- maturin >= 1.0 (Rust/PyO3 build backend)
- Python source in `src/`, Rust manifest at `src/rust/Cargo.toml`
- Module name: `earth2bufrio._lib`

### Development Tools

| Tool | Version | Purpose |
|------|---------|---------|
| ruff | >= 0.15 | Linting + formatting (line-length=120, py311, numpy docstrings) |
| ty | >= 0.0.26 | Type checking (python 3.12) |
| pytest | >= 8.0 | Testing |
| pytest-cov | >= 7.0 | Coverage (80% minimum) |
| interrogate | >= 1.7 | Docstring coverage (99% minimum) |
| pre-commit | >= 4.0 | Git hooks |
| uv | — | Package manager |

### Documentation

- Sphinx 9 + nvidia-sphinx-theme
- MyST parser (Markdown)
- sphinx-autoapi (auto-generate API reference)
- sphinx-copybutton
- numpydoc

### Ruff Configuration

```toml
select = [E, F, W, I, D, UP, N, B, A, C4, SIM, TCH, PTH, ERA]
convention = "numpy"
per-file-ignores: tests/ exempt from D (docstrings)
```

## 7. Future Backends

The `_decoder.py` module is the swap point for future backends:

- **Rust** (`_lib` module via PyO3/maturin): Replace `_decoder.decode()` with
  Rust implementation. The empty skeleton in `src/rust/` is ready.
- **Fortran** (via f2py or ctypes): Wrap existing Fortran BUFR libraries. Not
  scaffolded in v0.1.0 but the architecture supports it — the `_decoder`
  interface is the boundary.

Both backends would implement the same interface:

```python
def decode(
    expanded_descriptors: list[ExpandedDescriptor],
    data_bytes: bytes,
    num_subsets: int,
    compressed: bool,
) -> list[DecodedSubset]:
```

## 8. Constraints & Conventions

- SPDX Apache-2.0 header on every `.py` file
- All public functions fully typed
- NumPy-style docstrings on all public and private functions
- No `print()` — use `logging` module
- All file paths via `pathlib.Path`
