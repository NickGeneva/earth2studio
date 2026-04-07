# API Reference

The public API of earth2bufrio consists of a single entry-point function and one
exception class.  Full signatures and docstrings are generated automatically by
[sphinx-autoapi](https://sphinx-autoapi.readthedocs.io/) from the source code.

## `read_bufr`

The primary function for decoding BUFR files:

```python
import earth2bufrio

table = earth2bufrio.read_bufr(
    "observations.bufr",
    mnemonics=None,                  # Optional list of mnemonic names to extract
    filters={"data_category": 102},  # Optional message-level filters
    workers=1,                       # >1 enables multiprocess decoding
    backend="python",                # "python", "fortran", or "rust"
)
```

### Parameters

**path** (`str | Path`) -- *(required)* Path to a BUFR file.

**mnemonics** (`list[str] | None`) -- Mnemonic column names to include.
`None` extracts all available fields.

**filters** (`dict[str, Any] | None`) -- Message-level filters
(e.g. `{"data_category": 102}`).

**workers** (`int`) -- Number of parallel decode workers. Default `1`
runs in the current process.

**backend** (`str`) -- Decoding backend. `"python"` (default, pure
Python), `"fortran"` (NCEPLIBS-bufr via ctypes), or `"rust"` (compiled
Rust with Rayon parallelism).

### Output Schema (wide format)

`read_bufr` returns a `pyarrow.Table` in **wide format** — one row per subset,
one column per mnemonic.

**Fixed columns** (always present):

| Column | Type | Description |
| --- | --- | --- |
| `message_type` | `string` | BUFR message type identifier |
| `message_index` | `int32` | Zero-based message index in the file |
| `subset_index` | `int32` | Zero-based subset index within the message |
| `YEAR` | `int32` | Observation year |
| `MNTH` | `int32` | Observation month |
| `DAYS` | `int32` | Observation day |
| `HOUR` | `int32` | Observation hour |
| `MINU` | `int32` | Observation minute |
| `SECO` | `int32` | Observation second |

**Dynamic columns** (one per mnemonic):

Each mnemonic found in the data becomes a column.  The column type is inferred
from the data:

- **Scalar numeric** values → `float64`
- **String** values → `string`
- **Replicated** values (e.g. multi-channel brightness temperatures) → `list<float64>`

Missing values are represented as `null`.

### Backends

**Python backend** (`backend="python"`): Uses the built-in pure-Python BUFR
decoder.  Works on any WMO BUFR Edition 3/4 file.  No native dependencies.

**Fortran backend** (`backend="fortran"`): Uses NCEPLIBS-bufr via ctypes for
NCEP BUFR and PrepBUFR files.  Requires building the Fortran shared library
first (`make fortran`).  See {doc}`backends` for details.

**Rust backend** (`backend="rust"`): Uses the compiled Rust module for
high-performance decoding with Rayon message-level parallelism and zero-copy
Arrow FFI.  Requires building with `make rust` (needs Rust toolchain and
maturin).  The `workers` parameter is ignored — Rayon manages its own thread
pool.  See {doc}`backends` for details.

## `BufrDecodeError`

Raised when a BUFR message cannot be decoded — for example when the file is
truncated, contains an unsupported edition, or has malformed section headers.

```python
from earth2bufrio import BufrDecodeError

try:
    table = earth2bufrio.read_bufr("corrupt.bufr")
except BufrDecodeError as exc:
    print(f"Decode failed at byte {exc.offset}: {exc}")
```

## Auto-generated API

The sections below are produced by sphinx-autoapi from the package source.
Internal modules (prefixed with `_`) are excluded except where they define
public symbols re-exported by `earth2bufrio.__init__`.
