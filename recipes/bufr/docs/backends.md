# Backend Architecture

earth2bufr supports multiple decoding backends behind a unified `read_bufr()`
API.  The `backend=` parameter controls which implementation is used.

## Pure Python (default)

```python
table = earth2bufr.read_bufr("observations.bufr", backend="python")
```

The default backend is implemented entirely in Python.  It reads individual bits
from the data section using byte indexing and bitwise operations, applies the
Table B encoding parameters (reference value, scale, bit width), and produces
`DecodedSubset` objects that are converted into wide-format row dicts and then
into a PyArrow Table.

**Advantages:**

- Zero native dependencies — installs with `pip` on any platform.
- Supports all WMO BUFR Edition 3 and Edition 4 files.
- Easy to debug and extend.
- Adequate throughput for moderate file sizes (< 100 MB).

**Limitations:**

- Bit-level Python loops are inherently slow for very large files.
- Parallel decoding via `ProcessPoolExecutor` helps but adds serialisation
  overhead.

## Fortran Backend (NCEPLIBS-bufr)

```python
table = earth2bufr.read_bufr("prepbufr.gdas", backend="fortran")
```

The Fortran backend wraps NOAA's
[NCEPLIBS-bufr](https://github.com/NOAA-EMC/NCEPLIBS-bufr) library via
`ctypes` and ISO C bindings.  It is optimised for NCEP BUFR and PrepBUFR files
(satellite radiance, conventional observations).

**Supported message types:**

| Type | Message ID | Mnemonics |
| --- | --- | --- |
| ATMS | NC021203 | SAID, CLATH, CLONH, SAZA, SOZA, IANG, TMBR, CHNM |
| AMSU-A | NC021023 | SAID, CLAT, CLON, SAZA, SOZA, IANG, TMBR, CHNM |
| MHS | NC021027 | SAID, CLAT, CLON, SAZA, SOZA, IANG, TMBR, CHNM |
| PrepBUFR | (various) | YOB, XOB, DHR, ELV, TYP, POB, QOB, TOB, ZOB, UOB, VOB, PWO, TDO, PMO |

### Building the Fortran library

The Fortran backend requires a compiled shared library.  Build it with:

```bash
cd recipes/bufr
make fortran
```

This runs CMake to build NCEPLIBS-bufr (bundled or system-installed) and the
ISO C wrapper, producing `libearth2bufr_fort.so` in the package directory.

**Requirements:**

- Fortran compiler (gfortran 9+ or Intel ifx)
- CMake >= 3.15
- Make

### How it works

The Fortran wrapper (`src/fortran/earth2bufr_fort.f90`) exposes these
C-callable functions:

| Function | Purpose |
| --- | --- |
| `e2b_open` | Open a BUFR file (wraps `openbf`) |
| `e2b_next_message` | Read next message (wraps `readmg`) |
| `e2b_next_subset` | Read next subset (wraps `readsb`) |
| `e2b_read_values` | Read scalar mnemonics (wraps `ufbint`) |
| `e2b_read_replicated` | Read replicated mnemonics (wraps `ufbrep`) |
| `e2b_close` | Close file (wraps `closbf`) |
| `e2b_get_bmiss` | Get missing value sentinel |

The Python side (`_fortran_backend.py`) loads the shared library via `ctypes`,
iterates over messages and subsets, and converts the results into the same
wide-format PyArrow Table as the Python backend.

### Missing values

NCEPLIBS-bufr uses `10E10` (~1.0E+11) as the missing value sentinel.  The
Fortran backend converts these to `None` (null) in the output table.

### Thread safety

NCEPLIBS-bufr is **not thread-safe**.  The Fortran backend uses a single thread
for all I/O.  For parallel processing, use separate processes (not threads).

## Rust Backend (PyO3 / maturin)

```python
table = earth2bufr.read_bufr("observations.bufr", backend="rust")
```

The Rust backend reimplements the full BUFR pipeline in compiled Rust code.
It uses [Rayon](https://docs.rs/rayon/) for message-level parallelism and
[arrow-rs](https://docs.rs/arrow/) for zero-copy Arrow FFI transfer to
PyArrow via [pyo3-arrow](https://docs.rs/pyo3-arrow/).

**Advantages:**

- All BUFR parsing, descriptor expansion, and bit-level decoding in compiled
  code — no Python in the hot path.
- Rayon thread pool for automatic message-level parallelism (no GIL).
- Zero-copy Arrow transfer — the RecordBatch is shared between Rust and Python
  without serialization.
- Supports all WMO BUFR Edition 3 and Edition 4 files (same as Python backend).

**Limitations:**

- Requires Rust toolchain and maturin to compile.
- The `workers` parameter is ignored — Rayon manages its own thread pool
  internally using all available CPUs.

### Building the Rust module

```bash
cd recipes/bufr
make rust
```

This runs `maturin develop --release` to compile the Rust crate and install the
`earth2bufr._lib` extension module.

**Requirements:**

- Rust stable toolchain (edition 2021)
- maturin >= 1.0

## Backend Comparison

| Feature | Python | Fortran | Rust |
| --- | --- | --- | --- |
| WMO BUFR Ed3/Ed4 | Yes | NCEP formats only | Yes |
| Native deps | None | gfortran + CMake | Rust + maturin |
| Speed | Moderate | Fast | Fast |
| Install | `pip install` | `make fortran` | `make rust` |
| Parallel | ProcessPoolExecutor | Single-process | Rayon threads |
