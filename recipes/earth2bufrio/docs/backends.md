# Backend Architecture

earth2bufrio separates the high-level pipeline orchestration (message splitting,
section parsing, descriptor expansion, table building) from the performance-
critical inner loop — the **bit-level decoder** in `_decoder.py`.  This design
makes it straightforward to swap in an optimised native backend without changing
the public API.

## Current: Pure Python

The default backend is implemented entirely in Python.  It reads individual bits
from the data section using byte indexing and bitwise operations, applies the
Table B encoding parameters (reference value, scale, bit width), and produces
`DecodedSubset` objects that the Arrow stage converts into columnar arrays.

Advantages:

- Zero native dependencies — installs with `pip` on any platform.
- Easy to debug and extend.
- Adequate throughput for moderate file sizes (< 100 MB).

Limitations:

- Bit-level Python loops are inherently slow for very large files.
- Parallel decoding via `ProcessPoolExecutor` helps but adds serialisation
  overhead.

## Planned: Rust Backend (PyO3 / maturin)

A Rust implementation of the `decode()` function is planned as the primary
high-performance backend.  The Rust module would:

1. Accept the expanded descriptor list and raw `bytes` from Python.
2. Perform all bit extraction and value scaling in compiled code.
3. Return decoded values as Arrow-compatible arrays (via `arrow-rs` or plain
   buffers).

Integration path:

- Build with [maturin](https://www.maturin.rs/) as an optional extension
  (`earth2bufrio[rust]`).
- At import time, earth2bufrio checks whether the compiled module is available
  and transparently delegates `_decoder.decode()` to it.
- Falls back to the pure-Python decoder if the extension is not installed.

## Planned: Fortran Backend

For environments where a Fortran toolchain is available (e.g. HPC clusters),
a thin Fortran binding is under consideration:

- Wraps NCEP's existing `bufrlib` or a custom Fortran kernel.
- Exposes a C-compatible interface called via `ctypes` or `cffi`.
- Targets the same `decode()` entry point as the Rust backend.

## Swap Point: `_decoder.decode()`

All backends implement the same contract:

```python
def decode(
    expanded: list[ExpandedItem],
    data_bytes: bytes,
    num_subsets: int,
    compressed: bool,
) -> list[DecodedSubset]:
    ...
```

The `_api.py` module calls `decode()` without knowing which implementation is
active.  Backend selection happens at import time in `_decoder.py`:

```python
try:
    from earth2bufrio._rust_decoder import decode  # Rust
except ImportError:
    # fall back to pure Python
    ...
```

This pattern keeps the public API stable while allowing incremental performance
improvements.
