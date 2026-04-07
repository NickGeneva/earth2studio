# earth2bufrio Fortran Backend Design

## Goal

Add a Fortran backend to earth2bufrio that wraps NCEPLIBS-bufr via ctypes and
ISO\_C\_BINDING, enabling high-performance BUFR decoding for both satellite
radiance files (ATMS, AMSU-A, MHS) and PrepBUFR conventional observations.
Simultaneously, migrate the output schema from long-format (14 fixed columns)
to wide-format (one column per mnemonic).

---

## Scope

This spec covers three coordinated changes:

1. **Fortran backend** (`_fortran_backend.py` + `earth2bufrio_fort.f90` + CMake
   build).
2. **Wide-format output schema** for both Python and Fortran backends
   (`_arrow.py` rewrite).
3. **API changes** to `read_bufr()` (`backend=`, `mnemonics=`, drop `columns=`).

The pure-Python backend continues to work for any WMO BUFR file. The Fortran
backend is for NCEP-format files (PrepBUFR and NCEP satellite radiance BUFR)
where NCEPLIBS-bufr's mnemonic-based API is the natural fit.

---

## Architecture

### Integration Model

The Fortran backend is a **parallel pipeline**, not a drop-in replacement for
the `_decoder.decode()` swap point. The Python backend parses raw bits directly;
the Fortran backend delegates everything to NCEPLIBS-bufr's high-level
`openbf`/`readmg`/`readsb`/`ufbint`/`ufbrep` routines.

```text
read_bufr(path, backend="python")
  -> _reader -> _section -> _descriptors -> _decoder -> build_table()

read_bufr(path, backend="fortran")
  -> _fortran_backend.read_ncep(path, mnemonics, filters)
     -> ctypes calls to libearth2bufrio_fort.so
        -> openbf / readmg / readsb / ufbint / ufbrep / closbf
     -> numpy arrays -> build_table()
```

### File Layout

```text
recipes/earth2bufrio/
  src/earth2bufrio/
    _api.py                  # Modified: backend=, mnemonics=, drop columns=
    _arrow.py                # Rewritten: wide-format builder
    _fortran_backend.py      # NEW: ctypes wrapper around Fortran shared lib
    _types.py                # Minor: add FortranBackendError if needed
  src/fortran/
    CMakeLists.txt           # NEW: builds NCEPLIBS-bufr (static) + wrapper
    earth2bufrio_fort.f90    # NEW: ISO_C_BINDING module
    nceplibs-bufr/           # NEW: bundled NCEPLIBS-bufr source (~2 MB)
  Makefile                   # Modified: add `fortran` target
  pyproject.toml             # Modified: add numpy optional dep
```

---

## Public API Changes

### `read_bufr()` New Signature

```python
def read_bufr(
    path: str | Path,
    *,
    mnemonics: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    workers: int = 1,
    backend: str = "python",
) -> pa.Table:
```

| Parameter | Change | Notes |
|-----------|--------|-------|
| `columns` | **Removed** | Replaced by `mnemonics` |
| `mnemonics` | **New** | Optional list of BUFR mnemonic strings. `None` = all available fields. |
| `backend` | **New** | `"python"` (default) or `"fortran"`. |

### Backend Dispatch

```python
if backend == "python":
    # existing pipeline: read_messages -> parse -> expand -> decode -> build_table
elif backend == "fortran":
    from earth2bufrio._fortran_backend import read_ncep
    return read_ncep(path, mnemonics=mnemonics, filters=filters, workers=workers)
else:
    raise ValueError(f"Unknown backend: {backend!r}")
```

### Wide-Format Output Schema

Both backends produce a **wide-format** PyArrow Table. Instead of one row per
descriptor value, there is one row per subset, with columns named after BUFR
mnemonics.

Fixed columns (always present):

| Column | Type | Description |
|--------|------|-------------|
| `message_type` | `string` | BUFR message type (e.g., `"NC021203"`, `"ADPUPA"`) |
| `message_index` | `int32` | Zero-based message index in the file |
| `subset_index` | `int32` | Zero-based subset index within the message |

Time columns (always present):

| Column | Type |
|--------|------|
| `YEAR` | `int32` |
| `MNTH` | `int32` |
| `DAYS` | `int32` |
| `HOUR` | `int32` |
| `MINU` | `int32` |
| `SECO` | `int32` |

Mnemonic columns (dynamic, depend on file content and `mnemonics` parameter):

- Each unique mnemonic produces a `float64` column (or `string` for character
  mnemonics like `SID`).
- Replicated mnemonics (e.g., `TMBR` with 22 channels) produce array-typed
  columns: `pa.list_(pa.float64())`.
- Missing values are `null`.

Example table for satellite radiance (ATMS):

```text
msg_type | msg_idx | sub_idx | YEAR | CLATH | SAID | TMBR
NC021203 | 0       | 0       | 2024 | 45.2  | 224  | [210.1, ...]
NC021203 | 0       | 1       | 2024 | 45.3  | 224  | [209.8, ...]
```

Example table for PrepBUFR:

```text
msg_type | msg_idx | sub_idx | YEAR | YOB  | TYP | POB
ADPUPA   | 2       | 0       | 2024 | 40.1 | 120 | [1000, 925, ...]
```

---

## Fortran Wrapper: `earth2bufrio_fort.f90`

ISO\_C\_BINDING module exposing C-callable functions that wrap NCEPLIBS-bufr
routines:

```fortran
module earth2bufrio_fort
  use iso_c_binding
  implicit none
contains

  ! Open a BUFR file and return Fortran unit number
  integer(c_int) function e2b_open(filepath, filepath_len) bind(c)
    character(c_char), intent(in) :: filepath(*)
    integer(c_int), value, intent(in) :: filepath_len

  ! Read next message; returns 0=ok, 1=EOF
  integer(c_int) function e2b_next_message(lun, msg_type, msg_type_len, idate) bind(c)
    integer(c_int), value, intent(in) :: lun
    character(c_char), intent(out) :: msg_type(*)
    integer(c_int), intent(out) :: msg_type_len
    integer(c_int), intent(out) :: idate

  ! Read next subset; returns 0=ok, 1=no-more
  integer(c_int) function e2b_next_subset(lun) bind(c)
    integer(c_int), value, intent(in) :: lun

  ! Read scalar/vector values for a mnemonic string (wraps ufbint)
  integer(c_int) function e2b_read_values(lun, mnemonic, mnem_len, &
      values, max_values, nvalues) bind(c)
    integer(c_int), value, intent(in) :: lun
    character(c_char), intent(in) :: mnemonic(*)
    integer(c_int), value, intent(in) :: mnem_len
    real(c_double), intent(out) :: values(*)
    integer(c_int), value, intent(in) :: max_values
    integer(c_int), intent(out) :: nvalues

  ! Read replicated values (wraps ufbrep)
  integer(c_int) function e2b_read_replicated(lun, mnemonic, mnem_len, &
      values, max_values, nvalues) bind(c)
    ! same signature as e2b_read_values

  ! Close the BUFR file
  subroutine e2b_close(lun) bind(c)
    integer(c_int), value, intent(in) :: lun

  ! Get the BUFR missing-value sentinel
  real(c_double) function e2b_get_bmiss() bind(c)

end module
```

### Build System

`src/fortran/CMakeLists.txt`:

1. Build NCEPLIBS-bufr as a static library (`libbufr_4.a`) from the bundled
   source in `nceplibs-bufr/`.
2. Compile `earth2bufrio_fort.f90` into a shared library
   (`libearth2bufrio_fort.so`) linked against `libbufr_4.a`.
3. Install the `.so` into a discoverable location (e.g., `src/earth2bufrio/`).

Makefile target:

```makefile
.PHONY: fortran
fortran:
    cmake -S src/fortran -B build/fortran -DCMAKE_BUILD_TYPE=Release
    cmake --build build/fortran --parallel
    cp build/fortran/libearth2bufrio_fort.so src/earth2bufrio/
```

---

## Python ctypes Wrapper: `_fortran_backend.py`

### Library Loading

```python
import ctypes

def _load_lib() -> ctypes.CDLL:
    """Find and load libearth2bufrio_fort.so."""
    # Search order: package directory, LD_LIBRARY_PATH, system paths
    lib_name = "libearth2bufrio_fort.so"
    pkg_dir = Path(__file__).parent
    candidates = [pkg_dir / lib_name, lib_name]
    for path in candidates:
        try:
            return ctypes.CDLL(str(path))
        except OSError:
            continue
    msg = (
        f"Could not load {lib_name}. "
        "Build with 'make fortran' first."
    )
    raise RuntimeError(msg)
```

### Main Entry Point

```python
def read_ncep(
    path: str | Path,
    *,
    mnemonics: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    workers: int = 1,
) -> pa.Table:
    """Read an NCEP BUFR/PrepBUFR file using the Fortran backend.

    Parameters
    ----------
    path : str | Path
        Path to the BUFR file.
    mnemonics : list[str] | None
        Mnemonic strings to extract. None = discover all.
    filters : dict[str, Any] | None
        Message-level filters (e.g., ``{"message_type": "NC021203"}``).
    workers : int
        Unused (reserved for future multi-file parallelism).

    Returns
    -------
    pa.Table
        Wide-format table with one row per subset.
    """
```

### Data Flow

1. Call `e2b_open(filepath)` to get a unit number.
2. Loop: `e2b_next_message()` to iterate messages.
   - Extract `msg_type` and `idate`.
   - Apply `filters` (e.g., `message_type` filter).
3. Inner loop: `e2b_next_subset()` to iterate subsets.
   - For each mnemonic in the requested set:
     - Call `e2b_read_values()` (scalar fields: lat, lon, time, etc.).
     - Call `e2b_read_replicated()` (multi-level fields: TMBR, POB, etc.).
   - Replace BUFR missing sentinel (`bmiss`) with `None`/`NaN`.
   - Accumulate row as a dict.
4. Call `e2b_close(lun)`.
5. Convert accumulated rows to `build_table()` format.

### Mnemonic Discovery

When `mnemonics=None`, the Fortran backend tries a default set of well-known
mnemonics per message type:

```python
_DEFAULT_MNEMONICS: dict[str, list[str]] = {
    # Satellite radiance
    "NC021203": ["SAID", "CLATH", "CLONH", "SAZA", "SOZA", "IANG", "TMBR", "CHNM",
                 "YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO"],
    "NC021023": ["SAID", "CLAT", "CLON", "SAZA", "SOZA", "IANG", "TMBR", "CHNM",
                 "YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO"],
    # PrepBUFR
    "_PREPBUFR": ["YOB", "XOB", "DHR", "ELV", "TYP",
                  "POB", "QOB", "TOB", "ZOB", "UOB", "VOB", "PWO", "TDO",
                  "PMO", "XDR", "YDR", "HRDR"],
}
```

For unknown message types, the backend falls back to attempting all PrepBUFR
mnemonics and keeping only those that return data.

---

## Wide-Format `_arrow.py` Rewrite

### New `build_table()` Signature

```python
def build_table(
    rows: list[dict[str, Any]],
    mnemonics: list[str] | None = None,
) -> pa.Table:
    """Convert rows of mnemonic-keyed data into a PyArrow Table.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Each dict represents one subset. Keys are mnemonic names plus
        fixed keys (``message_type``, ``message_index``, ``subset_index``,
        ``YEAR``, ``MNTH``, ``DAYS``, ``HOUR``, ``MINU``, ``SECO``).
        Values are scalars (float, int, str) or lists (replicated data).
    mnemonics : list[str] | None
        If given, only include these mnemonic columns (plus fixed columns).

    Returns
    -------
    pa.Table
        Wide-format table.
    """
```

### Schema Construction

The schema is dynamic, built from the union of all mnemonic keys across rows:

1. Start with fixed columns: `message_type` (string), `message_index` (int32),
   `subset_index` (int32).
2. Add time columns: `YEAR`, `MNTH`, `DAYS`, `HOUR`, `MINU`, `SECO` (int32).
3. For each unique mnemonic key in the rows:
   - If all values for that key are scalars: `float64` (or `string` for
     character data).
   - If any value is a list: `list_(float64())`.
4. If `mnemonics` is specified, keep only those mnemonic columns.

### Python Backend Adapter

The Python backend currently produces `DecodedSubset` objects. A thin adapter
converts these to the row-dict format:

```python
def _python_subsets_to_rows(decoded_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Python backend output to wide-format rows."""
    rows = []
    for msg in decoded_messages:
        for subset_idx, subset in enumerate(msg["subsets"]):
            row = {
                "message_type": str(msg.get("message_type", "")),
                "message_index": msg["message_index"],
                "subset_index": subset_idx,
            }
            for desc, val in subset.values:
                name = desc.entry.name
                if name in row and isinstance(row[name], list):
                    row[name].append(val)
                elif name in row:
                    row[name] = [row[name], val]
                else:
                    row[name] = val
            rows.append(row)
    return rows
```

---

## Testing Strategy

### Unit Tests (mocked ctypes)

- `test_fortran_backend.py`: Mock `ctypes.CDLL` to simulate library calls.
  Verify `read_ncep()` produces correct PyArrow Tables from simulated data.
- `test_arrow_wide.py`: Test `build_table()` with hand-crafted row dicts.
  Cover scalar columns, list columns, missing values, `mnemonics` filtering.

### Integration Tests (`@pytest.mark.fortran`)

Require `libearth2bufrio_fort.so` to be built. Skipped if library not found.

- Read a real ATMS satellite BUFR file with the Fortran backend.
- Read a real PrepBUFR file with the Fortran backend.
- Verify output schema and value ranges.

### Cross-Validation Tests (`@pytest.mark.crossval`)

- Read the same BUFR file with both backends.
- Compare output tables: same number of subsets, same mnemonic values within
  tolerance.

### Existing Test Updates

- `test_api.py`: Update for new `read_bufr()` signature (no `columns`, add
  `mnemonics` and `backend`).
- `test_arrow.py`: Rewrite for wide-format schema.

---

## Migration Plan (Breaking Changes)

This is a **breaking change** to the `read_bufr()` API:

1. `columns` parameter is removed.
2. Output schema changes from long-format (14 columns) to wide-format (dynamic
   mnemonic columns).
3. `build_table()` input format changes from `decoded_messages` list to `rows`
   list.

Since earth2bufrio is version 0.1.0 and has no external consumers yet, this is
acceptable without a deprecation period.

---

## Dependencies

| Dependency | Scope | Notes |
|------------|-------|-------|
| `pyarrow >= 14.0` | Runtime | Already required |
| `numpy` | Optional runtime | Used by Fortran backend for ctypes array marshalling |
| NCEPLIBS-bufr source | Build-time | Bundled in `src/fortran/nceplibs-bufr/` |
| gfortran + CMake | Build-time | Required for `make fortran` |

---

## Non-Goals

- Thread safety for the Fortran backend (NCEPLIBS-bufr is not thread-safe).
- Multi-file parallelism (reserved for future work; `workers` param is ignored
  by Fortran backend).
- Automatic mnemonic discovery via DX table inspection (future enhancement;
  for now, fall back to default mnemonic sets).
- Rust backend changes (unchanged by this work).
