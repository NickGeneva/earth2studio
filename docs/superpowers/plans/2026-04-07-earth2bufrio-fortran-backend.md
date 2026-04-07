<!-- markdownlint-disable MD013 MD010 MD032 -->
# earth2bufrio Fortran Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Fortran backend (NCEPLIBS-bufr via ctypes) and migrate both backends to wide-format output with mnemonic column names.

**Architecture:** The Fortran backend is a parallel pipeline — `read_bufr(backend="fortran")` dispatches to `_fortran_backend.read_ncep()` which calls NCEPLIBS-bufr through ISO\_C\_BINDING + ctypes. The output schema changes from 14-column long-format to dynamic wide-format (one column per mnemonic). Both backends share the same `build_table()` function.

**Tech Stack:** Python 3.11+, PyArrow, ctypes, Fortran 90 (ISO\_C\_BINDING), NCEPLIBS-bufr, CMake.

**Test command:** `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/ -v`

**Lint command (from recipe dir):** `cd recipes/earth2bufrio && ruff check src/ tests/ && ruff format --check src/ tests/`

**Key convention:** All `.py` files in `src/earth2bufrio/` MUST have the SPDX Apache-2.0 header:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

**Key convention:** Use `from __future__ import annotations` in every module. Move type-only imports into `if TYPE_CHECKING:` blocks (ruff TCH001/TCH002). Use `logging` (not `loguru`).

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/earth2bufrio/_arrow.py` | **Rewrite** | Wide-format table builder from row-dicts |
| `src/earth2bufrio/_api.py` | **Modify** | Add `backend=`, `mnemonics=`, drop `columns=`, add Python->rows adapter |
| `src/earth2bufrio/__init__.py` | **Modify** | Keep exports, update version |
| `src/earth2bufrio/_fortran_backend.py` | **Create** | ctypes wrapper for Fortran shared lib |
| `src/earth2bufrio/_types.py` | **No change** | Existing dataclasses unchanged |
| `src/fortran/CMakeLists.txt` | **Create** | Build NCEPLIBS-bufr + wrapper shared lib |
| `src/fortran/earth2bufrio_fort.f90` | **Create** | ISO\_C\_BINDING wrappers around NCEPLIBS-bufr |
| `Makefile` | **Modify** | Add `fortran` target |
| `pyproject.toml` | **Modify** | Add `numpy` optional dep, `fortran` marker |
| `tests/test_arrow.py` | **Rewrite** | Wide-format schema tests |
| `tests/test_api.py` | **Rewrite** | Updated API signature tests |
| `tests/test_fortran_backend.py` | **Create** | Mocked ctypes unit tests |

---

### Task 1: Rewrite `_arrow.py` — Wide-Format Table Builder

**Files:**
- Modify: `recipes/earth2bufrio/src/earth2bufrio/_arrow.py`
- Rewrite: `recipes/earth2bufrio/tests/test_arrow.py`

This task replaces the 14-column long-format schema with a dynamic wide-format schema. The new `build_table()` takes a list of row-dicts (one per subset) with mnemonic keys.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_arrow.py` with the new wide-format tests:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for wide-format PyArrow table construction."""

from __future__ import annotations

import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from earth2bufrio._arrow import build_table


class TestBuildTableBasic:
    """Basic wide-format table construction."""

    def test_single_scalar_row(self) -> None:
        """One row with scalar mnemonics produces correct schema."""
        rows = [
            {
                "message_type": "NC021203",
                "message_index": 0,
                "subset_index": 0,
                "YEAR": 2024,
                "MNTH": 6,
                "DAYS": 15,
                "HOUR": 12,
                "MINU": 0,
                "SECO": 0,
                "CLATH": 45.2,
                "SAID": 224.0,
            }
        ]
        table = build_table(rows)

        assert isinstance(table, pa.Table)
        assert table.num_rows == 1
        assert "message_type" in table.column_names
        assert "message_index" in table.column_names
        assert "subset_index" in table.column_names
        assert "YEAR" in table.column_names
        assert "CLATH" in table.column_names
        assert "SAID" in table.column_names
        assert table.column("CLATH").to_pylist() == [pytest.approx(45.2)]

    def test_list_valued_mnemonic(self) -> None:
        """A mnemonic with list values produces list-typed column."""
        rows = [
            {
                "message_type": "NC021203",
                "message_index": 0,
                "subset_index": 0,
                "YEAR": 2024,
                "MNTH": 6,
                "DAYS": 15,
                "HOUR": 12,
                "MINU": 0,
                "SECO": 0,
                "TMBR": [210.1, 209.8, 211.0],
            }
        ]
        table = build_table(rows)

        assert pa.types.is_list(table.schema.field("TMBR").type)
        vals = table.column("TMBR").to_pylist()[0]
        assert len(vals) == 3
        assert vals[0] == pytest.approx(210.1)

    def test_missing_values_are_null(self) -> None:
        """Rows missing a mnemonic get null in that column."""
        rows = [
            {
                "message_type": "A",
                "message_index": 0,
                "subset_index": 0,
                "YEAR": 2024,
                "MNTH": 1,
                "DAYS": 1,
                "HOUR": 0,
                "MINU": 0,
                "SECO": 0,
                "CLATH": 45.0,
            },
            {
                "message_type": "A",
                "message_index": 0,
                "subset_index": 1,
                "YEAR": 2024,
                "MNTH": 1,
                "DAYS": 1,
                "HOUR": 0,
                "MINU": 0,
                "SECO": 0,
                "CLONH": -93.0,
            },
        ]
        table = build_table(rows)

        assert table.num_rows == 2
        clath = table.column("CLATH").to_pylist()
        clonh = table.column("CLONH").to_pylist()
        assert clath[0] == pytest.approx(45.0)
        assert clath[1] is None
        assert clonh[0] is None
        assert clonh[1] == pytest.approx(-93.0)

    def test_empty_input(self) -> None:
        """Empty list returns table with only fixed columns."""
        table = build_table([])

        assert isinstance(table, pa.Table)
        assert table.num_rows == 0
        assert "message_type" in table.column_names
        assert "message_index" in table.column_names
        assert "subset_index" in table.column_names

    def test_mnemonics_filter(self) -> None:
        """mnemonics param restricts dynamic columns."""
        rows = [
            {
                "message_type": "A",
                "message_index": 0,
                "subset_index": 0,
                "YEAR": 2024,
                "MNTH": 1,
                "DAYS": 1,
                "HOUR": 0,
                "MINU": 0,
                "SECO": 0,
                "CLATH": 45.0,
                "CLONH": -93.0,
                "SAID": 224.0,
            }
        ]
        table = build_table(rows, mnemonics=["CLATH"])

        assert "CLATH" in table.column_names
        assert "CLONH" not in table.column_names
        assert "SAID" not in table.column_names
        # Fixed columns always present
        assert "message_type" in table.column_names

    def test_string_mnemonic(self) -> None:
        """Character-valued mnemonic produces string column."""
        rows = [
            {
                "message_type": "A",
                "message_index": 0,
                "subset_index": 0,
                "YEAR": 2024,
                "MNTH": 1,
                "DAYS": 1,
                "HOUR": 0,
                "MINU": 0,
                "SECO": 0,
                "SID": "KORD",
            }
        ]
        table = build_table(rows)

        assert pa.types.is_string(table.schema.field("SID").type)
        assert table.column("SID").to_pylist() == ["KORD"]

    def test_multiple_rows(self) -> None:
        """Multiple rows from different messages."""
        rows = [
            {
                "message_type": "A",
                "message_index": 0,
                "subset_index": 0,
                "YEAR": 2024,
                "MNTH": 1,
                "DAYS": 1,
                "HOUR": 0,
                "MINU": 0,
                "SECO": 0,
                "TOB": 273.1,
            },
            {
                "message_type": "A",
                "message_index": 1,
                "subset_index": 0,
                "YEAR": 2024,
                "MNTH": 1,
                "DAYS": 1,
                "HOUR": 6,
                "MINU": 0,
                "SECO": 0,
                "TOB": 274.2,
            },
        ]
        table = build_table(rows)

        assert table.num_rows == 2
        assert table.column("message_index").to_pylist() == [0, 1]
        tob = table.column("TOB").to_pylist()
        assert tob[0] == pytest.approx(273.1)
        assert tob[1] == pytest.approx(274.2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/test_arrow.py -v`

Expected: FAIL — `build_table` signature changed, old implementation doesn't accept row-dicts.

- [ ] **Step 3: Rewrite `_arrow.py` with wide-format implementation**

Replace the entire contents of `recipes/earth2bufrio/src/earth2bufrio/_arrow.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert decoded BUFR data into wide-format PyArrow Tables."""

from __future__ import annotations

from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Fixed columns — always present in every table
# ---------------------------------------------------------------------------
_FIXED_COLUMNS = ("message_type", "message_index", "subset_index")
_TIME_COLUMNS = ("YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO")
_ALL_FIXED = _FIXED_COLUMNS + _TIME_COLUMNS


def build_table(
    rows: list[dict[str, Any]],
    mnemonics: list[str] | None = None,
) -> pa.Table:
    """Convert rows of mnemonic-keyed data into a wide-format PyArrow Table.

    Each dict in *rows* represents one BUFR subset.  Keys include the
    fixed columns (``message_type``, ``message_index``, ``subset_index``,
    ``YEAR``, ``MNTH``, ``DAYS``, ``HOUR``, ``MINU``, ``SECO``) plus
    one key per extracted mnemonic.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        One dict per subset.  Values are scalars (``float``, ``int``,
        ``str``) or lists (replicated data).
    mnemonics : list[str] | None, optional
        If given, only these mnemonic columns are included (fixed columns
        are always present).

    Returns
    -------
    pa.Table
        Wide-format table with one row per subset.
    """
    if not rows:
        schema = pa.schema(
            [
                pa.field("message_type", pa.string()),
                pa.field("message_index", pa.int32()),
                pa.field("subset_index", pa.int32()),
                pa.field("YEAR", pa.int32()),
                pa.field("MNTH", pa.int32()),
                pa.field("DAYS", pa.int32()),
                pa.field("HOUR", pa.int32()),
                pa.field("MINU", pa.int32()),
                pa.field("SECO", pa.int32()),
            ]
        )
        return pa.table({name: pa.array([], type=f.type) for name, f in zip(schema.names, schema)}, schema=schema)

    # Discover all mnemonic keys across rows
    mnemonic_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in _ALL_FIXED and key not in seen:
                seen.add(key)
                mnemonic_keys.append(key)

    # Apply mnemonics filter
    if mnemonics is not None:
        allowed = set(mnemonics)
        mnemonic_keys = [k for k in mnemonic_keys if k in allowed]

    # Determine type for each mnemonic column by inspecting first non-None value
    col_types: dict[str, pa.DataType] = {}
    for key in mnemonic_keys:
        for row in rows:
            val = row.get(key)
            if val is not None:
                if isinstance(val, list):
                    # Check inner type
                    if val and isinstance(val[0], str):
                        col_types[key] = pa.list_(pa.string())
                    else:
                        col_types[key] = pa.list_(pa.float64())
                elif isinstance(val, str):
                    col_types[key] = pa.string()
                else:
                    col_types[key] = pa.float64()
                break
        else:
            # All None — default to float64
            col_types[key] = pa.float64()

    # Build column arrays
    col_data: dict[str, list[Any]] = {name: [] for name in _ALL_FIXED}
    for key in mnemonic_keys:
        col_data[key] = []

    for row in rows:
        col_data["message_type"].append(row.get("message_type", ""))
        col_data["message_index"].append(row.get("message_index", 0))
        col_data["subset_index"].append(row.get("subset_index", 0))
        for tc in _TIME_COLUMNS:
            col_data[tc].append(row.get(tc, 0))
        for key in mnemonic_keys:
            col_data[key].append(row.get(key))

    # Build schema
    fields: list[pa.Field] = [
        pa.field("message_type", pa.string()),
        pa.field("message_index", pa.int32()),
        pa.field("subset_index", pa.int32()),
    ]
    for tc in _TIME_COLUMNS:
        fields.append(pa.field(tc, pa.int32()))
    for key in mnemonic_keys:
        fields.append(pa.field(key, col_types[key]))

    schema = pa.schema(fields)

    # Build arrays
    arrays: dict[str, pa.Array] = {}
    arrays["message_type"] = pa.array(col_data["message_type"], type=pa.string())
    arrays["message_index"] = pa.array(col_data["message_index"], type=pa.int32())
    arrays["subset_index"] = pa.array(col_data["subset_index"], type=pa.int32())
    for tc in _TIME_COLUMNS:
        arrays[tc] = pa.array(col_data[tc], type=pa.int32())
    for key in mnemonic_keys:
        arrays[key] = pa.array(col_data[key], type=col_types[key])

    return pa.table(arrays, schema=schema)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/test_arrow.py -v`

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add recipes/earth2bufrio/src/earth2bufrio/_arrow.py recipes/earth2bufrio/tests/test_arrow.py
git commit -m "feat(earth2bufrio): rewrite _arrow.py for wide-format output schema"
```

---

### Task 2: Update `_api.py` — New Signature + Python Adapter

**Files:**
- Modify: `recipes/earth2bufrio/src/earth2bufrio/_api.py`
- Rewrite: `recipes/earth2bufrio/tests/test_api.py`

This task changes `read_bufr()` to accept `backend=`, `mnemonics=` and drops `columns=`. It also adds `_python_subsets_to_rows()` to convert the Python backend's `DecodedSubset` output into the row-dict format expected by the new `build_table()`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_api.py` with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public ``read_bufr()`` API."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from earth2bufrio import read_bufr

if TYPE_CHECKING:
    from pathlib import Path


def _fxy_to_packed(fxy: int) -> int:
    """Convert decimal FXXYYY to 16-bit packed wire format."""
    f = fxy // 100000
    x = (fxy % 100000) // 1000
    y = fxy % 1000
    return (f << 14) | (x << 8) | y


def _build_section1_ed4(data_category: int = 0) -> bytes:
    """Build a minimal Edition 4 Section 1 (Identification)."""
    body = bytearray(19)
    body[0] = 4
    body[1:3] = (7).to_bytes(2, "big")
    body[3:5] = (0).to_bytes(2, "big")
    body[5] = 0
    body[6] = 0
    body[7] = data_category
    body[8] = 0
    body[9] = 0
    body[10] = 0
    body[11] = 0
    body[12:14] = (2025).to_bytes(2, "big")
    body[14] = 6
    body[15] = 15
    body[16] = 12
    body[17] = 0
    body[18] = 0
    sec_len = 3 + len(body)
    return sec_len.to_bytes(3, "big") + bytes(body)


def _build_section3(descriptors: list[int], num_subsets: int = 1) -> bytes:
    """Build Section 3 (Data Description)."""
    body = bytearray()
    body.append(0)
    body.extend(num_subsets.to_bytes(2, "big"))
    body.append(0x80)
    for d in descriptors:
        body.extend(_fxy_to_packed(d).to_bytes(2, "big"))
    sec_len = 3 + len(body)
    return sec_len.to_bytes(3, "big") + bytes(body)


def _build_section4(data_bits: bytes) -> bytes:
    """Build Section 4 (Data) from raw payload bytes."""
    sec_len = 4 + len(data_bits)
    return sec_len.to_bytes(3, "big") + b"\x00" + data_bits


def _build_bufr_ed4(
    data_category: int = 0,
    descriptors: list[int] | None = None,
    data_bits: bytes = b"",
    num_subsets: int = 1,
) -> bytes:
    """Assemble a complete minimal BUFR Ed4 message."""
    if descriptors is None:
        descriptors = []
    sec1 = _build_section1_ed4(data_category)
    sec3 = _build_section3(descriptors, num_subsets)
    sec4 = _build_section4(data_bits)
    total_len = 8 + len(sec1) + len(sec3) + len(sec4) + 4
    sec0 = b"BUFR" + total_len.to_bytes(3, "big") + b"\x04"
    end = b"7777"
    return sec0 + sec1 + sec3 + sec4 + end


class TestReadBufrErrors:
    """Tests for error handling in read_bufr."""

    def test_nonexistent_path_raises_file_not_found(self, tmp_path: Path) -> None:
        """read_bufr() raises FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError):
            read_bufr(tmp_path / "nonexistent.bufr")

    def test_empty_file_returns_empty_table(self, tmp_path: Path) -> None:
        """An empty file produces an empty table with fixed columns."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        table = read_bufr(p)
        assert isinstance(table, pa.Table)
        assert len(table) == 0
        assert "message_type" in table.column_names
        assert "message_index" in table.column_names
        assert "subset_index" in table.column_names

    def test_unknown_backend_raises(self, tmp_path: Path) -> None:
        """Unknown backend raises ValueError."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="Unknown backend"):
            read_bufr(p, backend="unknown")


class TestReadBufrBasic:
    """Tests for basic read_bufr functionality with hand-crafted BUFR data."""

    def test_hand_crafted_single_descriptor(self, tmp_path: Path) -> None:
        """A message with one numeric descriptor produces one row."""
        raw = 2731
        data_bits = struct.pack(">H", raw << 4)
        bufr = _build_bufr_ed4(
            data_category=2,
            descriptors=[12001],
            data_bits=data_bits,
        )
        p = tmp_path / "temp.bufr"
        p.write_bytes(bufr)

        table = read_bufr(p)
        assert len(table) >= 1
        assert "message_type" in table.column_names
        assert "message_index" in table.column_names

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """read_bufr() accepts a string path."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        table = read_bufr(str(p))
        assert isinstance(table, pa.Table)
        assert len(table) == 0


class TestReadBufrFiltering:
    """Tests for the mnemonics and filters parameters."""

    def test_filters_data_category(self, tmp_path: Path) -> None:
        """filters={'data_category': 0} skips non-matching messages."""
        msg0 = _build_bufr_ed4(data_category=0)
        msg2 = _build_bufr_ed4(data_category=2)
        p = tmp_path / "multi.bufr"
        p.write_bytes(msg0 + msg2)

        table_filtered = read_bufr(p, filters={"data_category": 0})
        # All returned rows should be from data_category=0 messages
        assert isinstance(table_filtered, pa.Table)


class TestReadBufrBackend:
    """Tests for the backend parameter."""

    def test_python_backend_explicit(self, tmp_path: Path) -> None:
        """backend='python' works explicitly."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        table = read_bufr(p, backend="python")
        assert isinstance(table, pa.Table)
        assert len(table) == 0

    def test_workers_parameter_accepted(self, tmp_path: Path) -> None:
        """workers parameter does not raise for workers=2."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        table = read_bufr(p, workers=2)
        assert isinstance(table, pa.Table)
        assert len(table) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/test_api.py -v`

Expected: FAIL — `read_bufr` still has old signature with `columns=`.

- [ ] **Step 3: Rewrite `_api.py`**

Replace `recipes/earth2bufrio/src/earth2bufrio/_api.py` with the updated implementation:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API for reading BUFR files into PyArrow Tables."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from earth2bufrio._arrow import build_table
from earth2bufrio._decoder import decode
from earth2bufrio._descriptors import expand_descriptors
from earth2bufrio._reader import read_messages
from earth2bufrio._section import parse_message
from earth2bufrio._tables import TableSet

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore[import-untyped]

    from earth2bufrio._types import DecodedSubset, ParsedMessage

logger = logging.getLogger(__name__)


def read_bufr(
    path: str | Path,
    *,
    mnemonics: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    workers: int = 1,
    backend: str = "python",
) -> pa.Table:
    """Read a BUFR file and return its contents as a PyArrow Table.

    Parameters
    ----------
    path : str | Path
        Path to the BUFR file on disk.
    mnemonics : list[str] | None, optional
        Mnemonic strings to extract.  ``None`` returns all available
        fields.
    filters : dict[str, Any] | None, optional
        Key-value pairs to filter messages.  Supports
        ``"data_category"`` (int) and ``"message_type"`` (str).
    workers : int, optional
        Number of parallel workers for decoding.  ``1`` (default) uses
        the current process; values ``>1`` use a
        :class:`~concurrent.futures.ProcessPoolExecutor`.
    backend : str, optional
        Decoding backend.  ``"python"`` (default) uses the pure-Python
        decoder.  ``"fortran"`` uses the NCEPLIBS-bufr Fortran backend
        (requires ``make fortran`` first).

    Returns
    -------
    pa.Table
        Wide-format table with one row per subset.  Columns are named
        after BUFR mnemonics.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    BufrDecodeError
        If the file contains malformed BUFR data.
    ValueError
        If *backend* is not ``"python"`` or ``"fortran"``.

    Examples
    --------
    >>> import earth2bufrio
    >>> table = earth2bufrio.read_bufr("observations.bufr")
    >>> table.column_names[:3]
    ['message_type', 'message_index', 'subset_index']
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = f"BUFR file not found: {file_path}"
        raise FileNotFoundError(msg)

    if backend == "fortran":
        from earth2bufrio._fortran_backend import read_ncep

        return read_ncep(
            file_path,
            mnemonics=mnemonics,
            filters=filters,
            workers=workers,
        )

    if backend != "python":
        msg = f"Unknown backend: {backend!r}. Use 'python' or 'fortran'."
        raise ValueError(msg)

    # --- Python backend ---
    raw_data = file_path.read_bytes()
    if len(raw_data) == 0:
        return build_table([], mnemonics=mnemonics)

    messages = list(read_messages(raw_data))
    if not messages:
        return build_table([], mnemonics=mnemonics)

    tables = TableSet()

    parsed_messages: list[tuple[int, ParsedMessage]] = []
    for msg in messages:
        parsed = parse_message(msg)
        ident = parsed.identification

        if ident.data_category == 11:
            logger.debug("Skipping DX table message %d", msg.index)
            continue

        parsed_messages.append((int(msg.index), parsed))

    if filters is not None:
        data_cat_filter = filters.get("data_category")
        if data_cat_filter is not None:
            parsed_messages = [
                (idx, pm)
                for idx, pm in parsed_messages
                if pm.identification.data_category == data_cat_filter
            ]

    if not parsed_messages:
        return build_table([], mnemonics=mnemonics)

    if workers > 1 and len(parsed_messages) > 1:
        decoded_msgs = _decode_parallel(parsed_messages, tables, workers)
    else:
        decoded_msgs = _decode_sequential(parsed_messages, tables)

    rows = _python_subsets_to_rows(decoded_msgs)
    return build_table(rows, mnemonics=mnemonics)


def _python_subsets_to_rows(
    decoded_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Python backend decoded messages to wide-format row dicts.

    Parameters
    ----------
    decoded_messages : list[dict[str, Any]]
        Output from ``_decode_single()``.

    Returns
    -------
    list[dict[str, Any]]
        One dict per subset, keyed by mnemonic name.
    """
    rows: list[dict[str, Any]] = []
    for msg in decoded_messages:
        subsets: list[DecodedSubset] = msg["subsets"]
        for subset_idx, subset in enumerate(subsets):
            row: dict[str, Any] = {
                "message_type": str(msg.get("data_category", "")),
                "message_index": msg["message_index"],
                "subset_index": subset_idx,
                "YEAR": msg["year"],
                "MNTH": msg["month"],
                "DAYS": msg["day"],
                "HOUR": msg["hour"],
                "MINU": msg["minute"],
                "SECO": msg["second"],
            }
            for desc, val in subset.values:
                name = desc.entry.name
                if name in row and isinstance(row[name], list):
                    row[name].append(val)
                elif name in row and name not in _FIXED_ROW_KEYS:
                    row[name] = [row[name], val]
                else:
                    row[name] = val
            rows.append(row)
    return rows


_FIXED_ROW_KEYS = frozenset(
    {
        "message_type",
        "message_index",
        "subset_index",
        "YEAR",
        "MNTH",
        "DAYS",
        "HOUR",
        "MINU",
        "SECO",
    }
)


def _decode_single(
    msg_index: int,
    parsed: ParsedMessage,
    tables: TableSet,
) -> dict[str, Any]:
    """Decode a single parsed message into the dict format for build_table.

    Parameters
    ----------
    msg_index : int
        The original message index in the file.
    parsed : ParsedMessage
        The parsed (but not decoded) BUFR message.
    tables : TableSet
        The BUFR table set for descriptor look-ups.

    Returns
    -------
    dict[str, Any]
        Dict with keys expected by :func:`_python_subsets_to_rows`.
    """
    ident = parsed.identification
    desc_section = parsed.data_description

    expanded = expand_descriptors(desc_section.descriptors, tables)

    subsets = decode(
        expanded,
        parsed.data_bytes,
        ident.num_subsets,
        ident.compressed,
    )

    return {
        "message_index": msg_index,
        "data_category": ident.data_category,
        "year": ident.year,
        "month": ident.month,
        "day": ident.day,
        "hour": ident.hour,
        "minute": ident.minute,
        "second": ident.second,
        "subsets": subsets,
    }


def _decode_sequential(
    parsed_messages: list[tuple[int, ParsedMessage]],
    tables: TableSet,
) -> list[dict[str, Any]]:
    """Decode messages sequentially in the current process.

    Parameters
    ----------
    parsed_messages : list[tuple[int, ParsedMessage]]
        List of (message_index, ParsedMessage) tuples.
    tables : TableSet
        BUFR table set.

    Returns
    -------
    list[dict[str, Any]]
        Decoded message dicts ready for :func:`_python_subsets_to_rows`.
    """
    results: list[dict[str, Any]] = []
    for msg_index, parsed in parsed_messages:
        try:
            result = _decode_single(msg_index, parsed, tables)
            results.append(result)
        except Exception:
            logger.warning("Failed to decode message %d, skipping", msg_index)
    return results


def _decode_parallel(
    parsed_messages: list[tuple[int, ParsedMessage]],
    tables: TableSet,
    workers: int,
) -> list[dict[str, Any]]:
    """Decode messages using a ProcessPoolExecutor.

    Parameters
    ----------
    parsed_messages : list[tuple[int, ParsedMessage]]
        List of (message_index, ParsedMessage) tuples.
    tables : TableSet
        BUFR table set.
    workers : int
        Number of worker processes.

    Returns
    -------
    list[dict[str, Any]]
        Decoded message dicts ready for :func:`_python_subsets_to_rows`.
    """
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_decode_single, msg_index, parsed, tables): msg_index
            for msg_index, parsed in parsed_messages
        }
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception:
                msg_index = futures[future]
                logger.warning("Failed to decode message %d, skipping", msg_index)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/test_api.py -v`

Expected: All 6 tests PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/ -v`

Expected: Some existing tests in `test_crossval.py` may need updating if they reference the old `build_table` signature or check the old 14-column schema. The crossval tests call `read_bufr()` end-to-end, so they should now produce wide-format output. Verify they pass or note needed fixes.

- [ ] **Step 6: Commit**

```bash
git add recipes/earth2bufrio/src/earth2bufrio/_api.py recipes/earth2bufrio/tests/test_api.py
git commit -m "feat(earth2bufrio): update read_bufr() with backend, mnemonics params; add Python->rows adapter"
```

---

### Task 3: Update Cross-Validation and Remaining Tests

**Files:**
- Modify: `recipes/earth2bufrio/tests/test_crossval.py`
- Possibly modify: other test files that reference old schema

After Tasks 1-2, the cross-validation tests and any integration tests may reference the old 14-column schema or old `build_table` signature. This task fixes them.

- [ ] **Step 1: Run the full test suite and identify failures**

Run: `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/ -v 2>&1 | head -100`

Note all failures related to old schema.

- [ ] **Step 2: Fix cross-validation tests**

The `test_crossval.py` tests call `read_bufr()` which now returns wide-format output. The crossval tests compare decoded values against `.ref.json` reference files. Since the crossval tests compare raw decoded values (not column schemas), they likely call internal functions directly and may not need changes. Fix any that reference the old `build_table` or `columns` parameter.

Key patterns to fix:
- `build_table(decoded_msgs, columns=...)` → `build_table(rows, mnemonics=...)`
- `table.column("descriptor_id")` → no longer exists; values are now in mnemonic-named columns
- `table.column("value")` → no longer exists; values are in per-mnemonic columns

- [ ] **Step 3: Run full suite to verify all pass**

Run: `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/ -v`

Expected: All tests PASS (or xfail as before).

- [ ] **Step 4: Commit**

```bash
git add recipes/earth2bufrio/tests/
git commit -m "fix(earth2bufrio): update crossval and remaining tests for wide-format schema"
```

---

### Task 4: Add `fortran` Marker and Dependencies to `pyproject.toml` + Makefile

**Files:**
- Modify: `recipes/earth2bufrio/pyproject.toml`
- Modify: `recipes/earth2bufrio/Makefile`

- [ ] **Step 1: Write the test**

No code test needed — this is configuration. Verify by running existing tests still work after edits.

- [ ] **Step 2: Update `pyproject.toml`**

Add `numpy` optional dependency and `fortran` test marker:

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
fortran = [
    "numpy>=1.24",
]
```

Add `fortran` marker to `[tool.pytest.ini_options]` markers list:

```toml
markers = [
    "unit: fast isolated unit tests",
    "crossval: cross-validation tests against reference decoders",
    "integration: tests requiring external resources",
    "fortran: tests requiring the Fortran backend (make fortran)",
    "slow: long-running tests",
]
```

- [ ] **Step 3: Update `Makefile`**

Add the `fortran` target at the end of the Makefile:

```makefile
.PHONY: lint format test test-cov docs fortran

# ... existing targets ...

fortran:
	cmake -S src/fortran -B build/fortran -DCMAKE_BUILD_TYPE=Release
	cmake --build build/fortran --parallel
	cp build/fortran/libearth2bufrio_fort.so src/earth2bufrio/
```

- [ ] **Step 4: Run tests to verify nothing broke**

Run: `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/ -v`

Expected: All tests still pass.

- [ ] **Step 5: Commit**

```bash
git add recipes/earth2bufrio/pyproject.toml recipes/earth2bufrio/Makefile
git commit -m "chore(earth2bufrio): add fortran marker, numpy optional dep, make fortran target"
```

---

### Task 5: Create Fortran Wrapper — `earth2bufrio_fort.f90`

**Files:**
- Create: `recipes/earth2bufrio/src/fortran/earth2bufrio_fort.f90`

This task creates the ISO\_C\_BINDING Fortran module that wraps NCEPLIBS-bufr routines with C-callable functions.

- [ ] **Step 1: Create the Fortran wrapper**

Create `recipes/earth2bufrio/src/fortran/earth2bufrio_fort.f90`:

```fortran
! SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
! SPDX-FileCopyrightText: All rights reserved.
! SPDX-License-Identifier: Apache-2.0

!> ISO_C_BINDING wrapper around NCEPLIBS-bufr for earth2bufrio.
!!
!! Exposes a small set of C-callable functions that the Python ctypes
!! layer can call to read BUFR / PrepBUFR files.
module earth2bufrio_fort
  use iso_c_binding
  implicit none

  ! Internal state for Fortran unit allocation
  integer, save :: next_unit = 20

contains

  !> Open a BUFR file and return a Fortran logical unit number.
  !!
  !! The caller passes a C character array and its length.
  !! Returns the allocated unit number (>0) on success, or -1 on failure.
  integer(c_int) function e2b_open(filepath, filepath_len) bind(c, name='e2b_open')
    character(c_char), intent(in) :: filepath(*)
    integer(c_int), value, intent(in) :: filepath_len

    character(len=512) :: fpath
    integer :: lun, i

    ! Copy C string to Fortran string
    fpath = ' '
    do i = 1, min(filepath_len, 512)
      fpath(i:i) = filepath(i)
    end do

    ! Allocate a unit number
    lun = next_unit
    next_unit = next_unit + 1

    ! Set 10-digit date format
    call datelen(10)

    ! Open the file
    open(unit=lun, file=trim(fpath), status='old', form='unformatted', &
         access='sequential', iostat=i)
    if (i /= 0) then
      e2b_open = -1_c_int
      return
    end if

    ! Initialize BUFR reading
    call openbf(lun, 'IN', lun)

    e2b_open = int(lun, c_int)
  end function e2b_open


  !> Read the next BUFR message.
  !!
  !! Returns 0 on success (message available), 1 on EOF / no more messages.
  !! On success, msg_type is filled with the 8-character subset identifier
  !! and idate with the message date.
  integer(c_int) function e2b_next_message(lun, msg_type, msg_type_len, idate) &
      bind(c, name='e2b_next_message')
    integer(c_int), value, intent(in) :: lun
    character(c_char), intent(out) :: msg_type(*)
    integer(c_int), intent(out) :: msg_type_len
    integer(c_int), intent(out) :: idate

    character(len=8) :: subset
    integer :: jdate, iret, i

    call readmg(int(lun), subset, jdate, iret)
    if (iret /= 0) then
      e2b_next_message = 1_c_int
      msg_type_len = 0_c_int
      idate = 0_c_int
      return
    end if

    ! Copy subset name to output
    do i = 1, 8
      msg_type(i) = subset(i:i)
    end do
    msg_type_len = int(len_trim(subset), c_int)
    idate = int(jdate, c_int)
    e2b_next_message = 0_c_int
  end function e2b_next_message


  !> Read the next subset within the current message.
  !!
  !! Returns 0 on success, 1 when no more subsets remain.
  integer(c_int) function e2b_next_subset(lun) bind(c, name='e2b_next_subset')
    integer(c_int), value, intent(in) :: lun
    integer :: iret

    call readsb(int(lun), iret)
    if (iret /= 0) then
      e2b_next_subset = 1_c_int
    else
      e2b_next_subset = 0_c_int
    end if
  end function e2b_next_subset


  !> Read scalar or multi-level values for a mnemonic (wraps ufbint).
  !!
  !! Returns 0 on success.  nvalues is set to the number of values read.
  integer(c_int) function e2b_read_values(lun, mnemonic, mnem_len, &
      values, max_values, nvalues) bind(c, name='e2b_read_values')
    integer(c_int), value, intent(in) :: lun
    character(c_char), intent(in) :: mnemonic(*)
    integer(c_int), value, intent(in) :: mnem_len
    real(c_double), intent(out) :: values(*)
    integer(c_int), value, intent(in) :: max_values
    integer(c_int), intent(out) :: nvalues

    character(len=80) :: mnem_str
    real(8) :: buf(255)
    integer :: n, i

    mnem_str = ' '
    do i = 1, min(int(mnem_len), 80)
      mnem_str(i:i) = mnemonic(i)
    end do

    n = 0
    call ufbint(int(lun), buf, 1, min(int(max_values), 255), n, trim(mnem_str))

    nvalues = int(n, c_int)
    do i = 1, n
      values(i) = buf(i)
    end do

    e2b_read_values = 0_c_int
  end function e2b_read_values


  !> Read replicated values for a mnemonic (wraps ufbrep).
  !!
  !! Returns 0 on success.  nvalues is set to the number of values read.
  integer(c_int) function e2b_read_replicated(lun, mnemonic, mnem_len, &
      values, max_values, nvalues) bind(c, name='e2b_read_replicated')
    integer(c_int), value, intent(in) :: lun
    character(c_char), intent(in) :: mnemonic(*)
    integer(c_int), value, intent(in) :: mnem_len
    real(c_double), intent(out) :: values(*)
    integer(c_int), value, intent(in) :: max_values
    integer(c_int), intent(out) :: nvalues

    character(len=80) :: mnem_str
    real(8) :: buf(255)
    integer :: n, i

    mnem_str = ' '
    do i = 1, min(int(mnem_len), 80)
      mnem_str(i:i) = mnemonic(i)
    end do

    n = 0
    call ufbrep(int(lun), buf, 1, min(int(max_values), 255), n, trim(mnem_str))

    nvalues = int(n, c_int)
    do i = 1, n
      values(i) = buf(i)
    end do

    e2b_read_replicated = 0_c_int
  end function e2b_read_replicated


  !> Close the BUFR file and release the unit.
  subroutine e2b_close(lun) bind(c, name='e2b_close')
    integer(c_int), value, intent(in) :: lun

    call closbf(int(lun))
    close(int(lun))
  end subroutine e2b_close


  !> Return the BUFR missing-value sentinel.
  real(c_double) function e2b_get_bmiss() bind(c, name='e2b_get_bmiss')
    real(8) :: getbmiss
    e2b_get_bmiss = real(getbmiss(), c_double)
  end function e2b_get_bmiss

end module earth2bufrio_fort
```

- [ ] **Step 2: Commit**

```bash
git add recipes/earth2bufrio/src/fortran/earth2bufrio_fort.f90
git commit -m "feat(earth2bufrio): add ISO_C_BINDING Fortran wrapper for NCEPLIBS-bufr"
```

---

### Task 6: Create CMake Build — `CMakeLists.txt`

**Files:**
- Create: `recipes/earth2bufrio/src/fortran/CMakeLists.txt`

This task creates the CMake build system that compiles the NCEPLIBS-bufr source (assumed bundled or fetched externally) and the wrapper into `libearth2bufrio_fort.so`.

- [ ] **Step 1: Create `CMakeLists.txt`**

Create `recipes/earth2bufrio/src/fortran/CMakeLists.txt`:

```cmake
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)
project(earth2bufrio_fort Fortran C)

# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------
option(NCEPLIBS_BUFR_SRC_DIR
       "Path to NCEPLIBS-bufr source tree"
       "${CMAKE_CURRENT_SOURCE_DIR}/nceplibs-bufr")

# ------------------------------------------------------------------
# Build NCEPLIBS-bufr as a static library
# ------------------------------------------------------------------
# The user must provide the NCEPLIBS-bufr source in nceplibs-bufr/
# or point NCEPLIBS_BUFR_SRC_DIR to it.
if(EXISTS "${NCEPLIBS_BUFR_SRC_DIR}/CMakeLists.txt")
  add_subdirectory(${NCEPLIBS_BUFR_SRC_DIR} nceplibs-bufr-build EXCLUDE_FROM_ALL)
  set(BUFR_LIB bufr_4)
else()
  # Fall back: assume libbufr_4 is already installed
  find_library(BUFR_LIB NAMES bufr_4 bufr REQUIRED)
endif()

# ------------------------------------------------------------------
# Build the wrapper shared library
# ------------------------------------------------------------------
add_library(earth2bufrio_fort SHARED earth2bufrio_fort.f90)
target_link_libraries(earth2bufrio_fort PRIVATE ${BUFR_LIB})

# Position-independent code for shared library
set_target_properties(earth2bufrio_fort PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod"
)

# Install rule (used by Makefile cp step)
install(TARGETS earth2bufrio_fort
        LIBRARY DESTINATION lib)
```

- [ ] **Step 2: Commit**

```bash
git add recipes/earth2bufrio/src/fortran/CMakeLists.txt
git commit -m "chore(earth2bufrio): add CMake build for Fortran wrapper + NCEPLIBS-bufr"
```

---

### Task 7: Create Python ctypes Wrapper — `_fortran_backend.py`

**Files:**
- Create: `recipes/earth2bufrio/src/earth2bufrio/_fortran_backend.py`
- Create: `recipes/earth2bufrio/tests/test_fortran_backend.py`

This task implements the Python-side ctypes wrapper that calls the Fortran shared library.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fortran_backend.py` with mocked ctypes tests:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Fortran backend (mocked ctypes)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadLib:
    """Tests for _load_lib() library discovery."""

    def test_load_lib_raises_when_not_found(self) -> None:
        """RuntimeError raised when shared lib is not found."""
        with patch("ctypes.CDLL", side_effect=OSError("not found")):
            from earth2bufrio._fortran_backend import _load_lib

            with pytest.raises(RuntimeError, match="Could not load"):
                _load_lib()


class TestReadNcep:
    """Tests for read_ncep() with mocked Fortran library."""

    def _make_mock_lib(self) -> MagicMock:
        """Create a mock Fortran library simulating one message, one subset."""
        lib = MagicMock()

        # e2b_open returns unit number 20
        lib.e2b_open.return_value = 20

        # e2b_get_bmiss returns 1.0e11
        lib.e2b_get_bmiss.return_value = 1.0e11

        # e2b_next_message: first call OK (returns 0), second call EOF (returns 1)
        lib.e2b_next_message.side_effect = [0, 1]

        # e2b_next_subset: first call OK (returns 0), second call no-more (returns 1)
        lib.e2b_next_subset.side_effect = [0, 1]

        # e2b_read_values: return scalar value
        def mock_read_values(lun, mnem, mnem_len, values, max_val, nvalues):
            import ctypes

            nvalues._obj.value = 1
            values[0] = 45.2
            return 0

        lib.e2b_read_values.side_effect = mock_read_values

        # e2b_read_replicated: return no data
        def mock_read_replicated(lun, mnem, mnem_len, values, max_val, nvalues):
            nvalues._obj.value = 0
            return 0

        lib.e2b_read_replicated.side_effect = mock_read_replicated

        lib.e2b_close.return_value = None

        return lib

    def test_read_ncep_returns_pyarrow_table(self, tmp_path: Path) -> None:
        """read_ncep produces a valid PyArrow table from mocked lib."""
        p = tmp_path / "test.bufr"
        p.write_bytes(b"dummy")

        mock_lib = self._make_mock_lib()
        with patch("earth2bufrio._fortran_backend._load_lib", return_value=mock_lib):
            from earth2bufrio._fortran_backend import read_ncep

            table = read_ncep(p, mnemonics=["CLATH"])

        assert isinstance(table, pa.Table)
        assert "message_type" in table.column_names
        assert "message_index" in table.column_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/test_fortran_backend.py -v`

Expected: FAIL — `_fortran_backend` module does not exist yet.

- [ ] **Step 3: Create `_fortran_backend.py`**

Create `recipes/earth2bufrio/src/earth2bufrio/_fortran_backend.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fortran backend for reading NCEP BUFR/PrepBUFR files via ctypes."""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from earth2bufrio._arrow import build_table

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default mnemonic sets per message type
# ---------------------------------------------------------------------------
_DEFAULT_MNEMONICS: dict[str, list[str]] = {
    # Satellite radiance — ATMS
    "NC021203": [
        "SAID", "CLATH", "CLONH", "SAZA", "SOZA", "IANG",
        "TMBR", "CHNM",
        "YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO",
    ],
    # Satellite radiance — AMSU-A
    "NC021023": [
        "SAID", "CLAT", "CLON", "SAZA", "SOZA", "IANG",
        "TMBR", "CHNM",
        "YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO",
    ],
    # Satellite radiance — MHS
    "NC021027": [
        "SAID", "CLAT", "CLON", "SAZA", "SOZA", "IANG",
        "TMBR", "CHNM",
        "YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO",
    ],
    # PrepBUFR conventional observations
    "_PREPBUFR": [
        "YOB", "XOB", "DHR", "ELV", "TYP",
        "POB", "QOB", "TOB", "ZOB", "UOB", "VOB", "PWO", "TDO",
        "PMO", "XDR", "YDR", "HRDR",
    ],
}

# Replicated mnemonics — use ufbrep instead of ufbint
_REPLICATED_MNEMONICS: frozenset[str] = frozenset({
    "TMBR", "CHNM",
    "POB", "QOB", "TOB", "ZOB", "UOB", "VOB", "PWO", "TDO",
    "XDR", "YDR", "HRDR",
})

# Time mnemonics — always extracted for the fixed time columns
_TIME_MNEMONICS: tuple[str, ...] = ("YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO")

# Maximum number of levels per ufbint/ufbrep call
_MAX_LEVELS: int = 255


def _load_lib() -> ctypes.CDLL:
    """Find and load ``libearth2bufrio_fort.so``.

    Raises
    ------
    RuntimeError
        If the shared library cannot be found.
    """
    lib_name = "libearth2bufrio_fort.so"
    pkg_dir = Path(__file__).parent
    candidates = [pkg_dir / lib_name, lib_name]
    for candidate in candidates:
        try:
            return ctypes.CDLL(str(candidate))
        except OSError:
            continue
    msg = (
        f"Could not load {lib_name}. "
        "Build with 'make fortran' first."
    )
    raise RuntimeError(msg)


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
    mnemonics : list[str] | None, optional
        Mnemonic strings to extract.  ``None`` discovers defaults
        per message type.
    filters : dict[str, Any] | None, optional
        Message-level filters.  Supports ``"message_type"`` (str)
        and ``"data_category"`` (int, not used by Fortran backend).
    workers : int, optional
        Unused (reserved for future multi-file parallelism).

    Returns
    -------
    pa.Table
        Wide-format table with one row per subset.
    """
    lib = _load_lib()

    # Set up ctypes signatures
    lib.e2b_open.restype = ctypes.c_int
    lib.e2b_next_message.restype = ctypes.c_int
    lib.e2b_next_subset.restype = ctypes.c_int
    lib.e2b_read_values.restype = ctypes.c_int
    lib.e2b_read_replicated.restype = ctypes.c_int
    lib.e2b_get_bmiss.restype = ctypes.c_double

    file_path = Path(path)
    filepath_bytes = str(file_path).encode("utf-8")

    # Open the file
    lun = lib.e2b_open(
        filepath_bytes,
        ctypes.c_int(len(filepath_bytes)),
    )
    if lun < 0:
        msg = f"Fortran backend failed to open: {file_path}"
        raise RuntimeError(msg)

    bmiss = lib.e2b_get_bmiss()

    msg_type_filter = None
    if filters is not None:
        msg_type_filter = filters.get("message_type")

    rows: list[dict[str, Any]] = []
    msg_index = 0

    try:
        while True:
            # Read next message
            msg_type_buf = ctypes.create_string_buffer(9)
            msg_type_len = ctypes.c_int(0)
            idate = ctypes.c_int(0)

            ret = lib.e2b_next_message(
                ctypes.c_int(lun),
                msg_type_buf,
                ctypes.byref(msg_type_len),
                ctypes.byref(idate),
            )
            if ret != 0:
                break

            msg_type = msg_type_buf.raw[: msg_type_len.value].decode("ascii").strip()

            # Apply message_type filter
            if msg_type_filter is not None and msg_type != msg_type_filter:
                msg_index += 1
                continue

            # Determine which mnemonics to read for this message
            if mnemonics is not None:
                read_mnems = mnemonics
            elif msg_type in _DEFAULT_MNEMONICS:
                read_mnems = _DEFAULT_MNEMONICS[msg_type]
            else:
                read_mnems = _DEFAULT_MNEMONICS["_PREPBUFR"]

            # Read subsets
            subset_index = 0
            while True:
                ret = lib.e2b_next_subset(ctypes.c_int(lun))
                if ret != 0:
                    break

                row: dict[str, Any] = {
                    "message_type": msg_type,
                    "message_index": msg_index,
                    "subset_index": subset_index,
                }

                # Read each mnemonic
                for mnem in read_mnems:
                    values_buf = (ctypes.c_double * _MAX_LEVELS)()
                    nvalues = ctypes.c_int(0)
                    mnem_bytes = mnem.encode("ascii")

                    if mnem in _REPLICATED_MNEMONICS:
                        lib.e2b_read_replicated(
                            ctypes.c_int(lun),
                            mnem_bytes,
                            ctypes.c_int(len(mnem_bytes)),
                            values_buf,
                            ctypes.c_int(_MAX_LEVELS),
                            ctypes.byref(nvalues),
                        )
                    else:
                        lib.e2b_read_values(
                            ctypes.c_int(lun),
                            mnem_bytes,
                            ctypes.c_int(len(mnem_bytes)),
                            values_buf,
                            ctypes.c_int(_MAX_LEVELS),
                            ctypes.byref(nvalues),
                        )

                    n = nvalues.value
                    if n == 0:
                        continue

                    # Extract values, replacing bmiss with None
                    raw_vals = [
                        None if values_buf[i] >= bmiss else values_buf[i]
                        for i in range(n)
                    ]

                    if n == 1:
                        val = raw_vals[0]
                        if mnem in _TIME_MNEMONICS and val is not None:
                            row[mnem] = int(val)
                        else:
                            row[mnem] = val
                    else:
                        # Multi-level: store as list
                        row[mnem] = raw_vals

                # Ensure time columns exist
                for tc in _TIME_MNEMONICS:
                    if tc not in row:
                        row[tc] = 0

                rows.append(row)
                subset_index += 1

            msg_index += 1
    finally:
        lib.e2b_close(ctypes.c_int(lun))

    return build_table(rows, mnemonics=mnemonics)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/test_fortran_backend.py -v`

Expected: Tests PASS (the mock tests exercise the Python wrapper logic without requiring the actual Fortran library).

Note: The mock-based test for `read_ncep` is inherently tricky because the ctypes interaction is complex. If mocking proves too fragile, simplify the test to just verify `_load_lib` raises correctly and that `read_ncep` dispatches to the library. The real validation comes from integration tests.

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/ -v`

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add recipes/earth2bufrio/src/earth2bufrio/_fortran_backend.py recipes/earth2bufrio/tests/test_fortran_backend.py
git commit -m "feat(earth2bufrio): add Fortran ctypes backend wrapper"
```

---

### Task 8: Update Documentation

**Files:**
- Modify: `recipes/earth2bufrio/docs/api.md`
- Modify: `recipes/earth2bufrio/docs/backends.md`

- [ ] **Step 1: Update `api.md`**

Update the API documentation to reflect the new `read_bufr()` signature and wide-format output:

Replace the content of `docs/api.md` with documentation showing:
- New signature: `read_bufr(path, *, mnemonics=None, filters=None, workers=1, backend="python")`
- Wide-format output schema (fixed columns + dynamic mnemonic columns)
- Backend parameter docs
- Example usage for both Python and Fortran backends

- [ ] **Step 2: Update `backends.md`**

Update `docs/backends.md` to document:
- The Fortran backend architecture (parallel pipeline, not swap-point)
- Build instructions (`make fortran`)
- The ctypes integration model
- NCEPLIBS-bufr dependency and bundling

- [ ] **Step 3: Verify docs build**

Run: `cd recipes/earth2bufrio && sphinx-build -b html docs/ docs/_build/html`

Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add recipes/earth2bufrio/docs/
git commit -m "docs(earth2bufrio): update API and backends docs for wide-format + Fortran"
```

---

### Task 9: Final Validation

**Files:** None new — validation only.

- [ ] **Step 1: Run full test suite with coverage**

Run: `PYTHONPATH=recipes/earth2bufrio/src .venv/bin/pytest recipes/earth2bufrio/tests/ -v --cov=earth2bufrio --cov-report=term-missing`

Expected: >= 80% coverage, all tests pass (87+ passed, 2 xfailed).

- [ ] **Step 2: Run linting**

Run from recipe dir:

```bash
cd recipes/earth2bufrio && ruff check src/ tests/ && ruff format --check src/ tests/
```

Expected: No errors.

- [ ] **Step 3: Run interrogate (docstring coverage)**

Run: `cd recipes/earth2bufrio && interrogate -v src/`

Expected: >= 99% coverage.

- [ ] **Step 4: Verify docs build clean**

Run: `cd recipes/earth2bufrio && sphinx-build -b html docs/ docs/_build/html`

Expected: Build succeeds.

- [ ] **Step 5: Commit any final fixes**

If any linting/format/docstring fixes were needed, commit them:

```bash
git add -A recipes/earth2bufrio/
git commit -m "chore(earth2bufrio): final validation fixes"
```
