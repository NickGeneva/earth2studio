# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-way backend consistency tests.

Runs every available backend (python, fortran, rust) on the same BUFR
fixtures and asserts that they produce **identical** PyArrow Tables.
The Python backend is always available and serves as the reference.
Fortran and Rust backends are included only when their native libraries
are importable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import earth2bufrio
import pytest

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Backend availability probes
# ---------------------------------------------------------------------------
try:
    from earth2bufrio._fortran_backend import _load_lib

    _load_lib()  # Actually try to open the .so — import alone is not enough
    HAS_FORTRAN = True
except Exception:
    HAS_FORTRAN = False

try:
    from earth2bufrio._lib import read_bufr_rust as _read_bufr_rust  # noqa: F401

    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def _available_backends() -> list[str]:
    """Return the list of backends that can actually be exercised."""
    backends = ["python"]
    if HAS_FORTRAN:
        backends.append("fortran")
    if HAS_RUST:
        backends.append("rust")
    return backends


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

BUFR_FIXTURES = [
    "profiler_european.bufr",
    "207003.bufr",
    "uegabe.bufr",
]

FIXED_COLUMNS = [
    "message_type",
    "message_index",
    "subset_index",
    "YEAR",
    "MNTH",
    "DAYS",
    "HOUR",
    "MINU",
    "SECO",
]

NUMERIC_RTOL = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _values_equal(a: object, b: object) -> bool:
    """Compare two scalar values with tolerance for floats.

    Parameters
    ----------
    a : object
        Value from one backend.
    b : object
        Value from another backend.

    Returns
    -------
    bool
        ``True`` when the values are considered equivalent.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if b == 0.0:
            return abs(a) < NUMERIC_RTOL
        return abs(a - b) / max(abs(b), 1e-15) < NUMERIC_RTOL
    return a == b


def _compare_tables(
    ref_table: pa.Table,
    cmp_table: pa.Table,
    ref_name: str,
    cmp_name: str,
    filename: str,
) -> list[str]:
    """Compare two PyArrow tables cell-by-cell, returning mismatch descriptions.

    Parameters
    ----------
    ref_table : pa.Table
        Reference table (typically from the Python backend).
    cmp_table : pa.Table
        Comparison table from another backend.
    ref_name : str
        Label for the reference backend (e.g. ``"python"``).
    cmp_name : str
        Label for the comparison backend (e.g. ``"rust"``).
    filename : str
        BUFR filename being tested (for error messages).

    Returns
    -------
    list[str]
        Human-readable mismatch descriptions. Empty list means the
        tables are equivalent.
    """
    mismatches: list[str] = []

    # Row count
    if ref_table.num_rows != cmp_table.num_rows:
        mismatches.append(
            f"{filename}: row count differs — {ref_name}={ref_table.num_rows}, {cmp_name}={cmp_table.num_rows}"
        )
        return mismatches  # No point comparing cells if rows differ

    # Column names (sorted for comparison since column order may differ)
    ref_cols = sorted(ref_table.column_names)
    cmp_cols = sorted(cmp_table.column_names)

    missing_in_cmp = set(ref_cols) - set(cmp_cols)
    extra_in_cmp = set(cmp_cols) - set(ref_cols)
    if missing_in_cmp:
        mismatches.append(
            f"{filename}: columns in {ref_name} but not {cmp_name}: {sorted(missing_in_cmp)}"
        )
    if extra_in_cmp:
        mismatches.append(
            f"{filename}: columns in {cmp_name} but not {ref_name}: {sorted(extra_in_cmp)}"
        )

    # Compare values for shared columns
    shared_cols = sorted(set(ref_cols) & set(cmp_cols))
    max_mismatches_per_col = 5
    for col in shared_cols:
        ref_vals = ref_table.column(col).to_pylist()
        cmp_vals = cmp_table.column(col).to_pylist()
        col_mismatches = 0
        for row_idx, (rv, cv) in enumerate(zip(ref_vals, cmp_vals, strict=True)):
            if not _values_equal(rv, cv):
                col_mismatches += 1
                if col_mismatches <= max_mismatches_per_col:
                    mismatches.append(
                        f"{filename}: col={col!r} row={row_idx}: {ref_name}={rv!r}, {cmp_name}={cv!r}"
                    )
        if col_mismatches > max_mismatches_per_col:
            mismatches.append(
                f"{filename}: col={col!r}: {col_mismatches - max_mismatches_per_col} more mismatches suppressed"
            )

    return mismatches


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.crossval
class TestBackendConsistency:
    """Assert all available backends produce identical output."""

    @pytest.mark.parametrize("filename", BUFR_FIXTURES)
    def test_all_backends_match(self, filename: str) -> None:
        """Every available backend produces the same table as Python."""
        bufr_path = DATA_DIR / filename
        if not bufr_path.exists():
            pytest.skip(f"Test fixture {filename} not found")

        backends = _available_backends()
        if len(backends) < 2:
            pytest.skip(
                "Only the Python backend is available — need at least one additional backend for consistency check"
            )

        tables: dict[str, pa.Table] = {}
        for backend in backends:
            tables[backend] = earth2bufrio.read_bufr(bufr_path, backend=backend)

        ref_name = "python"
        ref_table = tables[ref_name]

        all_mismatches: list[str] = []
        for backend, table in tables.items():
            if backend == ref_name:
                continue
            all_mismatches.extend(
                _compare_tables(ref_table, table, ref_name, backend, filename)
            )

        if all_mismatches:
            detail = "\n".join(all_mismatches[:100])
            total = len(all_mismatches)
            pytest.fail(
                f"{total} backend consistency mismatches in {filename}:\n{detail}"
                + (f"\n... and {total - 100} more" if total > 100 else "")
            )

    @pytest.mark.parametrize("filename", BUFR_FIXTURES)
    def test_column_names_identical(self, filename: str) -> None:
        """Every available backend produces the same column names."""
        bufr_path = DATA_DIR / filename
        if not bufr_path.exists():
            pytest.skip(f"Test fixture {filename} not found")

        backends = _available_backends()
        if len(backends) < 2:
            pytest.skip("Need at least two backends")

        ref_cols = sorted(
            earth2bufrio.read_bufr(bufr_path, backend="python").column_names
        )

        for backend in backends:
            if backend == "python":
                continue
            cmp_cols = sorted(
                earth2bufrio.read_bufr(bufr_path, backend=backend).column_names
            )
            assert ref_cols == cmp_cols, (
                f"{filename}: column names differ between python and {backend}\n"
                f"  python: {ref_cols}\n"
                f"  {backend}: {cmp_cols}"
            )

    @pytest.mark.parametrize("filename", BUFR_FIXTURES)
    def test_fixed_columns_identical(self, filename: str) -> None:
        """Fixed metadata columns match exactly across backends."""
        bufr_path = DATA_DIR / filename
        if not bufr_path.exists():
            pytest.skip(f"Test fixture {filename} not found")

        backends = _available_backends()
        if len(backends) < 2:
            pytest.skip("Need at least two backends")

        ref_table = earth2bufrio.read_bufr(bufr_path, backend="python")

        for backend in backends:
            if backend == "python":
                continue
            cmp_table = earth2bufrio.read_bufr(bufr_path, backend=backend)
            for col in FIXED_COLUMNS:
                if col not in ref_table.column_names:
                    continue
                if col not in cmp_table.column_names:
                    pytest.fail(
                        f"{filename}: fixed column {col!r} missing from {backend}"
                    )
                ref_vals = ref_table.column(col).to_pylist()
                cmp_vals = cmp_table.column(col).to_pylist()
                assert (
                    ref_vals == cmp_vals
                ), f"{filename}: fixed column {col!r} differs between python and {backend}"

    @pytest.mark.parametrize("filename", BUFR_FIXTURES)
    def test_mnemonic_columns_identical(self, filename: str) -> None:
        """Dynamic mnemonic columns match across backends with tolerance."""
        bufr_path = DATA_DIR / filename
        if not bufr_path.exists():
            pytest.skip(f"Test fixture {filename} not found")

        backends = _available_backends()
        if len(backends) < 2:
            pytest.skip("Need at least two backends")

        ref_table = earth2bufrio.read_bufr(bufr_path, backend="python")
        mnemonic_cols = [c for c in ref_table.column_names if c not in FIXED_COLUMNS]

        for backend in backends:
            if backend == "python":
                continue
            cmp_table = earth2bufrio.read_bufr(bufr_path, backend=backend)
            mismatches: list[str] = []
            for col in mnemonic_cols:
                if col not in cmp_table.column_names:
                    mismatches.append(f"column {col!r} missing from {backend}")
                    continue
                ref_vals = ref_table.column(col).to_pylist()
                cmp_vals = cmp_table.column(col).to_pylist()
                for row_idx, (rv, cv) in enumerate(
                    zip(ref_vals, cmp_vals, strict=True)
                ):
                    if not _values_equal(rv, cv):
                        mismatches.append(
                            f"col={col!r} row={row_idx}: python={rv!r}, {backend}={cv!r}"
                        )
                        if len(mismatches) >= 50:
                            break
                if len(mismatches) >= 50:
                    break

            if mismatches:
                detail = "\n".join(mismatches[:50])
                total = len(mismatches)
                pytest.fail(
                    f"{total} mnemonic mismatches in {filename} (python vs {backend}):\n{detail}"
                )
