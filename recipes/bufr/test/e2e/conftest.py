# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for end-to-end BUFR download-and-decode tests."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Backend availability probes (same pattern as test_backend_consistency.py)
# ---------------------------------------------------------------------------
try:
    from earth2bufr._fortran_backend import _load_lib

    _load_lib()
    HAS_FORTRAN = True
except Exception:
    HAS_FORTRAN = False

try:
    from earth2bufr._lib import read_bufr_rust as _read_bufr_rust  # noqa: F401

    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def available_backends(include_fortran: bool = True) -> list[str]:
    """Return backends that can actually be exercised.

    Parameters
    ----------
    include_fortran : bool
        If ``False``, exclude the Fortran backend.  Use this for
        standard WMO BUFR files that NCEPLIBS-bufr cannot decode.
    """
    backends = ["python"]
    if HAS_FORTRAN and include_fortran:
        backends.append("fortran")
    if HAS_RUST:
        backends.append("rust")
    return backends


# ---------------------------------------------------------------------------
# eccodes availability
# ---------------------------------------------------------------------------
try:
    import eccodes  # noqa: F401

    HAS_ECCODES = True
except ImportError:
    HAS_ECCODES = False


# ---------------------------------------------------------------------------
# pybufrkit availability
# ---------------------------------------------------------------------------
try:
    import pybufrkit  # noqa: F401

    HAS_PYBUFRKIT = True
except ImportError:
    HAS_PYBUFRKIT = False


# ---------------------------------------------------------------------------
# s3fs availability
# ---------------------------------------------------------------------------
try:
    import s3fs  # noqa: F401

    HAS_S3FS = True
except ImportError:
    HAS_S3FS = False


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
NUMERIC_RTOL = 1e-6

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


def values_equal(a: object, b: object) -> bool:
    """Compare two scalar values with tolerance for floats."""
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tmp_download_dir() -> Path:
    """Provide a session-scoped temporary directory for downloaded files."""
    with tempfile.TemporaryDirectory(prefix="e2e_bufr_") as d:
        yield Path(d)
