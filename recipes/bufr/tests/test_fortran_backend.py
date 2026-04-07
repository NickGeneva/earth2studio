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
            from earth2bufr._fortran_backend import _load_lib

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
        with patch("earth2bufr._fortran_backend._load_lib", return_value=mock_lib):
            from earth2bufr._fortran_backend import read_ncep

            table = read_ncep(p, mnemonics=["CLATH"])

        assert isinstance(table, pa.Table)
        assert "message_type" in table.column_names
        assert "message_index" in table.column_names
