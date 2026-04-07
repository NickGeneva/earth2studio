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
