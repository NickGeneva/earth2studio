# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PyArrow table construction from decoded BUFR data."""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest
from earth2bufrio._arrow import build_table
from earth2bufrio._types import (
    DecodedSubset,
    ExpandedDescriptor,
    TableBEntry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TEMP_ENTRY = TableBEntry(
    name="TEMPERATURE", units="K", scale=1, reference_value=0, bit_width=12
)
LAT_ENTRY = TableBEntry(
    name="LATITUDE (HIGH ACCURACY)",
    units="deg",
    scale=5,
    reference_value=-9000000,
    bit_width=25,
)
LON_ENTRY = TableBEntry(
    name="LONGITUDE (HIGH ACCURACY)",
    units="deg",
    scale=5,
    reference_value=-18000000,
    bit_width=26,
)
YEAR_ENTRY = TableBEntry(
    name="YEAR", units="a", scale=0, reference_value=0, bit_width=12
)
MONTH_ENTRY = TableBEntry(
    name="MONTH", units="mon", scale=0, reference_value=0, bit_width=4
)
DAY_ENTRY = TableBEntry(name="DAY", units="d", scale=0, reference_value=0, bit_width=6)
HOUR_ENTRY = TableBEntry(
    name="HOUR", units="h", scale=0, reference_value=0, bit_width=5
)
MINUTE_ENTRY = TableBEntry(
    name="MINUTE", units="min", scale=0, reference_value=0, bit_width=6
)
SECOND_ENTRY = TableBEntry(
    name="SECOND", units="s", scale=0, reference_value=0, bit_width=6
)
QUALITY_ENTRY = TableBEntry(
    name="% CONFIDENCE", units="NUMERIC", scale=0, reference_value=0, bit_width=7
)

ALL_COLUMNS = [
    "message_index",
    "subset_index",
    "data_category",
    "latitude",
    "longitude",
    "time",
    "station_id",
    "pressure",
    "elevation",
    "descriptor_id",
    "descriptor_name",
    "value",
    "units",
    "quality_mark",
]


def _make_msg(
    message_index: int = 0,
    data_category: int = 0,
    year: int = 2024,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    subsets: list[DecodedSubset] | None = None,
) -> dict:
    """Create a decoded message dict for ``build_table`` input."""
    return {
        "message_index": message_index,
        "data_category": data_category,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
        "subsets": subsets or [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildTableBasic:
    """Basic single-row table construction."""

    def test_build_table_basic(self) -> None:
        """One message, one subset, one temperature descriptor -> 1 row."""
        subset = DecodedSubset(
            values=((ExpandedDescriptor(fxy=12001, entry=TEMP_ENTRY), 273.1),)
        )
        msgs = [_make_msg(subsets=[subset])]
        table = build_table(msgs)

        assert isinstance(table, pa.Table)
        assert table.num_columns == 14
        assert table.column_names == ALL_COLUMNS
        assert table.num_rows == 1

        row = table.to_pydict()
        assert row["descriptor_id"] == ["012001"]
        assert row["descriptor_name"] == ["TEMPERATURE"]
        assert row["value"][0] == pytest.approx(273.1)
        assert row["units"] == ["K"]
        assert row["message_index"] == [0]
        assert row["subset_index"] == [0]


class TestWellKnownPromotion:
    """Well-known descriptors are promoted to named columns."""

    def test_well_known_promotion(self) -> None:
        """Lat/lon promoted to named columns, not appearing as rows."""
        subset = DecodedSubset(
            values=(
                (ExpandedDescriptor(fxy=5001, entry=LAT_ENTRY), 45.0),
                (ExpandedDescriptor(fxy=6001, entry=LON_ENTRY), -93.0),
                (ExpandedDescriptor(fxy=12001, entry=TEMP_ENTRY), 273.1),
            )
        )
        msgs = [_make_msg(subsets=[subset])]
        table = build_table(msgs)

        assert table.num_rows == 1
        d = table.to_pydict()
        assert d["latitude"] == [pytest.approx(45.0)]
        assert d["longitude"] == [pytest.approx(-93.0)]
        assert d["descriptor_id"] == ["012001"]
        # No rows for lat/lon descriptors
        assert "005001" not in d["descriptor_id"]
        assert "006001" not in d["descriptor_id"]


class TestTimePromotion:
    """Time-related descriptors are promoted and merged into a timestamp column."""

    def test_time_promotion(self) -> None:
        """Year/month/day/hour/minute/second -> single timestamp."""
        subset = DecodedSubset(
            values=(
                (ExpandedDescriptor(fxy=4001, entry=YEAR_ENTRY), 2024.0),
                (ExpandedDescriptor(fxy=4002, entry=MONTH_ENTRY), 3.0),
                (ExpandedDescriptor(fxy=4003, entry=DAY_ENTRY), 15.0),
                (ExpandedDescriptor(fxy=4004, entry=HOUR_ENTRY), 12.0),
                (ExpandedDescriptor(fxy=4005, entry=MINUTE_ENTRY), 30.0),
                (ExpandedDescriptor(fxy=4006, entry=SECOND_ENTRY), 0.0),
                (ExpandedDescriptor(fxy=12001, entry=TEMP_ENTRY), 273.1),
            )
        )
        msgs = [_make_msg(subsets=[subset])]
        table = build_table(msgs)

        assert table.num_rows == 1
        d = table.to_pydict()
        ts = d["time"][0]
        expected = datetime.datetime(2024, 3, 15, 12, 30, 0)
        assert ts == expected


class TestColumnFiltering:
    """Column filtering via the ``columns`` parameter."""

    def test_column_filtering(self) -> None:
        """Only requested columns are returned."""
        subset = DecodedSubset(
            values=((ExpandedDescriptor(fxy=12001, entry=TEMP_ENTRY), 273.1),)
        )
        msgs = [_make_msg(subsets=[subset])]
        table = build_table(msgs, columns=["value", "descriptor_id"])

        assert table.num_columns == 2
        assert set(table.column_names) == {"value", "descriptor_id"}


class TestEmptyInput:
    """Empty input produces a valid empty table."""

    def test_empty_input(self) -> None:
        """Empty message list -> empty table with full schema."""
        table = build_table([])

        assert isinstance(table, pa.Table)
        assert table.num_rows == 0
        assert table.num_columns == 14
        assert table.column_names == ALL_COLUMNS


class TestQualityMarkExtraction:
    """Quality mark descriptor is promoted to the quality_mark column."""

    def test_quality_mark_extraction(self) -> None:
        """Quality mark (33007) value applied to temperature row."""
        subset = DecodedSubset(
            values=(
                (ExpandedDescriptor(fxy=12001, entry=TEMP_ENTRY), 273.1),
                (ExpandedDescriptor(fxy=33007, entry=QUALITY_ENTRY), 2.0),
            )
        )
        msgs = [_make_msg(subsets=[subset])]
        table = build_table(msgs)

        assert table.num_rows == 1
        d = table.to_pydict()
        assert d["quality_mark"] == [2]


class TestMultipleMessages:
    """Multiple messages produce correct message_index values."""

    def test_multiple_messages(self) -> None:
        """Two messages with one subset each -> message_index [0, 1]."""
        subset0 = DecodedSubset(
            values=((ExpandedDescriptor(fxy=12001, entry=TEMP_ENTRY), 273.1),)
        )
        subset1 = DecodedSubset(
            values=((ExpandedDescriptor(fxy=12001, entry=TEMP_ENTRY), 274.2),)
        )
        msgs = [
            _make_msg(message_index=0, subsets=[subset0]),
            _make_msg(message_index=1, subsets=[subset1]),
        ]
        table = build_table(msgs)

        assert table.num_rows == 2
        d = table.to_pydict()
        assert d["message_index"] == [0, 1]
