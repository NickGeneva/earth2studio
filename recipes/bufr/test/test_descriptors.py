# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for descriptor expansion logic."""

import pytest
from earth2bufr._tables import TableSet
from earth2bufr._types import (
    BufrDecodeError,
    ExpandedDescriptor,
    TableBEntry,
    TableDEntry,
)


@pytest.fixture
def tables():
    """Create a TableSet with minimal test entries."""
    ts = TableSet()
    ts.add_b(
        12001,
        TableBEntry(
            name="TEMPERATURE", units="K", scale=1, reference_value=0, bit_width=12
        ),
    )
    ts.add_b(
        4001,
        TableBEntry(name="YEAR", units="a", scale=0, reference_value=0, bit_width=12),
    )
    ts.add_b(
        4002,
        TableBEntry(name="MONTH", units="mon", scale=0, reference_value=0, bit_width=4),
    )
    ts.add_b(
        4003,
        TableBEntry(name="DAY", units="d", scale=0, reference_value=0, bit_width=6),
    )
    ts.add_b(
        31001,
        TableBEntry(
            name="DELAYED DESCRIPTOR REPLICATION FACTOR",
            units="NUMERIC",
            scale=0,
            reference_value=0,
            bit_width=8,
        ),
    )
    ts.add_b(
        5001,
        TableBEntry(
            name="LATITUDE",
            units="deg",
            scale=5,
            reference_value=-9000000,
            bit_width=25,
        ),
    )
    ts.add_d(301011, TableDEntry(descriptors=(4001, 4002, 4003)))
    # Nested: 399001 -> [301011, 12001]
    ts.add_d(399001, TableDEntry(descriptors=(301011, 12001)))
    return ts


@pytest.mark.unit
class TestExpandSingleTableB:
    """F=0 (Table B element) expansion."""

    def test_single_element_returns_one_expanded(self, tables):
        from earth2bufr._descriptors import expand_descriptors

        result = expand_descriptors((12001,), tables)
        assert len(result) == 1
        assert isinstance(result[0], ExpandedDescriptor)
        assert result[0].fxy == 12001
        assert result[0].entry.name == "TEMPERATURE"
        assert result[0].entry.units == "K"
        assert result[0].entry.bit_width == 12

    def test_multiple_elements(self, tables):
        from earth2bufr._descriptors import expand_descriptors

        result = expand_descriptors((4001, 4002, 4003), tables)
        assert len(result) == 3
        assert result[0].fxy == 4001
        assert result[1].fxy == 4002
        assert result[2].fxy == 4003


@pytest.mark.unit
class TestExpandTableD:
    """F=3 (Table D sequence) expansion."""

    def test_table_d_expands_to_members(self, tables):
        from earth2bufr._descriptors import expand_descriptors

        result = expand_descriptors((301011,), tables)
        assert len(result) == 3
        assert result[0].fxy == 4001
        assert result[1].fxy == 4002
        assert result[2].fxy == 4003

    def test_nested_table_d_fully_flattens(self, tables):
        from earth2bufr._descriptors import expand_descriptors

        # 399001 -> [301011, 12001] -> [4001, 4002, 4003, 12001]
        result = expand_descriptors((399001,), tables)
        assert len(result) == 4
        assert result[0].fxy == 4001
        assert result[1].fxy == 4002
        assert result[2].fxy == 4003
        assert result[3].fxy == 12001


@pytest.mark.unit
class TestReplicationRegular:
    """F=1 regular replication."""

    def test_replicate_one_descriptor_three_times(self, tables):
        from earth2bufr._descriptors import expand_descriptors

        # 101003 = replicate next 1 descriptor 3 times, followed by 12001
        result = expand_descriptors((101003, 12001), tables)
        assert len(result) == 3
        for item in result:
            assert item.fxy == 12001
            assert item.entry.name == "TEMPERATURE"

    def test_replicate_two_descriptors_twice(self, tables):
        from earth2bufr._descriptors import expand_descriptors

        # 102002 = replicate next 2 descriptors 2 times
        result = expand_descriptors((102002, 4001, 4002), tables)
        assert len(result) == 4
        assert result[0].fxy == 4001
        assert result[1].fxy == 4002
        assert result[2].fxy == 4001
        assert result[3].fxy == 4002


@pytest.mark.unit
class TestReplicationDelayed:
    """F=1 delayed replication (Y=0)."""

    def test_delayed_replication_includes_factor_descriptor(self, tables):
        from earth2bufr._descriptors import expand_descriptors
        from earth2bufr._types import DelayedReplicationMarker

        # 101000 = delayed replication of 1 descriptor; next is 31001 (factor), then 12001
        result = expand_descriptors((101000, 31001, 12001), tables)
        # Should produce a single DelayedReplicationMarker
        assert len(result) == 1
        marker = result[0]
        assert isinstance(marker, DelayedReplicationMarker)
        assert marker.factor_desc.fxy == 31001
        assert marker.factor_desc.entry.name == "DELAYED DESCRIPTOR REPLICATION FACTOR"
        assert len(marker.group) == 1
        assert marker.group[0].fxy == 12001


@pytest.mark.unit
class TestOperator201ChangeWidth:
    """F=2 operator 201YYY — change data width."""

    def test_201_increases_bit_width(self, tables):
        from earth2bufr._descriptors import expand_descriptors

        # 201010: width delta = 10 - 128 = -118 ... let's use a sensible one
        # 201131: width delta = 131 - 128 = +3
        result = expand_descriptors((201131, 12001, 201000), tables)
        # 12001 normally has bit_width=12, with +3 it should be 15
        assert len(result) == 1
        assert result[0].fxy == 12001
        assert result[0].entry.bit_width == 15

    def test_201_reset_restores_original(self, tables):
        from earth2bufr._descriptors import expand_descriptors

        # 201131 applies +3, then 201000 resets
        result = expand_descriptors((201131, 12001, 201000, 5001), tables)
        assert len(result) == 2
        assert result[0].fxy == 12001
        assert result[0].entry.bit_width == 15  # 12 + 3
        assert result[1].fxy == 5001
        assert result[1].entry.bit_width == 25  # original, no delta


@pytest.mark.unit
class TestOperator202ChangeScale:
    """F=2 operator 202YYY — change scale."""

    def test_202_increases_scale(self, tables):
        from earth2bufr._descriptors import expand_descriptors

        # 202130: scale delta = 130 - 128 = +2
        result = expand_descriptors((202130, 12001, 202000), tables)
        # 12001 normally has scale=1, with +2 it should be 3
        assert len(result) == 1
        assert result[0].fxy == 12001
        assert result[0].entry.scale == 3

    def test_202_reset_restores_original(self, tables):
        from earth2bufr._descriptors import expand_descriptors

        result = expand_descriptors((202130, 12001, 202000, 5001), tables)
        assert len(result) == 2
        assert result[0].fxy == 12001
        assert result[0].entry.scale == 3  # 1 + 2
        assert result[1].fxy == 5001
        assert result[1].entry.scale == 5  # original, no delta


@pytest.mark.unit
class TestMaxDepthGuard:
    """Recursive expansion deeper than 50 levels should raise BufrDecodeError."""

    def test_deep_recursion_raises(self, tables):
        from earth2bufr._descriptors import expand_descriptors

        # Build a chain of 51 nested Table D entries: each references the next
        for i in range(51):
            fxy = 360001 + i
            next_fxy = 360002 + i
            if i == 50:
                # Terminal: points to a Table B element
                tables.add_d(fxy, TableDEntry(descriptors=(12001,)))
            else:
                tables.add_d(fxy, TableDEntry(descriptors=(next_fxy,)))

        with pytest.raises(BufrDecodeError, match="[Dd]epth|[Rr]ecursi"):
            expand_descriptors((360001,), tables)


@pytest.mark.unit
class TestUnknownDescriptor:
    """Unknown F=0 descriptor should raise an error."""

    def test_unknown_table_b_raises(self, tables):
        from earth2bufr._descriptors import expand_descriptors

        with pytest.raises((BufrDecodeError, KeyError)):
            expand_descriptors((99999,), tables)
