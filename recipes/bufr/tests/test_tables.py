# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for WMO table management."""

import pytest
from earth2bufrio._tables import TableSet, load_table_b, load_table_d
from earth2bufrio._types import TableBEntry, TableDEntry


@pytest.mark.unit
class TestLoadTableB:
    def test_returns_dict(self):
        result = load_table_b()
        assert isinstance(result, dict)
        assert len(result) > 100  # should have many entries

    def test_temperature_entry(self):
        table = load_table_b()
        assert 12001 in table  # 012001 = Temperature
        entry = table[12001]
        assert isinstance(entry, TableBEntry)
        assert "TEMPERATURE" in entry.name.upper()
        assert entry.units == "K"
        assert entry.scale == 1
        assert entry.bit_width == 12

    def test_latitude_entry(self):
        table = load_table_b()
        assert 5001 in table  # 005001 = Latitude high accuracy
        entry = table[5001]
        assert entry.scale == 5
        assert entry.reference_value == -9000000

    def test_keys_are_integers(self):
        table = load_table_b()
        for key in table:
            assert isinstance(key, int)


@pytest.mark.unit
class TestLoadTableD:
    def test_returns_dict(self):
        result = load_table_d()
        assert isinstance(result, dict)
        assert len(result) > 100

    def test_date_sequence(self):
        table = load_table_d()
        assert 301011 in table  # Year, month, day
        entry = table[301011]
        assert isinstance(entry, TableDEntry)
        assert entry.descriptors == (4001, 4002, 4003)

    def test_keys_are_integers(self):
        table = load_table_d()
        for key in table:
            assert isinstance(key, int)

    def test_members_are_integers(self):
        table = load_table_d()
        for entry in table.values():
            for d in entry.descriptors:
                assert isinstance(d, int)


@pytest.mark.unit
class TestTableSet:
    def test_lookup_b_returns_entry(self):
        ts = TableSet()
        entry = ts.lookup_b(12001)
        assert isinstance(entry, TableBEntry)
        assert "TEMPERATURE" in entry.name.upper()

    def test_lookup_b_missing_raises(self):
        ts = TableSet()
        with pytest.raises(KeyError):
            ts.lookup_b(999999)

    def test_lookup_d_returns_entry(self):
        ts = TableSet()
        entry = ts.lookup_d(301011)
        assert isinstance(entry, TableDEntry)
        assert entry.descriptors == (4001, 4002, 4003)

    def test_lookup_d_missing_raises(self):
        ts = TableSet()
        with pytest.raises(KeyError):
            ts.lookup_d(999999)

    def test_add_b_overrides(self):
        ts = TableSet()
        custom = TableBEntry(
            name="CUSTOM", units="X", scale=0, reference_value=0, bit_width=8
        )
        ts.add_b(12001, custom)
        assert ts.lookup_b(12001).name == "CUSTOM"

    def test_add_d_overrides(self):
        ts = TableSet()
        custom = TableDEntry(descriptors=(1, 2, 3))
        ts.add_d(301011, custom)
        assert ts.lookup_d(301011).descriptors == (1, 2, 3)

    def test_add_b_new_entry(self):
        ts = TableSet()
        custom = TableBEntry(
            name="NEW", units="Y", scale=2, reference_value=100, bit_width=16
        )
        ts.add_b(999999, custom)
        assert ts.lookup_b(999999).name == "NEW"

    def test_scope_context_manager(self):
        ts = TableSet()
        original = ts.lookup_b(12001)
        custom = TableBEntry(
            name="SCOPED", units="X", scale=0, reference_value=0, bit_width=8
        )
        with ts.scope():
            ts.add_b(12001, custom)
            assert ts.lookup_b(12001).name == "SCOPED"
        # After scope exits, original should be restored
        assert ts.lookup_b(12001).name == original.name
