# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from earth2bufrio._types import (
    BufrDecodeError,
    BufrMessage,
    DataDescriptionSection,
    DecodedSubset,
    ExpandedDescriptor,
    IdentificationSection,
    IndicatorSection,
    ParsedMessage,
    TableBEntry,
    TableDEntry,
)


# ---------------------------------------------------------------------------
# BufrDecodeError
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_bufr_decode_error_is_exception() -> None:
    assert issubclass(BufrDecodeError, Exception)


@pytest.mark.unit
def test_bufr_decode_error_with_offset() -> None:
    err = BufrDecodeError("bad data", offset=42)
    assert str(err) == "bad data"
    assert err.offset == 42


@pytest.mark.unit
def test_bufr_decode_error_without_offset() -> None:
    err = BufrDecodeError("something broke")
    assert str(err) == "something broke"
    assert err.offset is None


# ---------------------------------------------------------------------------
# BufrMessage
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_bufr_message() -> None:
    msg = BufrMessage(data=b"BUFR", offset=0, index=0)
    assert msg.data == b"BUFR"
    assert msg.offset == 0
    assert msg.index == 0


@pytest.mark.unit
def test_bufr_message_frozen() -> None:
    msg = BufrMessage(data=b"BUFR", offset=0, index=0)
    with pytest.raises(AttributeError):
        msg.offset = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# IndicatorSection
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_indicator_section() -> None:
    sec = IndicatorSection(length=100, edition=4)
    assert sec.edition == 4
    assert sec.length == 100


@pytest.mark.unit
def test_indicator_section_frozen() -> None:
    sec = IndicatorSection(length=100, edition=4)
    with pytest.raises(AttributeError):
        sec.edition = 3  # type: ignore[misc]


# ---------------------------------------------------------------------------
# IdentificationSection
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_identification_section() -> None:
    sec = IdentificationSection(
        originating_center=7,
        data_category=0,
        year=2024,
        month=1,
        day=15,
        hour=12,
        minute=0,
        second=0,
        num_subsets=1,
        observed=True,
        compressed=False,
        master_table_version=0,
        local_table_version=0,
    )
    assert sec.originating_center == 7
    assert sec.year == 2024
    assert sec.observed is True
    assert sec.compressed is False
    assert sec.master_table_version == 0
    assert sec.local_table_version == 0


# ---------------------------------------------------------------------------
# DataDescriptionSection
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_data_description_section() -> None:
    sec = DataDescriptionSection(descriptors=(1001, 1002, 10004))
    assert sec.descriptors == (1001, 1002, 10004)


@pytest.mark.unit
def test_data_description_section_frozen() -> None:
    sec = DataDescriptionSection(descriptors=(1001,))
    with pytest.raises(AttributeError):
        sec.descriptors = (9999,)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ParsedMessage
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_parsed_message() -> None:
    indicator = IndicatorSection(length=200, edition=4)
    identification = IdentificationSection(
        originating_center=7,
        data_category=0,
        year=2024,
        month=6,
        day=1,
        hour=0,
        minute=0,
        second=0,
        num_subsets=1,
        observed=True,
        compressed=False,
        master_table_version=0,
        local_table_version=0,
    )
    data_desc = DataDescriptionSection(descriptors=(301011,))
    parsed = ParsedMessage(
        indicator=indicator,
        identification=identification,
        data_description=data_desc,
        data_bytes=b"\x00\x01\x02",
    )
    assert parsed.indicator.edition == 4
    assert parsed.identification.originating_center == 7
    assert parsed.data_description.descriptors == (301011,)
    assert parsed.data_bytes == b"\x00\x01\x02"


# ---------------------------------------------------------------------------
# TableBEntry
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_table_b_entry() -> None:
    entry = TableBEntry(
        name="TEMPERATURE", units="K", scale=1, reference_value=0, bit_width=12
    )
    assert entry.scale == 1
    assert entry.name == "TEMPERATURE"
    assert entry.units == "K"
    assert entry.reference_value == 0
    assert entry.bit_width == 12


# ---------------------------------------------------------------------------
# TableDEntry
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_table_d_entry() -> None:
    entry = TableDEntry(descriptors=(4001, 4002, 4003))
    assert entry.descriptors == (4001, 4002, 4003)


# ---------------------------------------------------------------------------
# ExpandedDescriptor
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_expanded_descriptor() -> None:
    b_entry = TableBEntry(
        name="PRESSURE", units="Pa", scale=1, reference_value=0, bit_width=14
    )
    desc = ExpandedDescriptor(fxy=10004, entry=b_entry)
    assert desc.fxy == 10004
    assert desc.entry.name == "PRESSURE"


# ---------------------------------------------------------------------------
# DecodedSubset
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_decoded_subset() -> None:
    b_entry = TableBEntry(
        name="TEMP", units="K", scale=1, reference_value=0, bit_width=12
    )
    desc = ExpandedDescriptor(fxy=12101, entry=b_entry)
    subset = DecodedSubset(values=((desc, 293.5), (desc, None)))
    assert len(subset.values) == 2
    assert subset.values[0][1] == 293.5
    assert subset.values[1][1] is None


@pytest.mark.unit
def test_decoded_subset_frozen() -> None:
    subset = DecodedSubset(values=())
    with pytest.raises(AttributeError):
        subset.values = ()  # type: ignore[misc]
