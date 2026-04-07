# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from earth2bufrio._section import parse_message
from earth2bufrio._types import BufrMessage


# ---------------------------------------------------------------------------
# Helpers — hand-craft binary BUFR sections
# ---------------------------------------------------------------------------
def _build_indicator(length: int, edition: int) -> bytes:
    return b"BUFR" + length.to_bytes(3, "big") + bytes([edition])


def _build_section1_ed4(
    *,
    center: int = 7,
    data_cat: int = 0,
    year: int = 2024,
    month: int = 6,
    day: int = 15,
    hour: int = 12,
    minute: int = 30,
    second: int = 0,
    optional_section: bool = False,
) -> bytes:
    flags = 0x80 if optional_section else 0
    body = bytes(
        [
            0,  # master table
            (center >> 8) & 0xFF,
            center & 0xFF,  # center (2 bytes)
            0,
            0,  # sub-center
            0,  # update seq
            flags,  # flags
            data_cat,  # data category
            0,
            0,  # sub-categories
            0,
            0,  # master/local table versions
            (year >> 8) & 0xFF,
            year & 0xFF,
            month,
            day,
            hour,
            minute,
            second,
        ]
    )
    section_len = len(body) + 3
    return section_len.to_bytes(3, "big") + body


def _build_section1_ed3(
    *,
    center: int = 7,
    data_cat: int = 0,
    year: int = 24,
    month: int = 6,
    day: int = 15,
    hour: int = 12,
    minute: int = 30,
    optional_section: bool = False,
) -> bytes:
    flags = 0x80 if optional_section else 0
    body = bytes(
        [
            0,  # master table
            0,  # sub-center
            center,  # center (1 byte)
            0,  # update seq
            flags,  # flags
            data_cat,  # data category
            0,  # sub-category
            0,
            0,  # master/local table versions
            year,  # year of century
            month,
            day,
            hour,
            minute,
        ]
    )
    section_len = len(body) + 3
    return section_len.to_bytes(3, "big") + body


def _build_section2(payload: bytes = b"\x00\x00\x00") -> bytes:
    """Build an optional Section 2 with arbitrary payload."""
    body = bytes([0]) + payload
    section_len = len(body) + 3
    return section_len.to_bytes(3, "big") + body


def _fxy_to_packed(fxy: int) -> int:
    """Convert a decimal FXXYYY descriptor to its 16-bit packed wire format."""
    f = fxy // 100000
    x = (fxy % 100000) // 1000
    y = fxy % 1000
    return (f << 14) | (x << 8) | y


def _build_section3(
    num_subsets: int, observed: bool, compressed: bool, descriptors: list[int]
) -> bytes:
    flags = 0
    if observed:
        flags |= 0x80
    if compressed:
        flags |= 0x40
    desc_bytes = b"".join(_fxy_to_packed(d).to_bytes(2, "big") for d in descriptors)
    body = bytes([0]) + num_subsets.to_bytes(2, "big") + bytes([flags]) + desc_bytes
    section_len = len(body) + 3
    return section_len.to_bytes(3, "big") + body


def _build_section4(data: bytes) -> bytes:
    body = bytes([0]) + data
    section_len = len(body) + 3
    return section_len.to_bytes(3, "big") + body


END_SECTION = b"7777"


def _assemble_message(
    edition: int,
    sec1: bytes,
    sec3: bytes,
    sec4: bytes,
    sec2: bytes | None = None,
) -> bytes:
    """Assemble a complete BUFR message from its sections."""
    inner = sec1
    if sec2 is not None:
        inner += sec2
    inner += sec3 + sec4 + END_SECTION
    total_length = 8 + len(inner)  # 8 bytes for indicator
    indicator = _build_indicator(total_length, edition)
    return indicator + inner


# ---------------------------------------------------------------------------
# Tests — Indicator
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_parse_indicator() -> None:
    """Indicator section (Section 0) is parsed correctly."""
    from earth2bufrio._section import _parse_indicator

    data = _build_indicator(256, 4)
    result = _parse_indicator(data)
    assert result.length == 256
    assert result.edition == 4


# ---------------------------------------------------------------------------
# Tests — Identification (Ed4)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_parse_identification_ed4() -> None:
    """Edition 4 Section 1 fields are parsed correctly."""
    from earth2bufrio._section import _parse_identification_ed4

    sec1 = _build_section1_ed4(
        center=7, data_cat=2, year=2024, month=6, day=15, hour=12, minute=30, second=45
    )
    # Pretend this section starts at offset 0 in a buffer that is just sec1
    ident, next_offset = _parse_identification_ed4(sec1, 0)

    assert ident.originating_center == 7
    assert ident.data_category == 2
    assert ident.year == 2024
    assert ident.month == 6
    assert ident.day == 15
    assert ident.hour == 12
    assert ident.minute == 30
    assert ident.second == 45
    # Placeholders — filled from Section 3 later
    assert ident.num_subsets == 0
    assert ident.observed is False
    assert ident.compressed is False
    assert next_offset == len(sec1)


# ---------------------------------------------------------------------------
# Tests — Identification (Ed3)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_parse_identification_ed3() -> None:
    """Edition 3 Section 1 with year-of-century < 70 -> 2000+year."""
    from earth2bufrio._section import _parse_identification_ed3

    sec1 = _build_section1_ed3(
        center=7, data_cat=3, year=24, month=6, day=15, hour=12, minute=30
    )
    ident, next_offset = _parse_identification_ed3(sec1, 0)

    assert ident.originating_center == 7
    assert ident.data_category == 3
    assert ident.year == 2024
    assert ident.month == 6
    assert ident.day == 15
    assert ident.hour == 12
    assert ident.minute == 30
    assert ident.second == 0  # Ed3 has no second field
    assert next_offset == len(sec1)


@pytest.mark.unit
def test_parse_identification_ed3_year_70() -> None:
    """Edition 3 with year-of-century == 70 -> 1970."""
    from earth2bufrio._section import _parse_identification_ed3

    sec1 = _build_section1_ed3(year=70)
    ident, _ = _parse_identification_ed3(sec1, 0)
    assert ident.year == 1970


# ---------------------------------------------------------------------------
# Tests — Data Description (Section 3)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_parse_data_description() -> None:
    """Section 3 descriptors, num_subsets, and flags are parsed correctly."""
    from earth2bufrio._section import _parse_data_description

    sec3 = _build_section3(
        num_subsets=5, observed=True, compressed=False, descriptors=[301011, 12001]
    )
    desc, next_offset, num_subsets, observed, compressed = _parse_data_description(
        sec3, 0
    )

    assert desc.descriptors == (301011, 12001)
    assert num_subsets == 5
    assert observed is True
    assert compressed is False
    assert next_offset == len(sec3)


# ---------------------------------------------------------------------------
# Tests — Full message parsing
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_parse_message_ed4() -> None:
    """Complete Edition 4 message is parsed end-to-end."""
    sec1 = _build_section1_ed4(
        center=7, data_cat=0, year=2024, month=6, day=15, hour=12, minute=30, second=0
    )
    sec3 = _build_section3(
        num_subsets=1, observed=True, compressed=False, descriptors=[301011]
    )
    data_payload = b"\xab\xcd\xef"
    sec4 = _build_section4(data_payload)
    raw = _assemble_message(4, sec1, sec3, sec4)
    msg = BufrMessage(data=raw, offset=0, index=0)

    parsed = parse_message(msg)

    assert parsed.indicator.edition == 4
    assert parsed.indicator.length == len(raw)
    assert parsed.identification.originating_center == 7
    assert parsed.identification.year == 2024
    assert parsed.identification.month == 6
    assert parsed.identification.day == 15
    assert parsed.identification.hour == 12
    assert parsed.identification.minute == 30
    assert parsed.identification.second == 0
    assert parsed.identification.num_subsets == 1
    assert parsed.identification.observed is True
    assert parsed.identification.compressed is False
    assert parsed.data_description.descriptors == (301011,)
    assert parsed.data_bytes == data_payload


@pytest.mark.unit
def test_parse_message_ed3() -> None:
    """Complete Edition 3 message is parsed end-to-end."""
    sec1 = _build_section1_ed3(
        center=7, data_cat=1, year=24, month=3, day=10, hour=6, minute=0
    )
    sec3 = _build_section3(
        num_subsets=2, observed=False, compressed=True, descriptors=[12001, 11001]
    )
    data_payload = b"\x01\x02\x03\x04"
    sec4 = _build_section4(data_payload)
    raw = _assemble_message(3, sec1, sec3, sec4)
    msg = BufrMessage(data=raw, offset=0, index=0)

    parsed = parse_message(msg)

    assert parsed.indicator.edition == 3
    assert parsed.indicator.length == len(raw)
    assert parsed.identification.originating_center == 7
    assert parsed.identification.data_category == 1
    assert parsed.identification.year == 2024
    assert parsed.identification.month == 3
    assert parsed.identification.day == 10
    assert parsed.identification.hour == 6
    assert parsed.identification.minute == 0
    assert parsed.identification.second == 0
    assert parsed.identification.num_subsets == 2
    assert parsed.identification.observed is False
    assert parsed.identification.compressed is True
    assert parsed.data_description.descriptors == (12001, 11001)
    assert parsed.data_bytes == data_payload


@pytest.mark.unit
def test_parse_message_with_optional_section() -> None:
    """Ed4 message with optional Section 2 is parsed correctly (section skipped)."""
    sec1 = _build_section1_ed4(optional_section=True)
    sec2 = _build_section2(b"\xff\xff\xff\xff")
    sec3 = _build_section3(
        num_subsets=1, observed=True, compressed=False, descriptors=[301011]
    )
    data_payload = b"\x00\x01"
    sec4 = _build_section4(data_payload)
    raw = _assemble_message(4, sec1, sec3, sec4, sec2=sec2)
    msg = BufrMessage(data=raw, offset=0, index=0)

    parsed = parse_message(msg)

    assert parsed.indicator.edition == 4
    assert parsed.identification.num_subsets == 1
    assert parsed.identification.observed is True
    assert parsed.data_description.descriptors == (301011,)
    assert parsed.data_bytes == data_payload
