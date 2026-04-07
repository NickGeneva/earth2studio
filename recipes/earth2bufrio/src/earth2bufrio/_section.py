# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Section-level parsing for BUFR Edition 3 and Edition 4 messages."""

from __future__ import annotations

from dataclasses import replace

from earth2bufrio._types import (
    BufrDecodeError,
    BufrMessage,
    DataDescriptionSection,
    IdentificationSection,
    IndicatorSection,
    ParsedMessage,
)


def parse_message(msg: BufrMessage) -> ParsedMessage:
    """Parse a raw BUFR message into its component sections.

    Decodes the Indicator (Section 0), Identification (Section 1),
    optional Section 2, Data Description (Section 3), and Data
    (Section 4) from a :class:`BufrMessage`.

    Parameters
    ----------
    msg : BufrMessage
        A raw BUFR message extracted from a byte stream.

    Returns
    -------
    ParsedMessage
        The fully parsed (but not yet decoded) message.

    Raises
    ------
    BufrDecodeError
        If the message cannot be parsed.
    """
    data = msg.data

    # Section 0 — Indicator (8 bytes)
    indicator = _parse_indicator(data)
    offset = 8

    # Section 1 — Identification (edition-dependent)
    if indicator.edition == 4:
        identification, offset = _parse_identification_ed4(data, offset)
    elif indicator.edition == 3:
        identification, offset = _parse_identification_ed3(data, offset)
    else:
        raise BufrDecodeError(
            f"Unsupported BUFR edition {indicator.edition}",
            offset=msg.offset,
        )

    # Section 2 — Optional (check flag in Section 1 raw bytes)
    has_optional = _has_optional_section(data, indicator.edition)
    if has_optional:
        sec2_len = int.from_bytes(data[offset : offset + 3], "big")
        offset += sec2_len

    # Section 3 — Data Description
    data_desc, offset, num_subsets, observed, compressed = _parse_data_description(
        data, offset
    )

    # Section 4 — Data
    sec4_len = int.from_bytes(data[offset : offset + 3], "big")
    # Data bytes start at offset+4 (skip 3-byte length + 1-byte reserved)
    data_bytes = data[offset + 4 : offset + sec4_len]

    # Reconstruct identification with Section 3 values
    identification = replace(
        identification,
        num_subsets=num_subsets,
        observed=observed,
        compressed=compressed,
    )

    return ParsedMessage(
        indicator=indicator,
        identification=identification,
        data_description=data_desc,
        data_bytes=data_bytes,
    )


def _parse_indicator(data: bytes) -> IndicatorSection:
    """Parse Section 0 (Indicator) from the first 8 bytes.

    Parameters
    ----------
    data : bytes
        The complete raw BUFR message bytes.

    Returns
    -------
    IndicatorSection
        Parsed indicator with total message length and edition number.

    Raises
    ------
    BufrDecodeError
        If the magic bytes are not ``b"BUFR"``.
    """
    if data[:4] != b"BUFR":
        raise BufrDecodeError("Missing BUFR magic bytes", offset=0)
    length = int.from_bytes(data[4:7], "big")
    edition = data[7]
    return IndicatorSection(length=length, edition=edition)


def _parse_identification_ed4(
    data: bytes, offset: int
) -> tuple[IdentificationSection, int]:
    """Parse an Edition 4 Section 1 (Identification).

    Parameters
    ----------
    data : bytes
        The complete raw BUFR message bytes.
    offset : int
        Byte offset where Section 1 begins.

    Returns
    -------
    tuple[IdentificationSection, int]
        The parsed section and the byte offset immediately after it.
        Fields ``num_subsets``, ``observed``, and ``compressed`` are set
        to placeholder values (0 / False) and must be filled from
        Section 3 by the caller.
    """
    sec_len = int.from_bytes(data[offset : offset + 3], "big")
    base = offset + 3  # skip 3-byte length

    center = int.from_bytes(data[base + 1 : base + 3], "big")
    data_cat = data[base + 7]
    year = int.from_bytes(data[base + 12 : base + 14], "big")
    month = data[base + 14]
    day = data[base + 15]
    hour = data[base + 16]
    minute = data[base + 17]
    second = data[base + 18]

    section = IdentificationSection(
        originating_center=center,
        data_category=data_cat,
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
        num_subsets=0,
        observed=False,
        compressed=False,
    )
    return section, offset + sec_len


def _parse_identification_ed3(
    data: bytes, offset: int
) -> tuple[IdentificationSection, int]:
    """Parse an Edition 3 Section 1 (Identification).

    Parameters
    ----------
    data : bytes
        The complete raw BUFR message bytes.
    offset : int
        Byte offset where Section 1 begins.

    Returns
    -------
    tuple[IdentificationSection, int]
        The parsed section and the byte offset immediately after it.
        Year-of-century is converted: >= 70 -> 1900 + y, < 70 -> 2000 + y.
        Second is always 0 for Edition 3.  Fields ``num_subsets``,
        ``observed``, and ``compressed`` are placeholders filled from
        Section 3.
    """
    sec_len = int.from_bytes(data[offset : offset + 3], "big")
    base = offset + 3  # skip 3-byte length

    center = data[base + 2]
    data_cat = data[base + 5]
    year_of_century = data[base + 9]
    year = (
        (1900 + year_of_century) if year_of_century >= 70 else (2000 + year_of_century)
    )
    month = data[base + 10]
    day = data[base + 11]
    hour = data[base + 12]
    minute = data[base + 13]

    section = IdentificationSection(
        originating_center=center,
        data_category=data_cat,
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=0,
        num_subsets=0,
        observed=False,
        compressed=False,
    )
    return section, offset + sec_len


def _parse_data_description(
    data: bytes, offset: int
) -> tuple[DataDescriptionSection, int, int, bool, bool]:
    """Parse Section 3 (Data Description).

    Parameters
    ----------
    data : bytes
        The complete raw BUFR message bytes.
    offset : int
        Byte offset where Section 3 begins.

    Returns
    -------
    tuple[DataDescriptionSection, int, int, bool, bool]
        A 5-tuple of ``(section, next_offset, num_subsets, observed,
        compressed)``.
    """
    sec_len = int.from_bytes(data[offset : offset + 3], "big")
    base = offset + 3  # skip 3-byte length

    num_subsets = int.from_bytes(data[base + 1 : base + 3], "big")
    flags = data[base + 3]
    observed = bool(flags & 0x80)
    compressed = bool(flags & 0x40)

    # Descriptors start at base+4 (byte 7 of section), each is 2 bytes
    num_desc = (sec_len - 7) // 2
    descriptors: list[int] = []
    desc_offset = base + 4
    for _ in range(num_desc):
        packed = int.from_bytes(data[desc_offset : desc_offset + 2], "big")
        fxy = _packed_to_fxy(packed)
        descriptors.append(fxy)
        desc_offset += 2

    section = DataDescriptionSection(descriptors=tuple(descriptors))
    return section, offset + sec_len, num_subsets, observed, compressed


def _packed_to_fxy(packed: int) -> int:
    """Convert a 16-bit packed FXY descriptor to its decimal FXXYYY form.

    Parameters
    ----------
    packed : int
        The 16-bit on-wire descriptor (F in bits 15-14, X in bits 13-8,
        Y in bits 7-0).

    Returns
    -------
    int
        Decimal FXXYYY integer (e.g. ``301011``).
    """
    f = (packed >> 14) & 0x3
    x = (packed >> 8) & 0x3F
    y = packed & 0xFF
    return f * 100000 + x * 1000 + y


def _has_optional_section(data: bytes, edition: int) -> bool:
    """Check the Section 1 flag indicating an optional Section 2 is present.

    Parameters
    ----------
    data : bytes
        The complete raw BUFR message bytes.
    edition : int
        BUFR edition number (3 or 4).

    Returns
    -------
    bool
        ``True`` if the optional section flag is set.
    """
    if edition == 4:
        # Ed4: flag byte is at offset 8 (Section 1 start) + 3 (length) + 6 = 17
        # i.e. byte 9 of the section body (0-indexed: base+6)
        flag_byte = data[8 + 3 + 6]
        return bool(flag_byte & 0x80)
    elif edition == 3:
        # Ed3: flag byte is at offset 8 (Section 1 start) + 3 (length) + 4 = 15
        # i.e. byte 7 of the section body (0-indexed: base+4)
        flag_byte = data[8 + 3 + 4]
        return bool(flag_byte & 0x80)
    return False
