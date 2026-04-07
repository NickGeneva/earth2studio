# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core dataclasses and exceptions for BUFR message decoding."""

from __future__ import annotations

from dataclasses import dataclass


class BufrDecodeError(Exception):
    """Raised when a BUFR message cannot be decoded.

    Parameters
    ----------
    message : str
        Human-readable error description.
    offset : int | None, optional
        Byte offset within the BUFR stream where the error occurred.
    """

    def __init__(self, message: str, *, offset: int | None = None) -> None:
        super().__init__(message)
        self.offset = offset


# ---------------------------------------------------------------------------
# Raw message container
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BufrMessage:
    """A single raw BUFR message extracted from a byte stream.

    Parameters
    ----------
    data : bytes
        The complete raw bytes of the BUFR message (BUFR … 7777).
    offset : int
        Byte offset of the message start within the original stream.
    index : int
        Zero-based message index within the stream.
    """

    data: bytes
    offset: int
    index: int


# ---------------------------------------------------------------------------
# Parsed section containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class IndicatorSection:
    """BUFR Section 0 — Indicator Section.

    Parameters
    ----------
    length : int
        Total length of the BUFR message in bytes.
    edition : int
        BUFR edition number (typically 3 or 4).
    """

    length: int
    edition: int


@dataclass(frozen=True, slots=True)
class IdentificationSection:
    """BUFR Section 1 — Identification Section.

    Parameters
    ----------
    originating_center : int
        Originating / generating centre code (Common Code Table C-1).
    data_category : int
        Data category (BUFR Table A).
    year : int
        Year of the observation / message.
    month : int
        Month (1–12).
    day : int
        Day (1–31).
    hour : int
        Hour (0–23).
    minute : int
        Minute (0–59).
    second : int
        Second (0–59).
    num_subsets : int
        Number of data subsets.
    observed : bool
        ``True`` if the data is observed (not forecast).
    compressed : bool
        ``True`` if data compression is used.
    """

    originating_center: int
    data_category: int
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    num_subsets: int
    observed: bool
    compressed: bool


@dataclass(frozen=True, slots=True)
class DataDescriptionSection:
    """BUFR Section 3 — Data Description Section.

    Parameters
    ----------
    descriptors : tuple[int, ...]
        Unexpanded BUFR descriptor sequence (FXY integers).
    """

    descriptors: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ParsedMessage:
    """A fully parsed (but not yet decoded) BUFR message.

    Parameters
    ----------
    indicator : IndicatorSection
        Section 0.
    identification : IdentificationSection
        Section 1.
    data_description : DataDescriptionSection
        Section 3.
    data_bytes : bytes
        Raw bytes of Section 4 (data section).
    """

    indicator: IndicatorSection
    identification: IdentificationSection
    data_description: DataDescriptionSection
    data_bytes: bytes


# ---------------------------------------------------------------------------
# Table entries
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TableBEntry:
    """A single BUFR Table B element descriptor entry.

    Parameters
    ----------
    name : str
        Element name / mnemonic.
    units : str
        Unit string (e.g. ``"K"``, ``"Pa"``).
    scale : int
        Decimal scale factor.
    reference_value : int
        Reference (minimum) value.
    bit_width : int
        Data width in bits.
    """

    name: str
    units: str
    scale: int
    reference_value: int
    bit_width: int


@dataclass(frozen=True, slots=True)
class TableDEntry:
    """A single BUFR Table D sequence descriptor entry.

    Parameters
    ----------
    descriptors : tuple[int, ...]
        The sequence of FXY descriptors this entry expands to.
    """

    descriptors: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExpandedDescriptor:
    """A fully expanded element descriptor paired with its Table B entry.

    Parameters
    ----------
    fxy : int
        The FXY descriptor integer.
    entry : TableBEntry
        Corresponding Table B metadata.
    """

    fxy: int
    entry: TableBEntry


@dataclass(frozen=True, slots=True)
class DecodedSubset:
    """One decoded BUFR data subset.

    Parameters
    ----------
    values : tuple[tuple[ExpandedDescriptor, float | str | None], ...]
        Sequence of ``(descriptor, value)`` pairs. Values are ``float``
        for numeric elements, ``str`` for character data, or ``None``
        for missing.
    """

    values: tuple[tuple[ExpandedDescriptor, float | str | None], ...]
