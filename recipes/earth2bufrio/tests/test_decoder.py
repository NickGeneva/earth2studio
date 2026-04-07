# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bit-level BUFR decoder."""

from __future__ import annotations

import pytest
from earth2bufrio._decoder import (
    _decode_value,
    _is_missing,
    _read_bits,
    decode,
)
from earth2bufrio._types import (
    DecodedSubset,
    ExpandedDescriptor,
    TableBEntry,
)


# ---------------------------------------------------------------------------
# Helper: pack bits into bytes
# ---------------------------------------------------------------------------
def pack_bits(*bit_groups: tuple[int, int]) -> bytes:
    """Pack (value, num_bits) pairs into a bytes object.

    Parameters
    ----------
    *bit_groups : tuple[int, int]
        Each is (value, num_bits) to pack sequentially.

    Returns
    -------
    bytes
        The packed byte string.
    """
    bits: list[int] = []
    for value, num_bits in bit_groups:
        for i in range(num_bits - 1, -1, -1):
            bits.append((value >> i) & 1)
    # Pad to byte boundary
    while len(bits) % 8 != 0:
        bits.append(0)
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        result.append(byte)
    return bytes(result)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
TEMP_ENTRY = TableBEntry(
    name="TEMPERATURE", units="K", scale=1, reference_value=0, bit_width=12
)
TEMP_DESC = ExpandedDescriptor(fxy=12001, entry=TEMP_ENTRY)

LAT_ENTRY = TableBEntry(
    name="LATITUDE", units="deg", scale=2, reference_value=-9000, bit_width=15
)
LAT_DESC = ExpandedDescriptor(fxy=5002, entry=LAT_ENTRY)

STR_ENTRY = TableBEntry(
    name="STATION ID", units="CCITT IA5", scale=0, reference_value=0, bit_width=32
)
STR_DESC = ExpandedDescriptor(fxy=1015, entry=STR_ENTRY)


# ---------------------------------------------------------------------------
# _read_bits
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_read_bits() -> None:
    """Extract correct unsigned integer from a byte buffer."""
    # 0xAB = 0b10101011
    data = b"\xab"
    # First 4 bits: 0b1010 = 10
    assert _read_bits(data, 0, 4) == 10
    # Next 4 bits: 0b1011 = 11
    assert _read_bits(data, 4, 4) == 11
    # Full byte
    assert _read_bits(data, 0, 8) == 0xAB


# ---------------------------------------------------------------------------
# _is_missing
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_is_missing() -> None:
    """All bits set to 1 for a given width is detected as missing."""
    # 12-bit all-ones = 0xFFF = 4095
    assert _is_missing(0xFFF, 12) is True
    # Not all ones
    assert _is_missing(0xFFE, 12) is False
    # 1-bit missing
    assert _is_missing(1, 1) is True
    assert _is_missing(0, 1) is False


# ---------------------------------------------------------------------------
# Numeric value decoding
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_decode_value_numeric() -> None:
    """Numeric decode: (raw + reference_value) / 10^scale."""
    # TEMP_ENTRY: scale=1, ref=0  =>  2731 / 10 = 273.1
    result = _decode_value(2731, TEMP_ENTRY)
    assert result == pytest.approx(273.1)


@pytest.mark.unit
def test_decode_value_missing() -> None:
    """All-ones raw value should decode to None."""
    result = _decode_value(2**12 - 1, TEMP_ENTRY)
    assert result is None


# ---------------------------------------------------------------------------
# Uncompressed mode
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_uncompressed_single_subset() -> None:
    """Single subset, two descriptors — uncompressed."""
    # temp raw=2731 (12 bits), lat raw=1000 (15 bits)
    data = pack_bits((2731, 12), (1000, 15))
    expanded = [TEMP_DESC, LAT_DESC]
    result = decode(expanded, data, num_subsets=1, compressed=False)

    assert len(result) == 1
    subset = result[0]
    assert isinstance(subset, DecodedSubset)
    assert len(subset.values) == 2

    # temp = 2731 / 10 = 273.1
    assert subset.values[0][0] is TEMP_DESC
    assert subset.values[0][1] == pytest.approx(273.1)

    # lat = (1000 + (-9000)) / 100 = -80.0
    assert subset.values[1][0] is LAT_DESC
    assert subset.values[1][1] == pytest.approx(-80.0)


@pytest.mark.unit
def test_uncompressed_two_subsets() -> None:
    """Two subsets decoded sequentially — uncompressed."""
    # subset 1: temp=2731, lat=1000
    # subset 2: temp=2932, lat=5000
    data = pack_bits(
        (2731, 12),
        (1000, 15),
        (2932, 12),
        (5000, 15),
    )
    expanded = [TEMP_DESC, LAT_DESC]
    result = decode(expanded, data, num_subsets=2, compressed=False)

    assert len(result) == 2

    # Subset 1
    assert result[0].values[0][1] == pytest.approx(273.1)
    assert result[0].values[1][1] == pytest.approx(-80.0)

    # Subset 2: temp = 2932/10 = 293.2, lat = (5000-9000)/100 = -40.0
    assert result[1].values[0][1] == pytest.approx(293.2)
    assert result[1].values[1][1] == pytest.approx(-40.0)


# ---------------------------------------------------------------------------
# Compressed mode
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_compressed_same_values() -> None:
    """Compressed: NBINC=0 means all subsets share the same value."""
    # For TEMP_DESC (12-bit): R0=2731, then NBINC=0 (6 bits)
    # For LAT_DESC (15-bit): R0=1000, then NBINC=0 (6 bits)
    data = pack_bits(
        (2731, 12),
        (0, 6),  # NBINC=0 for temp
        (1000, 15),
        (0, 6),  # NBINC=0 for lat
    )
    expanded = [TEMP_DESC, LAT_DESC]
    result = decode(expanded, data, num_subsets=2, compressed=True)

    assert len(result) == 2
    for subset in result:
        assert subset.values[0][1] == pytest.approx(273.1)
        assert subset.values[1][1] == pytest.approx(-80.0)


@pytest.mark.unit
def test_compressed_different_values() -> None:
    """Compressed: NBINC>0 with per-subset increments."""
    # TEMP_DESC (12-bit): R0=2731, NBINC=4
    #   subset 0 increment=0  => 2731 => 273.1
    #   subset 1 increment=10 => 2741 => 274.1
    data = pack_bits(
        (2731, 12),
        (4, 6),  # NBINC=4
        (0, 4),  # subset 0 increment
        (10, 4),  # subset 1 increment
    )
    expanded = [TEMP_DESC]
    result = decode(expanded, data, num_subsets=2, compressed=True)

    assert len(result) == 2
    assert result[0].values[0][1] == pytest.approx(273.1)
    assert result[1].values[0][1] == pytest.approx(274.1)


# ---------------------------------------------------------------------------
# String decode
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_string_decode() -> None:
    """Uncompressed string field: read bytes, decode ASCII, strip trailing."""
    # "ABCD" = 0x41 0x42 0x43 0x44, 32 bits
    data = pack_bits(
        (0x41, 8),
        (0x42, 8),
        (0x43, 8),
        (0x44, 8),
    )
    expanded = [STR_DESC]
    result = decode(expanded, data, num_subsets=1, compressed=False)

    assert len(result) == 1
    assert result[0].values[0][0] is STR_DESC
    assert result[0].values[0][1] == "ABCD"


@pytest.mark.unit
def test_missing_string() -> None:
    """All-ones for a string field should decode to None."""
    # 32 bits all 1s = 0xFFFFFFFF
    data = pack_bits(
        (0xFFFFFFFF, 32),
    )
    expanded = [STR_DESC]
    result = decode(expanded, data, num_subsets=1, compressed=False)

    assert len(result) == 1
    assert result[0].values[0][1] is None
