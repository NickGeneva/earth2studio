# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bit-level BUFR data section decoder.

Reads the raw data bits from Section 4 using the expanded descriptor
list to produce :class:`~earth2bufr._types.DecodedSubset` objects.
"""

from __future__ import annotations

from earth2bufr._types import (
    DecodedSubset,
    DelayedReplicationMarker,
    ExpandedDescriptor,
    TableBEntry,
)

# Union type matching _descriptors.ExpandedItem
ExpandedItem = ExpandedDescriptor | DelayedReplicationMarker


def _read_bits(data: bytes, bit_offset: int, num_bits: int) -> int:
    """Extract an unsigned integer from *data* starting at *bit_offset*.

    Parameters
    ----------
    data : bytes
        The byte buffer to read from.
    bit_offset : int
        Zero-based bit position of the first bit to read.
    num_bits : int
        Number of bits to extract.

    Returns
    -------
    int
        The extracted unsigned integer value.
    """
    result = 0
    for i in range(num_bits):
        byte_idx, bit_idx = divmod(bit_offset + i, 8)
        result = (result << 1) | ((data[byte_idx] >> (7 - bit_idx)) & 1)
    return result


def _is_missing(raw: int, num_bits: int) -> bool:
    """Check whether *raw* represents a BUFR missing-value indicator.

    A value is missing when all *num_bits* bits are set to 1.  As a
    special case, 1-bit descriptors (such as 031031 DATA PRESENT
    INDICATOR) are *never* treated as missing because the all-ones
    pattern (value 1) is a valid data value.

    Parameters
    ----------
    raw : int
        The raw unsigned integer extracted from the data section.
    num_bits : int
        The bit width of the descriptor.

    Returns
    -------
    bool
        ``True`` if *raw* is the all-ones missing indicator.
    """
    if num_bits <= 1:
        return False
    return raw == (1 << num_bits) - 1


def _decode_value(raw: int, entry: TableBEntry) -> float | None:
    """Decode a raw integer into a physical value using Table B metadata.

    Parameters
    ----------
    raw : int
        The raw unsigned integer from the data section.
    entry : TableBEntry
        The Table B entry describing scale, reference and width.

    Returns
    -------
    float | None
        The decoded physical value, or ``None`` if the value is missing.
    """
    if _is_missing(raw, entry.bit_width):
        return None
    return (raw + entry.reference_value) / (10**entry.scale)


def _decode_string(data: bytes, bit_offset: int, num_bytes: int) -> str | None:
    """Decode a character (CCITT IA5) field from the data section.

    Parameters
    ----------
    data : bytes
        The byte buffer to read from.
    bit_offset : int
        Zero-based bit position of the first bit of the string.
    num_bytes : int
        Number of characters (bytes) to read.

    Returns
    -------
    str | None
        The decoded string with trailing spaces/nulls stripped,
        or ``None`` if all bits are set (missing indicator).
    """
    raw_bytes = bytearray(num_bytes)
    all_ones = True
    for i in range(num_bytes):
        byte_val = _read_bits(data, bit_offset + i * 8, 8)
        raw_bytes[i] = byte_val
        if byte_val != 0xFF:
            all_ones = False
    if all_ones:
        return None
    return bytes(raw_bytes).decode("ascii", errors="replace").rstrip(" \x00")


def decode(
    expanded: list[ExpandedItem],
    data_bytes: bytes,
    num_subsets: int,
    compressed: bool,
) -> list[DecodedSubset]:
    """Decode Section 4 data bits into :class:`DecodedSubset` objects.

    Parameters
    ----------
    expanded : list[ExpandedItem]
        The expanded descriptor sequence from Section 3, which may
        include :class:`DelayedReplicationMarker` items.
    data_bytes : bytes
        Raw bytes of the data section payload.
    num_subsets : int
        Number of data subsets to decode.
    compressed : bool
        ``True`` for compressed (DRS) mode, ``False`` for uncompressed.

    Returns
    -------
    list[DecodedSubset]
        One :class:`DecodedSubset` per data subset.
    """
    if compressed:
        return _decode_compressed(expanded, data_bytes, num_subsets)
    return _decode_uncompressed(expanded, data_bytes, num_subsets)


# ---------------------------------------------------------------------------
# Uncompressed decoding
# ---------------------------------------------------------------------------
def _decode_items_uncompressed(
    items: list[ExpandedItem] | tuple[ExpandedItem, ...],
    data_bytes: bytes,
    bit_offset: int,
    values: list[tuple[ExpandedDescriptor, float | str | None]],
) -> int:
    """Decode a sequence of expanded items (uncompressed mode).

    Handles both plain descriptors and delayed-replication markers.

    Returns the updated bit_offset.
    """
    for item in items:
        if isinstance(item, DelayedReplicationMarker):
            bit_offset = _decode_delayed_replication_uncompressed(
                item, data_bytes, bit_offset, values
            )
        else:
            entry = item.entry
            if entry.bit_width == 0:
                continue
            if entry.units == "CCITT IA5":
                num_bytes = entry.bit_width // 8
                string_val = _decode_string(data_bytes, bit_offset, num_bytes)
                values.append((item, string_val))
                bit_offset += entry.bit_width
            else:
                raw = _read_bits(data_bytes, bit_offset, entry.bit_width)
                bit_offset += entry.bit_width
                val = _decode_value(raw, entry)
                values.append((item, val))
    return bit_offset


def _decode_delayed_replication_uncompressed(
    marker: DelayedReplicationMarker,
    data_bytes: bytes,
    bit_offset: int,
    values: list[tuple[ExpandedDescriptor, float | str | None]],
) -> int:
    """Handle a delayed-replication marker in uncompressed mode.

    Reads the replication factor, emits it as a value, then
    decodes the group that many times.
    """
    factor_desc = marker.factor_desc
    entry = factor_desc.entry
    raw = _read_bits(data_bytes, bit_offset, entry.bit_width)
    bit_offset += entry.bit_width
    factor_val = _decode_value(raw, entry)
    values.append((factor_desc, factor_val))

    # The replication count is the raw integer value (not scaled)
    replication_count = raw + entry.reference_value
    group = list(marker.group)

    for _ in range(replication_count):
        bit_offset = _decode_items_uncompressed(group, data_bytes, bit_offset, values)
    return bit_offset


def _decode_uncompressed(
    expanded: list[ExpandedItem],
    data_bytes: bytes,
    num_subsets: int,
) -> list[DecodedSubset]:
    """Decode uncompressed data subsets.

    Parameters
    ----------
    expanded : list[ExpandedItem]
        The expanded descriptor list (may include markers).
    data_bytes : bytes
        Raw data section bytes.
    num_subsets : int
        Number of subsets.

    Returns
    -------
    list[DecodedSubset]
        Decoded subsets.
    """
    subsets: list[DecodedSubset] = []
    bit_offset = 0

    for _ in range(num_subsets):
        values: list[tuple[ExpandedDescriptor, float | str | None]] = []
        bit_offset = _decode_items_uncompressed(
            expanded, data_bytes, bit_offset, values
        )
        subsets.append(DecodedSubset(values=tuple(values)))

    return subsets


# ---------------------------------------------------------------------------
# Compressed decoding
# ---------------------------------------------------------------------------
def _decode_items_compressed(
    items: list[ExpandedItem] | tuple[ExpandedItem, ...],
    data_bytes: bytes,
    bit_offset: int,
    num_subsets: int,
    subset_values: list[list[tuple[ExpandedDescriptor, float | str | None]]],
) -> int:
    """Decode a sequence of expanded items (compressed mode).

    Returns the updated bit_offset.
    """
    for item in items:
        if isinstance(item, DelayedReplicationMarker):
            bit_offset = _decode_delayed_replication_compressed(
                item, data_bytes, bit_offset, num_subsets, subset_values
            )
        else:
            entry = item.entry
            if entry.bit_width == 0:
                continue

            if entry.units == "CCITT IA5":
                # String field: R0 is the common character bytes
                num_bytes = entry.bit_width // 8
                r0_string = _decode_string(data_bytes, bit_offset, num_bytes)
                bit_offset += entry.bit_width

                # Read NBINC (6 bits)
                nbinc = _read_bits(data_bytes, bit_offset, 6)
                bit_offset += 6

                if nbinc == 0:
                    for s in range(num_subsets):
                        subset_values[s].append((item, r0_string))
                else:
                    for s in range(num_subsets):
                        sub_string = _decode_string(data_bytes, bit_offset, nbinc)
                        bit_offset += nbinc * 8
                        subset_values[s].append((item, sub_string))
            else:
                # Numeric field: read R0 (bit_width bits)
                r0 = _read_bits(data_bytes, bit_offset, entry.bit_width)
                bit_offset += entry.bit_width

                # Read NBINC (6 bits)
                nbinc = _read_bits(data_bytes, bit_offset, 6)
                bit_offset += 6

                if nbinc == 0:
                    val = _decode_value(r0, entry)
                    for s in range(num_subsets):
                        subset_values[s].append((item, val))
                else:
                    for s in range(num_subsets):
                        increment = _read_bits(data_bytes, bit_offset, nbinc)
                        bit_offset += nbinc
                        if _is_missing(increment, nbinc):
                            subset_values[s].append((item, None))
                        else:
                            combined = r0 + increment
                            val = _decode_value(combined, entry)
                            subset_values[s].append((item, val))
    return bit_offset


def _decode_delayed_replication_compressed(
    marker: DelayedReplicationMarker,
    data_bytes: bytes,
    bit_offset: int,
    num_subsets: int,
    subset_values: list[list[tuple[ExpandedDescriptor, float | str | None]]],
) -> int:
    """Handle a delayed-replication marker in compressed mode.

    In compressed mode the replication factor is encoded the same way
    as any other compressed numeric: R0 + NBINC + per-subset increments.
    All subsets must have the same replication count (NBINC should be 0).
    """
    factor_desc = marker.factor_desc
    entry = factor_desc.entry

    # Read R0
    r0 = _read_bits(data_bytes, bit_offset, entry.bit_width)
    bit_offset += entry.bit_width

    # Read NBINC
    nbinc = _read_bits(data_bytes, bit_offset, 6)
    bit_offset += 6

    factor_val = _decode_value(r0, entry)
    replication_count = r0 + entry.reference_value

    if nbinc == 0:
        # All subsets same count
        for s in range(num_subsets):
            subset_values[s].append((factor_desc, factor_val))
    else:
        # Per-subset counts (rare but possible)
        for s in range(num_subsets):
            increment = _read_bits(data_bytes, bit_offset, nbinc)
            bit_offset += nbinc
            combined = r0 + increment
            val = _decode_value(combined, entry)
            subset_values[s].append((factor_desc, val))
        # Use first subset's count for the group
        replication_count = r0 + entry.reference_value

    group = list(marker.group)
    for _ in range(replication_count):
        bit_offset = _decode_items_compressed(
            group, data_bytes, bit_offset, num_subsets, subset_values
        )
    return bit_offset


def _decode_compressed(
    expanded: list[ExpandedItem],
    data_bytes: bytes,
    num_subsets: int,
) -> list[DecodedSubset]:
    """Decode compressed data subsets.

    Parameters
    ----------
    expanded : list[ExpandedItem]
        The expanded descriptor list (may include markers).
    data_bytes : bytes
        Raw data section bytes.
    num_subsets : int
        Number of subsets.

    Returns
    -------
    list[DecodedSubset]
        Decoded subsets.
    """
    subset_values: list[list[tuple[ExpandedDescriptor, float | str | None]]] = [
        [] for _ in range(num_subsets)
    ]

    _decode_items_compressed(expanded, data_bytes, 0, num_subsets, subset_values)

    return [DecodedSubset(values=tuple(sv)) for sv in subset_values]
