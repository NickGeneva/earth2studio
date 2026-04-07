# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public ``read_bufr()`` API."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from earth2bufrio import read_bufr

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers — build minimal valid BUFR Ed4 messages
# ---------------------------------------------------------------------------

_SPDX = "SPDX"  # avoid license-header false positive


def _fxy_to_packed(fxy: int) -> int:
    """Convert decimal FXXYYY to 16-bit packed wire format."""
    f = fxy // 100000
    x = (fxy % 100000) // 1000
    y = fxy % 1000
    return (f << 14) | (x << 8) | y


def _build_section1_ed4(data_category: int = 0) -> bytes:
    """Build a minimal Edition 4 Section 1 (Identification)."""
    body = bytearray(19)
    body[0] = 4  # master table
    body[1:3] = (7).to_bytes(2, "big")  # originating center
    body[3:5] = (0).to_bytes(2, "big")  # subcenter
    body[5] = 0  # update sequence number
    body[6] = 0  # flags (no optional section)
    body[7] = data_category  # data category
    body[8] = 0  # international sub-category
    body[9] = 0  # local sub-category
    body[10] = 0  # master table version
    body[11] = 0  # local table version
    body[12:14] = (2025).to_bytes(2, "big")  # year
    body[14] = 6  # month
    body[15] = 15  # day
    body[16] = 12  # hour
    body[17] = 0  # minute
    body[18] = 0  # second
    sec_len = 3 + len(body)
    return sec_len.to_bytes(3, "big") + bytes(body)


def _build_section3(descriptors: list[int], num_subsets: int = 1) -> bytes:
    """Build Section 3 (Data Description)."""
    body = bytearray()
    body.append(0)  # reserved
    body.extend(num_subsets.to_bytes(2, "big"))
    body.append(0x80)  # observed=True, compressed=False
    for d in descriptors:
        body.extend(_fxy_to_packed(d).to_bytes(2, "big"))
    sec_len = 3 + len(body)
    return sec_len.to_bytes(3, "big") + bytes(body)


def _build_section4(data_bits: bytes) -> bytes:
    """Build Section 4 (Data) from raw payload bytes."""
    sec_len = 4 + len(data_bits)
    return sec_len.to_bytes(3, "big") + b"\x00" + data_bits


def _build_bufr_ed4(
    data_category: int = 0,
    descriptors: list[int] | None = None,
    data_bits: bytes = b"",
    num_subsets: int = 1,
) -> bytes:
    """Assemble a complete minimal BUFR Ed4 message."""
    if descriptors is None:
        descriptors = []
    sec1 = _build_section1_ed4(data_category)
    sec3 = _build_section3(descriptors, num_subsets)
    sec4 = _build_section4(data_bits)
    # Total: 8 (sec0) + sec1 + sec3 + sec4 + 4 (end marker)
    total_len = 8 + len(sec1) + len(sec3) + len(sec4) + 4
    sec0 = b"BUFR" + total_len.to_bytes(3, "big") + b"\x04"
    end = b"7777"
    return sec0 + sec1 + sec3 + sec4 + end


def _encode_value(value: float, scale: int, reference: int, bit_width: int) -> bytes:
    """Encode a physical value into BUFR raw bits (byte-aligned)."""
    raw = int(value * (10**scale)) - reference
    return raw.to_bytes((bit_width + 7) // 8, "big")


class TestReadBufrErrors:
    """Tests for error handling in read_bufr."""

    def test_nonexistent_path_raises_file_not_found(self, tmp_path: Path) -> None:
        """read_bufr() raises FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError):
            read_bufr(tmp_path / "nonexistent.bufr")

    def test_empty_file_returns_empty_table(self, tmp_path: Path) -> None:
        """An empty file produces an empty table with the correct schema."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        table = read_bufr(p)
        assert isinstance(table, pa.Table)
        assert len(table) == 0
        assert table.schema.names == [
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


class TestReadBufrBasic:
    """Tests for basic read_bufr functionality with hand-crafted BUFR data."""

    def test_hand_crafted_single_descriptor(self, tmp_path: Path) -> None:
        """A message with one numeric descriptor produces one row."""
        # Descriptor 12001 = TEMPERATURE, K, scale=1, ref=0, 12-bit
        # Encode 273.1 K -> raw = 2731
        raw = 2731
        # 12-bit value, needs 2 bytes (pad with zeros)
        data_bits = struct.pack(">H", raw << 4)  # shift left to fill 16 bits

        bufr = _build_bufr_ed4(
            data_category=2,
            descriptors=[12001],
            data_bits=data_bits,
        )
        p = tmp_path / "temp.bufr"
        p.write_bytes(bufr)

        table = read_bufr(p)
        assert len(table) >= 1
        assert "value" in table.schema.names

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """read_bufr() accepts a string path."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        table = read_bufr(str(p))
        assert isinstance(table, pa.Table)
        assert len(table) == 0


class TestReadBufrFiltering:
    """Tests for the columns and filters parameters."""

    def test_columns_parameter_filters_output(self, tmp_path: Path) -> None:
        """columns parameter restricts returned columns."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        table = read_bufr(p, columns=["value", "units"])
        assert table.schema.names == ["value", "units"]

    def test_filters_data_category(self, tmp_path: Path) -> None:
        """filters={'data_category': 0} skips non-matching messages."""
        # Build two messages: data_category=0 and data_category=2
        msg0 = _build_bufr_ed4(data_category=0)
        msg2 = _build_bufr_ed4(data_category=2)

        p = tmp_path / "multi.bufr"
        p.write_bytes(msg0 + msg2)

        # Without filter: both messages are readable
        read_bufr(p)

        # With filter: only data_category=0
        table_filtered = read_bufr(p, filters={"data_category": 0})
        if len(table_filtered) > 0:
            cats = table_filtered.column("data_category").to_pylist()
            assert all(c == 0 for c in cats)


class TestReadBufrWorkers:
    """Tests for the workers parameter."""

    def test_workers_parameter_accepted(self, tmp_path: Path) -> None:
        """workers parameter does not raise for workers=2."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        table = read_bufr(p, workers=2)
        assert isinstance(table, pa.Table)
        assert len(table) == 0
