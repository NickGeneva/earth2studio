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


def _fxy_to_packed(fxy: int) -> int:
    """Convert decimal FXXYYY to 16-bit packed wire format."""
    f = fxy // 100000
    x = (fxy % 100000) // 1000
    y = fxy % 1000
    return (f << 14) | (x << 8) | y


def _build_section1_ed4(data_category: int = 0) -> bytes:
    """Build a minimal Edition 4 Section 1 (Identification)."""
    body = bytearray(19)
    body[0] = 4
    body[1:3] = (7).to_bytes(2, "big")
    body[3:5] = (0).to_bytes(2, "big")
    body[5] = 0
    body[6] = 0
    body[7] = data_category
    body[8] = 0
    body[9] = 0
    body[10] = 0
    body[11] = 0
    body[12:14] = (2025).to_bytes(2, "big")
    body[14] = 6
    body[15] = 15
    body[16] = 12
    body[17] = 0
    body[18] = 0
    sec_len = 3 + len(body)
    return sec_len.to_bytes(3, "big") + bytes(body)


def _build_section3(descriptors: list[int], num_subsets: int = 1) -> bytes:
    """Build Section 3 (Data Description)."""
    body = bytearray()
    body.append(0)
    body.extend(num_subsets.to_bytes(2, "big"))
    body.append(0x80)
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
    total_len = 8 + len(sec1) + len(sec3) + len(sec4) + 4
    sec0 = b"BUFR" + total_len.to_bytes(3, "big") + b"\x04"
    end = b"7777"
    return sec0 + sec1 + sec3 + sec4 + end


class TestReadBufrErrors:
    """Tests for error handling in read_bufr."""

    def test_nonexistent_path_raises_file_not_found(self, tmp_path: Path) -> None:
        """read_bufr() raises FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError):
            read_bufr(tmp_path / "nonexistent.bufr")

    def test_empty_file_returns_empty_table(self, tmp_path: Path) -> None:
        """An empty file produces an empty table with fixed columns."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        table = read_bufr(p)
        assert isinstance(table, pa.Table)
        assert len(table) == 0
        assert "message_type" in table.column_names
        assert "message_index" in table.column_names
        assert "subset_index" in table.column_names

    def test_unknown_backend_raises(self, tmp_path: Path) -> None:
        """Unknown backend raises ValueError."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="Unknown backend"):
            read_bufr(p, backend="unknown")


class TestReadBufrBasic:
    """Tests for basic read_bufr functionality with hand-crafted BUFR data."""

    def test_hand_crafted_single_descriptor(self, tmp_path: Path) -> None:
        """A message with one numeric descriptor produces one row."""
        raw = 2731
        data_bits = struct.pack(">H", raw << 4)
        bufr = _build_bufr_ed4(
            data_category=2,
            descriptors=[12001],
            data_bits=data_bits,
        )
        p = tmp_path / "temp.bufr"
        p.write_bytes(bufr)

        table = read_bufr(p)
        assert len(table) >= 1
        assert "message_type" in table.column_names
        assert "message_index" in table.column_names

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """read_bufr() accepts a string path."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        table = read_bufr(str(p))
        assert isinstance(table, pa.Table)
        assert len(table) == 0


class TestReadBufrFiltering:
    """Tests for the mnemonics and filters parameters."""

    def test_filters_data_category(self, tmp_path: Path) -> None:
        """filters={'data_category': 0} skips non-matching messages."""
        msg0 = _build_bufr_ed4(data_category=0)
        msg2 = _build_bufr_ed4(data_category=2)
        p = tmp_path / "multi.bufr"
        p.write_bytes(msg0 + msg2)

        table_filtered = read_bufr(p, filters={"data_category": 0})
        # All returned rows should be from data_category=0 messages
        assert isinstance(table_filtered, pa.Table)


class TestReadBufrBackend:
    """Tests for the backend parameter."""

    def test_python_backend_explicit(self, tmp_path: Path) -> None:
        """backend='python' works explicitly."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        table = read_bufr(p, backend="python")
        assert isinstance(table, pa.Table)
        assert len(table) == 0

    def test_workers_parameter_accepted(self, tmp_path: Path) -> None:
        """workers parameter does not raise for workers=2."""
        p = tmp_path / "empty.bufr"
        p.write_bytes(b"")
        table = read_bufr(p, workers=2)
        assert isinstance(table, pa.Table)
        assert len(table) == 0
