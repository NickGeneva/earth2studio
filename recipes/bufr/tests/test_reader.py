# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from earth2bufrio._reader import read_messages
from earth2bufrio._types import BufrDecodeError, BufrMessage


def _make_valid_message() -> bytes:
    """Hand-craft a minimal valid 12-byte BUFR edition-4 message."""
    # 4 (BUFR) + 3 (length) + 1 (edition) + 4 (7777) = 12 bytes
    return b"BUFR" + (12).to_bytes(3, "big") + b"\x04" + b"7777"


# ---------------------------------------------------------------------------
# Single valid message
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_single_valid_message() -> None:
    data = _make_valid_message()
    messages = list(read_messages(data))
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, BufrMessage)
    assert msg.data == data
    assert msg.offset == 0
    assert msg.index == 0


# ---------------------------------------------------------------------------
# Two concatenated messages
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_two_concatenated_messages() -> None:
    single = _make_valid_message()
    data = single + single
    messages = list(read_messages(data))
    assert len(messages) == 2
    assert messages[0].index == 0
    assert messages[0].offset == 0
    assert messages[0].data == single
    assert messages[1].index == 1
    assert messages[1].offset == 12
    assert messages[1].data == single


# ---------------------------------------------------------------------------
# Empty bytes
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_empty_bytes() -> None:
    messages = list(read_messages(b""))
    assert messages == []


# ---------------------------------------------------------------------------
# No BUFR marker
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_no_bufr_marker() -> None:
    messages = list(read_messages(b"not bufr data at all"))
    assert messages == []


# ---------------------------------------------------------------------------
# Truncated message
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_truncated_message() -> None:
    data = b"BUFR" + (100).to_bytes(3, "big") + b"\x04"
    with pytest.raises(BufrDecodeError):
        list(read_messages(data))


# ---------------------------------------------------------------------------
# Bad end marker
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_bad_end_marker() -> None:
    data = b"BUFR" + (12).to_bytes(3, "big") + b"\x04" + b"XXXX"
    with pytest.raises(BufrDecodeError):
        list(read_messages(data))


# ---------------------------------------------------------------------------
# Garbage between messages
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_garbage_before_message() -> None:
    junk = b"\x00" * 5
    valid = _make_valid_message()
    data = junk + valid
    messages = list(read_messages(data))
    assert len(messages) == 1
    assert messages[0].offset == 5
    assert messages[0].data == valid
    assert messages[0].index == 0
