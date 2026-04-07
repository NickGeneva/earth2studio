# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level splitting of a byte stream into individual BUFR messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from earth2bufr._types import BufrDecodeError, BufrMessage

_BUFR_MARKER = b"BUFR"
_END_MARKER = b"7777"
_MIN_MSG_LEN = 12


def read_messages(data: bytes) -> Iterator[BufrMessage]:
    """Scan *data* and yield each complete BUFR message found.

    The function searches for ``b"BUFR"`` start markers, reads the 3-byte
    big-endian total length field, validates the ``b"7777"`` end marker,
    and yields a :class:`~earth2bufr._types.BufrMessage` for every valid
    message.

    Parameters
    ----------
    data : bytes
        Raw bytes that may contain one or more concatenated BUFR messages,
        possibly separated by non-BUFR padding.

    Yields
    ------
    BufrMessage
        One message per BUFR record found in *data*.

    Raises
    ------
    BufrDecodeError
        If a BUFR start marker is found but the message is truncated
        (declared length exceeds available bytes) or the end marker is
        not ``b"7777"``.
    """
    pos = 0
    counter = 0
    length = len(data)

    while pos < length:
        start = data.find(_BUFR_MARKER, pos)
        if start == -1:
            return

        # Need at least 8 bytes to read the length field (4 marker + 3 len + 1 edition)
        if start + 7 > length:
            raise BufrDecodeError(
                f"Truncated BUFR header at offset {start}: need 8 bytes but only {length - start} available",
                offset=start,
            )

        msg_len = int.from_bytes(data[start + 4 : start + 7], "big")

        if start + msg_len > length:
            raise BufrDecodeError(
                f"Truncated BUFR message at offset {start}: "
                f"declared length {msg_len} but only {length - start} bytes available",
                offset=start,
            )

        msg_bytes = data[start : start + msg_len]

        if msg_bytes[-4:] != _END_MARKER:
            raise BufrDecodeError(
                f"Bad end marker at offset {start}: expected b'7777', got {msg_bytes[-4:]!r}",
                offset=start,
            )

        yield BufrMessage(data=msg_bytes, offset=start, index=counter)
        counter += 1
        pos = start + msg_len
