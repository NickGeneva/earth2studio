# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests that download a real GDAS PrepBUFR file from NOAA NOMADS.

Each test downloads a partial GDAS PrepBUFR file from NOMADS, validates
the BUFR message structure, and verifies that earth2bufr correctly
identifies and skips DX table messages (data_category=11).

.. note::

    GDAS PrepBUFR is an NCEP-proprietary format that uses embedded DX
    descriptor tables (class 063 descriptors) not in the standard WMO
    tables.  Neither eccodes nor earth2bufr's standard Python backend
    can decode the data messages.  The Fortran backend (NCEPLIBS-bufr)
    is required for full decoding.  These tests therefore validate file
    structure and DX-table handling rather than value-level decoding.

Data source
-----------
HTTP     : ``https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/``
URL      : ``gdas.{YYYYMMDD}/gdas.t{HH}z.prepbufr.nr``
Access   : Anonymous HTTP
Retention: ~2 days rolling window
File size: ~60-70 MB per 6-hourly cycle
Cycles   : 00z, 06z, 12z, 18z
"""

from __future__ import annotations

import urllib.request
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import earth2bufr
import pyarrow as pa
import pytest
from earth2bufr._reader import read_messages
from earth2bufr._section import parse_message

if TYPE_CHECKING:
    from pathlib import Path

from .conftest import (
    FIXED_COLUMNS,
    HAS_ECCODES,
)

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(300)]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Download enough bytes to get the DX table messages and a few data messages.
# PrepBUFR files start with DX table messages (~10 KB each), followed by
# data messages (~10-40 KB each).  2 MB is enough for 50+ messages.
_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024  # 2 MB


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
_NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod"


def _find_available_gdas_url() -> str:
    """Probe NOMADS for the most recent available GDAS PrepBUFR URL.

    Tries today, yesterday, and two days ago at cycles 00z, 06z, 12z,
    18z in reverse chronological order.

    Returns
    -------
    str
        Full URL to a GDAS PrepBUFR file.

    Raises
    ------
    pytest.skip
        If no valid URL can be found within the search window.
    """
    now = datetime.now(tz=UTC)

    for days_ago in range(3):
        date = now - timedelta(days=days_ago)
        ymd = date.strftime("%Y%m%d")
        for cycle in ["18", "12", "06", "00"]:
            url = f"{_NOMADS_BASE}/gdas.{ymd}/gdas.t{cycle}z.prepbufr.nr"
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        return url
            except Exception:
                continue

    pytest.skip("Could not find an available GDAS PrepBUFR file on NOMADS")
    return ""  # unreachable, for type checkers


def _truncate_to_complete_messages(data: bytes) -> bytes:
    """Return the prefix of *data* that contains only complete BUFR messages.

    Scans forward through concatenated BUFR messages, stopping at the
    first message whose declared length exceeds the remaining bytes or
    whose end-marker (``7777``) is missing.

    Parameters
    ----------
    data : bytes
        Raw bytes that may contain truncated trailing messages.

    Returns
    -------
    bytes
        A prefix of *data* ending at the last complete BUFR message.
    """
    pos = 0
    last_good = 0

    while pos < len(data):
        start = data.find(b"BUFR", pos)
        if start == -1:
            break
        if start + 7 > len(data):
            break

        msg_len = int.from_bytes(data[start + 4 : start + 7], "big")
        msg_end = start + msg_len

        if msg_end > len(data):
            break
        if data[msg_end - 4 : msg_end] != b"7777":
            break

        last_good = msg_end
        pos = msg_end

    return data[:last_good]


def _download_gdas_prepbufr(
    dest_dir: Path,
    max_bytes: int = _MAX_DOWNLOAD_BYTES,
) -> Path:
    """Download a GDAS PrepBUFR file (or partial head) from NOMADS.

    The downloaded data is truncated to the last complete BUFR message
    boundary to avoid ``BufrDecodeError`` from the earth2bufr reader.

    Parameters
    ----------
    dest_dir : Path
        Directory to save the downloaded file.
    max_bytes : int
        Maximum number of bytes to download.  ``0`` means full file.

    Returns
    -------
    Path
        Local path to the downloaded file.
    """
    url = _find_available_gdas_url()
    fname = url.rsplit("/", 1)[-1]
    local_path = dest_dir / fname

    if local_path.exists():
        return local_path

    req = urllib.request.Request(url)
    if max_bytes > 0:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()

    data = _truncate_to_complete_messages(data)
    if len(data) == 0:
        pytest.skip("Downloaded GDAS data contained no complete BUFR messages")

    local_path.write_bytes(data)
    return local_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestGdasRemote:
    """Download a real GDAS PrepBUFR file from NOMADS and validate structure."""

    @pytest.fixture(scope="class")
    def gdas_bufr_path(self, tmp_download_dir: Path) -> Path:
        """Download a GDAS PrepBUFR file (cached across tests in class)."""
        return _download_gdas_prepbufr(tmp_download_dir)

    # -- File structure validation -----------------------------------------

    def test_file_contains_bufr_messages(self, gdas_bufr_path: Path) -> None:
        """The downloaded file contains multiple valid BUFR messages."""
        raw = gdas_bufr_path.read_bytes()
        messages = list(read_messages(raw))
        assert len(messages) > 0, "No BUFR messages found"

        for msg in messages:
            assert msg.data[-4:] == b"7777", f"Message {msg.index} missing end marker"

    def test_file_contains_dx_table_messages(self, gdas_bufr_path: Path) -> None:
        """PrepBUFR files start with DX table messages (data_category=11).

        These messages contain the NCEP-specific descriptor tables that
        are needed to decode the subsequent data messages.
        """
        raw = gdas_bufr_path.read_bytes()
        messages = list(read_messages(raw))

        dx_count = 0
        for msg in messages:
            parsed = parse_message(msg)
            if parsed.identification.data_category == 11:
                dx_count += 1

        assert dx_count > 0, "Expected at least one DX table message (data_category=11)"

    def test_earth2bufr_skips_dx_tables(self, gdas_bufr_path: Path) -> None:
        """earth2bufr correctly filters out DX table messages.

        DX table messages (data_category=11) should be skipped during
        decoding since they contain descriptor table definitions, not
        observational data.
        """
        table = earth2bufr.read_bufr(gdas_bufr_path, backend="python")
        assert isinstance(table, pa.Table)

        # The table should have fixed columns even if empty
        for col in FIXED_COLUMNS:
            assert col in table.column_names, f"Missing fixed column: {col}"

        # If any rows were decoded, none should be DX table messages
        if table.num_rows > 0:
            msg_types = table.column("message_type").to_pylist()
            assert (
                "11" not in msg_types
            ), "DX table messages (type 11) should be filtered out"

    def test_file_has_data_messages(self, gdas_bufr_path: Path) -> None:
        """The file contains data messages beyond DX tables.

        PrepBUFR data messages use NCEP-specific descriptors (class 063)
        which require the Fortran backend (NCEPLIBS-bufr) for decoding.
        This test validates that data messages exist in the file.
        """
        raw = gdas_bufr_path.read_bytes()
        messages = list(read_messages(raw))

        data_msg_count = 0
        data_categories: set[int] = set()
        for msg in messages:
            parsed = parse_message(msg)
            cat = parsed.identification.data_category
            if cat != 11:
                data_msg_count += 1
                data_categories.add(cat)

        assert data_msg_count > 0, "Expected data messages beyond DX tables"
        assert len(data_categories) > 0, "Expected at least one non-DX data category"

    def test_message_metadata_plausible(self, gdas_bufr_path: Path) -> None:
        """BUFR message headers contain plausible metadata values."""
        raw = gdas_bufr_path.read_bytes()
        messages = list(read_messages(raw))
        assert len(messages) > 0

        # Check the first non-DX message's identification section
        for msg in messages:
            parsed = parse_message(msg)
            ident = parsed.identification
            if ident.data_category == 11:
                continue

            # Year should be recent
            assert 2020 <= ident.year <= 2030, f"Unexpected year: {ident.year}"
            # Month in range
            assert 1 <= ident.month <= 12, f"Unexpected month: {ident.month}"
            # Day in range
            assert 1 <= ident.day <= 31, f"Unexpected day: {ident.day}"
            # Edition should be 3 (PrepBUFR is typically BUFR Edition 3)
            assert parsed.indicator.edition in (
                3,
                4,
            ), f"Unexpected BUFR edition: {parsed.indicator.edition}"

    # -- Cross-validation against eccodes ----------------------------------

    @pytest.mark.skipif(not HAS_ECCODES, reason="eccodes not installed")
    def test_eccodes_agrees_on_message_count(self, gdas_bufr_path: Path) -> None:
        """eccodes and earth2bufr find the same number of BUFR messages.

        PrepBUFR data messages use NCEP-specific descriptors that
        neither eccodes nor earth2bufr's standard Python backend can
        fully decode.  However, both should agree on the total number
        of messages in the file.
        """
        import eccodes

        # Count with eccodes
        eccodes_count = 0
        with gdas_bufr_path.open("rb") as f:
            while True:
                msgid = eccodes.codes_bufr_new_from_file(f)
                if msgid is None:
                    break
                eccodes_count += 1
                eccodes.codes_release(msgid)

        # Count with earth2bufr
        raw = gdas_bufr_path.read_bytes()
        e2b_count = len(list(read_messages(raw)))

        assert (
            eccodes_count == e2b_count
        ), f"Message count differs: eccodes={eccodes_count}, earth2bufr={e2b_count}"
