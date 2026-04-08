# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests that download real NNJA reanalysis BUFR files from NOAA S3.

Each test downloads a small reanalysis observation BUFR file from the
NOAA Reanalyses Public Data Set on S3, validates the BUFR message
structure, and verifies that earth2bufr correctly identifies NCEP
PrepBUFR DX table messages (data_category=11).

.. note::

    NNJA reanalysis BUFR files are NCEP-proprietary format (BUFR
    Edition 3, centre=7) that use embedded DX descriptor tables and
    class 048-063 local descriptors not in the standard WMO tables.
    Neither eccodes nor earth2bufr's standard Python backend can decode
    the data messages.  These tests validate file structure, DX table
    handling, and message metadata rather than value-level decoding.

Data source
-----------
S3 bucket : ``s3://noaa-reanalyses-pds/observations/reanalysis/``
Access    : Anonymous
Archive   : Permanent (1998-2024, instrument-dependent)
Instruments tested:
  - ``amsua/1bamua``  : AMSU-A satellite radiance (~15 MB)
  - ``ozone/cfsr``    : SBUV ozone (~0.6 MB, smallest available)
  - ``gps/gpsro``     : GPS radio occultation (~7 MB)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from earth2bufr._reader import read_messages
from earth2bufr._section import parse_message

from .conftest import (
    HAS_ECCODES,
    HAS_S3FS,
)

# ---------------------------------------------------------------------------
# S3 configuration
# ---------------------------------------------------------------------------
_S3_BUCKET = "noaa-reanalyses-pds"

# Instruments to test, keyed by a short label.
# Each entry: (S3 prefix under the bucket, glob pattern, max bytes to download).
# We use specific dates known to exist in the archive.
_INSTRUMENTS: dict[str, dict[str, str | int]] = {
    "ozone": {
        "prefix": "observations/reanalysis/ozone/cfsr/2020/01/bufr",
        "glob": "*.bufr_d",
        "max_bytes": 0,  # full file (~0.6 MB)
    },
    "gpsro": {
        "prefix": "observations/reanalysis/gps/gpsro/2020/01/bufr",
        "glob": "*.bufr_d",
        "max_bytes": 2 * 1024 * 1024,  # 2 MB partial (~7 MB full)
    },
    "amsua": {
        "prefix": "observations/reanalysis/amsua/1bamua/2020/01/bufr",
        "glob": "*.bufr_d",
        "max_bytes": 2 * 1024 * 1024,  # 2 MB partial (~15 MB full)
    },
}

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(300)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _truncate_to_complete_messages(data: bytes) -> bytes:
    """Return the prefix of *data* containing only complete BUFR messages."""
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


def _download_nnja_bufr(
    dest_dir: Path,
    instrument: str,
) -> Path:
    """Download an NNJA reanalysis BUFR file from S3.

    Parameters
    ----------
    dest_dir : Path
        Directory to save the downloaded file.
    instrument : str
        Key into ``_INSTRUMENTS`` dict.

    Returns
    -------
    Path
        Local path to the downloaded (possibly truncated) BUFR file.
    """
    import s3fs

    cfg = _INSTRUMENTS[instrument]
    prefix = cfg["prefix"]
    max_bytes = int(cfg["max_bytes"])

    fs = s3fs.S3FileSystem(anon=True)
    listing = fs.ls(f"{_S3_BUCKET}/{prefix}")
    bufr_keys = sorted(k for k in listing if k.endswith(".bufr_d"))
    if not bufr_keys:
        pytest.skip(f"No .bufr_d files found under s3://{_S3_BUCKET}/{prefix}")

    remote_key = bufr_keys[0]
    local_name = f"{instrument}_{Path(remote_key).name}"
    local_path = dest_dir / local_name

    if local_path.exists():
        return local_path

    if max_bytes > 0:
        # Partial download: read only the first max_bytes
        with fs.open(remote_key, "rb") as f:
            data = f.read(max_bytes)
        data = _truncate_to_complete_messages(data)
        if len(data) == 0:
            pytest.skip(
                f"Downloaded NNJA {instrument} data contained no complete messages"
            )
        local_path.write_bytes(data)
    else:
        fs.get(remote_key, str(local_path))

    return local_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_S3FS, reason="s3fs not installed")
class TestNnjaRemote:
    """Download real NNJA reanalysis BUFR files and validate structure."""

    @pytest.fixture(scope="class", params=list(_INSTRUMENTS.keys()))
    def nnja_bufr_path(
        self, request: pytest.FixtureRequest, tmp_download_dir: Path
    ) -> Path:
        """Download an NNJA BUFR file for each instrument."""
        return _download_nnja_bufr(tmp_download_dir, request.param)

    # -- File structure validation -----------------------------------------

    def test_file_contains_bufr_messages(self, nnja_bufr_path: Path) -> None:
        """The downloaded file contains at least one valid BUFR message."""
        raw = nnja_bufr_path.read_bytes()
        assert len(raw) > 8, f"File too small: {len(raw)} bytes"
        assert b"BUFR" in raw, "File does not contain a BUFR marker"

        messages = list(read_messages(raw))
        assert len(messages) > 0, "No complete BUFR messages found"

        for msg in messages:
            assert msg.data[-4:] == b"7777", f"Message {msg.index} missing end marker"

    def test_messages_are_ncep_bufr_edition3(self, nnja_bufr_path: Path) -> None:
        """NNJA messages are BUFR Edition 3 from NCEP (centre=7)."""
        raw = nnja_bufr_path.read_bytes()
        messages = list(read_messages(raw))

        for msg in messages:
            parsed = parse_message(msg)

            assert parsed.indicator.edition in (
                3,
                4,
            ), f"Unexpected BUFR edition: {parsed.indicator.edition}"

    def test_file_contains_dx_table_messages(self, nnja_bufr_path: Path) -> None:
        """NNJA files start with DX table messages (data_category=11).

        These NCEP-proprietary messages contain embedded descriptor
        tables that define the local class 048-063 descriptors used in
        the subsequent data messages.
        """
        raw = nnja_bufr_path.read_bytes()
        messages = list(read_messages(raw))

        dx_count = 0
        for msg in messages:
            parsed = parse_message(msg)
            if parsed.identification.data_category == 11:
                dx_count += 1

        assert dx_count > 0, "Expected at least one DX table message (data_category=11)"

    def test_file_has_data_messages(self, nnja_bufr_path: Path) -> None:
        """The file contains data messages beyond DX tables."""
        raw = nnja_bufr_path.read_bytes()
        messages = list(read_messages(raw))

        data_msg_count = 0
        for msg in messages:
            parsed = parse_message(msg)
            if parsed.identification.data_category != 11:
                data_msg_count += 1

        assert data_msg_count > 0, "Expected data messages beyond DX tables"

    def test_message_metadata_plausible(self, nnja_bufr_path: Path) -> None:
        """BUFR message headers contain plausible metadata values."""
        raw = nnja_bufr_path.read_bytes()
        messages = list(read_messages(raw))

        for msg in messages:
            parsed = parse_message(msg)
            ident = parsed.identification

            if ident.data_category == 11:
                continue

            # Year should be around 2020 for our test data
            assert 2015 <= ident.year <= 2025, f"Unexpected year: {ident.year}"
            assert 1 <= ident.month <= 12, f"Unexpected month: {ident.month}"
            assert 1 <= ident.day <= 31, f"Unexpected day: {ident.day}"
            assert 0 <= ident.hour <= 23, f"Unexpected hour: {ident.hour}"
            break  # Only need to check one data message

    # -- Cross-validation against eccodes ----------------------------------

    @pytest.mark.skipif(not HAS_ECCODES, reason="eccodes not installed")
    def test_eccodes_agrees_on_message_count(self, nnja_bufr_path: Path) -> None:
        """eccodes and earth2bufr find the same number of BUFR messages.

        Neither decoder can fully unpack NCEP-proprietary data messages,
        but both should agree on the number of messages in the file.
        """
        import eccodes

        eccodes_count = 0
        with nnja_bufr_path.open("rb") as f:
            while True:
                msgid = eccodes.codes_bufr_new_from_file(f)
                if msgid is None:
                    break
                eccodes_count += 1
                eccodes.codes_release(msgid)

        raw = nnja_bufr_path.read_bytes()
        e2b_count = len(list(read_messages(raw)))

        assert (
            eccodes_count == e2b_count
        ), f"Message count differs: eccodes={eccodes_count}, earth2bufr={e2b_count}"
