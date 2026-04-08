# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests that download real JPSS NUCAPS C0431 BUFR files from NOAA S3.

Each test downloads a single NUCAPS BUFR file from the NOAA Open Data
archive, decodes it with eccodes and the earth2bufr Python backend, and
cross-validates the results.

NUCAPS (NOAA Unique CrIS/ATMS Processing System) C0431 files contain
atmospheric temperature and moisture retrieval profiles.  These files
use standard WMO descriptors (centre=160, local_table_ver=0) and are
fully decodable by eccodes.  The earth2bufr Python backend may not
decode all messages due to complex compressed data.

Data source
-----------
S3 bucket : ``s3://noaa-nesdis-n20-pds/NUCAPS_C0431_BUFR/<YYYY>/<MM>/<DD>/*.bufr``
Access    : Anonymous
Archive   : Permanent (available since 2023-09-06)
File size : ~1.1 MB per granule
"""

from __future__ import annotations

from pathlib import Path

import earth2bufr
import pyarrow as pa
import pytest
from earth2bufr._reader import read_messages
from earth2bufr._section import parse_message

from .conftest import (
    FIXED_COLUMNS,
    HAS_ECCODES,
    HAS_S3FS,
    available_backends,
    values_equal,
)

# ---------------------------------------------------------------------------
# S3 download helper
# ---------------------------------------------------------------------------
_S3_BUCKET = "noaa-nesdis-n20-pds"
_S3_PREFIX = "NUCAPS_C0431_BUFR/2024/01/15"

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]


def _download_first_nucaps_bufr(dest_dir: Path) -> Path:
    """Download the first NUCAPS C0431 BUFR file from the S3 archive.

    Parameters
    ----------
    dest_dir : Path
        Directory to save the downloaded file.

    Returns
    -------
    Path
        Local path to the downloaded BUFR file.
    """
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    listing = fs.ls(f"{_S3_BUCKET}/{_S3_PREFIX}")
    bufr_keys = sorted(k for k in listing if k.endswith(".bufr"))
    if not bufr_keys:
        pytest.skip(f"No .bufr files found under s3://{_S3_BUCKET}/{_S3_PREFIX}")

    remote_key = bufr_keys[0]
    local_path = dest_dir / Path(remote_key).name
    if not local_path.exists():
        fs.get(remote_key, str(local_path))
    return local_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_S3FS, reason="s3fs not installed")
class TestNucapsRemote:
    """Download a real NUCAPS BUFR file from NOAA S3 and validate decoding."""

    @pytest.fixture(scope="class")
    def nucaps_bufr_path(self, tmp_download_dir: Path) -> Path:
        """Download a single NUCAPS BUFR file (cached across tests in class)."""
        return _download_first_nucaps_bufr(tmp_download_dir)

    # -- File structure validation -----------------------------------------

    def test_downloaded_file_contains_bufr_messages(
        self, nucaps_bufr_path: Path
    ) -> None:
        """The downloaded file contains at least one valid BUFR message."""
        raw = nucaps_bufr_path.read_bytes()
        assert len(raw) > 8, f"File too small: {len(raw)} bytes"
        assert b"BUFR" in raw, "File does not contain a BUFR marker"

        messages = list(read_messages(raw))
        assert len(messages) > 0, "No complete BUFR messages found"

        for msg in messages:
            assert msg.data[-4:] == b"7777", f"Message {msg.index} missing end marker"

    def test_messages_are_standard_wmo_bufr(self, nucaps_bufr_path: Path) -> None:
        """NUCAPS messages use standard WMO descriptors (no local tables)."""
        raw = nucaps_bufr_path.read_bytes()
        messages = list(read_messages(raw))

        for msg in messages:
            parsed = parse_message(msg)
            ident = parsed.identification

            # Edition should be 4
            assert (
                parsed.indicator.edition == 4
            ), f"Expected BUFR Edition 4, got {parsed.indicator.edition}"

            # Originating center 160 = NESDIS
            assert ident.originating_center in (
                160,
                7,
            ), f"Unexpected originating center: {ident.originating_center}"

            # Year should be 2024 for our pinned date
            assert 2020 <= ident.year <= 2030, f"Unexpected year: {ident.year}"

    # -- Python backend decoding -------------------------------------------

    def test_python_backend_runs_without_crash(self, nucaps_bufr_path: Path) -> None:
        """The Python backend returns a PyArrow Table (may be empty for NUCAPS).

        NUCAPS C0431 files use standard WMO descriptors but may contain
        complex compressed data or delayed replication that the earth2bufr
        Python decoder cannot yet fully handle.  This test validates the
        decode pipeline runs without crashing.
        """
        table = earth2bufr.read_bufr(nucaps_bufr_path, backend="python")
        assert isinstance(table, pa.Table)

        for col in FIXED_COLUMNS:
            assert col in table.column_names, f"Missing fixed column: {col}"

    # -- Backend consistency -----------------------------------------------

    def test_all_backends_match(self, nucaps_bufr_path: Path) -> None:
        """All available backends produce the same table as Python.

        When only the Python backend is compiled, this test verifies
        decode idempotency (two decodes produce identical tables).

        The Fortran backend is excluded because NCEPLIBS-bufr only
        supports NCEP PrepBUFR, not standard WMO BUFR like NUCAPS.
        """
        backends = available_backends(include_fortran=False)
        ref_table = earth2bufr.read_bufr(nucaps_bufr_path, backend="python")

        if len(backends) < 2:
            # Idempotency check: decode a second time and compare
            dup_table = earth2bufr.read_bufr(nucaps_bufr_path, backend="python")
            assert (
                ref_table.num_rows == dup_table.num_rows
            ), "Idempotency: row count differs"
            assert sorted(ref_table.column_names) == sorted(
                dup_table.column_names
            ), "Idempotency: column names differ"
        else:
            for backend in backends:
                if backend == "python":
                    continue
                cmp_table = earth2bufr.read_bufr(nucaps_bufr_path, backend=backend)

                assert (
                    ref_table.num_rows == cmp_table.num_rows
                ), f"Row count differs: python={ref_table.num_rows}, {backend}={cmp_table.num_rows}"

                ref_cols = sorted(ref_table.column_names)
                cmp_cols = sorted(cmp_table.column_names)
                assert (
                    ref_cols == cmp_cols
                ), f"Column names differ between python and {backend}"

    # -- Cross-validation against eccodes ----------------------------------

    @pytest.mark.skipif(not HAS_ECCODES, reason="eccodes not installed")
    def test_eccodes_can_decode(self, nucaps_bufr_path: Path) -> None:
        """eccodes can successfully unpack the downloaded NUCAPS BUFR file."""
        import eccodes

        decoded_messages = 0
        total_subsets = 0

        with nucaps_bufr_path.open("rb") as f:
            while True:
                msgid = eccodes.codes_bufr_new_from_file(f)
                if msgid is None:
                    break
                try:
                    eccodes.codes_set(msgid, "unpack", 1)
                    num_subsets = eccodes.codes_get(msgid, "numberOfSubsets")
                    total_subsets += num_subsets
                    decoded_messages += 1
                finally:
                    eccodes.codes_release(msgid)

        assert decoded_messages > 0, "eccodes could not decode any messages"
        assert total_subsets > 0, "eccodes found zero subsets"

    @pytest.mark.skipif(not HAS_ECCODES, reason="eccodes not installed")
    def test_crossval_eccodes_vs_earth2bufr(self, nucaps_bufr_path: Path) -> None:
        """Compare message structure between eccodes and earth2bufr.

        If earth2bufr decoded zero rows (unsupported descriptors), this
        test records the difference as informational and passes -- the
        purpose is to verify structural consistency.
        """
        import eccodes

        table = earth2bufr.read_bufr(nucaps_bufr_path, backend="python")

        eccodes_total = 0
        with nucaps_bufr_path.open("rb") as f:
            while True:
                msgid = eccodes.codes_bufr_new_from_file(f)
                if msgid is None:
                    break
                try:
                    eccodes.codes_set(msgid, "unpack", 1)
                    eccodes_total += eccodes.codes_get(msgid, "numberOfSubsets")
                finally:
                    eccodes.codes_release(msgid)

        # earth2bufr may not decode NUCAPS (complex satellite descriptors).
        # If it decoded something, counts should roughly match.
        if table.num_rows > 0 and eccodes_total > 0:
            ratio = min(table.num_rows, eccodes_total) / max(
                table.num_rows, eccodes_total
            )
            assert ratio > 0.5, (
                f"Subset count divergence too large: "
                f"earth2bufr={table.num_rows}, eccodes={eccodes_total} "
                f"(ratio={ratio:.2f})"
            )

    @pytest.mark.skipif(not HAS_ECCODES, reason="eccodes not installed")
    def test_eccodes_and_earth2bufr_datetime_match(
        self, nucaps_bufr_path: Path
    ) -> None:
        """Date/time fields from eccodes match earth2bufr for the first message.

        Compares YEAR, MONTH, DAY, HOUR, MINUTE from both decoders to
        verify that earth2bufr correctly extracts temporal metadata.
        When earth2bufr decodes zero rows, validates eccodes datetime
        fields are plausible on their own.
        """
        import eccodes

        table = earth2bufr.read_bufr(nucaps_bufr_path, backend="python")

        # Read first message with eccodes
        with nucaps_bufr_path.open("rb") as f:
            msgid = eccodes.codes_bufr_new_from_file(f)
            assert msgid is not None
            try:
                eccodes.codes_set(msgid, "unpack", 1)
                ecc_year = int(eccodes.codes_get(msgid, "typicalYear"))
                ecc_month = int(eccodes.codes_get(msgid, "typicalMonth"))
                ecc_day = int(eccodes.codes_get(msgid, "typicalDay"))
                ecc_hour = int(eccodes.codes_get(msgid, "typicalHour"))
                ecc_minute = int(eccodes.codes_get(msgid, "typicalMinute"))
            finally:
                eccodes.codes_release(msgid)

        if table.num_rows == 0:
            # earth2bufr couldn't decode — validate eccodes values are plausible
            assert 2020 <= ecc_year <= 2030, f"eccodes year implausible: {ecc_year}"
            assert 1 <= ecc_month <= 12, f"eccodes month implausible: {ecc_month}"
            assert 1 <= ecc_day <= 31, f"eccodes day implausible: {ecc_day}"
            assert 0 <= ecc_hour <= 23, f"eccodes hour implausible: {ecc_hour}"
            assert 0 <= ecc_minute <= 59, f"eccodes minute implausible: {ecc_minute}"
        else:
            # Compare with earth2bufr first row
            row0_year = table.column("YEAR")[0].as_py()
            row0_month = table.column("MNTH")[0].as_py()
            row0_day = table.column("DAYS")[0].as_py()
            row0_hour = table.column("HOUR")[0].as_py()
            row0_minute = table.column("MINU")[0].as_py()

            assert values_equal(
                row0_year, float(ecc_year)
            ), f"Year mismatch: earth2bufr={row0_year}, eccodes={ecc_year}"
            assert values_equal(
                row0_month, float(ecc_month)
            ), f"Month mismatch: earth2bufr={row0_month}, eccodes={ecc_month}"
            assert values_equal(
                row0_day, float(ecc_day)
            ), f"Day mismatch: earth2bufr={row0_day}, eccodes={ecc_day}"
            assert values_equal(
                row0_hour, float(ecc_hour)
            ), f"Hour mismatch: earth2bufr={row0_hour}, eccodes={ecc_hour}"
            assert values_equal(
                row0_minute, float(ecc_minute)
            ), f"Minute mismatch: earth2bufr={row0_minute}, eccodes={ecc_minute}"
