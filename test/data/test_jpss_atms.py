# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import shutil
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pytest

from earth2studio.data import JPSS_ATMS


# ---------------------------------------------------------------------------
# Network / slow tests
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=6, day=1, hour=12),
        [datetime(year=2024, month=6, day=1, hour=12)],
    ],
)
@pytest.mark.parametrize("variable", ["atms", ["atms"]])
def test_jpss_atms_fetch(time, variable):
    ds = JPSS_ATMS(
        satellites=["n20"],
        time_tolerance=timedelta(minutes=5),
        cache=False,
        verbose=False,
    )
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset({"atms"})
    assert "observation" in df.columns
    assert "satellite" in df.columns
    assert "sensor_index" in df.columns

    if not df.empty:
        assert df["sensor_index"].between(1, 22).all()
        assert df["lat"].between(-90, 90).all()
        assert df["lon"].between(0, 360).all()
        assert (df["observation"] > 0).all()
        assert (df["observation"] < 400).all()  # BT in K


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
def test_jpss_atms_schema_fields():
    ds = JPSS_ATMS(
        satellites=["n20"],
        time_tolerance=timedelta(minutes=5),
        cache=False,
        verbose=False,
    )
    time = datetime(2024, 6, 1, 12)

    df_full = ds(time, ["atms"], fields=None)
    assert list(df_full.columns) == ds.SCHEMA.names

    subset_fields = ["time", "lat", "lon", "observation", "variable"]
    df_subset = ds(time, ["atms"], fields=subset_fields)
    assert list(df_subset.columns) == subset_fields


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("cache", [True, False])
def test_jpss_atms_cache(cache):
    ds = JPSS_ATMS(
        satellites=["n20"],
        time_tolerance=timedelta(minutes=5),
        cache=cache,
        verbose=False,
    )
    df = ds(datetime(2024, 6, 1, 12), ["atms"])
    assert list(df.columns) == ds.SCHEMA.names
    assert pathlib.Path(ds.cache).is_dir() == cache

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Mock / offline tests (no network required)
# ---------------------------------------------------------------------------
def _make_mock_atms_batch(n_fov=4, n_channels=22):
    """Create a mock PyArrow RecordBatch mimicking earth2bufr output for ATMS.

    Without flatten, earth2bufr returns one row per subset (FOV) with replicated
    fields (like brightnessTemperature) as list<double> columns.
    """
    # BT per FOV: (n_fov, n_channels) — each FOV has a list of n_channels values
    bt = np.random.uniform(200, 300, size=(n_fov, n_channels)).astype(np.float64)

    # Build list arrays for replicated columns
    bt_list = pa.array([bt[i].tolist() for i in range(n_fov)], type=pa.list_(pa.float64()))
    cqf_list = pa.array(
        [np.zeros(n_channels, dtype=np.int64).tolist() for _ in range(n_fov)],
        type=pa.list_(pa.int64()),
    )

    batch = pa.record_batch(
        {
            "latitude": pa.array(np.linspace(30, 40, n_fov)),
            "longitude": pa.array(np.linspace(-100, -80, n_fov)),
            "fieldOfViewNumber": pa.array(
                np.arange(1, n_fov + 1, dtype=np.float64)
            ),
            "solarZenithAngle": pa.array(np.full(n_fov, 45.0)),
            "solarAzimuth": pa.array(np.full(n_fov, 180.0)),
            "satelliteZenithAngle": pa.array(np.full(n_fov, 30.0)),
            "bearingOrAzimuth": pa.array(np.full(n_fov, 90.0)),
            "brightnessTemperature": bt_list,
            "channelDataQualityFlags": cqf_list,
            "year": pa.array(np.full(n_fov, 2024, dtype=np.int64)),
            "month": pa.array(np.full(n_fov, 6, dtype=np.int64)),
            "day": pa.array(np.full(n_fov, 1, dtype=np.int64)),
            "hour": pa.array(np.full(n_fov, 12, dtype=np.int64)),
            "minute": pa.array(np.zeros(n_fov, dtype=np.int64)),
            "second": pa.array(np.zeros(n_fov, dtype=np.int64)),
            "satelliteIdentifier": pa.array(np.full(n_fov, 225, dtype=np.int64)),
        }
    )
    return batch, bt


def test_jpss_atms_call_mock(tmp_path):
    """Exercise the full __call__ path without any network access."""
    n_fov, n_channels = 4, 22
    batch, bt = _make_mock_atms_batch(n_fov, n_channels)

    # Create a fake cached BUFR file (content irrelevant, earth2bufr is mocked)
    fake_bufr = tmp_path / "fakefile.bufr"
    fake_bufr.write_bytes(b"\x00" * 32)

    # Mock _fetch_remote_file to be a no-op, _cache_path to return our fake file
    with (
        patch.object(JPSS_ATMS, "_fetch_remote_file", return_value=None),
        patch.object(
            JPSS_ATMS,
            "_cache_path",
            return_value=str(fake_bufr),
        ),
        patch.object(
            JPSS_ATMS,
            "_create_tasks",
            return_value=[
                MagicMock(
                    s3_uri="s3://fake/file.bufr",
                    datetime_min=datetime(2024, 6, 1, 11, 30),
                    datetime_max=datetime(2024, 6, 1, 12, 30),
                    satellite="n20",
                    variable="atms",
                    bufr_key="brightnessTemperature",
                    modifier=lambda x: x,
                )
            ],
        ),
        patch("earth2studio.data.jpss_atms.earth2bufr") as mock_earth2bufr,
    ):
        # earth2bufr.read_bufr returns a list of RecordBatches
        mock_earth2bufr.read_bufr.return_value = [batch]

        ds = JPSS_ATMS(satellites=["n20"], cache=False, verbose=False)
        df = ds(datetime(2024, 6, 1, 12), ["atms"])

    assert not df.empty
    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()) == {"atms"}
    assert df["sensor_index"].between(1, 22).all()
    assert len(df) == n_fov * n_channels
    assert df["satellite"].iloc[0] == "n20"
    assert "quality" in df.columns
    assert (df["quality"] == 0).all()


# ---------------------------------------------------------------------------
# Validation / error tests (no network)
# ---------------------------------------------------------------------------
@pytest.mark.timeout(15)
def test_jpss_atms_available():
    assert JPSS_ATMS.available(datetime(2024, 6, 1, 12))
    assert not JPSS_ATMS.available(datetime(2015, 1, 1))
    assert JPSS_ATMS.available(np.datetime64("2024-06-01T12:00"))
    assert not JPSS_ATMS.available(np.datetime64("2015-01-01T00:00"))


@pytest.mark.timeout(15)
def test_jpss_atms_validate_time():
    with pytest.raises(ValueError):
        ds = JPSS_ATMS(satellites=["n20"], cache=False)
        ds(datetime(2015, 1, 1), ["atms"])


@pytest.mark.timeout(15)
def test_jpss_atms_invalid_satellite():
    with pytest.raises(ValueError, match="Invalid satellite"):
        JPSS_ATMS(satellites=["invalid"])


def test_jpss_atms_exceptions():
    ds = JPSS_ATMS(satellites=["n20"], cache=False, verbose=False)

    with pytest.raises(KeyError):
        ds(datetime(2024, 6, 1, 12), ["invalid_variable"])

    with pytest.raises(KeyError):
        ds(
            datetime(2024, 6, 1, 12),
            ["atms"],
            fields=["observation", "variable", "invalid_field"],
        )

    invalid_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("nonexistent", pa.float32()),
        ]
    )
    with pytest.raises(KeyError):
        ds(datetime(2024, 6, 1, 12), ["atms"], fields=invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),
        ]
    )
    with pytest.raises(TypeError):
        ds(datetime(2024, 6, 1, 12), ["atms"], fields=wrong_type_schema)


def test_jpss_atms_tolerance_conversion():
    ds_td = JPSS_ATMS(time_tolerance=timedelta(minutes=30), cache=False, verbose=False)
    assert ds_td._tolerance_lower == timedelta(minutes=-30)
    assert ds_td._tolerance_upper == timedelta(minutes=30)

    ds_np = JPSS_ATMS(
        time_tolerance=np.timedelta64(30, "m"), cache=False, verbose=False
    )
    assert ds_np._tolerance_lower == timedelta(minutes=-30)
    assert ds_np._tolerance_upper == timedelta(minutes=30)

    ds_asym = JPSS_ATMS(
        time_tolerance=(np.timedelta64(-10, "m"), np.timedelta64(60, "m")),
        cache=False,
        verbose=False,
    )
    assert ds_asym._tolerance_lower == timedelta(minutes=-10)
    assert ds_asym._tolerance_upper == timedelta(minutes=60)


def test_jpss_atms_parse_filename():
    # Valid filename
    t = JPSS_ATMS._parse_filename_time(
        "ATMS_v1r0_j01_s20240601120000_e20240601120100_c20240601130000.bufr"
    )
    assert t == datetime(2024, 6, 1, 12, 0, 0)

    # Invalid filename
    assert JPSS_ATMS._parse_filename_time("random_file.bufr") is None

    # Truncated timestamp
    assert JPSS_ATMS._parse_filename_time("ATMS_v1r0_j01_s2024.bufr") is None
