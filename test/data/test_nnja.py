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

import shutil
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from earth2studio.data import NNJAObsConv

pytest.importorskip("earth2bufr", reason="earth2bufr not installed")


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time",
    [datetime(year=2024, month=1, day=1, hour=0)],
)
@pytest.mark.parametrize(
    "variable, tol",
    [
        (["t"], timedelta(0)),
        (["u", "v"], timedelta(0)),
    ],
)
def test_nnja_obs_conv_fetch(time, variable, tol):
    ds = NNJAObsConv(time_tolerance=tol, cache=False, verbose=False)
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert "observation" in df.columns
    assert not df.empty


@pytest.mark.parametrize("cache", [True, False])
def test_nnja_obs_conv_cache_mock(cache, tmp_path):
    """Test NNJAObsConv cache behavior with mocked S3 fetch."""
    # Create a minimal mock DataFrame matching NNJA output schema
    mock_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:00"]),
            "pres": [85000.0, 92500.0],
            "elev": [100.0, 50.0],
            "type": [120, 120],
            "class": ["ADPUPA", "ADPUPA"],
            "lat": [40.0, 41.0],
            "lon": [250.0, 251.0],
            "station": ["72469", "72469"],
            "station_elev": [1000.0, 1000.0],
            "observation": [273.15, 280.0],
            "variable": ["t", "t"],
        }
    )

    with patch("earth2studio.data.nnja._sync_async") as mock_sync:
        mock_sync.return_value = mock_df

        ds = NNJAObsConv(time_tolerance=timedelta(0), cache=cache, verbose=False)

        # First fetch
        df = ds(datetime(2024, 1, 1, 0), ["t"])
        assert list(df.columns) == ds.SCHEMA.names

        # Second fetch (should use cache if enabled)
        df2 = ds(datetime(2024, 1, 1, 0), ["t"])
        assert list(df2.columns) == ds.SCHEMA.names

    # Clean up
    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


def test_nnja_obs_conv_exceptions():

    # Invalid source
    with pytest.raises(ValueError):
        NNJAObsConv(source="not_a_source", cache=False, verbose=False)

    # Invalid variable - test via lexicon lookup directly (avoids network)
    with pytest.raises(KeyError):
        from earth2studio.lexicon import NNJAObsConvLexicon

        NNJAObsConvLexicon["invalid_variable"]

    # Invalid fields - test via resolve_fields directly (avoids network)
    with pytest.raises(KeyError):
        NNJAObsConv.resolve_fields(["observation", "variable", "invalid_field"])

    invalid_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("nonexistent", pa.float32()),
        ]
    )
    with pytest.raises(KeyError):
        NNJAObsConv.resolve_fields(invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),
        ]
    )
    with pytest.raises(TypeError):
        NNJAObsConv.resolve_fields(wrong_type_schema)


def test_nnja_obs_conv_validate_time():
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 0)])
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 6)])
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 12)])
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 18)])

    with pytest.raises(ValueError):
        NNJAObsConv._validate_time([datetime(2024, 1, 1, 1)])

    with pytest.raises(ValueError):
        NNJAObsConv._validate_time([datetime(2024, 1, 1, 0, 30)])

    with pytest.raises(ValueError):
        NNJAObsConv._validate_time([datetime(1970, 1, 1, 0)])


def test_nnja_obs_conv_tolerance_conversion():

    ds = NNJAObsConv(time_tolerance=timedelta(hours=1), cache=False, verbose=False)
    assert ds._tolerance_lower == timedelta(hours=-1)
    assert ds._tolerance_upper == timedelta(hours=1)

    ds_np = NNJAObsConv(
        time_tolerance=np.timedelta64(2, "h"), cache=False, verbose=False
    )
    assert ds_np._tolerance_lower == timedelta(hours=-2)
    assert ds_np._tolerance_upper == timedelta(hours=2)

    ds_asym = NNJAObsConv(
        time_tolerance=(np.timedelta64(-3, "h"), np.timedelta64(1, "h")),
        cache=False,
        verbose=False,
    )
    assert ds_asym._tolerance_lower == timedelta(hours=-3)
    assert ds_asym._tolerance_upper == timedelta(hours=1)


def test_nnja_obs_conv_resolve_fields():
    schema_full = NNJAObsConv.resolve_fields(None)
    assert schema_full.names == NNJAObsConv.SCHEMA.names

    schema_subset = NNJAObsConv.resolve_fields(
        ["time", "lat", "lon", "observation", "variable"]
    )
    assert schema_subset.names == ["time", "lat", "lon", "observation", "variable"]

    schema_str = NNJAObsConv.resolve_fields("time")
    assert schema_str.names == ["time"]

    sub = pa.schema(
        [
            NNJAObsConv.SCHEMA.field("time"),
            NNJAObsConv.SCHEMA.field("observation"),
        ]
    )
    out = NNJAObsConv.resolve_fields(sub)
    assert out.names == ["time", "observation"]

    with pytest.raises(KeyError):
        NNJAObsConv.resolve_fields(["nonexistent"])

    bad_schema = pa.schema([pa.field("nonexistent", pa.float32())])
    with pytest.raises(KeyError):
        NNJAObsConv.resolve_fields(bad_schema)

    wrong_type = pa.schema([pa.field("time", pa.string())])
    with pytest.raises(TypeError):
        NNJAObsConv.resolve_fields(wrong_type)


def test_nnja_obs_conv_mock_fetch():
    """Test NNJAObsConv data processing with mocked S3 fetch."""

    # Create a minimal mock DataFrame matching NNJA output schema
    mock_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:00"]),
            "pres": [85000.0, 92500.0],
            "elev": [100.0, 50.0],
            "type": [120, 120],
            "class": ["ADPUPA", "ADPUPA"],
            "lat": [40.0, 41.0],
            "lon": [250.0, 251.0],
            "station": ["72469", "72469"],
            "station_elev": [1000.0, 1000.0],
            "observation": [273.15, 280.0],
            "variable": ["t", "t"],
        }
    )

    with patch.object(NNJAObsConv, "fetch") as mock_fetch:
        mock_fetch.return_value = mock_df

        ds = NNJAObsConv(time_tolerance=timedelta(0), cache=False, verbose=False)

        # Patch _sync_async to call the mock directly
        with patch("earth2studio.data.nnja._sync_async") as mock_sync:
            mock_sync.return_value = mock_df
            df = ds(datetime(2024, 1, 1, 0), ["t"])

    assert list(df.columns) == ds.SCHEMA.names
    assert len(df) == 2
    assert set(df["variable"].unique()) == {"t"}
    assert df["observation"].iloc[0] == pytest.approx(273.15)


def test_nnja_obs_conv_available():
    """Test NNJAObsConv.available() classmethod with both datetime types."""
    # Valid 6-hourly cycle times
    assert NNJAObsConv.available(datetime(2024, 1, 1, 0)) is True
    assert NNJAObsConv.available(datetime(2024, 1, 1, 6)) is True
    assert NNJAObsConv.available(datetime(2024, 1, 1, 12)) is True
    assert NNJAObsConv.available(datetime(2024, 1, 1, 18)) is True

    # Invalid hours
    assert NNJAObsConv.available(datetime(2024, 1, 1, 1)) is False
    assert NNJAObsConv.available(datetime(2024, 1, 1, 7)) is False

    # Before MIN_DATE
    assert NNJAObsConv.available(datetime(1970, 1, 1, 0)) is False

    # np.datetime64 input - valid
    assert NNJAObsConv.available(np.datetime64("2024-01-01T00:00:00")) is True
    assert NNJAObsConv.available(np.datetime64("2024-01-01T06:00:00")) is True

    # np.datetime64 input - invalid
    assert NNJAObsConv.available(np.datetime64("2024-01-01T01:00:00")) is False
    assert NNJAObsConv.available(np.datetime64("1970-01-01T00:00:00")) is False


def test_nnja_obs_conv_build_uris():
    """Test URI building methods for prepbufr and gpsro."""
    ds = NNJAObsConv(cache=False, verbose=False)

    # Test prepbufr URI
    cycle = datetime(2024, 1, 15, 6)
    prepbufr_uri = ds._build_prepbufr_uri(cycle)
    assert "2024" in prepbufr_uri
    assert "01" in prepbufr_uri
    assert "20240115" in prepbufr_uri
    assert "t06z" in prepbufr_uri
    assert "prepbufr" in prepbufr_uri

    # Test gpsro URI
    gpsro_uri = ds._build_gpsro_uri(cycle)
    assert "2024" in gpsro_uri
    assert "01" in gpsro_uri
    assert "20240115" in gpsro_uri
    assert "t06z" in gpsro_uri
    assert "gpsro" in gpsro_uri

    # Test backward compat alias
    assert ds._build_uri(cycle) == ds._build_prepbufr_uri(cycle)


def test_nnja_obs_conv_create_tasks():
    """Test _create_tasks method for prepbufr variables."""
    from earth2studio.data.nnja import _NNJAConvTask

    # Use zero tolerance to get exactly one task per cycle
    ds = NNJAObsConv(time_tolerance=timedelta(0), cache=False, verbose=False)

    # Test prepbufr-only variable
    tasks = ds._create_tasks([datetime(2024, 1, 1, 0)], ["t"])
    assert len(tasks) == 1
    assert isinstance(tasks[0], _NNJAConvTask)
    assert "prepbufr" in tasks[0].s3_uri
    assert tasks[0].datetime_file == datetime(2024, 1, 1, 0)

    # Test multiple variables (same route)
    tasks_multi = ds._create_tasks([datetime(2024, 1, 1, 0)], ["t", "u", "v"])
    assert len(tasks_multi) == 1  # Same file, combined var_plan
    assert "t" in tasks_multi[0].var_plan
    assert "u" in tasks_multi[0].var_plan
    assert "v" in tasks_multi[0].var_plan

    # Test multiple times
    tasks_times = ds._create_tasks(
        [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 6)], ["t"]
    )
    assert len(tasks_times) == 2  # Two different cycles


def test_nnja_flat_batch_to_nnja():
    """Test the PyArrow-based wide-to-long transformation."""
    from earth2studio.data.nnja import _flat_batch_to_nnja
    from earth2studio.lexicon import NNJAObsConvLexicon

    cycle_time = datetime(2024, 1, 1, 0)
    dt_min = datetime(2024, 1, 1, 0)
    dt_max = datetime(2024, 1, 1, 0)

    # Get the modifier for temperature
    _, modifier_t = NNJAObsConvLexicon["t"]
    _, modifier_u = NNJAObsConvLexicon["u"]

    var_plan = {
        "t": ("TOB", modifier_t),
        "u": ("wind::u", modifier_u),
    }

    # Create a wide-format table similar to earth2bufr recursive-flattened output
    wide_table = pa.table(
        {
            "_data_category": pa.array([109, 109, 102], type=pa.int64()),
            "SID": pa.array(["72469", "72469", "72451"], type=pa.utf8()),
            "XOB": pa.array([250.0, 251.0, 260.0], type=pa.float64()),
            "YOB": pa.array([40.0, 41.0, 35.0], type=pa.float64()),
            "DHR": pa.array([0.0, 0.0, 0.0], type=pa.float64()),
            "ELV": pa.array([1000.0, 500.0, 200.0], type=pa.float64()),
            "TYP": pa.array([120, 120, 220], type=pa.float64()),
            "POB": pa.array([850.0, 925.0, 700.0], type=pa.float64()),
            "TOB": pa.array([15.0, 20.0, None], type=pa.float64()),
            "UOB": pa.array([5.0, None, 10.0], type=pa.float64()),
            "VOB": pa.array([3.0, None, -2.0], type=pa.float64()),
        }
    )

    result = _flat_batch_to_nnja(
        wide_table,
        cycle_time=cycle_time,
        dt_min=dt_min,
        dt_max=dt_max,
        var_plan=var_plan,
    )

    assert result is not None
    assert result.num_rows > 0
    # Should have rows for "t" (2 non-null TOB) and "u" (2 non-null UOB)
    assert set(result.column("variable").to_pylist()) == {"t", "u"}

    # Temperature rows
    t_mask = pc.equal(result.column("variable"), pa.scalar("t"))
    t_rows = result.filter(t_mask)
    assert t_rows.num_rows == 2
    # Observations should be raw (before modifier)
    assert t_rows.column("observation")[0].as_py() == pytest.approx(15.0)

    # Wind u rows
    u_mask = pc.equal(result.column("variable"), pa.scalar("u"))
    u_rows = result.filter(u_mask)
    assert u_rows.num_rows == 2


def test_nnja_apply_modifiers_arrow():
    """Test the PyArrow-based unit conversion."""
    from earth2studio.data.nnja import _NNJA_CONV_SCHEMA, _apply_modifiers_arrow
    from earth2studio.lexicon import NNJAObsConvLexicon

    _, modifier_t = NNJAObsConvLexicon["t"]
    _, modifier_q = NNJAObsConvLexicon["q"]
    _, modifier_pres = NNJAObsConvLexicon["pres"]

    var_plan = {
        "t": ("TOB", modifier_t),
        "q": ("QOB", modifier_q),
        "pres": ("POB", modifier_pres),
    }

    # Create a table with raw values
    table = pa.table(
        {
            "time": pa.array(
                [np.datetime64("2024-01-01", "us")] * 3,
                type=pa.timestamp("us"),
            ),
            "pres": pa.array([850.0, 925.0, 700.0], type=pa.float32()),
            "elev": pa.array([None, None, None], type=pa.float32()),
            "type": pa.array([120, 120, 120], type=pa.uint16()),
            "class": pa.array(["ADPSFC", "ADPSFC", "ADPSFC"], type=pa.utf8()),
            "lat": pa.array([40.0, 41.0, 42.0], type=pa.float32()),
            "lon": pa.array([250.0, 251.0, 252.0], type=pa.float32()),
            "station": pa.array(["72469", "72469", "72469"], type=pa.utf8()),
            "station_elev": pa.array([1000.0, 500.0, 200.0], type=pa.float32()),
            "observation": pa.array([15.0, 5000.0, 850.0], type=pa.float32()),
            "variable": pa.array(["t", "q", "pres"], type=pa.utf8()),
        },
        schema=_NNJA_CONV_SCHEMA,
    )

    result = _apply_modifiers_arrow(table, var_plan, convert_pres_mb_to_pa=True)

    obs = result.column("observation").to_pylist()
    # t: 15°C → 288.15K
    assert obs[0] == pytest.approx(288.15, rel=1e-4)
    # q: 5000 mg/kg → 0.005 kg/kg
    assert obs[1] == pytest.approx(0.005, rel=1e-4)
    # pres (observation): 850 MB → 85000 Pa
    assert obs[2] == pytest.approx(85000.0, rel=1e-4)

    # pres column (level pressure): 850 MB → 85000 Pa
    pres_vals = result.column("pres").to_pylist()
    assert pres_vals[0] == pytest.approx(85000.0, rel=1e-4)
    assert pres_vals[1] == pytest.approx(92500.0, rel=1e-4)
    assert pres_vals[2] == pytest.approx(70000.0, rel=1e-4)


def test_nnja_obs_conv_finalize_empty():
    """Test that _decode_prepbufr_file handles empty data gracefully."""
    from unittest.mock import patch as _patch

    from earth2studio.data.nnja import _NNJAConvTask
    from earth2studio.lexicon import NNJAObsConvLexicon

    ds = NNJAObsConv(cache=False, verbose=False)
    _, modifier_t = NNJAObsConvLexicon["t"]

    task = _NNJAConvTask(
        s3_uri="s3://test/file",
        datetime_file=datetime(2024, 1, 1, 0),
        datetime_min=datetime(2024, 1, 1, 0),
        datetime_max=datetime(2024, 1, 1, 0),
        var_plan={"t": ("TOB", modifier_t)},
    )

    # Mock earth2bufr to return empty batches
    with _patch("earth2bufr.read_prepbufr", return_value=[]):
        result = ds._decode_prepbufr_file("/fake/path", task)
        assert result is None
