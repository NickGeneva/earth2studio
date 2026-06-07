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
"""Agent-friendly summary: synthetic fixtures for viz tests.

Key APIs: pytest fixtures create deterministic xarray DataArrays, Datasets,
regional cubes, terrain arrays, and pandas DataFrames without mocks.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def sample_dataarray() -> xr.DataArray:
    times = pd.date_range("2026-06-07", periods=2, freq="6h")
    lead_times = pd.to_timedelta([0, 6], unit="h")
    variables = ["t2m", "u10m"]
    lat = np.linspace(30.0, 32.0, 3)
    lon = np.linspace(-120.0, -117.0, 4)
    values = np.arange(2 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 2, 3, 4)
    return xr.DataArray(
        values,
        dims=("time", "lead_time", "variable", "lat", "lon"),
        coords={
            "time": times,
            "lead_time": lead_times,
            "variable": variables,
            "lat": lat,
            "lon": lon,
        },
        name="fields",
        attrs={"units": "K"},
    )


@pytest.fixture
def sample_dataset(sample_dataarray: xr.DataArray) -> xr.Dataset:
    t2m = sample_dataarray.sel(variable="t2m").drop_vars("variable")
    u10m = sample_dataarray.sel(variable="u10m").drop_vars("variable")
    return xr.Dataset({"t2m": t2m, "u10m": u10m, "v10m": u10m * 0.5})


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latitude": [30.0, 31.0, 32.0],
            "longitude": [-120.0, -119.0, -118.0],
            "valid_time": pd.date_range("2026-06-07", periods=3, freq="h"),
            "temperature": [290.0, 291.5, 292.0],
            "status": [1, 0, 1],
        }
    )


@pytest.fixture
def terrain_dataarray() -> xr.DataArray:
    y = np.linspace(0.0, 2_000.0, 3)
    x = np.linspace(0.0, 3_000.0, 4)
    values = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [15.0, 30.0, 45.0, 55.0],
            [20.0, 35.0, 50.0, 70.0],
        ],
        dtype=np.float32,
    )
    return xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}, name="dem")


@pytest.fixture
def cube_dataarray() -> xr.DataArray:
    time = pd.date_range("2026-06-07", periods=2, freq="h")
    z = np.array([100.0, 500.0])
    y = np.linspace(0.0, 1.0, 3)
    x = np.linspace(0.0, 1.0, 4)
    values = np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
    return xr.DataArray(
        values,
        dims=("time", "z", "y", "x"),
        coords={"time": time, "z": z, "y": y, "x": x},
        name="q850",
    )
