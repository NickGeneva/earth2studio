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
"""Agent-friendly summary: tests for xarray and dataframe viz adapters.

Key APIs under test: `select_xarray`, `infer_spatial_reference`,
`XarrayAdapter.to_raster_view`, and `DataFrameAdapter.to_frame_view`.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from earth2studio.viz.adapters.dataframe import DataFrameAdapter
from earth2studio.viz.adapters.xarray import XarrayAdapter
from earth2studio.viz.selection import infer_spatial_reference, select_xarray


def test_select_xarray_from_dataarray(sample_dataarray: xr.DataArray) -> None:
    selected = select_xarray(
        sample_dataarray,
        variable="t2m",
        time=0,
        lead_time=1,
    )

    assert selected.dims == ("lat", "lon")
    assert selected.name == "fields"
    assert selected.attrs["units"] == "K"


def test_select_xarray_from_dataset(sample_dataset: xr.Dataset) -> None:
    selected = select_xarray(sample_dataset, variable="t2m", time=0, lead_time=0)

    assert selected.dims == ("lat", "lon")
    assert selected.name == "t2m"


def test_select_xarray_single_variable_dataset(sample_dataset: xr.Dataset) -> None:
    selected = select_xarray(sample_dataset[["t2m"]], time=0, lead_time=0)

    assert selected.name == "t2m"


def test_select_xarray_requires_dataset_variable(sample_dataset: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="variable must be provided"):
        select_xarray(sample_dataset)


def test_select_xarray_rejects_missing_dataset_variable(
    sample_dataset: xr.Dataset,
) -> None:
    with pytest.raises(KeyError, match="not found in Dataset"):
        select_xarray(sample_dataset, variable="q999")


def test_select_xarray_rejects_missing_variable(sample_dataarray: xr.DataArray) -> None:
    with pytest.raises(KeyError, match="not found"):
        select_xarray(sample_dataarray, variable="q999")


def test_select_xarray_named_dataarray(sample_dataarray: xr.DataArray) -> None:
    selected = sample_dataarray.sel(variable="t2m").drop_vars("variable")

    assert select_xarray(selected, variable="fields") is selected


def test_xarray_adapter_regular_grid(sample_dataarray: xr.DataArray) -> None:
    view = XarrayAdapter(sample_dataarray).to_raster_view(
        variable="t2m",
        time=0,
        lead_time=0,
    )

    assert view.shape_2d == (3, 4)
    assert view.y_dim == "lat"
    assert view.x_dim == "lon"
    assert view.y_coord == "lat"
    assert view.x_coord == "lon"
    assert view.variable == "t2m"
    assert view.device == "cpu"
    assert view.as_2d().dims == ("lat", "lon")


def test_xarray_adapter_curvilinear_grid() -> None:
    lon2d, lat2d = np.meshgrid(np.linspace(-2.0, 2.0, 3), np.linspace(40.0, 42.0, 2))
    data = xr.DataArray(
        np.ones((2, 3), dtype=np.float32),
        dims=("y", "x"),
        coords={"lat": (("y", "x"), lat2d), "lon": (("y", "x"), lon2d)},
    )

    assert infer_spatial_reference(data) == ("y", "x", "lat", "lon")
    view = XarrayAdapter(data).to_raster_view()
    assert view.shape_2d == (2, 3)


def test_infer_spatial_reference_explicit_xy(terrain_dataarray: xr.DataArray) -> None:
    assert infer_spatial_reference(terrain_dataarray, x="x", y="y") == (
        "y",
        "x",
        "y",
        "x",
    )


def test_infer_spatial_reference_falls_back_to_last_dims() -> None:
    data = xr.DataArray(np.ones((2, 3), dtype=np.float32), dims=("row", "col"))

    assert infer_spatial_reference(data) == ("row", "col", "row", "col")


def test_infer_spatial_reference_rejects_1d_raster() -> None:
    data = xr.DataArray(np.ones((3,), dtype=np.float32), dims=("sample",))

    with pytest.raises(ValueError, match="at least two"):
        infer_spatial_reference(data)


def test_infer_spatial_reference_rejects_missing_explicit_coord(
    terrain_dataarray: xr.DataArray,
) -> None:
    with pytest.raises(KeyError, match="not found"):
        infer_spatial_reference(terrain_dataarray, x="missing", y="y")


def test_infer_spatial_reference_rejects_scalar_coord() -> None:
    data = xr.DataArray(
        np.ones((2, 3), dtype=np.float32),
        dims=("y", "x"),
        coords={"height": 10.0},
    )

    with pytest.raises(ValueError, match="scalar"):
        infer_spatial_reference(data, x="x", y="height")


def test_xarray_adapter_rejects_unselected_dimension(
    sample_dataarray: xr.DataArray,
) -> None:
    with pytest.raises(ValueError, match="needs an explicit selection"):
        XarrayAdapter(sample_dataarray).to_raster_view(variable="t2m")


def test_dataframe_adapter_infers_lon_lat(sample_frame: pd.DataFrame) -> None:
    view = DataFrameAdapter(sample_frame).to_frame_view()

    assert view.size == 3
    assert view.x == "longitude"
    assert view.y == "latitude"
    assert view.lon == "longitude"
    assert view.lat == "latitude"
    assert view.time == "valid_time"
    assert view.fields == ("temperature", "status")
    assert view.device == "cpu"


def test_dataframe_adapter_accepts_explicit_xy() -> None:
    frame = pd.DataFrame({"x": [0.0], "y": [1.0], "value": [2.0]})

    view = DataFrameAdapter(frame).to_frame_view(fields=["value"])

    assert view.x == "x"
    assert view.y == "y"
    assert view.fields == ("value",)


def test_dataframe_adapter_explicit_fields_and_time(sample_frame: pd.DataFrame) -> None:
    view = DataFrameAdapter(sample_frame).to_frame_view(
        lat="latitude",
        lon="longitude",
        time="valid_time",
        fields=("temperature",),
    )

    assert view.fields == ("temperature",)


def test_dataframe_adapter_rejects_missing_columns(sample_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="Columns not found"):
        DataFrameAdapter(sample_frame).to_frame_view(lat="missing")


def test_dataframe_adapter_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas or cuDF-like"):
        DataFrameAdapter({"lat": [1.0]})
