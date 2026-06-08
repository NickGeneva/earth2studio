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

Key APIs under test: legacy `select_xarray`, `infer_spatial_reference`,
selected-data `XarrayAdapter.to_raster_view`, and
`DataFrameAdapter.to_frame_view`.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from earth2studio.viz.adapters.dataframe import DataFrameAdapter
from earth2studio.viz.adapters.xarray import RasterSequenceView, XarrayAdapter
from earth2studio.viz.grids import GridSpec, infer_grid_spec_from_xarray
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
    field = sample_dataarray.sel(variable="t2m").isel(time=0, lead_time=0)
    view = XarrayAdapter(field).to_raster_view()

    assert view.shape_2d == (3, 4)
    assert view.y_dim == "lat"
    assert view.x_dim == "lon"
    assert view.y_coord == "lat"
    assert view.x_coord == "lon"
    assert view.variable == "fields"
    assert view.device == "cpu"
    assert view.grid is not None
    assert view.grid.kind == "regular_latlon"
    assert view.grid.projection == "latlon"
    assert view.as_2d().dims == ("lat", "lon")


def test_xarray_adapter_sequence_view_over_lead_time(
    sample_dataarray: xr.DataArray,
) -> None:
    field = sample_dataarray.sel(variable="t2m").isel(time=0)
    view = XarrayAdapter(field).to_raster_layer_view()

    assert isinstance(view, RasterSequenceView)
    assert view.frame_dims == ("lead_time",)
    assert view.frame_count == 2
    frames = list(view.iter_frames())
    assert frames[0][0] == "time=2026-06-07T00:00:00"
    assert frames[0][1].shape_2d == (3, 4)
    assert frames[1][0] == "time=2026-06-07T06:00:00"


def test_xarray_adapter_sequence_requires_non_time_selection() -> None:
    data = xr.DataArray(
        np.ones((2, 2, 3, 4), dtype=np.float32),
        dims=("ensemble", "lead_time", "lat", "lon"),
        coords={
            "ensemble": [0, 1],
            "lead_time": [np.timedelta64(0, "h"), np.timedelta64(6, "h")],
            "lat": [0.0, 1.0, 2.0],
            "lon": [10.0, 11.0, 12.0, 13.0],
        },
        name="tcwv",
    )

    with pytest.raises(ValueError, match="ensemble"):
        XarrayAdapter(data).to_raster_layer_view()

    view = XarrayAdapter(data.sel(ensemble=0)).to_raster_layer_view()

    assert isinstance(view, RasterSequenceView)
    assert view.frame_count == 2


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
    assert view.grid is not None
    assert view.grid.kind == "curvilinear_latlon"


def test_grid_spec_detects_hpx_diamond_goes_and_geohash() -> None:
    raster = xr.DataArray(np.ones((2, 3), dtype=np.float32), dims=("y", "x"))

    hpx = raster.assign_attrs(projection="hpx")
    diamond = raster.assign_attrs(projection="diamond")
    goes = raster.assign_attrs(projection="goes")
    geohash = xr.DataArray(
        np.ones((2,), dtype=np.float32),
        dims=("geohash",),
        coords={"geohash": ["9q8yy", "9q8yz"]},
    )

    assert infer_grid_spec_from_xarray(hpx).kind == "healpix"
    assert infer_grid_spec_from_xarray(hpx).projection == "hpx"
    assert infer_grid_spec_from_xarray(diamond).kind == "diamond"
    assert infer_grid_spec_from_xarray(goes).kind == "goes"
    assert infer_grid_spec_from_xarray(geohash).kind == "geohash"
    assert infer_grid_spec_from_xarray(geohash).index_coord == "geohash"


def test_grid_spec_detects_face_tiled_cubed_sphere() -> None:
    data = xr.DataArray(
        np.ones((6, 2, 3), dtype=np.float32),
        dims=("face", "height", "width"),
        attrs={"grid": "cubed_sphere"},
    )

    grid = infer_grid_spec_from_xarray(data)

    assert grid.kind == "cubed_sphere"
    assert grid.projection == "cubed_sphere"
    assert grid.index_coord == "face"
    assert grid.tile_shape == (2, 3)


def test_grid_spec_projected_and_serializable() -> None:
    data = xr.DataArray(
        np.ones((2, 3), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [10.0, 11.0, 12.0]},
        attrs={"crs": "EPSG:3857"},
    )
    grid = infer_grid_spec_from_xarray(
        data,
        y_dim="y",
        x_dim="x",
        y_coord="y",
        x_coord="x",
    )

    assert isinstance(grid, GridSpec)
    assert grid.kind == "projected"
    assert grid.crs == "epsg:3857"
    assert grid.as_dict()["x_dim"] == "x"


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
        XarrayAdapter(sample_dataarray.sel(variable="t2m")).to_raster_view()


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
