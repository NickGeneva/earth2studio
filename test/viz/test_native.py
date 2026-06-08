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
"""Agent-friendly summary: tests for native-grid heatmap visualization.

Key APIs under test: `infer_grid_spec_from_xarray`, `native_grid_heatmap`,
`can_native_heatmap`, and `XarrayAdapter.to_raster_view` for cBottle-style
HEALPix vectors and cubed-sphere/face-tiled rasters.
"""

import numpy as np
import pytest
import xarray as xr

from earth2studio import viz
from earth2studio.viz.adapters.xarray import XarrayAdapter


def test_native_heatmap_tiles_cbottle_hpx_vector() -> None:
    data = xr.DataArray(
        np.arange(48, dtype=np.float32),
        dims=("hpx",),
        coords={"hpx": np.arange(48)},
        name="tcwv",
    )

    grid = viz.infer_grid_spec_from_xarray(data)
    heatmap = viz.native_grid_heatmap(data, grid)

    assert grid.kind == "healpix"
    assert grid.tile_shape == (2, 2)
    assert grid.metadata["nside"] == 2
    assert heatmap.dims == ("native_y", "native_x")
    assert heatmap.shape == (6, 8)
    np.testing.assert_array_equal(heatmap.values[:2, :2], np.array([[0, 1], [2, 3]]))
    np.testing.assert_array_equal(
        heatmap.values[4:6, 6:8],
        np.array([[44, 45], [46, 47]]),
    )
    assert heatmap.attrs["viz_native_heatmap"]["grid"]["kind"] == "healpix"


def test_xarray_adapter_rasterizes_hpx_before_spatial_inference() -> None:
    data = xr.DataArray(
        np.arange(96, dtype=np.float32).reshape(2, 1, 48),
        dims=("time", "variable", "hpx"),
        coords={
            "time": [0, 1],
            "variable": ["tcwv"],
            "hpx": np.arange(48),
        },
        name="fields",
    )

    view = XarrayAdapter(data.sel(variable="tcwv").isel(time=1)).to_raster_view()

    assert view.grid is not None
    assert view.grid.kind == "healpix"
    assert view.shape_2d == (6, 8)
    assert view.y_dim == "native_y"
    assert view.x_dim == "native_x"
    assert view.as_2d().values[0, 0] == 48


def test_native_heatmap_tiles_face_tiled_healpix() -> None:
    data = xr.DataArray(
        np.arange(48, dtype=np.float32).reshape(12, 2, 2),
        dims=("face", "height", "width"),
        coords={"face": np.arange(12)},
        name="t2m",
    )

    grid = viz.infer_grid_spec_from_xarray(data)
    heatmap = viz.native_grid_heatmap(data, grid)

    assert grid.kind == "healpix"
    assert grid.index_coord == "face"
    assert heatmap.shape == (6, 8)
    np.testing.assert_array_equal(
        heatmap.values[2:4, 0:2],
        np.array([[16, 17], [18, 19]]),
    )


def test_native_heatmap_tiles_cubed_sphere_faces() -> None:
    data = xr.DataArray(
        np.arange(24, dtype=np.float32).reshape(6, 2, 2),
        dims=("face", "height", "width"),
        coords={"face": np.arange(6)},
        attrs={"grid": "cubed_sphere"},
        name="z500",
    )

    grid = viz.infer_grid_spec_from_xarray(data)
    heatmap = viz.native_grid_heatmap(data, grid)

    assert grid.kind == "cubed_sphere"
    assert viz.can_native_heatmap(data, grid)
    assert heatmap.shape == (4, 6)
    assert heatmap.attrs["viz_native_heatmap"]["face_layout"] == (2, 3)


def test_plot_raster_grid_accepts_native_hpx_panel() -> None:
    data = xr.DataArray(
        np.arange(48, dtype=np.float32),
        dims=("hpx",),
        coords={"hpx": np.arange(48)},
        name="tcwv",
    )

    fig = viz.plot_raster_grid([viz.raster_panel(data, title="Native HPX")])

    assert fig.axes[0].get_title() == "Native HPX"
    assert fig.axes[0].get_xlabel() == "native_x"
    assert fig.axes[0].get_ylabel() == "native_y"


def test_native_heatmap_rejects_regular_latlon_grid() -> None:
    data = xr.DataArray(
        np.ones((2, 3), dtype=np.float32),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0, 12.0]},
    )
    grid = viz.infer_grid_spec_from_xarray(
        data,
        y_dim="lat",
        x_dim="lon",
        y_coord="lat",
        x_coord="lon",
    )

    assert not viz.can_native_heatmap(data, grid)
    with pytest.raises(ValueError, match="cannot be drawn"):
        viz.native_grid_heatmap(data, grid)


def test_native_heatmap_uses_square_index_fallback_for_diamond() -> None:
    data = xr.DataArray(
        np.arange(9, dtype=np.float32),
        dims=("diamond",),
        coords={"diamond": np.arange(9)},
    )

    heatmap = viz.native_grid_heatmap(data)

    assert heatmap.shape == (3, 3)
    np.testing.assert_array_equal(heatmap.values[-1], np.array([6, 7, 8]))


def test_native_heatmap_uses_row_fallback_for_irregular_hpx_index() -> None:
    data = xr.DataArray(
        np.arange(10, dtype=np.float32),
        dims=("hpx",),
        coords={"hpx": np.arange(10)},
    )

    heatmap = viz.native_grid_heatmap(data)

    assert heatmap.shape == (1, 10)
    np.testing.assert_array_equal(heatmap.values[0], np.arange(10))


def test_native_heatmap_rejects_unselected_non_spatial_dimension() -> None:
    data = xr.DataArray(
        np.ones((2, 48), dtype=np.float32),
        dims=("sample", "hpx"),
        coords={"sample": [0, 1], "hpx": np.arange(48)},
    )

    with pytest.raises(ValueError, match="needs an explicit selection"):
        viz.native_grid_heatmap(data)


def test_can_native_heatmap_rejects_projection_metadata_without_native_shape() -> None:
    data = xr.DataArray(
        np.ones((2, 3), dtype=np.float32),
        dims=("y", "x"),
        attrs={"projection": "hpx"},
    )
    grid = viz.infer_grid_spec_from_xarray(data)

    assert grid.kind == "healpix"
    assert not viz.can_native_heatmap(data, grid)
