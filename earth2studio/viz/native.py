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
"""Agent-friendly summary: native-grid heatmap rasterization helpers.

Key APIs: `supports_native_heatmap` and `can_native_heatmap` identify grid
descriptors that can be rendered as 2D native heatmaps; `native_grid_heatmap`
converts HEALPix vectors, HEALPix PAD_XY/face-tiled arrays, cubed-sphere arrays,
and diamond face stacks into xarray rasters with `native_y`/`native_x`
coordinates for quick plotting.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import xarray as xr

from earth2studio.viz.grids import GridSpec, infer_grid_spec_from_xarray

_NATIVE_HEATMAP_KINDS = {"healpix", "cubed_sphere", "diamond"}
_FACE_DIM_ALIASES = ("face", "faces", "diamond", "tile")
_Y_DIM_ALIASES = ("height", "row", "rows", "tile_y", "y")
_X_DIM_ALIASES = ("width", "col", "cols", "tile_x", "x")


def supports_native_heatmap(grid: GridSpec | None) -> bool:
    """Return whether a grid can be drawn as a native heatmap mosaic."""
    return grid is not None and grid.kind in _NATIVE_HEATMAP_KINDS


def can_native_heatmap(data: xr.DataArray, grid: GridSpec | None = None) -> bool:
    """Return whether data has a native indexed or face-tiled heatmap shape."""
    grid = grid or infer_grid_spec_from_xarray(data)
    if not supports_native_heatmap(grid):
        return False
    if grid.index_coord is not None and grid.index_coord in data.dims:
        return len(_spatial_dims(data, grid)) == 1 or (
            (grid.y_dim in data.dims) and (grid.x_dim in data.dims)
        )
    return (
        _first_dim(data, _FACE_DIM_ALIASES) is not None
        and _first_dim(data, _Y_DIM_ALIASES) is not None
        and _first_dim(data, _X_DIM_ALIASES) is not None
    )


def native_grid_heatmap(
    data: xr.DataArray,
    grid: GridSpec | None = None,
) -> xr.DataArray:
    """Convert a native indexed or face-tiled grid into a 2D heatmap raster."""
    grid = grid or infer_grid_spec_from_xarray(data)
    if not supports_native_heatmap(grid):
        raise ValueError(f"Grid kind {grid.kind!r} cannot be drawn as a native heatmap")

    if grid.index_coord is not None and grid.index_coord in data.dims:
        if len(_spatial_dims(data, grid)) == 1:
            return _indexed_heatmap(data, grid)

    return _face_tiled_heatmap(data, grid)


def _indexed_heatmap(data: xr.DataArray, grid: GridSpec) -> xr.DataArray:
    index_dim = grid.index_coord
    if index_dim is None:
        raise ValueError("Indexed native heatmaps require an index coordinate")

    array = _squeeze_to_dims(data, (index_dim,))
    values = _as_numpy(array.transpose(index_dim).data)
    if values.ndim != 1:
        raise ValueError("Indexed native heatmaps require one spatial index dimension")

    face_count, tile_y, tile_x = _indexed_tile_shape(values.size, grid)
    if face_count > 1:
        tiled = values.reshape(face_count, tile_y, tile_x)
        return _mosaic_faces(
            tiled,
            grid,
            name=data.name,
            attrs=data.attrs,
            source_dims=(index_dim,),
        )

    return _heatmap_array(
        values.reshape(tile_y, tile_x),
        grid,
        name=data.name,
        attrs=data.attrs,
        source_dims=(index_dim,),
    )


def _face_tiled_heatmap(data: xr.DataArray, grid: GridSpec) -> xr.DataArray:
    face_dim = grid.index_coord or _first_dim(data, _FACE_DIM_ALIASES)
    y_dim = grid.y_dim or _first_dim(data, _Y_DIM_ALIASES)
    x_dim = grid.x_dim or _first_dim(data, _X_DIM_ALIASES)
    if face_dim is None or y_dim is None or x_dim is None:
        raise ValueError("Face-tiled native heatmaps require face, y, and x dimensions")

    array = _squeeze_to_dims(data, (face_dim, y_dim, x_dim))
    values = _as_numpy(array.transpose(face_dim, y_dim, x_dim).data)
    if values.ndim != 3:
        raise ValueError("Face-tiled native heatmaps require three spatial dimensions")

    return _mosaic_faces(
        values,
        grid,
        name=data.name,
        attrs=data.attrs,
        source_dims=(face_dim, y_dim, x_dim),
    )


def _mosaic_faces(
    values: np.ndarray,
    grid: GridSpec,
    *,
    name: str | None,
    attrs: Any,
    source_dims: tuple[str, ...],
) -> xr.DataArray:
    face_count, tile_y, tile_x = values.shape
    rows, cols = _tile_layout(face_count)
    dtype = np.result_type(values.dtype, np.float32)
    mosaic = np.full((rows * tile_y, cols * tile_x), np.nan, dtype=dtype)
    for face in range(face_count):
        row = face // cols
        col = face % cols
        y_slice = slice(row * tile_y, (row + 1) * tile_y)
        x_slice = slice(col * tile_x, (col + 1) * tile_x)
        mosaic[y_slice, x_slice] = values[face]
    return _heatmap_array(
        mosaic,
        grid,
        name=name,
        attrs=attrs,
        source_dims=source_dims,
        face_layout=(rows, cols),
    )


def _heatmap_array(
    values: np.ndarray,
    grid: GridSpec,
    *,
    name: str | None,
    attrs: Any,
    source_dims: tuple[str, ...],
    face_layout: tuple[int, int] | None = None,
) -> xr.DataArray:
    metadata = {
        "grid": grid.as_dict(),
        "source_dims": source_dims,
    }
    if face_layout is not None:
        metadata["face_layout"] = face_layout

    array_attrs = dict(attrs)
    array_attrs["viz_native_heatmap"] = metadata
    return xr.DataArray(
        values,
        dims=("native_y", "native_x"),
        coords={
            "native_y": np.arange(values.shape[0]),
            "native_x": np.arange(values.shape[1]),
        },
        name=name,
        attrs=array_attrs,
    )


def _indexed_tile_shape(index_size: int, grid: GridSpec) -> tuple[int, int, int]:
    if grid.kind == "healpix":
        tile_shape = grid.tile_shape or _healpix_tile_shape(index_size)
        if tile_shape is not None:
            return 12, tile_shape[0], tile_shape[1]
    side = int(math.sqrt(index_size))
    if side * side == index_size:
        return 1, side, side
    return 1, 1, index_size


def _healpix_tile_shape(index_size: int) -> tuple[int, int] | None:
    if index_size <= 0 or index_size % 12:
        return None
    nside = int(math.sqrt(index_size // 12))
    if 12 * nside * nside != index_size:
        return None
    return nside, nside


def _tile_layout(face_count: int) -> tuple[int, int]:
    if face_count == 12:
        return 3, 4
    if face_count == 10:
        return 2, 5
    if face_count == 6:
        return 2, 3
    cols = int(math.ceil(math.sqrt(face_count)))
    rows = int(math.ceil(face_count / cols))
    return rows, cols


def _squeeze_to_dims(data: xr.DataArray, spatial_dims: tuple[str, ...]) -> xr.DataArray:
    array = data
    for dim in list(array.dims):
        if dim in spatial_dims:
            continue
        if array.sizes[dim] != 1:
            raise ValueError(
                f"Dimension {dim!r} has size {array.sizes[dim]} and needs an explicit selection"
            )
        array = array.isel({dim: 0})
    return array


def _spatial_dims(data: xr.DataArray, grid: GridSpec) -> tuple[str, ...]:
    dims = tuple(
        dim
        for dim in (grid.index_coord, grid.y_dim, grid.x_dim)
        if dim is not None and dim in data.dims
    )
    return dims


def _first_dim(data: xr.DataArray, aliases: tuple[str, ...]) -> str | None:
    lower_to_dim = {str(dim).lower(): str(dim) for dim in data.dims}
    for alias in aliases:
        if alias in lower_to_dim:
            return lower_to_dim[alias]
    return None


def _as_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        values = values.detach().cpu().numpy()
    elif hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)
