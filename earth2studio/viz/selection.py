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
"""Agent-friendly summary: xarray selection and spatial inference helpers.

Key APIs: `select_xarray` picks variables/time/lead_time from DataArray or
Dataset inputs; `infer_spatial_reference` identifies y/x dimensions and
coordinate names for regular or 2D lat/lon grids.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

X_COORD_ALIASES = ("x", "lon", "long", "lng", "longitude")
Y_COORD_ALIASES = ("y", "lat", "latitude")


def select_xarray(
    data: xr.DataArray | xr.Dataset,
    *,
    variable: str | None = None,
    time: Any | None = None,
    lead_time: Any | None = None,
) -> xr.DataArray:
    """Select a renderable DataArray from an xarray object."""
    array = _select_variable(data, variable)
    array = _select_named_coordinate(array, "time", time)
    array = _select_named_coordinate(array, "lead_time", lead_time)
    return array


def infer_spatial_reference(
    data: xr.DataArray,
    *,
    x: str | None = None,
    y: str | None = None,
) -> tuple[str, str, str, str]:
    """Infer `(y_dim, x_dim, y_coord, x_coord)` from an xarray DataArray."""
    if x is not None and y is not None:
        y_dim = _dim_for_coord(data, y, prefer_axis=0)
        x_dim = _dim_for_coord(data, x, prefer_axis=-1)
        return y_dim, x_dim, y, x

    y_coord = _first_present(data, Y_COORD_ALIASES)
    x_coord = _first_present(data, X_COORD_ALIASES)
    if y_coord is not None and x_coord is not None:
        y_dim = _dim_for_coord(data, y_coord, prefer_axis=0)
        x_dim = _dim_for_coord(data, x_coord, prefer_axis=-1)
        return y_dim, x_dim, y_coord, x_coord

    if data.ndim < 2:
        raise ValueError("A raster needs at least two spatial dimensions")

    y_dim = data.dims[-2]
    x_dim = data.dims[-1]
    return y_dim, x_dim, y_dim, x_dim


def squeeze_non_spatial_singletons(
    data: xr.DataArray, y_dim: str, x_dim: str
) -> xr.DataArray:
    """Drop singleton non-spatial dimensions and reject unresolved dimensions."""
    array = data
    for dim in list(array.dims):
        if dim in (y_dim, x_dim):
            continue
        if array.sizes[dim] != 1:
            raise ValueError(
                f"Dimension {dim!r} has size {array.sizes[dim]} and needs an explicit selection"
            )
        array = array.isel({dim: 0})
    return array


def _select_variable(
    data: xr.DataArray | xr.Dataset, variable: str | None
) -> xr.DataArray:
    if isinstance(data, xr.Dataset):
        if variable is None:
            if len(data.data_vars) != 1:
                raise ValueError(
                    "variable must be provided for multi-variable Datasets"
                )
            variable = next(iter(data.data_vars))
        if variable not in data:
            raise KeyError(f"Variable {variable!r} was not found in Dataset")
        return data[variable]

    if variable is None:
        return data
    if "variable" in data.dims:
        variable_values = (
            data.coords["variable"].values if "variable" in data.coords else []
        )
        if variable not in set(variable_values):
            raise KeyError(f"Variable {variable!r} was not found in DataArray")
        return _select_named_coordinate(data, "variable", variable)
    if data.name == variable:
        return data
    raise KeyError(f"Variable {variable!r} was not found in DataArray")


def _select_named_coordinate(
    data: xr.DataArray,
    coord: str,
    value: Any | None,
) -> xr.DataArray:
    if value is None:
        return data
    if coord not in data.coords and coord not in data.dims:
        raise KeyError(f"Coordinate {coord!r} was not found")
    if isinstance(value, int):
        return data.isel({coord: value})
    return data.sel({coord: value})


def _first_present(data: xr.DataArray, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in data.coords:
            return alias
    for alias in aliases:
        if alias in data.dims:
            return alias
    return None


def _dim_for_coord(data: xr.DataArray, coord: str, *, prefer_axis: int) -> str:
    if coord in data.dims:
        return coord
    if coord not in data.coords:
        raise KeyError(f"Coordinate {coord!r} was not found")
    dims = data.coords[coord].dims
    if not dims:
        raise ValueError(
            f"Coordinate {coord!r} is scalar and cannot define a spatial axis"
        )
    return dims[prefer_axis]
