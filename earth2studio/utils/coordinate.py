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

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike

from earth2studio.utils.grid import (
    E2S_GRID_ID,
    E2S_SPATIAL_DIMS,
    _materialize_grid_indexes,
    _resolve_grid,
)

E2S_KIND = "earth2studio_kind"
E2S_SCHEMA_VERSION = "earth2studio_schema_version"
E2S_DYNAMIC_DIMS = "earth2studio_dynamic_dims"
E2S_STATISTICS = "earth2studio_statistics"
_STATISTIC_METHODS = {"mean", "max", "min", "sum"}


class _CoordinateArray:
    __array_priority__ = 100

    def __init__(self, shape: Sequence[int], dtype: DTypeLike) -> None:
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        return 0

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise TypeError("Coordinate arrays do not contain field values")

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        return NotImplemented

    def __array_ufunc__(
        self, ufunc: Any, method: str, *args: Any, **kwargs: Any
    ) -> Any:
        return NotImplemented

    def __getitem__(self, key: Any) -> _CoordinateArray:
        key = getattr(key, "tuple", key)
        items = key if isinstance(key, tuple) else (key,)
        shape: list[int] = []
        axis = 0
        for item in items:
            if item is Ellipsis:
                count = self.ndim - len(items) + 1
                shape.extend(self.shape[axis : axis + count])
                axis += count
            elif item is None:
                shape.append(1)
            elif isinstance(item, slice):
                shape.append(len(range(*item.indices(self.shape[axis]))))
                axis += 1
            elif isinstance(item, (int, np.integer)):
                axis += 1
            elif isinstance(item, np.ndarray) and item.ndim == 1:
                shape.append(
                    int(np.count_nonzero(item))
                    if item.dtype == bool
                    else int(item.size)
                )
                axis += 1
            else:
                raise TypeError("Unsupported coordinate-array indexer")
        shape.extend(self.shape[axis:])
        return type(self)(shape, self.dtype)

    def transpose(self, axes: Sequence[int] | None = None) -> _CoordinateArray:
        axes = tuple(reversed(range(self.ndim))) if axes is None else tuple(axes)
        return type(self)(tuple(self.shape[axis] for axis in axes), self.dtype)


def _coordinate_sizes(coordinates: Mapping[Hashable, Any]) -> dict[Hashable, int]:
    sizes: dict[Hashable, int] = {}
    for name, value in coordinates.items():
        if isinstance(value, (xr.DataArray, xr.Variable)):
            dimensions = value.dims
            shape = value.shape
        elif isinstance(value, tuple) and len(value) >= 2:
            dimensions = (value[0],) if isinstance(value[0], str) else tuple(value[0])
            shape = np.asarray(value[1]).shape
        else:
            array = np.asarray(value)
            dimensions = (name,) if array.ndim == 1 else ()
            shape = array.shape
        if len(dimensions) != len(shape):
            raise ValueError(f"Coordinate '{name}' dimensions do not match its shape")
        for dimension, size in zip(dimensions, shape, strict=True):
            if dimension in sizes and sizes[dimension] != size:
                raise ValueError(f"Coordinates disagree on size of '{dimension}'")
            sizes[dimension] = int(size)
    return sizes


def _parse_offset(value: str) -> int:
    match = re.fullmatch(r"([+-]?)(\d+)h", value)
    if match is None:
        raise ValueError("Statistic offsets must use integer hours, such as '24h'")
    sign = -1 if match.group(1) == "-" else 1
    return sign * int(match.group(2))


def _format_offset(hours: int) -> str:
    if hours == 0:
        return "PT0S"
    sign = "-" if hours < 0 else ""
    return f"{sign}PT{abs(hours)}H"


def _parse_statistic(modifier: str) -> dict[str, str]:
    parts = modifier.split(":")
    if len(parts) not in (2, 3) or parts[0] not in _STATISTIC_METHODS:
        raise ValueError(f"Invalid statistic modifier '{modifier}'")
    if len(parts) == 2:
        window = _parse_offset(parts[1])
        if window <= 0:
            raise ValueError("Statistic windows must be positive")
        start, end = -window, 0
    else:
        start, end = _parse_offset(parts[1]), _parse_offset(parts[2])
        if end <= start:
            raise ValueError("Statistic end offset must follow its start offset")
    return {
        "modifier": modifier,
        "method": parts[0],
        "window": _format_offset(end - start),
        "start_offset": _format_offset(start),
        "end_offset": _format_offset(end),
        "closed": "left",
    }


def _get_statistic(array: xr.DataArray, variable: str) -> str | None:
    metadata = array.attrs.get(E2S_STATISTICS, {})
    return metadata.get(variable, {}).get("modifier")


def coord_array(
    dims: Sequence[Hashable],
    coords: Mapping[Hashable, Any] | None = None,
    *,
    dynamic: Sequence[Hashable] = (),
    sizes: Mapping[Hashable, int] | None = None,
    grid: str | None = None,
    statistics: Mapping[str, str] | None = None,
    dtype: DTypeLike = np.float32,
    name: Hashable | None = None,
    attrs: Mapping[Hashable, Any] | None = None,
) -> xr.DataArray:
    """Create an allocation-free coordinate DataArray.

    Parameters
    ----------
    dims : Sequence[Hashable]
        Ordered dimensions.
    coords : Mapping[Hashable, Any] | None, optional
        Fixed xarray coordinates, by default None
    dynamic : Sequence[Hashable], optional
        Zero-length wildcard dimensions, by default ()
    sizes : Mapping[Hashable, int] | None, optional
        Sizes not supplied by coordinates or a grid, by default None
    grid : str | None, optional
        Registered grid identifier, by default None
    statistics : Mapping[str, str] | None, optional
        Variable statistic modifiers, by default None
    dtype : DTypeLike, optional
        Declared field dtype, by default np.float32
    name : Hashable | None, optional
        DataArray name, by default None
    attrs : Mapping[Hashable, Any] | None, optional
        Additional attributes, by default None

    Returns
    -------
    xr.DataArray
        Coordinate signature with no allocated field values.
    """
    dimensions = tuple(dims)
    dynamic_dims = tuple(dynamic)
    coordinates = dict(coords or {})
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Dimensions must be unique")
    if not set(dynamic_dims).issubset(dimensions):
        raise ValueError("Dynamic dimensions must be present in dims")

    resolved_grid = _resolve_grid(grid) if grid is not None else None
    resolved_sizes: dict[Hashable, int] = {}
    candidates = dict(sizes or {})
    coordinate_sizes = _coordinate_sizes(coordinates)
    if resolved_grid is not None:
        grid_definition = resolved_grid[1]
        grid_sizes = dict(zip(grid_definition.dims, grid_definition.shape, strict=True))
        missing_grid_dims = set(grid_sizes) - set(dimensions)
        if missing_grid_dims:
            raise ValueError(
                f"Grid dimensions are missing from dims: {sorted(missing_grid_dims)}"
            )
        conflicting = {
            dim
            for dim, size in grid_sizes.items()
            if dim in candidates and candidates[dim] != size
        }
        if conflicting:
            raise ValueError(
                f"Declared sizes conflict with grid: {sorted(conflicting)}"
            )
        candidates.update(grid_sizes)
    for dim in dimensions:
        coord_size = coordinate_sizes.get(dim)
        size = coord_size if coord_size is not None else candidates.get(dim)
        if dim in dynamic_dims:
            if size not in (None, 0):
                raise ValueError(f"Dynamic dimension '{dim}' must have size zero")
            size = 0
        if size is None:
            raise ValueError(f"Missing size for dimension '{dim}'")
        if (
            coord_size is not None
            and dim in candidates
            and coord_size != candidates[dim]
        ):
            raise ValueError(f"Coordinate and declared size differ for '{dim}'")
        resolved_sizes[dim] = int(size)

    metadata = dict(attrs or {})
    metadata.update(
        {
            E2S_KIND: "coordinate_array",
            E2S_SCHEMA_VERSION: 1,
            E2S_DYNAMIC_DIMS: dynamic_dims,
        }
    )
    array = xr.DataArray(
        _CoordinateArray(tuple(resolved_sizes[dim] for dim in dimensions), dtype),
        dims=dimensions,
        coords=coordinates,
        name=name,
        attrs=metadata,
    )

    if resolved_grid is not None:
        grid_id, grid_definition = resolved_grid
        array.attrs.update(
            {
                E2S_GRID_ID: grid_id,
                E2S_SPATIAL_DIMS: grid_definition.dims,
            }
        )
        array = _materialize_grid_indexes(array)

    if statistics:
        if "variable" not in array.coords:
            raise ValueError("Statistics require a variable coordinate")
        variables = set(array.coords["variable"].values.tolist())
        missing = set(statistics) - variables
        if missing:
            raise ValueError(
                f"Statistics reference unknown variables: {sorted(missing)}"
            )
        array.attrs[E2S_STATISTICS] = {
            variable: _parse_statistic(modifier)
            for variable, modifier in statistics.items()
        }
    return array
