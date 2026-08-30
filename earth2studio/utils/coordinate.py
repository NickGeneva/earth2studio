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
from copy import deepcopy
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from pyproj import CRS, Transformer

E2S_KIND = "earth2studio_kind"
E2S_SCHEMA_VERSION = "earth2studio_schema_version"
E2S_DYNAMIC_DIMS = "earth2studio_dynamic_dims"
E2S_GRID_ID = "earth2studio_grid_id"
E2S_SPATIAL_DIMS = "earth2studio_spatial_dims"
E2S_STATISTICS = "earth2studio_statistics"

_GRID_REGISTRY: dict[str, dict[str, Any]] = {
    "latlon-0.25deg": {
        "aliases": ("latlon025",),
        "crs": "EPSG:4326",
        "spatial_dims": ("lat", "lon"),
        "sizes": {"lat": 721, "lon": 1440},
        "lat_start": 90.0,
        "lat_step": -0.25,
        "lon_start": 0.0,
        "lon_step": 0.25,
        "resolution": 0.25,
        "topology": "rectilinear",
    },
    "fcn-global-0.25deg": {
        "aliases": ("fcn",),
        "crs": "EPSG:4326",
        "spatial_dims": ("lat", "lon"),
        "sizes": {"lat": 720, "lon": 1440},
        "lat_start": 90.0,
        "lat_step": -0.25,
        "lon_start": 0.0,
        "lon_step": 0.25,
        "resolution": 0.25,
        "topology": "rectilinear",
    },
    "hrrr-conus-3km": {
        "aliases": ("hrrr",),
        "crs": (
            "+proj=lcc +lon_0=262.5 +lat_0=38.5 +lat_1=38.5 "
            "+lat_2=38.5 +R=6371229 +units=m +type=crs"
        ),
        "spatial_dims": ("hrrr_y", "hrrr_x"),
        "sizes": {"hrrr_y": 1059, "hrrr_x": 1799},
        "x_start": -2697520.1425219304,
        "x_step": 3000.0,
        "y_start": -1587306.1525566636,
        "y_step": 3000.0,
        "resolution": 3000,
        "topology": "projected",
    },
    "healpix-l6-nested": {
        "aliases": ("hpx6",),
        "crs": "EPSG:4326",
        "spatial_dims": ("hpx",),
        "sizes": {"hpx": 49_152},
        "level": 6,
        "nside": 64,
        "ordering": "nested",
        "topology": "healpix",
    },
}
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
            else:
                raise TypeError("Coordinate arrays support only basic indexing")
        shape.extend(self.shape[axis:])
        return type(self)(shape, self.dtype)

    def transpose(self, axes: Sequence[int] | None = None) -> _CoordinateArray:
        axes = tuple(reversed(range(self.ndim))) if axes is None else tuple(axes)
        return type(self)(tuple(self.shape[axis] for axis in axes), self.dtype)


def _resolve_grid(grid: str) -> tuple[str, dict[str, Any]]:
    for name, spec in _GRID_REGISTRY.items():
        if grid == name or grid in spec["aliases"]:
            return name, spec
    raise ValueError(f"Unknown Earth2Studio grid '{grid}'")


def _coordinate_size(name: Hashable, value: Any) -> int | None:
    if isinstance(value, (xr.DataArray, xr.Variable)):
        return value.sizes.get(name)
    if isinstance(value, tuple) and len(value) >= 2:
        coord_dims = (value[0],) if isinstance(value[0], str) else tuple(value[0])
        if coord_dims != (name,):
            return None
        value = value[1]
    array = np.asarray(value)
    return int(array.shape[0]) if array.ndim == 1 else None


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


def _grid_metadata(array: xr.DataArray) -> dict[str, Any] | None:
    grid_id = array.attrs.get(E2S_GRID_ID)
    if grid_id is None:
        return None
    spec = _GRID_REGISTRY[grid_id]
    spatial_dims = tuple(array.attrs[E2S_SPATIAL_DIMS])
    crs = CRS.from_user_input(spec["crs"])
    metadata = {
        "id": grid_id,
        "crs": _crs_label(crs),
        "spatial_dims": spatial_dims,
        "shape": tuple(array.sizes[dim] for dim in spatial_dims),
    }
    metadata.update(
        {
            key: value
            for key, value in spec.items()
            if key not in {"aliases", "crs", "spatial_dims", "sizes"}
        }
    )
    return metadata


def _crs_label(crs: CRS) -> str:
    authority = crs.to_authority()
    if authority is not None:
        return ":".join(authority)
    if crs.coordinate_operation is not None:
        return crs.coordinate_operation.method_name
    return crs.name


def _array_crs(array: xr.DataArray) -> CRS | None:
    grid_id = array.attrs.get(E2S_GRID_ID)
    if grid_id is None:
        return None
    return CRS.from_user_input(_GRID_REGISTRY[grid_id]["crs"])


def _get_statistic(array: xr.DataArray, variable: str) -> str | None:
    metadata = array.attrs.get(E2S_STATISTICS, {})
    return metadata.get(variable, {}).get("modifier")


def _healpix_coordinates(level: int) -> tuple[np.ndarray, np.ndarray]:
    nside = 2**level
    pixels = np.arange(12 * nside**2, dtype=np.int64)
    local = pixels % nside**2
    x = np.zeros_like(local)
    y = np.zeros_like(local)
    for bit in range(level):
        x |= ((local >> (2 * bit)) & 1) << bit
        y |= ((local >> (2 * bit + 1)) & 1) << bit

    face = pixels // nside**2
    x = (x + 0.5) / nside
    y = (y + 0.5) / nside
    x_origin = np.array([1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 3])
    y_origin = np.array([1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5])
    x_rot = x_origin[face] + x
    y_rot = -y_origin[face] + y
    xs = (x_rot - y_rot - 1) * np.pi / 4
    ys = (x_rot + y_rot) * np.pi / 4

    polar = np.abs(ys) > np.pi / 4
    longitude = xs.copy()
    longitude[polar] -= (
        (np.abs(ys[polar]) - np.pi / 4)
        / (np.abs(ys[polar]) - np.pi / 2)
        * (np.mod(xs[polar], np.pi / 2) - np.pi / 4)
    )
    z = 8 * ys / (3 * np.pi)
    term = 2 - 4 * np.abs(ys[polar]) / np.pi
    z[polar] = (1 - term**2 / 3) * np.sign(ys[polar])
    latitude = 90 - np.degrees(np.arccos(np.clip(z, -1, 1)))
    return latitude, np.mod(np.degrees(longitude), 360)


def _materialize_grid_coords(array: xr.DataArray) -> xr.DataArray:
    grid_id = array.attrs.get(E2S_GRID_ID)
    if grid_id is None:
        raise ValueError("DataArray does not contain Earth2Studio grid metadata")
    spec = _GRID_REGISTRY[grid_id]
    if spec.get("topology") == "rectilinear":
        return array.assign_coords(
            lat=spec["lat_start"] + spec["lat_step"] * np.arange(spec["sizes"]["lat"]),
            lon=spec["lon_start"] + spec["lon_step"] * np.arange(spec["sizes"]["lon"]),
        )
    if grid_id == "hrrr-conus-3km":
        x = spec["x_start"] + spec["x_step"] * np.arange(spec["sizes"]["hrrr_x"])
        y = spec["y_start"] + spec["y_step"] * np.arange(spec["sizes"]["hrrr_y"])
        mesh: tuple[np.ndarray, np.ndarray] = np.meshgrid(x, y)
        xx: np.ndarray = mesh[0]
        yy: np.ndarray = mesh[1]
        longitude, latitude = Transformer.from_crs(
            CRS.from_user_input(spec["crs"]),
            CRS.from_epsg(4326),
            always_xy=True,
        ).transform(xx, yy)
        return array.assign_coords(
            hrrr_x=("hrrr_x", x),
            hrrr_y=("hrrr_y", y),
            lat=(("hrrr_y", "hrrr_x"), latitude),
            lon=(("hrrr_y", "hrrr_x"), np.mod(longitude, 360)),
        )
    latitude, longitude = _healpix_coordinates(spec["level"])
    return array.assign_coords(
        hpx=np.arange(spec["sizes"]["hpx"]),
        lat=("hpx", latitude),
        lon=("hpx", longitude),
    )


def known_grids() -> tuple[str, ...]:
    """List the built-in coordinate grids.

    Returns
    -------
    tuple[str, ...]
        Canonical grid identifiers.
    """
    return tuple(_GRID_REGISTRY)


def resolve_grid(grid: str) -> dict[str, Any]:
    """Resolve a registered grid to its definition.

    Parameters
    ----------
    grid : str
        Canonical grid identifier or alias.

    Returns
    -------
    dict[str, Any]
        Independent copy of the canonical registry entry.
    """
    grid_id, spec = _resolve_grid(grid)
    return {"id": grid_id, **deepcopy(spec)}


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
    if resolved_grid is not None:
        grid_sizes = resolved_grid[1]["sizes"]
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
        candidates = {**candidates, **grid_sizes}
    for dim in dimensions:
        coord_size = (
            _coordinate_size(dim, coordinates[dim]) if dim in coordinates else None
        )
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
        grid_id, grid_spec = resolved_grid
        array.attrs.update(
            {
                E2S_GRID_ID: grid_id,
                E2S_SPATIAL_DIMS: grid_spec["spatial_dims"],
            }
        )

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
