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
"""Agent-friendly summary: backend-neutral grid descriptors for viz data.

Key APIs: `GridSpec` records how an xarray/dataframe layer is spatially indexed;
`infer_grid_spec_from_xarray` detects regular lat/lon, curvilinear lat/lon,
projected/native grids, HPX/HEALPix-style grids, Command Center diamond/GOES
projection hints, and geohash-indexed data without importing renderer packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import xarray as xr

_GEOHASH_NAMES = {"geohash", "geo_hash", "geohashes", "gh"}
_HEALPIX_NAMES = {"hpx", "healpix", "healpix_index", "nside"}
_DIAMOND_NAMES = {"diamond", "diamond_idx", "diamond_subidx"}
_GOES_NAMES = {"goes", "geostationary", "geos"}
_LAT_NAMES = {"lat", "latitude"}
_LON_NAMES = {"lon", "long", "lng", "longitude"}
_PROJECTED_X_NAMES = {"x", "easting", "projection_x_coordinate"}
_PROJECTED_Y_NAMES = {"y", "northing", "projection_y_coordinate"}


@dataclass(frozen=True, kw_only=True)
class GridSpec:
    """Backend-neutral spatial grid/indexing description."""

    kind: str
    projection: str
    crs: str | None = None
    y_dim: str | None = None
    x_dim: str | None = None
    y_coord: str | None = None
    x_coord: str | None = None
    index_coord: str | None = None
    tile_shape: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable grid summary."""
        return {
            "kind": self.kind,
            "projection": self.projection,
            "crs": self.crs,
            "y_dim": self.y_dim,
            "x_dim": self.x_dim,
            "y_coord": self.y_coord,
            "x_coord": self.x_coord,
            "index_coord": self.index_coord,
            "tile_shape": self.tile_shape,
            "metadata": dict(self.metadata),
        }


def infer_grid_spec_from_xarray(
    data: xr.DataArray,
    *,
    y_dim: str | None = None,
    x_dim: str | None = None,
    y_coord: str | None = None,
    x_coord: str | None = None,
) -> GridSpec:
    """Infer a grid descriptor from xarray dimensions, coordinates, and attrs."""
    names = _name_set(data)
    projection = _attr_text(data, "projection")
    grid = _attr_text(data, "grid") or _attr_text(data, "grid_type")
    crs = _attr_text(data, "crs") or _attr_text(data, "spatial_ref")

    if _matches(projection, _HEALPIX_NAMES) or _matches(grid, _HEALPIX_NAMES):
        return _indexed_grid("healpix", "hpx", data, _HEALPIX_NAMES, crs=crs)
    if _matches(projection, _DIAMOND_NAMES) or _matches(grid, _DIAMOND_NAMES):
        return _indexed_grid("diamond", "diamond", data, _DIAMOND_NAMES, crs=crs)
    if _matches(projection, _GOES_NAMES) or _matches(grid, _GOES_NAMES):
        return GridSpec(
            kind="goes",
            projection="goes",
            crs=crs,
            y_dim=y_dim,
            x_dim=x_dim,
            y_coord=y_coord,
            x_coord=x_coord,
            metadata={"source": "attrs"},
        )
    if names & _GEOHASH_NAMES:
        return _indexed_grid("geohash", "geohash", data, _GEOHASH_NAMES, crs=crs)

    if y_coord in _LAT_NAMES and x_coord in _LON_NAMES:
        coord_ndim = max(_coord_ndim(data, y_coord), _coord_ndim(data, x_coord))
        return GridSpec(
            kind="curvilinear_latlon" if coord_ndim == 2 else "regular_latlon",
            projection="latlon",
            crs=crs or "EPSG:4326",
            y_dim=y_dim,
            x_dim=x_dim,
            y_coord=y_coord,
            x_coord=x_coord,
        )

    if {y_dim, x_dim} & _PROJECTED_Y_NAMES or {y_dim, x_dim} & _PROJECTED_X_NAMES:
        return GridSpec(
            kind="projected",
            projection=projection or "projected",
            crs=crs,
            y_dim=y_dim,
            x_dim=x_dim,
            y_coord=y_coord,
            x_coord=x_coord,
            metadata={"grid_mapping": _attr_text(data, "grid_mapping")},
        )

    return GridSpec(
        kind="native",
        projection=projection or grid or "native",
        crs=crs,
        y_dim=y_dim,
        x_dim=x_dim,
        y_coord=y_coord,
        x_coord=x_coord,
    )


def _indexed_grid(
    kind: str,
    projection: str,
    data: xr.DataArray,
    candidates: set[str],
    *,
    crs: str | None,
) -> GridSpec:
    index_coord = _first_matching_name(data, candidates)
    return GridSpec(
        kind=kind,
        projection=projection,
        crs=crs,
        index_coord=index_coord,
        metadata={"index_family": projection},
    )


def _name_set(data: xr.DataArray) -> set[str]:
    return {str(name).lower() for name in (*data.dims, *data.coords)}


def _first_matching_name(data: xr.DataArray, candidates: set[str]) -> str | None:
    for name in (*data.dims, *data.coords):
        if str(name).lower() in candidates:
            return str(name)
    return None


def _attr_text(data: xr.DataArray, key: str) -> str | None:
    value = data.attrs.get(key)
    if value is None:
        return None
    return str(value).lower()


def _matches(value: str | None, candidates: set[str]) -> bool:
    if value is None:
        return False
    return any(candidate in value for candidate in candidates)


def _coord_ndim(data: xr.DataArray, coord: str | None) -> int:
    if coord is None or coord not in data.coords:
        return 0
    return len(data.coords[coord].dims)
