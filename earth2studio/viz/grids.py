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
projected/native grids, cubed-sphere/face-tiled grids, HPX/HEALPix-style grids,
Command Center diamond/GOES projection hints, and geohash-indexed data without
importing renderer packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import xarray as xr

_GEOHASH_NAMES = {"geohash", "geo_hash", "geohashes", "gh"}
_HEALPIX_NAMES = {"hpx", "healpix", "healpix_index", "nside"}
_CUBED_SPHERE_NAMES = {
    "cubed_sphere",
    "cube_sphere",
    "cubesphere",
    "cubed-sphere",
}
_FACE_NAMES = {"face", "faces"}
_FACE_Y_NAMES = {"height", "row", "rows", "tile_y", "y"}
_FACE_X_NAMES = {"width", "col", "cols", "tile_x", "x"}
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
    if names & _HEALPIX_NAMES or _contains_any(names, _HEALPIX_NAMES):
        return _indexed_grid("healpix", "hpx", data, _HEALPIX_NAMES, crs=crs)
    if _matches(projection, _CUBED_SPHERE_NAMES) or _matches(grid, _CUBED_SPHERE_NAMES):
        return _face_tiled_grid("cubed_sphere", "cubed_sphere", data, crs=crs)
    if names & _FACE_NAMES and (names & _FACE_Y_NAMES) and (names & _FACE_X_NAMES):
        face_dim = _first_matching_name(data, _FACE_NAMES)
        face_count = data.sizes.get(face_dim, 0) if face_dim is not None else 0
        if face_count == 12:
            return _face_tiled_grid("healpix", "hpx", data, crs=crs)
        return _face_tiled_grid("cubed_sphere", "cubed_sphere", data, crs=crs)
    if (
        _matches(projection, _DIAMOND_NAMES)
        or _matches(grid, _DIAMOND_NAMES)
        or names & _DIAMOND_NAMES
    ):
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
    tile_shape, metadata = _indexed_metadata(kind, data, index_coord, projection)
    return GridSpec(
        kind=kind,
        projection=projection,
        crs=crs,
        index_coord=index_coord,
        tile_shape=tile_shape,
        metadata=metadata,
    )


def _face_tiled_grid(
    kind: str,
    projection: str,
    data: xr.DataArray,
    *,
    crs: str | None,
) -> GridSpec:
    face_dim = _first_matching_name(data, _FACE_NAMES)
    y_dim = _first_matching_name(data, _FACE_Y_NAMES)
    x_dim = _first_matching_name(data, _FACE_X_NAMES)
    tile_shape = None
    if y_dim is not None and x_dim is not None:
        tile_shape = (data.sizes[y_dim], data.sizes[x_dim])
    metadata: dict[str, Any] = {"index_family": projection}
    if face_dim is not None:
        metadata["face_dim"] = face_dim
        metadata["face_count"] = data.sizes[face_dim]
    return GridSpec(
        kind=kind,
        projection=projection,
        crs=crs,
        y_dim=y_dim,
        x_dim=x_dim,
        y_coord=y_dim,
        x_coord=x_dim,
        index_coord=face_dim,
        tile_shape=tile_shape,
        metadata=metadata,
    )


def _name_set(data: xr.DataArray) -> set[str]:
    return {str(name).lower() for name in (*data.dims, *data.coords)}


def _first_matching_name(data: xr.DataArray, candidates: set[str]) -> str | None:
    for name in (*data.dims, *data.coords):
        if str(name).lower() in candidates:
            return str(name)
    return None


def _indexed_metadata(
    kind: str,
    data: xr.DataArray,
    index_coord: str | None,
    projection: str,
) -> tuple[tuple[int, int] | None, dict[str, Any]]:
    metadata: dict[str, Any] = {"index_family": projection}
    if index_coord is None or index_coord not in data.sizes:
        return None, metadata
    index_size = data.sizes[index_coord]
    metadata["index_size"] = index_size
    if kind == "healpix":
        tile_shape = _healpix_tile_shape(index_size)
        if tile_shape is not None:
            nside = tile_shape[0]
            metadata["nside"] = nside
            metadata["level"] = nside.bit_length() - 1
        return tile_shape, metadata
    return None, metadata


def _healpix_tile_shape(index_size: int) -> tuple[int, int] | None:
    if index_size <= 0 or index_size % 12:
        return None
    nside = int((index_size // 12) ** 0.5)
    if nside * nside * 12 != index_size:
        return None
    return nside, nside


def _attr_text(data: xr.DataArray, key: str) -> str | None:
    value = data.attrs.get(key)
    if value is None:
        return None
    return str(value).lower()


def _matches(value: str | None, candidates: set[str]) -> bool:
    if value is None:
        return False
    return any(candidate in value for candidate in candidates)


def _contains_any(names: set[str], candidates: set[str]) -> bool:
    return any(candidate in name for name in names for candidate in candidates)


def _coord_ndim(data: xr.DataArray, coord: str | None) -> int:
    if coord is None or coord not in data.coords:
        return 0
    return len(data.coords[coord].dims)
