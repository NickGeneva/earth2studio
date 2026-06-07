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
"""Agent-friendly summary: local/regional spatial contracts for viz scenes.

Key APIs: `RegionSpec` stores CRS, bounds, vertical datum, origin, and local
frame metadata; `RegionSpec.from_lonlat_bounds` builds a regional spec with an
auto-selected UTM CRS when requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Bounds = tuple[float, float, float, float]
Origin = tuple[float, float, float]


@dataclass(frozen=True)
class RegionSpec:
    """Spatial contract for bounded regional and digital-twin style scenes."""

    name: str
    crs: str
    bounds: Bounds
    vertical_datum: str = "ellipsoid"
    z_units: str = "m"
    origin: Origin = (0.0, 0.0, 0.0)
    local_frame: str = "enu"
    source_crs: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate region geometry."""
        xmin, ymin, xmax, ymax = self.bounds
        if xmax <= xmin:
            raise ValueError("RegionSpec bounds must have xmax > xmin")
        if ymax <= ymin:
            raise ValueError("RegionSpec bounds must have ymax > ymin")
        if len(self.origin) != 3:
            raise ValueError("RegionSpec origin must contain three values")

    @classmethod
    def from_lonlat_bounds(
        cls,
        *,
        name: str,
        west: float,
        south: float,
        east: float,
        north: float,
        target_crs: str = "auto_utm",
        vertical_datum: str = "ellipsoid",
        z_units: str = "m",
        local_frame: str = "enu",
        metadata: dict[str, Any] | None = None,
    ) -> "RegionSpec":
        """Create a regional spec from lon/lat bounds.

        The bounds remain in the source CRS for this MVP. The selected `crs`
        records the target local CRS to use when an exporter or renderer performs
        projection.
        """
        if east <= west:
            raise ValueError("east must be greater than west")
        if north <= south:
            raise ValueError("north must be greater than south")
        lon_center = (west + east) / 2.0
        lat_center = (south + north) / 2.0
        crs = (
            cls._auto_utm_crs(lon_center, lat_center)
            if target_crs == "auto_utm"
            else target_crs
        )
        return cls(
            name=name,
            crs=crs,
            bounds=(west, south, east, north),
            vertical_datum=vertical_datum,
            z_units=z_units,
            origin=(lon_center, lat_center, 0.0),
            local_frame=local_frame,
            source_crs="EPSG:4326",
            metadata={} if metadata is None else dict(metadata),
        )

    @staticmethod
    def _auto_utm_crs(lon: float, lat: float) -> str:
        """Return the EPSG code for the UTM zone containing lon/lat."""
        zone = int((lon + 180.0) // 6.0) + 1
        zone = max(1, min(zone, 60))
        epsg = 32600 + zone if lat >= 0.0 else 32700 + zone
        return f"EPSG:{epsg}"

    @property
    def width(self) -> float:
        """Return the width of the region bounds."""
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> float:
        """Return the height of the region bounds."""
        return self.bounds[3] - self.bounds[1]

    def contains_xy(self, x: float, y: float) -> bool:
        """Return whether a point lies inside the region bounds."""
        xmin, ymin, xmax, ymax = self.bounds
        return xmin <= x <= xmax and ymin <= y <= ymax

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable region summary."""
        return {
            "name": self.name,
            "crs": self.crs,
            "bounds": self.bounds,
            "vertical_datum": self.vertical_datum,
            "z_units": self.z_units,
            "origin": self.origin,
            "local_frame": self.local_frame,
            "source_crs": self.source_crs,
            "metadata": dict(self.metadata),
        }
