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
"""Agent-friendly summary: semantic layer objects for Earth2 Studio scenes.

Key APIs: `Layer` is the base registry item; specialized layers include
`RasterLayer`, `PointLayer`, `VectorLayer`, `TerrainLayer`,
`DrapedRasterLayer`, `ImageLayer`, `GeoTiffLayer`, `MeshLayer`,
`RegionCubeLayer`, and `VolumeLayer`. Layers carry data views plus
backend-neutral style, projection, visibility, and time metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from earth2studio.viz.styles import LayerStyle, ProjectionSpec

TimeExtent = tuple[Any, Any] | None


@dataclass(kw_only=True)
class Layer:
    """Backend-neutral visual layer registered in a `Scene`."""

    id: str
    name: str
    data: Any
    kind: str = "layer"
    visible: bool = True
    style: LayerStyle = field(default_factory=LayerStyle)
    projection: ProjectionSpec = field(default_factory=ProjectionSpec)
    time_extent: TimeExtent = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def show(self) -> "Layer":
        """Mark this layer visible and return it."""
        self.visible = True
        return self

    def hide(self) -> "Layer":
        """Mark this layer hidden and return it."""
        self.visible = False
        return self

    def summary(self) -> dict[str, Any]:
        """Return a small serializable layer summary."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "visible": self.visible,
            "style": self.style.as_dict(),
            "projection": self.projection.as_dict(),
            "time_extent": self.time_extent,
            "metadata": dict(self.metadata),
        }


@dataclass(kw_only=True)
class RasterLayer(Layer):
    """Dense 2D raster field, usually backed by an xarray view."""

    kind: str = field(default="raster", init=False)


@dataclass(kw_only=True)
class PointLayer(Layer):
    """Sparse points, observations, sensors, or stations."""

    kind: str = field(default="points", init=False)


@dataclass(kw_only=True)
class TrackLayer(Layer):
    """Time-ordered paths or grouped curves."""

    kind: str = field(default="tracks", init=False)


@dataclass(kw_only=True)
class VectorLayer(Layer):
    """Vector fields, glyphs, barbs, streamlines, or flow-derived geometry."""

    kind: str = field(default="vectors", init=False)


@dataclass(kw_only=True)
class TerrainLayer(Layer):
    """Regional elevation, bathymetry, DSM, or topography surface."""

    kind: str = field(default="terrain", init=False)


@dataclass(kw_only=True)
class DrapedRasterLayer(Layer):
    """Raster projected onto a terrain surface or regional plane."""

    kind: str = field(default="draped_raster", init=False)


@dataclass(kw_only=True)
class ImageLayer(Layer):
    """External image or image-texture asset layer."""

    kind: str = field(default="image", init=False)


@dataclass(kw_only=True)
class GeoTiffLayer(Layer):
    """GeoTIFF or Cloud Optimized GeoTIFF asset layer."""

    kind: str = field(default="geotiff", init=False)


@dataclass(kw_only=True)
class MeshLayer(Layer):
    """External or in-memory mesh asset layer for 3D backends."""

    kind: str = field(default="mesh", init=False)


@dataclass(kw_only=True)
class RegionCubeLayer(Layer):
    """Bounded local 3D cube for atmosphere, ocean, or scenario fields."""

    kind: str = field(default="region_cube", init=False)


@dataclass(kw_only=True)
class VolumeLayer(Layer):
    """Scalar or vector volume representation for 3D renderer backends."""

    kind: str = field(default="volume", init=False)
