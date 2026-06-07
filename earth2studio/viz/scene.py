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
"""Agent-friendly summary: scene registry and user-facing layer API.

Key APIs: `Scene.add_raster`, `add_points`, `add_vectors`, `add_terrain`,
`add_draped_raster`, `add_image`, `add_default_texture`, `add_geotiff`, `add_mesh`,
`add_region_cube`, `show`, `render`, `save`, `animate`, and `summary`. Scenes
dispatch to registered backends lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from earth2studio.viz.adapters.dataframe import DataFrameAdapter
from earth2studio.viz.adapters.xarray import XarrayAdapter
from earth2studio.viz.assets import AssetSource, MeshSource, TextureSource
from earth2studio.viz.backends.base import RenderResult, get_backend
from earth2studio.viz.camera import Camera
from earth2studio.viz.domains import TextureDomain, default_texture_domain
from earth2studio.viz.layers import (
    DrapedRasterLayer,
    GeoTiffLayer,
    ImageLayer,
    Layer,
    MeshLayer,
    PointLayer,
    RasterLayer,
    RegionCubeLayer,
    TerrainLayer,
    VectorLayer,
)
from earth2studio.viz.regional import RegionSpec
from earth2studio.viz.styles import LayerStyle, ProjectionSpec
from earth2studio.viz.textures import TextureSequence
from earth2studio.viz.timeline import (
    Timeline,
    infer_frames_from_dataframe,
    infer_frames_from_xarray,
)


@dataclass
class Scene:
    """Ordered visualization scene with shared timeline and camera state."""

    title: str | None = None
    region: RegionSpec | None = None
    layers: list[Layer] = field(default_factory=list)
    timeline: Timeline = field(default_factory=Timeline)
    camera: Camera = field(default_factory=Camera)
    metadata: dict[str, Any] = field(default_factory=dict)
    _id_counter: int = 0

    @property
    def visible_layers(self) -> list[Layer]:
        """Return visible layers in draw order."""
        return [layer for layer in self.layers if layer.visible]

    def add_layer(self, layer: Layer) -> Layer:
        """Add an already constructed layer to the scene."""
        if any(existing.id == layer.id for existing in self.layers):
            raise ValueError(f"Layer id {layer.id!r} already exists")
        self.layers.append(layer)
        return layer

    def add_raster(
        self,
        data: xr.DataArray | xr.Dataset,
        *,
        variable: str | None = None,
        time: Any | None = None,
        lead_time: Any | None = None,
        x: str | None = None,
        y: str | None = None,
        name: str | None = None,
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> RasterLayer:
        """Add a dense xarray raster layer."""
        view = XarrayAdapter(data).to_raster_view(
            variable=variable,
            time=time,
            lead_time=lead_time,
            x=x,
            y=y,
        )
        self.timeline.add_frames(infer_frames_from_xarray(data))
        layer = RasterLayer(
            id=self._next_id("raster"),
            name=name or view.variable or "Raster",
            data=view,
            style=_style(style, **style_kwargs),
            projection=projection or ProjectionSpec(kind="latlon"),
            time_extent=self.timeline.range(),
        )
        return self.add_layer(layer)  # type: ignore[return-value]

    def add_points(
        self,
        table: Any,
        *,
        lat: str | None = None,
        lon: str | None = None,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        time: str | None = None,
        fields: Sequence[str] | None = None,
        name: str = "Points",
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> PointLayer:
        """Add sparse point or observation data from a dataframe."""
        view = DataFrameAdapter(table).to_frame_view(
            lat=lat,
            lon=lon,
            x=x,
            y=y,
            z=z,
            time=time,
            fields=fields,
        )
        self.timeline.add_frames(infer_frames_from_dataframe(table, time=view.time))
        layer = PointLayer(
            id=self._next_id("points"),
            name=name,
            data=view,
            style=_style(style, **style_kwargs),
            projection=projection or ProjectionSpec(kind="latlon"),
            time_extent=self.timeline.range(),
        )
        return self.add_layer(layer)  # type: ignore[return-value]

    def add_vectors(
        self,
        data: Any,
        *,
        vector: tuple[str, str] | tuple[str, str, str] | None = None,
        mode: str = "quiver",
        name: str = "Vectors",
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> VectorLayer:
        """Add vector, glyph, barb, or flow-derived layer intent."""
        payload = _vector_payload(data, vector=vector, mode=mode)
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            self.timeline.add_frames(infer_frames_from_xarray(data))
        layer = VectorLayer(
            id=self._next_id("vectors"),
            name=name,
            data=payload,
            style=_style(style, **style_kwargs),
            projection=projection or ProjectionSpec(kind="latlon"),
            time_extent=self.timeline.range(),
            metadata={"mode": mode, "vector": vector},
        )
        return self.add_layer(layer)  # type: ignore[return-value]

    def add_terrain(
        self,
        data: xr.DataArray | xr.Dataset,
        *,
        variable: str | None = None,
        texture: Any | None = None,
        name: str = "Terrain",
        vertical_exaggeration: float = 1.0,
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> TerrainLayer:
        """Add regional terrain, elevation, bathymetry, DSM, or topography."""
        view = XarrayAdapter(data).to_raster_view(variable=variable)
        layer = TerrainLayer(
            id=self._next_id("terrain"),
            name=name,
            data=view,
            style=_style(style, **style_kwargs),
            projection=projection
            or ProjectionSpec(
                kind="local", crs=self.region.crs if self.region else None
            ),
            metadata={
                "texture": texture,
                "vertical_exaggeration": vertical_exaggeration,
            },
        )
        return self.add_layer(layer)  # type: ignore[return-value]

    def add_draped_raster(
        self,
        data: xr.DataArray | xr.Dataset,
        *,
        variable: str | None = None,
        time: Any | None = None,
        lead_time: Any | None = None,
        name: str | None = None,
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> DrapedRasterLayer:
        """Add a raster intended to be projected onto terrain or a local plane."""
        view = XarrayAdapter(data).to_raster_view(
            variable=variable,
            time=time,
            lead_time=lead_time,
        )
        self.timeline.add_frames(infer_frames_from_xarray(data))
        layer = DrapedRasterLayer(
            id=self._next_id("draped"),
            name=name or view.variable or "Draped raster",
            data=view,
            style=_style(style, **style_kwargs),
            projection=projection
            or ProjectionSpec(
                kind="local", crs=self.region.crs if self.region else None
            ),
            time_extent=self.timeline.range(),
        )
        return self.add_layer(layer)  # type: ignore[return-value]

    def add_image(
        self,
        source: Any,
        *,
        name: str | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        crs: str | None = None,
        time: Any | None = None,
        mime_type: str | None = None,
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> ImageLayer:
        """Add an image, texture, or time-varying texture sequence layer."""
        if isinstance(source, TextureSequence):
            self._add_sequence_frames(source)
            metadata = {"streaming": True, "sequence": source.as_dict()}
            layer = ImageLayer(
                id=self._next_id("image"),
                name=name or source.name or "Image",
                data=source,
                style=_style(style, **style_kwargs),
                projection=projection or ProjectionSpec(kind="latlon", crs=crs),
                time_extent=source.time_extent,
                metadata=metadata,
            )
            return self.add_layer(layer)  # type: ignore[return-value]

        asset = _texture_source(
            source,
            kind="image",
            name=name,
            bounds=bounds,
            crs=crs,
            time=time,
            mime_type=mime_type,
        )
        self._add_asset_time(asset.time)
        layer = ImageLayer(
            id=self._next_id("image"),
            name=name or asset.name or "Image",
            data=asset,
            style=_style(style, **style_kwargs),
            projection=projection or ProjectionSpec(kind="latlon", crs=asset.crs),
            time_extent=_single_time_extent(asset.time),
            metadata={"streaming": True, "asset": asset.as_dict()},
        )
        return self.add_layer(layer)  # type: ignore[return-value]

    def add_default_texture(
        self,
        asset: str = "global_base_color",
        *,
        domain: TextureDomain | None = None,
        name: str | None = None,
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> ImageLayer:
        """Add a default global texture from the versioned viz cache domain."""
        texture_domain = domain or default_texture_domain()
        source = texture_domain.source(asset)
        layer = self.add_image(
            source,
            name=name or source.name,
            style=style,
            projection=projection,
            **style_kwargs,
        )
        layer.metadata["texture_domain"] = texture_domain.as_dict()
        return layer

    def add_geotiff(
        self,
        source: Any,
        *,
        role: str = "raster",
        name: str | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        crs: str | None = None,
        time: Any | None = None,
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> GeoTiffLayer:
        """Add a GeoTIFF or Cloud Optimized GeoTIFF asset layer."""
        if role not in {"raster", "terrain", "texture", "draped_raster"}:
            raise ValueError(
                "GeoTIFF role must be raster, terrain, texture, or draped_raster"
            )
        asset = _texture_source(
            source,
            kind="geotiff",
            name=name,
            bounds=bounds,
            crs=crs,
            time=time,
        )
        self._add_asset_time(asset.time)
        layer = GeoTiffLayer(
            id=self._next_id("geotiff"),
            name=name or asset.name or "GeoTIFF",
            data=asset,
            style=_style(style, **style_kwargs),
            projection=projection or ProjectionSpec(kind="asset", crs=asset.crs),
            time_extent=_single_time_extent(asset.time),
            metadata={"streaming": True, "role": role, "asset": asset.as_dict()},
        )
        return self.add_layer(layer)  # type: ignore[return-value]

    def add_mesh(
        self,
        source: Any,
        *,
        name: str | None = None,
        crs: str | None = None,
        transform: Sequence[float] | None = None,
        material: dict[str, Any] | None = None,
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> MeshLayer:
        """Add an external or in-memory 3D mesh asset layer."""
        asset = _mesh_source(
            source,
            name=name,
            crs=crs,
            transform=transform,
            material=material,
        )
        layer = MeshLayer(
            id=self._next_id("mesh"),
            name=name or asset.name or "Mesh",
            data=asset,
            style=_style(style, **style_kwargs),
            projection=projection or ProjectionSpec(kind="local", crs=asset.crs),
            metadata={"streaming": False, "asset": asset.as_dict()},
        )
        return self.add_layer(layer)  # type: ignore[return-value]

    def add_region_cube(
        self,
        data: xr.DataArray | xr.Dataset,
        *,
        variable: str | None = None,
        vertical: str | None = None,
        mode: str = "slices",
        levels: Sequence[Any] | None = None,
        name: str | None = None,
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> RegionCubeLayer:
        """Add bounded 3D regional cube data for slices or future volumes."""
        selected = data[variable] if isinstance(data, xr.Dataset) and variable else data
        self.timeline.add_frames(infer_frames_from_xarray(selected))
        layer = RegionCubeLayer(
            id=self._next_id("cube"),
            name=name or variable or "Region cube",
            data=selected,
            style=_style(style, **style_kwargs),
            projection=projection
            or ProjectionSpec(
                kind="local", crs=self.region.crs if self.region else None
            ),
            time_extent=self.timeline.range(),
            metadata={
                "vertical": vertical,
                "mode": mode,
                "levels": tuple(levels or ()),
            },
        )
        return self.add_layer(layer)  # type: ignore[return-value]

    def get_layer(self, key: str) -> Layer:
        """Return a layer by id or name."""
        for layer in self.layers:
            if layer.id == key or layer.name == key:
                return layer
        raise KeyError(f"Layer {key!r} was not found")

    def remove_layer(self, key: str) -> Layer:
        """Remove and return a layer by id or name."""
        layer = self.get_layer(key)
        self.layers.remove(layer)
        return layer

    def render(self, backend: str = "summary", **kwargs: Any) -> RenderResult:
        """Render this scene with a registered backend."""
        return get_backend(backend).render(self, **kwargs)

    def show(self, backend: str = "summary", **kwargs: Any) -> Any:
        """Show this scene using a registered backend."""
        return get_backend(backend).show(self, **kwargs)

    def save(
        self, path: str | Path, *, backend: str = "summary", **kwargs: Any
    ) -> Path:
        """Save this scene using a registered backend."""
        return get_backend(backend).save(self, path, **kwargs)

    def animate(
        self, path: str | Path, *, backend: str = "summary", **kwargs: Any
    ) -> Path:
        """Animate this scene using a registered backend."""
        return get_backend(backend).animate(self, path, **kwargs)

    def summary(self) -> dict[str, Any]:
        """Return a serializable scene summary."""
        return {
            "title": self.title,
            "region": None if self.region is None else self.region.as_dict(),
            "layers": [layer.summary() for layer in self.layers],
            "timeline": {
                "frames": list(self.timeline.frames),
                "current": self.timeline.current,
                "mode": self.timeline.mode,
            },
            "camera": self.camera.as_dict(),
            "metadata": dict(self.metadata),
        }

    def _next_id(self, prefix: str) -> str:
        self._id_counter += 1
        return f"{prefix}-{self._id_counter:03d}"

    def _add_asset_time(self, time: Any | None) -> None:
        if time is not None:
            self.timeline.add_frames([time])

    def _add_sequence_frames(self, sequence: TextureSequence) -> None:
        frames = [
            frame.timestamp for frame in sequence.frames if frame.timestamp is not None
        ]
        self.timeline.add_frames(frames)


def _style(style: LayerStyle | None = None, **style_kwargs: Any) -> LayerStyle:
    base = LayerStyle() if style is None else style
    return base.merged(**style_kwargs)


def _texture_source(
    source: Any,
    *,
    kind: str,
    name: str | None,
    bounds: tuple[float, float, float, float] | None,
    crs: str | None,
    time: Any | None,
    mime_type: str | None = None,
) -> TextureSource:
    if isinstance(source, TextureSource):
        return TextureSource(
            uri=source.uri,
            data=source.data,
            kind=kind,
            name=name or source.name,
            mime_type=mime_type or source.mime_type,
            crs=crs or source.crs,
            bounds=bounds or source.bounds,
            time=time if time is not None else source.time,
            metadata=dict(source.metadata),
            codec=source.codec,
            channels=source.channels,
            tile_size=source.tile_size,
            levels=source.levels,
        )
    if isinstance(source, AssetSource):
        return TextureSource(
            uri=source.uri,
            data=source.data,
            kind=kind,
            name=name or source.name,
            mime_type=mime_type or source.mime_type,
            crs=crs or source.crs,
            bounds=bounds or source.bounds,
            time=time if time is not None else source.time,
            metadata=dict(source.metadata),
        )
    if isinstance(source, (str, Path)):
        return TextureSource(
            uri=source,
            kind=kind,
            name=name or Path(str(source).split("?", maxsplit=1)[0]).name,
            mime_type=mime_type,
            crs=crs,
            bounds=bounds,
            time=time,
        )
    return TextureSource(
        data=source,
        kind=kind,
        name=name,
        mime_type=mime_type,
        crs=crs,
        bounds=bounds,
        time=time,
    )


def _mesh_source(
    source: Any,
    *,
    name: str | None,
    crs: str | None,
    transform: Sequence[float] | None,
    material: dict[str, Any] | None,
) -> MeshSource:
    if isinstance(source, MeshSource):
        return MeshSource(
            uri=source.uri,
            data=source.data,
            name=name or source.name,
            mime_type=source.mime_type,
            crs=crs or source.crs,
            bounds=source.bounds,
            time=source.time,
            metadata=dict(source.metadata),
            transform=tuple(transform) if transform is not None else source.transform,
            material=dict(material or source.material),
        )
    if isinstance(source, AssetSource):
        return MeshSource(
            uri=source.uri,
            data=source.data,
            name=name or source.name,
            mime_type=source.mime_type,
            crs=crs or source.crs,
            bounds=source.bounds,
            time=source.time,
            metadata=dict(source.metadata),
            transform=tuple(transform) if transform is not None else None,
            material=dict(material or {}),
        )
    if isinstance(source, (str, Path)):
        return MeshSource(
            uri=source,
            name=name or Path(str(source).split("?", maxsplit=1)[0]).name,
            crs=crs,
            transform=tuple(transform) if transform is not None else None,
            material=dict(material or {}),
        )
    return MeshSource(
        data=source,
        name=name,
        crs=crs,
        transform=tuple(transform) if transform is not None else None,
        material=dict(material or {}),
    )


def _single_time_extent(time: Any | None) -> tuple[Any, Any] | None:
    if time is None:
        return None
    return time, time


def _vector_payload(
    data: Any,
    *,
    vector: tuple[str, str] | tuple[str, str, str] | None,
    mode: str,
) -> dict[str, Any]:
    if isinstance(data, dict):
        payload = dict(data)
        payload.setdefault("mode", mode)
        return payload
    if isinstance(data, xr.Dataset) and vector is not None:
        payload = {"data": data, "mode": mode}
        payload["u"] = data[vector[0]]
        payload["v"] = data[vector[1]]
        if len(vector) == 3:
            payload["w"] = data[vector[2]]
        return payload
    return {"data": data, "vector": vector, "mode": mode}
