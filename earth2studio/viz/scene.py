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

import pandas as pd
import xarray as xr

from earth2studio.viz.adapters.dataframe import DataFrameAdapter, FrameView
from earth2studio.viz.adapters.xarray import (
    RasterSequenceView,
    RasterView,
    XarrayAdapter,
)
from earth2studio.viz.assets import AssetSource, MeshSource, TextureSource
from earth2studio.viz.backends.base import RenderResult, get_backend
from earth2studio.viz.camera import Camera
from earth2studio.viz.domains import TextureDomain, default_texture_domain
from earth2studio.viz.grids import GridSpec
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
from earth2studio.viz.textures import TextureFrame, TextureSequence
from earth2studio.viz.timeline import (
    Timeline,
    infer_frames_from_dataframe,
    infer_frames_from_xarray,
)


@dataclass(frozen=True, kw_only=True)
class SceneEvent:
    """Backend-facing description of one scene or layer mutation."""

    kind: str
    scene: "Scene"
    layer: Layer | None = None
    payload: dict[str, Any] = field(default_factory=dict)


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
    _sessions: list[Any] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Connect timeline notifications to the scene event stream."""
        self.timeline._notify = self._timeline_changed

    @property
    def visible_layers(self) -> list[Layer]:
        """Return visible layers in draw order."""
        return [layer for layer in self.layers if layer.visible]

    def add_layer(self, layer: Layer) -> Layer:
        """Add an already constructed layer to the scene."""
        if any(existing.id == layer.id for existing in self.layers):
            raise ValueError(f"Layer id {layer.id!r} already exists")
        self.layers.append(layer)
        self._attach_layer(layer)
        self._emit("layer_added", layer)
        return layer

    def add_raster(
        self,
        data: xr.DataArray | xr.Dataset,
        *,
        x: str | None = None,
        y: str | None = None,
        name: str | None = None,
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> RasterLayer:
        """Add an already-selected xarray raster layer or raster time series."""
        view = XarrayAdapter(data).to_raster_layer_view(
            x=x,
            y=y,
        )
        self.timeline.add_frames(infer_frames_from_xarray(view.data))
        layer = RasterLayer(
            id=self._next_id("raster"),
            name=name or view.variable or "Raster",
            data=view,
            style=_style(style, **style_kwargs),
            projection=projection or _projection_for_grid(view.grid),
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
        texture: Any | None = None,
        name: str = "Terrain",
        vertical_exaggeration: float = 1.0,
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> TerrainLayer:
        """Add regional terrain, elevation, bathymetry, DSM, or topography."""
        view = XarrayAdapter(data).to_raster_view()
        layer = TerrainLayer(
            id=self._next_id("terrain"),
            name=name,
            data=view,
            style=_style(style, **style_kwargs),
            projection=projection
            or _projection_for_grid(
                view.grid,
                fallback_kind="local",
                fallback_crs=self.region.crs if self.region else None,
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
        name: str | None = None,
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> DrapedRasterLayer:
        """Add an already-selected raster to terrain or a local plane."""
        view = XarrayAdapter(data).to_raster_view()
        self.timeline.add_frames(infer_frames_from_xarray(view.data))
        layer = DrapedRasterLayer(
            id=self._next_id("draped"),
            name=name or view.variable or "Draped raster",
            data=view,
            style=_style(style, **style_kwargs),
            projection=projection
            or _projection_for_grid(
                view.grid,
                fallback_kind="local",
                fallback_crs=self.region.crs if self.region else None,
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
        vertical: str | None = None,
        mode: str = "slices",
        levels: Sequence[Any] | None = None,
        name: str | None = None,
        style: LayerStyle | None = None,
        projection: ProjectionSpec | None = None,
        **style_kwargs: Any,
    ) -> RegionCubeLayer:
        """Add bounded 3D regional cube data for slices or future volumes."""
        selected = _selected_dataarray(data)
        self.timeline.add_frames(infer_frames_from_xarray(selected))
        layer = RegionCubeLayer(
            id=self._next_id("cube"),
            name=name or selected.name or "Region cube",
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
        self._emit("layer_removed", layer)
        layer._scene = None
        return layer

    def render(self, backend: str = "summary", **backend_kwargs: Any) -> RenderResult:
        """Render this scene with a registered backend."""
        return get_backend(backend).render(self, **backend_kwargs)

    def show(
        self,
        backend: str = "summary",
        *,
        streaming: bool = False,
        auto_flush: bool = True,
        **backend_kwargs: Any,
    ) -> Any:
        """Show this scene using a registered backend."""
        session_or_result = get_backend(backend).show(
            self,
            streaming=streaming,
            auto_flush=auto_flush,
            **backend_kwargs,
        )
        if streaming:
            self._attach_session(session_or_result)
        return session_or_result

    def save(
        self, path: str | Path, *, backend: str = "summary", **backend_kwargs: Any
    ) -> Path:
        """Save this scene using a registered backend."""
        return get_backend(backend).save(self, path, **backend_kwargs)

    def animate(
        self, path: str | Path, *, backend: str = "summary", **backend_kwargs: Any
    ) -> Path:
        """Animate this scene using a registered backend."""
        return get_backend(backend).animate(self, path, **backend_kwargs)

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

    def _attach_layer(self, layer: Layer) -> None:
        layer._scene = self

    def _attach_session(self, session: Any) -> None:
        if not _is_scene_session(session) or session in self._sessions:
            return
        self._sessions.append(session)

    def _detach_session(self, session: Any) -> None:
        if session in self._sessions:
            self._sessions.remove(session)

    def _emit(
        self,
        kind: str,
        layer: Layer | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self._sessions:
            return
        event = SceneEvent(
            kind=kind,
            scene=self,
            layer=layer,
            payload={} if payload is None else payload,
        )
        for session in list(self._sessions):
            if getattr(session, "closed", False):
                self._detach_session(session)
                continue
            session.update(event)

    def _timeline_changed(self, kind: str, payload: dict[str, Any]) -> None:
        self._emit("timeline_changed", payload={"change": kind, **payload})

    def _set_layer_visible(self, layer: Layer, visible: bool) -> None:
        old_visible = layer.visible
        layer.visible = visible
        if old_visible != visible:
            self._emit(
                "layer_visibility_changed",
                layer,
                {"old_visible": old_visible, "visible": visible},
            )

    def _update_layer(
        self,
        layer: Layer,
        data: Any,
        *,
        time: Any | None = None,
        **metadata: Any,
    ) -> None:
        layer.data = _coerce_layer_data(layer, data)
        if metadata:
            layer.metadata.update(metadata)
        self._add_layer_frames(layer, data, time=time)
        layer.time_extent = self.timeline.range()
        self._emit(
            "layer_updated",
            layer,
            {"data_type": type(data).__name__, "time": time},
        )

    def _append_layer(
        self,
        layer: Layer,
        data: Any,
        *,
        time: Any | None = None,
        **metadata: Any,
    ) -> None:
        layer.data = _append_layer_data(layer, data, time=time)
        if metadata:
            layer.metadata.update(metadata)
        self._add_layer_frames(layer, data, time=time)
        layer.time_extent = self.timeline.range()
        self._emit(
            "layer_appended",
            layer,
            {"data_type": type(data).__name__, "time": time},
        )

    def _add_layer_frames(
        self,
        layer: Layer,
        data: Any,
        *,
        time: Any | None,
    ) -> None:
        if time is not None:
            self.timeline.add_frames([time])
            return
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            self.timeline.add_frames(infer_frames_from_xarray(data))
            return
        if isinstance(data, TextureFrame) and data.timestamp is not None:
            self.timeline.add_frames([data.timestamp])
            return
        if isinstance(layer.data, TextureSequence):
            self._add_sequence_frames(layer.data)
            return
        if hasattr(data, "columns"):
            self.timeline.add_frames(infer_frames_from_dataframe(data))

    def _add_asset_time(self, time: Any | None) -> None:
        if time is not None:
            self.timeline.add_frames([time])

    def _add_sequence_frames(self, sequence: TextureSequence) -> None:
        frames = [
            frame.timestamp for frame in sequence.frames if frame.timestamp is not None
        ]
        self.timeline.add_frames(frames)


def _is_scene_session(candidate: Any) -> bool:
    return all(
        hasattr(candidate, name)
        for name in ("update", "flush", "close", "closed", "auto_flush")
    )


def _selected_dataarray(data: xr.DataArray | xr.Dataset) -> xr.DataArray:
    if isinstance(data, xr.DataArray):
        return data
    if len(data.data_vars) != 1:
        raise ValueError(
            "Select one Dataset variable before passing xarray data to viz"
        )
    return data[next(iter(data.data_vars))]


def _coerce_layer_data(layer: Layer, data: Any) -> Any:
    if isinstance(layer, (RasterLayer, TerrainLayer, DrapedRasterLayer)):
        return _coerce_raster_data(layer, data)
    if isinstance(layer, PointLayer):
        return _coerce_point_data(layer, data)
    if isinstance(layer, ImageLayer) and isinstance(data, TextureFrame):
        return TextureSequence(frames=[data], name=layer.name)
    return data


def _append_layer_data(layer: Layer, data: Any, *, time: Any | None) -> Any:
    if isinstance(layer, (RasterLayer, TerrainLayer, DrapedRasterLayer)):
        return _append_raster_data(layer, data, time=time)
    if isinstance(layer, PointLayer):
        return _append_point_data(layer, data)
    if isinstance(layer, ImageLayer) and isinstance(layer.data, TextureSequence):
        return _append_texture_data(layer.data, data, time=time)
    return _coerce_layer_data(layer, data)


def _coerce_raster_data(
    layer: RasterLayer | TerrainLayer | DrapedRasterLayer,
    data: Any,
) -> Any:
    if isinstance(data, (RasterView, RasterSequenceView)):
        return data
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return _xarray_view_for_layer(layer, data)
    return data


def _append_raster_data(
    layer: RasterLayer | TerrainLayer | DrapedRasterLayer,
    data: Any,
    *,
    time: Any | None,
) -> Any:
    existing = layer.data
    incoming = _coerce_raster_data(layer, data)
    if not isinstance(incoming, (RasterView, RasterSequenceView)):
        return incoming
    if isinstance(existing, RasterSequenceView):
        return _append_to_raster_sequence(existing, incoming, time=time)
    if isinstance(existing, RasterView):
        return _raster_views_to_sequence(existing, incoming, time=time)
    return incoming


def _xarray_view_for_layer(
    layer: RasterLayer | TerrainLayer | DrapedRasterLayer,
    data: xr.DataArray | xr.Dataset,
) -> RasterView | RasterSequenceView:
    existing = layer.data
    x_coord = getattr(existing, "x_coord", None)
    y_coord = getattr(existing, "y_coord", None)
    attempts = (
        {"x": x_coord, "y": y_coord},
        {"x": None, "y": None},
    )
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return XarrayAdapter(data).to_raster_layer_view(**attempt)
        except (KeyError, ValueError) as exc:
            last_error = exc
    if last_error is None:
        raise ValueError("Could not adapt xarray data for raster layer")
    raise last_error


def _append_to_raster_sequence(
    existing: RasterSequenceView,
    incoming: RasterView | RasterSequenceView,
    *,
    time: Any | None,
) -> RasterSequenceView:
    if len(existing.frame_dims) != 1:
        raise ValueError(
            "RasterLayer.append supports one frame dimension; select extra "
            "dimensions before streaming new frames"
        )
    frame_dim = existing.frame_dims[0]
    if isinstance(incoming, RasterSequenceView):
        if incoming.frame_dims != existing.frame_dims:
            raise ValueError("Appended raster sequence frame dimensions must match")
        incoming_data = incoming.data
    else:
        incoming_data = _expand_frame(
            incoming.data,
            frame_dim,
            time if time is not None else _next_frame_index(existing.data, frame_dim),
        )
    data = xr.concat([existing.data, incoming_data], dim=frame_dim)
    return RasterSequenceView(
        data=data,
        y_dim=existing.y_dim,
        x_dim=existing.x_dim,
        y_coord=existing.y_coord,
        x_coord=existing.x_coord,
        frame_dims=existing.frame_dims,
        variable=existing.variable,
        device=_device_for_data(data.data),
        grid=existing.grid,
        attrs=dict(data.attrs),
        native_heatmap=existing.native_heatmap,
    )


def _raster_views_to_sequence(
    existing: RasterView,
    incoming: RasterView | RasterSequenceView,
    *,
    time: Any | None,
) -> RasterSequenceView:
    frame_dim = _raster_frame_dim(existing, incoming, time=time)
    existing_data = _expand_frame(
        existing.data,
        frame_dim,
        _frame_value(existing.data, frame_dim, fallback=0),
    )
    if isinstance(incoming, RasterSequenceView):
        if len(incoming.frame_dims) != 1 or incoming.frame_dims[0] != frame_dim:
            raise ValueError("Appended raster sequence frame dimensions must match")
        incoming_data = incoming.data
    else:
        incoming_data = _expand_frame(
            incoming.data,
            frame_dim,
            time if time is not None else _next_frame_index(existing_data, frame_dim),
        )
    data = xr.concat([existing_data, incoming_data], dim=frame_dim)
    return RasterSequenceView(
        data=data,
        y_dim=existing.y_dim,
        x_dim=existing.x_dim,
        y_coord=existing.y_coord,
        x_coord=existing.x_coord,
        frame_dims=(frame_dim,),
        variable=existing.variable,
        device=_device_for_data(data.data),
        grid=existing.grid,
        attrs=dict(data.attrs),
    )


def _raster_frame_dim(
    existing: RasterView,
    incoming: RasterView | RasterSequenceView,
    *,
    time: Any | None,
) -> str:
    if time is not None:
        return "time"
    if existing.time_coord is not None:
        return existing.time_coord
    if isinstance(incoming, RasterView) and incoming.time_coord is not None:
        return incoming.time_coord
    if isinstance(incoming, RasterSequenceView) and len(incoming.frame_dims) == 1:
        return incoming.frame_dims[0]
    return "frame"


def _expand_frame(data: xr.DataArray, dim: str, value: Any) -> xr.DataArray:
    if dim in data.dims:
        return data
    array = data
    if dim in array.coords:
        scalar_value = _frame_value(array, dim, fallback=value)
        value = scalar_value if value is None else value
        array = array.drop_vars(dim)
    return array.expand_dims({dim: [value]})


def _frame_value(data: xr.DataArray, dim: str, *, fallback: Any) -> Any:
    if dim not in data.coords:
        return fallback
    coord = data.coords[dim]
    values = coord.values
    if getattr(values, "shape", ()) == ():
        return values.item()
    if getattr(values, "size", 0):
        return values[0]
    return fallback


def _next_frame_index(data: xr.DataArray, dim: str) -> int:
    return int(data.sizes[dim]) if dim in data.sizes else 0


def _coerce_point_data(layer: PointLayer, data: Any) -> Any:
    if isinstance(data, FrameView):
        return data
    if not hasattr(data, "columns"):
        return data
    existing = layer.data
    kwargs: dict[str, Any] = {}
    if isinstance(existing, FrameView):
        kwargs = {
            "lat": existing.lat,
            "lon": existing.lon,
            "x": existing.x,
            "y": existing.y,
            "z": existing.z,
            "time": existing.time,
            "fields": existing.fields,
        }
    return DataFrameAdapter(data).to_frame_view(**kwargs)


def _append_point_data(layer: PointLayer, data: Any) -> Any:
    existing = layer.data
    incoming = _coerce_point_data(layer, data)
    if not isinstance(existing, FrameView) or not isinstance(incoming, FrameView):
        return incoming
    table = _concat_tables(existing.table, incoming.table)
    return DataFrameAdapter(table).to_frame_view(
        lat=existing.lat,
        lon=existing.lon,
        x=existing.x,
        y=existing.y,
        z=existing.z,
        time=existing.time,
        fields=existing.fields,
    )


def _append_texture_data(
    sequence: TextureSequence,
    data: Any,
    *,
    time: Any | None,
) -> TextureSequence:
    if isinstance(data, TextureSequence):
        for frame in data.frames:
            sequence.append(frame)
        return sequence
    if isinstance(data, TextureFrame):
        frame = data
    else:
        frame = TextureFrame(
            source=data,
            index=len(sequence.frames),
            timestamp=time,
        )
    sequence.append(frame)
    return sequence


def _concat_tables(left: Any, right: Any) -> Any:
    module = type(left).__module__.split(".", maxsplit=1)[0]
    if module == "cudf":
        import cudf

        return cudf.concat([left, right], ignore_index=True)
    return pd.concat([left, right], ignore_index=True)


def _device_for_data(data: Any) -> str:
    if hasattr(data, "__cuda_array_interface__"):
        return "cuda"
    module = type(data).__module__.split(".", maxsplit=1)[0]
    if module in {"cupy", "cudf"}:
        return "cuda"
    return "cpu"


def _style(style: LayerStyle | None = None, **style_kwargs: Any) -> LayerStyle:
    base = LayerStyle() if style is None else style
    return base.merged(**style_kwargs)


def _projection_for_grid(
    grid: GridSpec | None,
    *,
    fallback_kind: str = "latlon",
    fallback_crs: str | None = None,
) -> ProjectionSpec:
    if grid is None:
        return ProjectionSpec(kind=fallback_kind, crs=fallback_crs)
    return ProjectionSpec(
        kind=grid.projection,
        crs=grid.crs or fallback_crs,
        metadata={"grid": grid.as_dict()},
    )


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
