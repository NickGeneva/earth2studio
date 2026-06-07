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
"""Agent-friendly summary: Cartopy static map backend.

Key APIs: `CartopyBackend.render` draws raster time-series grids, raster
comparison rows, and point layers on Cartopy projections described by existing
`ProjectionSpec` objects. Projection-specific details stay in layer metadata so
examples can keep simple `Scene.add_raster` and `Scene.add_points` calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from earth2studio.viz.adapters.dataframe import FrameView
from earth2studio.viz.adapters.xarray import RasterSequenceView, RasterView
from earth2studio.viz.backends.base import (
    BackendCapabilities,
    RenderResult,
    VizDependencyError,
)
from earth2studio.viz.backends.matplotlib import _color_norm, _pyplot, _remap_values
from earth2studio.viz.layers import (
    DrapedRasterLayer,
    PointLayer,
    RasterLayer,
    TerrainLayer,
)
from earth2studio.viz.styles import LayerStyle, ProjectionSpec


class CartopyBackend:
    """Static 2D geospatial plotting backend using Cartopy projections."""

    name = "cartopy"
    capabilities = BackendCapabilities(
        raster=True,
        points=True,
        vectors=False,
        terrain=True,
        images=False,
        meshes=False,
        volumes=False,
        texture_streaming=False,
        animation=False,
        interactive=False,
        export=True,
        metadata={"projections": True, "coastlines": True},
    )

    def supports(self, scene: Any) -> bool:
        """Return whether all visible layer kinds have basic Cartopy support."""
        supported = {"raster", "points", "terrain", "draped_raster"}
        return all(layer.kind in supported for layer in scene.visible_layers)

    def render(self, scene: Any, **kwargs: Any) -> RenderResult:
        """Render visible scene layers to a Cartopy-backed Matplotlib figure."""
        ccrs, _ = _cartopy()
        plt = _pyplot()
        layers = _drawable_layers(scene)
        frame_sets = [_layer_frames(layer) for layer in layers]
        nrows = max(len(layers), 1)
        ncols = max((len(frames) for frames in frame_sets), default=1)
        projection = _cartopy_projection(
            _scene_projection(scene, layers, kwargs),
            ccrs,
        )
        fig, axes = plt.subplots(
            nrows,
            ncols,
            subplot_kw={"projection": projection},
            figsize=kwargs.get("figsize") or (4.8 * ncols, 3.4 * nrows),
            squeeze=False,
            tight_layout=True,
        )
        title = kwargs.get("title", scene.title)
        if title:
            fig.suptitle(title)
        for row, (layer, frames) in enumerate(zip(layers, frame_sets)):
            for col in range(ncols):
                ax = axes[row][col]
                if col >= len(frames):
                    ax.set_axis_off()
                    continue
                label, payload = frames[col]
                artist = None
                can_colorbar = False
                if isinstance(payload, RasterView):
                    artist = _draw_raster(ax, payload, layer.style, ccrs)
                    can_colorbar = True
                elif isinstance(payload, FrameView):
                    artist = _draw_points(ax, payload, layer.style, ccrs)
                _decorate_axis(ax, layer.projection)
                ax.set_title(_subplot_title(layer, label))
                if (
                    can_colorbar
                    and artist is not None
                    and kwargs.get("colorbar", False)
                ):
                    fig.colorbar(artist, ax=ax, shrink=0.74, pad=0.04)
        return RenderResult(backend=self.name, output=fig, metadata={"axes": axes})

    def show(self, scene: Any, **kwargs: Any) -> Any:
        """Render and return the Matplotlib figure."""
        return self.render(scene, **kwargs).output

    def save(self, scene: Any, path: str | Path, **kwargs: Any) -> Path:
        """Save the Cartopy-backed Matplotlib figure to disk."""
        result = self.render(scene, **kwargs)
        output_path = Path(path)
        result.output.savefig(output_path)
        return output_path

    def animate(self, scene: Any, path: str | Path, **kwargs: Any) -> Path:
        """Animation is intentionally deferred for this initial backend."""
        raise NotImplementedError("Cartopy animation is not implemented yet")


def _cartopy() -> tuple[Any, Any]:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise VizDependencyError("cartopy", "cartopy") from exc
    return ccrs, cfeature


def _drawable_layers(scene: Any) -> list[Any]:
    return [
        layer
        for layer in scene.visible_layers
        if isinstance(layer, (RasterLayer, TerrainLayer, DrapedRasterLayer, PointLayer))
    ]


def _layer_frames(layer: Any) -> list[tuple[str | None, RasterView | FrameView]]:
    data = layer.data
    if isinstance(data, RasterSequenceView):
        return list(data.iter_frames())
    if isinstance(data, RasterView):
        return [(None, data)]
    if isinstance(data, FrameView):
        return [(None, data)]
    return []


def _scene_projection(
    scene: Any,
    layers: list[Any],
    kwargs: dict[str, Any],
) -> ProjectionSpec:
    projection = kwargs.get("projection")
    if isinstance(projection, ProjectionSpec):
        return projection
    if isinstance(projection, str):
        return ProjectionSpec(kind=projection)
    for layer in layers:
        if layer.projection.kind not in {"latlon", "regular_latlon"}:
            return layer.projection
        cartopy_kind = layer.projection.metadata.get("cartopy_projection")
        if cartopy_kind:
            return ProjectionSpec(
                kind=cartopy_kind, metadata=dict(layer.projection.metadata)
            )
    return ProjectionSpec(kind="platecarree")


def _cartopy_projection(spec: ProjectionSpec, ccrs: Any) -> Any:
    metadata = dict(spec.metadata)
    kind = metadata.get("cartopy_projection", spec.kind).lower().replace("-", "_")
    if kind in {"latlon", "regular_latlon", "platecarree", "plate_carree"}:
        return ccrs.PlateCarree(
            central_longitude=metadata.get("central_longitude", 0.0)
        )
    if kind == "robinson":
        return ccrs.Robinson(central_longitude=metadata.get("central_longitude", 0.0))
    if kind == "mollweide":
        return ccrs.Mollweide(central_longitude=metadata.get("central_longitude", 0.0))
    if kind == "orthographic":
        return ccrs.Orthographic(
            central_longitude=metadata.get("central_longitude", 0.0),
            central_latitude=metadata.get("central_latitude", 0.0),
        )
    if kind in {"lambertconformal", "lambert_conformal"}:
        kwargs: dict[str, Any] = {
            "central_longitude": metadata.get("central_longitude", 0.0),
            "central_latitude": metadata.get("central_latitude", 0.0),
        }
        if "standard_parallels" in metadata:
            kwargs["standard_parallels"] = tuple(metadata["standard_parallels"])
        if "globe_semimajor_axis" in metadata or "globe_semiminor_axis" in metadata:
            kwargs["globe"] = ccrs.Globe(
                semimajor_axis=metadata.get("globe_semimajor_axis"),
                semiminor_axis=metadata.get("globe_semiminor_axis"),
            )
        return ccrs.LambertConformal(**kwargs)
    return ccrs.PlateCarree()


def _draw_raster(
    ax: Any,
    view: RasterView,
    style: LayerStyle,
    ccrs: Any,
) -> Any:
    array = view.as_2d()
    values = _remap_values(array.values, style)
    lon, lat = _lon_lat_for(view, array)
    norm = _color_norm(style)
    return ax.pcolormesh(
        lon,
        lat,
        values,
        transform=ccrs.PlateCarree(),
        cmap=style.colormap,
        vmin=None if norm is not None else style.vmin,
        vmax=None if norm is not None else style.vmax,
        norm=norm,
        alpha=style.alpha,
        shading="auto",
    )


def _lon_lat_for(view: RasterView, array: Any) -> tuple[Any, Any]:
    if view.x_coord in view.data.coords and view.y_coord in view.data.coords:
        x_coord = view.data.coords[view.x_coord]
        y_coord = view.data.coords[view.y_coord]
        return x_coord.values, y_coord.values
    return array[view.x_dim].values, array[view.y_dim].values


def _draw_points(
    ax: Any,
    view: FrameView,
    style: LayerStyle,
    ccrs: Any,
) -> Any:
    table = view.table
    values = table[view.fields[0]] if view.fields else style.color
    return ax.scatter(
        table[view.x],
        table[view.y],
        c=values,
        s=style.size,
        alpha=style.alpha,
        transform=ccrs.PlateCarree(),
    )


def _decorate_axis(ax: Any, projection: ProjectionSpec) -> None:
    metadata = projection.metadata
    extent = metadata.get("extent")
    if extent is not None:
        ccrs, _ = _cartopy()
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    if metadata.get("coastlines", True):
        ax.coastlines(
            resolution=metadata.get("coastline_resolution", "110m"),
            linewidth=metadata.get("coastline_linewidth", 0.8),
        )
    if metadata.get("states", False):
        _, cfeature = _cartopy()
        scale = metadata.get("states_scale", "50m")
        ax.add_feature(
            cfeature.STATES.with_scale(scale),
            linewidth=metadata.get("states_linewidth", 0.5),
            edgecolor=metadata.get("states_edgecolor", "black"),
            zorder=2,
        )
    if metadata.get("land", False):
        _, cfeature = _cartopy()
        ax.add_feature(
            cfeature.LAND,
            facecolor=metadata.get("land_facecolor", "lightyellow"),
        )
    if metadata.get("gridlines", True):
        ax.gridlines(
            draw_labels=metadata.get("gridline_labels", False),
            linewidth=metadata.get("gridline_width", 0.5),
            alpha=metadata.get("gridline_alpha", 0.5),
            linestyle=metadata.get("gridline_style", "-"),
        )


def _subplot_title(layer: Any, frame_label: str | None) -> str:
    if frame_label:
        return f"{layer.name} | {frame_label}"
    return str(layer.name)
