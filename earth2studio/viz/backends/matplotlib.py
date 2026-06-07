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
"""Agent-friendly summary: Matplotlib static plotting backend.

Key APIs: `MatplotlibBackend.render` draws raster, point, terrain, draped
raster, and simple vector layers into a Matplotlib figure. Imports Matplotlib
lazily so `earth2studio.viz` stays lightweight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from earth2studio.viz.adapters.dataframe import FrameView
from earth2studio.viz.adapters.xarray import RasterView
from earth2studio.viz.backends.base import (
    BackendCapabilities,
    RenderResult,
    VizDependencyError,
)
from earth2studio.viz.layers import (
    DrapedRasterLayer,
    PointLayer,
    RasterLayer,
    TerrainLayer,
    VectorLayer,
)
from earth2studio.viz.styles import LayerStyle


class MatplotlibBackend:
    """Static 2D plotting backend."""

    name = "matplotlib"
    capabilities = BackendCapabilities(
        raster=True,
        points=True,
        vectors=True,
        terrain=True,
        images=False,
        meshes=False,
        volumes=False,
        texture_streaming=False,
        animation=False,
        interactive=False,
        export=True,
    )

    def supports(self, scene: Any) -> bool:
        """Return whether all visible layer kinds have basic support."""
        supported = {"raster", "points", "terrain", "draped_raster", "vectors"}
        return all(layer.kind in supported for layer in scene.visible_layers)

    def render(self, scene: Any, **kwargs: Any) -> RenderResult:
        """Render visible scene layers to a Matplotlib figure."""
        plt = _pyplot()
        fig, ax = plt.subplots()
        title = kwargs.get("title", scene.title)
        if title:
            ax.set_title(title)
        for layer in scene.visible_layers:
            if isinstance(layer, (RasterLayer, TerrainLayer, DrapedRasterLayer)):
                self._draw_raster(ax, layer)
            elif isinstance(layer, PointLayer):
                self._draw_points(ax, layer)
            elif isinstance(layer, VectorLayer):
                self._draw_vectors(ax, layer)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return RenderResult(backend=self.name, output=fig, metadata={"axes": ax})

    def show(self, scene: Any, **kwargs: Any) -> Any:
        """Render and return the Matplotlib figure."""
        return self.render(scene, **kwargs).output

    def save(self, scene: Any, path: str | Path, **kwargs: Any) -> Path:
        """Save the Matplotlib figure to disk."""
        result = self.render(scene, **kwargs)
        output_path = Path(path)
        result.output.savefig(output_path)
        return output_path

    def animate(self, scene: Any, path: str | Path, **kwargs: Any) -> Path:
        """Animation is intentionally deferred for this initial backend."""
        raise NotImplementedError("Matplotlib animation is not implemented yet")

    def _draw_raster(
        self, ax: Any, layer: RasterLayer | TerrainLayer | DrapedRasterLayer
    ) -> None:
        view = layer.data
        if not isinstance(view, RasterView):
            return
        array = view.as_2d()
        values = _remap_values(array.values, layer.style)
        norm = _color_norm(layer.style)
        ax.imshow(
            values,
            origin="lower",
            cmap=layer.style.colormap,
            vmin=None if norm is not None else layer.style.vmin,
            vmax=None if norm is not None else layer.style.vmax,
            norm=norm,
            alpha=layer.style.alpha,
        )

    def _draw_points(self, ax: Any, layer: PointLayer) -> None:
        view = layer.data
        if not isinstance(view, FrameView):
            return
        table = view.table
        ax.scatter(
            table[view.x],
            table[view.y],
            c=table[view.fields[0]] if view.fields else layer.style.color,
            s=layer.style.size,
            alpha=layer.style.alpha,
        )

    def _draw_vectors(self, ax: Any, layer: VectorLayer) -> None:
        data = layer.data
        if not isinstance(data, dict):
            return
        x = data.get("x")
        y = data.get("y")
        u = data.get("u")
        v = data.get("v")
        if x is None or y is None or u is None or v is None:
            return
        ax.quiver(x, y, u, v, alpha=layer.style.alpha)


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise VizDependencyError("matplotlib", "matplotlib") from exc
    return plt


def _remap_values(values: Any, style: LayerStyle) -> Any:
    if style.input_range is None and style.output_range is None:
        return values
    import numpy as np

    array = np.asarray(values, dtype=float)
    if style.input_range is None:
        input_min = np.nanmin(array)
        input_max = np.nanmax(array)
    else:
        input_min, input_max = style.input_range
    output_min, output_max = style.output_range or (0.0, 1.0)
    span = input_max - input_min
    if span == 0:
        return np.full_like(array, output_min, dtype=float)
    normalized = np.clip((array - input_min) / span, 0.0, 1.0)
    return output_min + normalized * (output_max - output_min)


def _color_norm(style: LayerStyle) -> Any:
    if style.gamma is None:
        return None
    try:
        from matplotlib.colors import PowerNorm
    except ImportError as exc:
        raise VizDependencyError("matplotlib", "matplotlib") from exc
    return PowerNorm(gamma=style.gamma, vmin=style.vmin, vmax=style.vmax)
