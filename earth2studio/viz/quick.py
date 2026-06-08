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
"""Agent-friendly summary: concise plotting helpers for examples and notebooks.

Key APIs: `RasterPanel`, `SeriesPanel`, `PointPanel`, `TrackPanel`,
`raster_dataarray`, `raster_panel`, `series_panel`, `point_panel`,
`track_panel`, `plot_raster_grid`, `save_raster_grid`, `plot_series`,
`save_series`, `plot_points`, `save_points`, `plot_point_sets`,
`save_point_sets`, `plot_tracks`, and `save_tracks`. These helpers keep example scripts
xarray/dataframe-native while moving repeated Matplotlib figure layout code into
the viz module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xarray as xr

from earth2studio.viz.adapters.dataframe import DataFrameAdapter
from earth2studio.viz.adapters.xarray import RasterView, XarrayAdapter
from earth2studio.viz.backends.matplotlib import _pyplot
from earth2studio.viz.selection import select_xarray


@dataclass(frozen=True, kw_only=True)
class RasterPanel:
    """One xarray raster panel in a static grid plot."""

    data: xr.DataArray | xr.Dataset
    variable: str | None = None
    time: Any | None = None
    lead_time: Any | None = None
    x: str | None = None
    y: str | None = None
    title: str | None = None
    colormap: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    colorbar: bool = True
    colorbar_label: str | None = None
    alpha: float = 1.0


@dataclass(frozen=True, kw_only=True)
class SeriesPanel:
    """One line series panel for scalar diagnostics or summary statistics."""

    x: Any
    y: Any
    label: str | None = None
    title: str | None = None
    color: str | None = None
    linewidth: float | None = None


@dataclass(frozen=True, kw_only=True)
class PointPanel:
    """One dataframe-backed point panel."""

    table: Any
    lat: str | None = None
    lon: str | None = None
    x: str | None = None
    y: str | None = None
    fields: Sequence[str] | None = None
    color: str | None = None
    title: str | None = None
    size: float | None = None
    alpha: float = 1.0


@dataclass(frozen=True, kw_only=True)
class TrackPanel:
    """One dataframe-backed track panel."""

    table: Any
    x: str | None = None
    y: str | None = None
    lat: str | None = None
    lon: str | None = None
    group: str | None = None
    label: str | None = None
    color: str | None = None
    linewidth: float | None = None


def raster_panel(
    data: xr.DataArray | xr.Dataset,
    *,
    variable: str | None = None,
    time: Any | None = None,
    lead_time: Any | None = None,
    x: str | None = None,
    y: str | None = None,
    title: str | None = None,
    colormap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    alpha: float = 1.0,
) -> RasterPanel:
    """Create a raster panel descriptor from xarray data."""
    return RasterPanel(
        data=data,
        variable=variable,
        time=time,
        lead_time=lead_time,
        x=x,
        y=y,
        title=title,
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
        alpha=alpha,
    )


def raster_dataarray(
    values: Any,
    *,
    lat: Any | None = None,
    lon: Any | None = None,
    y: Any | None = None,
    x: Any | None = None,
    dims: tuple[str, str] | None = None,
    name: str | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> xr.DataArray:
    """Wrap a 2D array and optional coordinates as an xarray DataArray."""
    coords: dict[str, Any] = {}
    if lat is not None or lon is not None:
        if _coord_ndim(lat) > 1 or _coord_ndim(lon) > 1:
            dims = dims or ("y", "x")
            if lat is not None:
                coords["lat"] = (dims, lat)
            if lon is not None:
                coords["lon"] = (dims, lon)
        else:
            dims = dims or ("lat", "lon")
            if lat is not None:
                coords[dims[0]] = lat
            if lon is not None:
                coords[dims[1]] = lon
    elif y is not None or x is not None:
        dims = dims or ("y", "x")
        if y is not None:
            coords[dims[0]] = y
        if x is not None:
            coords[dims[1]] = x
    else:
        dims = dims or ("y", "x")
    return xr.DataArray(values, dims=dims, coords=coords, name=name, attrs=attrs or {})


def series_panel(
    x: Any,
    y: Any,
    *,
    label: str | None = None,
    title: str | None = None,
    color: str | None = None,
    linewidth: float | None = None,
) -> SeriesPanel:
    """Create a line series panel descriptor."""
    return SeriesPanel(
        x=x,
        y=y,
        label=label,
        title=title,
        color=color,
        linewidth=linewidth,
    )


def point_panel(
    table: Any,
    *,
    lat: str | None = None,
    lon: str | None = None,
    x: str | None = None,
    y: str | None = None,
    fields: Sequence[str] | None = None,
    color: str | None = None,
    title: str | None = None,
    size: float | None = None,
    alpha: float = 1.0,
) -> PointPanel:
    """Create a dataframe-backed point panel descriptor."""
    return PointPanel(
        table=table,
        lat=lat,
        lon=lon,
        x=x,
        y=y,
        fields=fields,
        color=color,
        title=title,
        size=size,
        alpha=alpha,
    )


def track_panel(
    table: Any,
    *,
    x: str | None = None,
    y: str | None = None,
    lat: str | None = None,
    lon: str | None = None,
    group: str | None = None,
    label: str | None = None,
    color: str | None = None,
    linewidth: float | None = None,
) -> TrackPanel:
    """Create a dataframe-backed track panel descriptor."""
    return TrackPanel(
        table=table,
        x=x,
        y=y,
        lat=lat,
        lon=lon,
        group=group,
        label=label,
        color=color,
        linewidth=linewidth,
    )


def plot_raster_grid(
    panels: Sequence[RasterPanel | xr.DataArray | xr.Dataset],
    *,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    backend: str = "matplotlib",
) -> Any:
    """Plot one or more xarray rasters into a compact static grid."""
    _require_matplotlib(backend)
    plt = _pyplot()
    normalized = [_coerce_raster_panel(panel) for panel in panels]
    nrows, ncols = _grid_shape(len(normalized), ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize or (5.0 * ncols, 3.8 * nrows),
        squeeze=False,
        tight_layout=True,
    )
    for axis in axes.ravel()[len(normalized) :]:
        axis.set_axis_off()
    for axis, panel in zip(axes.ravel(), normalized):
        selected = select_xarray(
            panel.data,
            variable=panel.variable,
            time=panel.time,
            lead_time=panel.lead_time,
        )
        view = XarrayAdapter(selected).to_raster_view(
            x=panel.x,
            y=panel.y,
        )
        artist = _draw_raster(
            axis,
            view,
            colormap=panel.colormap,
            vmin=panel.vmin,
            vmax=panel.vmax,
            alpha=panel.alpha,
        )
        axis.set_title(panel.title or view.variable or "")
        axis.set_xlabel(view.x_coord)
        axis.set_ylabel(view.y_coord)
        if panel.colorbar:
            fig.colorbar(
                artist,
                ax=axis,
                shrink=0.74,
                pad=0.04,
                label=panel.colorbar_label or _label_for(view),
            )
    if title:
        fig.suptitle(title)
    return fig


def save_raster_grid(
    panels: Sequence[RasterPanel | xr.DataArray | xr.Dataset],
    path: str | Path,
    *,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    backend: str = "matplotlib",
    **savefig_kwargs: Any,
) -> Path:
    """Save a static raster grid and return the output path."""
    fig = plot_raster_grid(
        panels,
        ncols=ncols,
        figsize=figsize,
        title=title,
        backend=backend,
    )
    output_path = Path(path)
    fig.savefig(output_path, **savefig_kwargs)
    _pyplot().close(fig)
    return output_path


def plot_series(
    panels: Sequence[SeriesPanel],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (10.0, 4.0),
    backend: str = "matplotlib",
) -> Any:
    """Plot one or more one-dimensional series on a shared axis."""
    _require_matplotlib(backend)
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=figsize, tight_layout=True)
    for panel in panels:
        axis.plot(
            panel.x,
            panel.y,
            label=panel.label,
            color=panel.color,
            linewidth=panel.linewidth,
        )
    if any(panel.label for panel in panels):
        axis.legend()
    if title:
        axis.set_title(title)
    elif len(panels) == 1 and panels[0].title:
        axis.set_title(panels[0].title)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    return fig


def save_series(
    panels: Sequence[SeriesPanel],
    path: str | Path,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (10.0, 4.0),
    backend: str = "matplotlib",
    **savefig_kwargs: Any,
) -> Path:
    """Save a one-dimensional series plot and return the output path."""
    fig = plot_series(
        panels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        backend=backend,
    )
    output_path = Path(path)
    fig.savefig(output_path, **savefig_kwargs)
    _pyplot().close(fig)
    return output_path


def plot_points(
    table: Any,
    *,
    lat: str | None = None,
    lon: str | None = None,
    x: str | None = None,
    y: str | None = None,
    fields: Sequence[str] | None = None,
    color: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    backend: str = "matplotlib",
) -> Any:
    """Plot dataframe-backed points with inferred lat/lon or x/y columns."""
    _require_matplotlib(backend)
    plt = _pyplot()
    view = DataFrameAdapter(table).to_frame_view(
        lat=lat,
        lon=lon,
        x=x,
        y=y,
        fields=fields,
    )
    colors = _field_values(view.table, view.fields[0]) if view.fields else color
    fig, axis = plt.subplots(figsize=figsize, tight_layout=True)
    artist = axis.scatter(view.table[view.x], view.table[view.y], c=colors)
    if view.fields:
        fig.colorbar(artist, ax=axis, shrink=0.8, label=view.fields[0])
    if title:
        axis.set_title(title)
    axis.set_xlabel(xlabel or view.x)
    axis.set_ylabel(ylabel or view.y)
    return fig


def save_points(
    table: Any,
    path: str | Path,
    *,
    lat: str | None = None,
    lon: str | None = None,
    x: str | None = None,
    y: str | None = None,
    fields: Sequence[str] | None = None,
    color: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    backend: str = "matplotlib",
    **savefig_kwargs: Any,
) -> Path:
    """Save dataframe-backed points and return the output path."""
    fig = plot_points(
        table,
        lat=lat,
        lon=lon,
        x=x,
        y=y,
        fields=fields,
        color=color,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        backend=backend,
    )
    output_path = Path(path)
    fig.savefig(output_path, **savefig_kwargs)
    _pyplot().close(fig)
    return output_path


def plot_point_sets(
    panels: Sequence[PointPanel],
    *,
    ncols: int = 2,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str = "matplotlib",
) -> Any:
    """Plot multiple dataframe-backed point panels in a grid."""
    _require_matplotlib(backend)
    plt = _pyplot()
    nrows, ncols = _grid_shape(len(panels), ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize or (5.0 * ncols, 3.8 * nrows),
        squeeze=False,
        tight_layout=True,
    )
    for axis in axes.ravel()[len(panels) :]:
        axis.set_axis_off()
    for axis, panel in zip(axes.ravel(), panels):
        view = DataFrameAdapter(panel.table).to_frame_view(
            lat=panel.lat,
            lon=panel.lon,
            x=panel.x,
            y=panel.y,
            fields=panel.fields,
        )
        colors = (
            _field_values(view.table, view.fields[0]) if view.fields else panel.color
        )
        artist = axis.scatter(
            view.table[view.x],
            view.table[view.y],
            c=colors,
            s=panel.size,
            alpha=panel.alpha,
        )
        if view.fields:
            fig.colorbar(artist, ax=axis, shrink=0.8, label=view.fields[0])
        if panel.title:
            axis.set_title(panel.title)
        axis.set_xlabel(view.x)
        axis.set_ylabel(view.y)
    if title:
        fig.suptitle(title)
    return fig


def save_point_sets(
    panels: Sequence[PointPanel],
    path: str | Path,
    *,
    ncols: int = 2,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str = "matplotlib",
    **savefig_kwargs: Any,
) -> Path:
    """Save multiple dataframe-backed point panels and return the output path."""
    fig = plot_point_sets(
        panels,
        ncols=ncols,
        title=title,
        figsize=figsize,
        backend=backend,
    )
    output_path = Path(path)
    fig.savefig(output_path, **savefig_kwargs)
    _pyplot().close(fig)
    return output_path


def plot_tracks(
    panels: Sequence[TrackPanel] | Mapping[str, Any],
    *,
    title: str | None = None,
    xlabel: str | None = "longitude",
    ylabel: str | None = "latitude",
    figsize: tuple[float, float] = (8.0, 6.0),
    backend: str = "matplotlib",
) -> Any:
    """Plot one or more dataframe-backed trajectory collections."""
    _require_matplotlib(backend)
    plt = _pyplot()
    normalized = _coerce_track_panels(panels)
    fig, axis = plt.subplots(figsize=figsize, tight_layout=True)
    for panel in normalized:
        view = DataFrameAdapter(panel.table).to_frame_view(
            lat=panel.lat,
            lon=panel.lon,
            x=panel.x,
            y=panel.y,
            fields=(),
        )
        groups = _track_groups(view.table, panel.group)
        for index, (_, group) in enumerate(groups):
            label = panel.label if index == 0 else None
            axis.plot(
                group[view.x],
                group[view.y],
                label=label,
                color=panel.color,
                linewidth=panel.linewidth,
            )
    if any(panel.label for panel in normalized):
        axis.legend()
    if title:
        axis.set_title(title)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    return fig


def save_tracks(
    panels: Sequence[TrackPanel] | Mapping[str, Any],
    path: str | Path,
    *,
    title: str | None = None,
    xlabel: str | None = "longitude",
    ylabel: str | None = "latitude",
    figsize: tuple[float, float] = (8.0, 6.0),
    backend: str = "matplotlib",
    **savefig_kwargs: Any,
) -> Path:
    """Save dataframe-backed tracks and return the output path."""
    fig = plot_tracks(
        panels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        backend=backend,
    )
    output_path = Path(path)
    fig.savefig(output_path, **savefig_kwargs)
    _pyplot().close(fig)
    return output_path


def _draw_raster(
    axis: Any,
    view: RasterView,
    *,
    colormap: str | None,
    vmin: float | None,
    vmax: float | None,
    alpha: float,
) -> Any:
    array = view.as_2d()
    x_values = _coordinate_values(array, view.x_coord)
    y_values = _coordinate_values(array, view.y_coord)
    if x_values is not None and y_values is not None:
        return axis.pcolormesh(
            x_values,
            y_values,
            array.values,
            shading="auto",
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
        )
    return axis.imshow(
        array.values,
        origin="lower",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
    )


def _coerce_raster_panel(panel: RasterPanel | xr.DataArray | xr.Dataset) -> RasterPanel:
    if isinstance(panel, RasterPanel):
        return panel
    return RasterPanel(data=panel)


def _coerce_track_panels(
    panels: Sequence[TrackPanel] | Mapping[str, Any],
) -> list[TrackPanel]:
    if isinstance(panels, Mapping):
        return [TrackPanel(table=table, label=label) for label, table in panels.items()]
    return list(panels)


def _grid_shape(count: int, ncols: int) -> tuple[int, int]:
    if count < 1:
        raise ValueError("At least one panel is required")
    if ncols < 1:
        raise ValueError("ncols must be at least 1")
    ncols = min(ncols, count)
    nrows = math.ceil(count / ncols)
    return nrows, ncols


def _require_matplotlib(backend: str) -> None:
    if backend != "matplotlib":
        raise NotImplementedError(
            "Quick plotting helpers currently support only "
            f"'matplotlib', got {backend!r}"
        )


def _coordinate_values(array: xr.DataArray, coordinate: str) -> Any | None:
    if coordinate not in array.coords:
        return None
    coord = array.coords[coordinate]
    if coord.ndim > 1:
        return coord.values
    return coord.values if coord.size > 0 else None


def _coord_ndim(value: Any | None) -> int:
    if value is None:
        return 0
    if hasattr(value, "ndim"):
        return int(value.ndim)
    if hasattr(value, "shape"):
        return len(value.shape)
    return 1


def _label_for(view: RasterView) -> str:
    units = view.attrs.get("units")
    if units:
        return f"{view.variable or ''} ({units})".strip()
    return view.variable or ""


def _field_values(table: Any, field: str) -> Any:
    return table[field]


def _track_groups(table: Any, group: str | None) -> list[tuple[Any, Any]]:
    if group is None or group not in table.columns:
        return [(None, table)]
    return list(table.groupby(group))
