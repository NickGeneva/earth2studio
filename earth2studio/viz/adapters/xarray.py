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
"""Agent-friendly summary: xarray-to-raster view adapter.

Key APIs: `XarrayAdapter.to_raster_view` and
`XarrayAdapter.to_raster_layer_view` select variables/time/lead_time,
infer spatial axes and grid descriptors, convert native indexed grids to
heatmap rasters when needed, preserve attributes, and return `RasterView` or
`RasterSequenceView` for scenes and backends.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import xarray as xr

from earth2studio.viz.grids import GridSpec, infer_grid_spec_from_xarray
from earth2studio.viz.native import (
    can_native_heatmap,
    native_grid_heatmap,
    supports_native_heatmap,
)
from earth2studio.viz.selection import (
    infer_spatial_reference,
    select_xarray,
    squeeze_non_spatial_singletons,
)

_DEFAULT_FRAME_DIMS = ("time", "lead_time")


@dataclass(frozen=True)
class RasterView:
    """Normalized raster view backed by an xarray DataArray."""

    data: xr.DataArray
    y_dim: str
    x_dim: str
    y_coord: str
    x_coord: str
    variable: str | None = None
    time_coord: str | None = None
    lead_time_coord: str | None = None
    device: str = "cpu"
    grid: GridSpec | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def shape_2d(self) -> tuple[int, int]:
        """Return the spatial `(y, x)` shape."""
        return self.data.sizes[self.y_dim], self.data.sizes[self.x_dim]

    def as_2d(self) -> xr.DataArray:
        """Return the raster transposed as `(y_dim, x_dim)`."""
        return self.data.transpose(self.y_dim, self.x_dim)


@dataclass(frozen=True)
class RasterSequenceView:
    """Normalized raster time series backed by an xarray DataArray."""

    data: xr.DataArray
    y_dim: str
    x_dim: str
    y_coord: str
    x_coord: str
    frame_dims: tuple[str, ...]
    variable: str | None = None
    device: str = "cpu"
    grid: GridSpec | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    native_heatmap: bool = False

    @property
    def frame_count(self) -> int:
        """Return the number of 2D frames represented by this sequence."""
        count = 1
        for dim in self.frame_dims:
            count *= self.data.sizes[dim]
        return count

    def iter_frames(self) -> Iterator[tuple[str | None, RasterView]]:
        """Yield `(label, RasterView)` pairs in xarray dimension order."""
        ranges = [range(self.data.sizes[dim]) for dim in self.frame_dims]
        for indices in product(*ranges):
            indexer = dict(zip(self.frame_dims, indices))
            frame = self.data.isel(indexer)
            label = _frame_label(self.data, indexer)
            if self.native_heatmap:
                frame = native_grid_heatmap(frame, self.grid)
            yield label, RasterView(
                data=frame,
                y_dim=self.y_dim,
                x_dim=self.x_dim,
                y_coord=self.y_coord,
                x_coord=self.x_coord,
                variable=self.variable or frame.name,
                device=_device_for_array(frame.data),
                grid=self.grid,
                attrs=dict(frame.attrs),
            )


class XarrayAdapter:
    """Adapter for xarray DataArray and Dataset inputs."""

    def __init__(self, data: xr.DataArray | xr.Dataset):
        self.data = data

    def to_raster_view(
        self,
        *,
        variable: str | None = None,
        time: Any | None = None,
        lead_time: Any | None = None,
        x: str | None = None,
        y: str | None = None,
    ) -> RasterView:
        """Convert xarray data into a renderable raster view."""
        view = self.to_raster_layer_view(
            variable=variable,
            time=time,
            lead_time=lead_time,
            x=x,
            y=y,
            allow_sequence=False,
        )
        if isinstance(view, RasterSequenceView):
            raise ValueError("A single raster view cannot include time-series frames")
        return view

    def to_raster_layer_view(
        self,
        *,
        variable: str | None = None,
        time: Any | None = None,
        lead_time: Any | None = None,
        x: str | None = None,
        y: str | None = None,
        frame_dims: Sequence[str] = _DEFAULT_FRAME_DIMS,
        allow_sequence: bool = True,
    ) -> RasterView | RasterSequenceView:
        """Convert xarray data into a single raster or raster time series."""
        selected = select_xarray(
            self.data,
            variable=variable,
            time=time,
            lead_time=lead_time,
        )
        grid = infer_grid_spec_from_xarray(selected)
        if can_native_heatmap(selected, grid):
            selected = native_grid_heatmap(selected, grid)
            y_dim, x_dim, y_coord, x_coord = infer_spatial_reference(selected, x=x, y=y)
            return RasterView(
                data=selected,
                y_dim=y_dim,
                x_dim=x_dim,
                y_coord=y_coord,
                x_coord=x_coord,
                variable=variable or selected.name,
                time_coord="time" if "time" in selected.coords else None,
                lead_time_coord="lead_time" if "lead_time" in selected.coords else None,
                device=_device_for_array(selected.data),
                grid=grid,
                attrs=dict(selected.attrs),
            )
        if allow_sequence and _can_native_sequence(selected, grid, frame_dims):
            return RasterSequenceView(
                data=selected,
                y_dim="native_y",
                x_dim="native_x",
                y_coord="native_y",
                x_coord="native_x",
                frame_dims=_remaining_frame_dims(
                    selected,
                    frame_dims,
                    _native_spatial_dims(selected, grid),
                ),
                variable=variable or selected.name,
                device=_device_for_array(selected.data),
                grid=grid,
                attrs=dict(selected.attrs),
                native_heatmap=True,
            )
        y_dim, x_dim, y_coord, x_coord = infer_spatial_reference(selected, x=x, y=y)
        if allow_sequence:
            selected, sequence_dims = _squeeze_or_sequence(
                selected,
                y_dim,
                x_dim,
                frame_dims,
            )
            if sequence_dims:
                return RasterSequenceView(
                    data=selected,
                    y_dim=y_dim,
                    x_dim=x_dim,
                    y_coord=y_coord,
                    x_coord=x_coord,
                    frame_dims=sequence_dims,
                    variable=variable or selected.name,
                    device=_device_for_array(selected.data),
                    grid=grid,
                    attrs=dict(selected.attrs),
                )
        else:
            selected = squeeze_non_spatial_singletons(selected, y_dim, x_dim)
        grid = infer_grid_spec_from_xarray(
            selected,
            y_dim=y_dim,
            x_dim=x_dim,
            y_coord=y_coord,
            x_coord=x_coord,
        )
        return RasterView(
            data=selected,
            y_dim=y_dim,
            x_dim=x_dim,
            y_coord=y_coord,
            x_coord=x_coord,
            variable=variable or selected.name,
            time_coord="time" if "time" in selected.coords else None,
            lead_time_coord="lead_time" if "lead_time" in selected.coords else None,
            device=_device_for_array(selected.data),
            grid=grid,
            attrs=dict(selected.attrs),
        )


def _device_for_array(array: Any) -> str:
    if hasattr(array, "__cuda_array_interface__"):
        return "cuda"
    module = type(array).__module__.split(".", maxsplit=1)[0]
    if module in {"cupy", "cudf"}:
        return "cuda"
    return "cpu"


def _squeeze_or_sequence(
    data: xr.DataArray,
    y_dim: str,
    x_dim: str,
    frame_dims: Sequence[str],
) -> tuple[xr.DataArray, tuple[str, ...]]:
    array = data
    sequence_dims: list[str] = []
    allowed = set(frame_dims)
    for dim in list(array.dims):
        if dim in (y_dim, x_dim):
            continue
        if dim in allowed:
            sequence_dims.append(dim)
            continue
        if array.sizes[dim] != 1:
            raise ValueError(
                f"Dimension {dim!r} has size {array.sizes[dim]} and needs an explicit selection"
            )
        array = array.isel({dim: 0})
    return array, tuple(sequence_dims)


def _can_native_sequence(
    data: xr.DataArray,
    grid: GridSpec,
    frame_dims: Sequence[str],
) -> bool:
    if not supports_native_heatmap(grid):
        return False
    spatial_dims = _native_spatial_dims(data, grid)
    if not spatial_dims:
        return False
    non_spatial = [dim for dim in data.dims if dim not in spatial_dims]
    return bool(non_spatial) and all(dim in frame_dims for dim in non_spatial)


def _native_spatial_dims(data: xr.DataArray, grid: GridSpec) -> tuple[str, ...]:
    dims = tuple(
        dim
        for dim in (grid.index_coord, grid.y_dim, grid.x_dim)
        if dim is not None and dim in data.dims
    )
    if len(dims) == 1 or len(dims) == 3:
        return dims
    return ()


def _remaining_frame_dims(
    data: xr.DataArray,
    frame_dims: Sequence[str],
    spatial_dims: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        dim for dim in data.dims if dim in frame_dims and dim not in spatial_dims
    )


def _frame_label(data: xr.DataArray, indexer: dict[str, int]) -> str | None:
    if not indexer:
        return None
    parts = []
    for dim, index in indexer.items():
        if dim in data.coords:
            value = data.coords[dim].values[index]
        else:
            value = index
        parts.append(f"{dim}={_format_frame_value(value)}")
    return ", ".join(parts)


def _format_frame_value(value: Any) -> str:
    if isinstance(value, np.timedelta64):
        hours = value / np.timedelta64(1, "h")
        if float(hours).is_integer():
            return f"{int(hours)} h"
        return f"{float(hours):g} h"
    return str(value)
