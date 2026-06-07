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

Key APIs: `XarrayAdapter.to_raster_view` selects variables/time/lead_time,
infers spatial axes and grid descriptors, preserves attributes, and returns
`RasterView` for scenes and backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import xarray as xr

from earth2studio.viz.grids import GridSpec, infer_grid_spec_from_xarray
from earth2studio.viz.selection import (
    infer_spatial_reference,
    select_xarray,
    squeeze_non_spatial_singletons,
)


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
        selected = select_xarray(
            self.data,
            variable=variable,
            time=time,
            lead_time=lead_time,
        )
        y_dim, x_dim, y_coord, x_coord = infer_spatial_reference(selected, x=x, y=y)
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
