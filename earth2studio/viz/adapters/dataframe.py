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
"""Agent-friendly summary: pandas/cuDF-like dataframe adapter.

Key APIs: `DataFrameAdapter.to_frame_view` infers lat/lon/x/y/time/value columns
and returns `FrameView` while preserving the original table object and device
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

LAT_ALIASES = ("lat", "latitude")
LON_ALIASES = ("lon", "longitude")
X_ALIASES = ("x",)
Y_ALIASES = ("y",)
TIME_ALIASES = ("time", "valid_time", "datetime", "date")


@dataclass(frozen=True)
class FrameView:
    """Normalized sparse geospatial table view."""

    table: Any
    x: str
    y: str
    lat: str | None = None
    lon: str | None = None
    z: str | None = None
    time: str | None = None
    fields: tuple[str, ...] = ()
    device: str = "cpu"
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Return the number of rows in the frame view."""
        return len(self.table)


class DataFrameAdapter:
    """Adapter for pandas and cuDF-like tabular data."""

    def __init__(self, table: Any):
        if not hasattr(table, "columns"):
            raise TypeError("DataFrameAdapter expects a pandas or cuDF-like table")
        self.table = table

    def to_frame_view(
        self,
        *,
        lat: str | None = None,
        lon: str | None = None,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        time: str | None = None,
        fields: Sequence[str] | None = None,
    ) -> FrameView:
        """Convert a dataframe into a sparse frame view."""
        columns = tuple(str(column) for column in self.table.columns)
        lat = _pick(columns, lat, LAT_ALIASES)
        lon = _pick(columns, lon, LON_ALIASES)
        x = _pick(columns, x, X_ALIASES) or lon
        y = _pick(columns, y, Y_ALIASES) or lat
        time = _pick(columns, time, TIME_ALIASES)
        if x is None or y is None:
            raise ValueError("Could not infer x/y or lon/lat columns")
        if fields is None:
            excluded = {x, y, lat, lon, z, time, None}
            fields = tuple(column for column in columns if column not in excluded)
        else:
            _require_columns(columns, fields)
        return FrameView(
            table=self.table,
            x=x,
            y=y,
            lat=lat,
            lon=lon,
            z=z,
            time=time,
            fields=tuple(fields),
            device=_device_for_table(self.table),
            attrs={},
        )


def _pick(
    columns: tuple[str, ...], requested: str | None, aliases: tuple[str, ...]
) -> str | None:
    if requested is not None:
        _require_columns(columns, (requested,))
        return requested
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def _require_columns(columns: tuple[str, ...], requested: Sequence[str]) -> None:
    missing = [column for column in requested if column not in columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")


def _device_for_table(table: Any) -> str:
    module = type(table).__module__.split(".", maxsplit=1)[0]
    if module == "cudf":
        return "cuda"
    return "cpu"
