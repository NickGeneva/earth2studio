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
"""Agent-friendly summary: timeline inference for xarray and dataframe data.

Key APIs: `Timeline` stores ordered frames and current playback state;
`infer_frames_from_xarray` and `infer_frames_from_dataframe` produce frames from
`time`, `lead_time`, valid-time, or tabular time columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd
import xarray as xr


@dataclass
class Timeline:
    """Ordered timeline frames with mutable current frame state."""

    frames: list[Any] = field(default_factory=list)
    current_index: int = 0
    mode: str = "time"

    def __post_init__(self) -> None:
        """Deduplicate frames and validate the current index."""
        self.frames = _dedupe(self.frames)
        self._validate_index()

    @property
    def current(self) -> Any | None:
        """Return the active frame or None for a timeless scene."""
        if not self.frames:
            return None
        return self.frames[self.current_index]

    def add_frames(self, frames: Iterable[Any]) -> "Timeline":
        """Merge new frames into the timeline and return it."""
        self.frames = _dedupe([*self.frames, *frames])
        self._validate_index()
        return self

    def set(self, *, time: Any | None = None, index: int | None = None) -> "Timeline":
        """Set the active frame by value or index."""
        if index is not None:
            self.current_index = index
            self._validate_index()
            return self
        if time is None:
            return self
        for i, frame in enumerate(self.frames):
            if _frame_equal(frame, time):
                self.current_index = i
                return self
        raise KeyError(f"Timeline frame {time!r} was not found")

    def use_valid_time(self) -> "Timeline":
        """Mark the timeline as using forecast valid time."""
        self.mode = "valid_time"
        return self

    def range(self) -> tuple[Any, Any] | None:
        """Return the first and last frame, or None for a timeless scene."""
        if not self.frames:
            return None
        return self.frames[0], self.frames[-1]

    def _validate_index(self) -> None:
        if not self.frames:
            self.current_index = 0
            return
        if self.current_index < 0 or self.current_index >= len(self.frames):
            raise IndexError("Timeline current_index is out of range")


def infer_frames_from_xarray(
    data: xr.DataArray | xr.Dataset, *, valid_time: bool = False
) -> list[Any]:
    """Infer frames from xarray `time` and `lead_time` coordinates."""
    coords = data.coords
    if valid_time and "time" in coords and "lead_time" in coords:
        times = pd.to_datetime(_coord_values(data, "time"))
        lead_times = pd.to_timedelta(_coord_values(data, "lead_time"))
        return [time + lead_time for time in times for lead_time in lead_times]
    if "time" in coords:
        return list(_coord_values(data, "time"))
    if "lead_time" in coords:
        return list(_coord_values(data, "lead_time"))
    return []


def infer_frames_from_dataframe(table: Any, *, time: str | None = None) -> list[Any]:
    """Infer frames from a pandas or cuDF-like dataframe time column."""
    if time is None:
        time = _infer_time_column(table)
    if time is None:
        return []
    if time not in table.columns:
        raise KeyError(f"Time column {time!r} was not found")
    series = table[time]
    if hasattr(series, "drop_duplicates"):
        series = series.drop_duplicates()
    if hasattr(series, "to_pandas"):
        series = series.to_pandas()
    if hasattr(series, "tolist"):
        return series.tolist()
    return list(series)


def _coord_values(data: xr.DataArray | xr.Dataset, coord: str) -> list[Any]:
    values = data.coords[coord].values
    if getattr(values, "shape", ()) == ():
        return [values.item()]
    return list(values)


def _infer_time_column(table: Any) -> str | None:
    for candidate in ("time", "valid_time", "datetime", "date"):
        if candidate in table.columns:
            return candidate
    return None


def _dedupe(frames: Iterable[Any]) -> list[Any]:
    deduped: list[Any] = []
    for frame in frames:
        if not any(_frame_equal(existing, frame) for existing in deduped):
            deduped.append(frame)
    return deduped


def _frame_equal(left: Any, right: Any) -> bool:
    try:
        return bool(pd.Timestamp(left) == pd.Timestamp(right))
    except (TypeError, ValueError):
        return bool(left == right)
