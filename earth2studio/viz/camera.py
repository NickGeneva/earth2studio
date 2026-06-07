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
"""Agent-friendly summary: backend-neutral camera state for viz scenes.

Key APIs: `Camera.set` updates globe/local camera parameters, `Camera.orbit`
sets an orbital view around a target, and `Camera.as_dict` returns a serializable
camera payload for renderers or exporters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Camera:
    """Camera description shared by static and interactive backends."""

    lon: float = 0.0
    lat: float = 20.0
    distance: float = 2.5
    heading: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    projection: str = "globe"
    target: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def set(
        self,
        *,
        lon: float | None = None,
        lat: float | None = None,
        distance: float | None = None,
        heading: float | None = None,
        pitch: float | None = None,
        roll: float | None = None,
        projection: str | None = None,
        target: Any | None = None,
        **metadata: Any,
    ) -> "Camera":
        """Update camera fields in place and return this camera."""
        if lon is not None:
            self.lon = lon
        if lat is not None:
            self.lat = lat
        if distance is not None:
            self.distance = distance
        if heading is not None:
            self.heading = heading
        if pitch is not None:
            self.pitch = pitch
        if roll is not None:
            self.roll = roll
        if projection is not None:
            self.projection = projection
        if target is not None:
            self.target = target
        self.metadata.update(metadata)
        return self

    def orbit(
        self,
        *,
        target: Any | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        projection: str = "local",
    ) -> "Camera":
        """Set a local orbit-style camera and return this camera."""
        return self.set(
            heading=azimuth,
            pitch=elevation,
            distance=distance,
            projection=projection,
            target=target,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable camera summary."""
        return {
            "lon": self.lon,
            "lat": self.lat,
            "distance": self.distance,
            "heading": self.heading,
            "pitch": self.pitch,
            "roll": self.roll,
            "projection": self.projection,
            "target": self.target,
            "metadata": dict(self.metadata),
        }
