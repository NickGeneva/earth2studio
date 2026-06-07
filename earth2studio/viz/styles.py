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
"""Agent-friendly summary: backend-neutral style and projection specs.

Key APIs: `LayerStyle` captures colormap, scalar ranges, opacity, gamma,
remapping, point size, and metadata; `ProjectionSpec` captures the intended
map/local/globe projection without importing CartoPy or renderer packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProjectionSpec:
    """Backend-neutral projection description for a layer or scene."""

    kind: str = "latlon"
    crs: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable projection summary."""
        return {
            "kind": self.kind,
            "crs": self.crs,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LayerStyle:
    """Backend-neutral layer styling choices."""

    colormap: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    alpha: float = 1.0
    gamma: float | None = None
    input_range: tuple[float, float] | None = None
    output_range: tuple[float, float] | None = None
    color: str | None = None
    size: float | None = None
    width: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def merged(self, **overrides: Any) -> "LayerStyle":
        """Return a copy with non-None overrides applied."""
        values = {
            "colormap": self.colormap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "input_range": self.input_range,
            "output_range": self.output_range,
            "color": self.color,
            "size": self.size,
            "width": self.width,
            "metadata": dict(self.metadata),
        }
        metadata = overrides.pop("metadata", None)
        values.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        if metadata:
            values["metadata"].update(metadata)
        return LayerStyle(**values)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable style summary."""
        return {
            "colormap": self.colormap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "input_range": self.input_range,
            "output_range": self.output_range,
            "color": self.color,
            "size": self.size,
            "width": self.width,
            "metadata": dict(self.metadata),
        }
