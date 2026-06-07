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
"""Agent-friendly summary: convenience APIs over the Scene object.

Key APIs: `plot` creates a temporary `Scene`, adds a raster layer, and renders it
with a selected backend. Use `Scene` directly for multi-layer scenes.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from earth2studio.viz.backends.base import RenderResult
from earth2studio.viz.scene import Scene


def plot(
    data: xr.DataArray | xr.Dataset,
    *,
    variable: str | None = None,
    time: Any | None = None,
    lead_time: Any | None = None,
    backend: str = "matplotlib",
    **kwargs: Any,
) -> Any:
    """Plot a single xarray raster using a visualization backend."""
    scene = Scene(title=kwargs.pop("title", None))
    scene.add_raster(
        data,
        variable=variable,
        time=time,
        lead_time=lead_time,
        **kwargs,
    )
    result = scene.render(backend=backend)
    if isinstance(result, RenderResult):
        return result.output
    return result
