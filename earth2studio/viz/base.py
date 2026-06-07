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
"""Agent-friendly summary: structural protocols for viz extensibility.

Key APIs: `LayerProtocol`, `SceneProtocol`, `BackendProtocol`,
`SceneEventProtocol`, `SceneSessionProtocol`, `AssetSourceProtocol`, and
`TextureManagerProtocol` define the small contracts downstream packages need for
layers, scenes, renderer backends, streaming sessions, external assets, and
backend-internal texture streaming.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ContextManager, Protocol, runtime_checkable


@runtime_checkable
class LayerProtocol(Protocol):
    """Minimal contract for a backend-neutral visual layer."""

    id: str
    name: str
    kind: str
    visible: bool
    data: Any
    metadata: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return a serializable layer summary."""


@runtime_checkable
class SceneProtocol(Protocol):
    """Minimal contract consumed by visualization backends."""

    title: str | None
    layers: list[LayerProtocol]
    visible_layers: list[LayerProtocol]
    metadata: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return a serializable scene summary."""


@runtime_checkable
class SceneEventProtocol(Protocol):
    """Backend-facing description of one scene or layer mutation."""

    kind: str
    scene: SceneProtocol
    layer: LayerProtocol | None
    payload: dict[str, Any]


@runtime_checkable
class SceneSessionProtocol(Protocol):
    """Minimal contract for backend-owned streaming scene sessions."""

    backend: str
    scene: SceneProtocol
    auto_flush: bool
    closed: bool

    def update(self, event: SceneEventProtocol) -> None:
        """Receive a scene mutation event from the owning scene."""

    def flush(self) -> Any:
        """Reconcile pending scene changes with the backend representation."""

    def hold(self) -> ContextManager["SceneSessionProtocol"]:
        """Temporarily collect updates without flushing."""

    def close(self) -> None:
        """Release backend resources and detach the session."""


@runtime_checkable
class AssetSourceProtocol(Protocol):
    """Minimal contract for path-like or in-memory visual assets."""

    uri: str | Path | None
    data: Any | None
    kind: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable asset summary."""


@runtime_checkable
class TextureManagerProtocol(Protocol):
    """Backend-owned texture manager for staging, prefetch, and GPU residency."""

    policy: Any

    def resolve(self, frame: Any, **kwargs: Any) -> Any:
        """Resolve or upload a texture frame and return a backend handle."""

    def prefetch(self, frames: Iterable[Any], **kwargs: Any) -> None:
        """Stage candidate frames without changing the public layer API."""

    def release_layer(self, layer_id: str) -> None:
        """Release cached resources owned by a logical layer."""

    def clear(self) -> None:
        """Release all backend-owned texture resources."""


@runtime_checkable
class BackendProtocol(Protocol):
    """Minimal contract for pluggable visualization backends."""

    name: str
    capabilities: Any

    def supports(self, scene: SceneProtocol) -> bool:
        """Return whether the backend can render the scene."""

    def render(self, scene: SceneProtocol, **backend_kwargs: Any) -> Any:
        """Render a scene and return a backend-specific result."""

    def show(
        self,
        scene: SceneProtocol,
        *,
        streaming: bool = False,
        auto_flush: bool = True,
        **backend_kwargs: Any,
    ) -> Any:
        """Show a scene using a backend-native view."""

    def save(
        self,
        scene: SceneProtocol,
        path: str | Path,
        **backend_kwargs: Any,
    ) -> Path:
        """Save a scene artifact."""

    def animate(
        self,
        scene: SceneProtocol,
        path: str | Path,
        **backend_kwargs: Any,
    ) -> Path:
        """Save a scene animation artifact."""
