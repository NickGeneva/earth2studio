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
"""Agent-friendly summary: backend protocol, registry, and summary backend.

Key APIs: `VizBackend` defines the backend contract; `register_backend` and
`get_backend` manage lazy backend factories; `SummaryBackend` is a deterministic
no-dependency backend for tests, debugging, and metadata inspection.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendCapabilities:
    """Backend capability flags used for route selection."""

    raster: bool = False
    points: bool = False
    vectors: bool = False
    terrain: bool = False
    images: bool = False
    meshes: bool = False
    volumes: bool = False
    texture_streaming: bool = False
    animation: bool = False
    interactive: bool = False
    export: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderResult:
    """Result returned by backend render/save/animate operations."""

    backend: str
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class VizDependencyError(ImportError):
    """Raised when a visualization backend dependency is not installed."""

    def __init__(self, backend: str, package: str):
        super().__init__(
            f"Visualization backend {backend!r} requires optional package {package!r}. "
            "Install the consolidated visualization extra, for example "
            "`pip install earth2studio[viz]`."
        )
        self.backend = backend
        self.package = package


class VizBackend(Protocol):
    """Protocol implemented by visualization backends."""

    name: str
    capabilities: BackendCapabilities

    def supports(self, scene: Any) -> bool:
        """Return whether the backend can render the provided scene."""

    def render(self, scene: Any, **kwargs: Any) -> RenderResult:
        """Render the scene into an in-memory result."""

    def show(self, scene: Any, **kwargs: Any) -> Any:
        """Show the scene in the backend's natural representation."""

    def save(self, scene: Any, path: str | Path, **kwargs: Any) -> Path:
        """Save a scene representation to disk."""

    def animate(self, scene: Any, path: str | Path, **kwargs: Any) -> Path:
        """Save an animation to disk."""


BackendFactory = Callable[[], VizBackend]
_BACKENDS: dict[str, BackendFactory] = {}
_DEFAULTS_REGISTERED = False


def register_backend(
    name: str, factory: BackendFactory, *, replace: bool = False
) -> None:
    """Register a backend factory by name."""
    normalized = name.lower()
    if not replace and normalized in _BACKENDS:
        raise ValueError(f"Backend {name!r} is already registered")
    _BACKENDS[normalized] = factory


def get_backend(name: str) -> VizBackend:
    """Instantiate a registered backend by name."""
    ensure_default_backends()
    normalized = name.lower()
    if normalized not in _BACKENDS:
        known = ", ".join(available_backends())
        raise KeyError(
            f"Unknown visualization backend {name!r}. Available backends: {known}"
        )
    return _BACKENDS[normalized]()


def available_backends() -> list[str]:
    """Return sorted backend names."""
    ensure_default_backends()
    return sorted(_BACKENDS)


def ensure_default_backends() -> None:
    """Register built-in lazy backend factories once."""
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    register_backend("summary", SummaryBackend)
    register_backend("matplotlib", _matplotlib_factory)
    _DEFAULTS_REGISTERED = True


class SummaryBackend:
    """No-dependency backend that returns scene metadata."""

    name = "summary"
    capabilities = BackendCapabilities(
        raster=True,
        points=True,
        vectors=True,
        terrain=True,
        images=True,
        meshes=True,
        volumes=True,
        animation=False,
        interactive=False,
        export=True,
    )

    def supports(self, scene: Any) -> bool:
        """Return True for all scenes because this backend only summarizes."""
        return True

    def render(self, scene: Any, **kwargs: Any) -> RenderResult:
        """Render the scene as a serializable summary dictionary."""
        return RenderResult(
            backend=self.name,
            output=scene.summary(),
            metadata={"kwargs": dict(kwargs)},
        )

    def show(self, scene: Any, **kwargs: Any) -> RenderResult:
        """Return the same result as `render` for deterministic inspection."""
        return self.render(scene, **kwargs)

    def save(self, scene: Any, path: str | Path, **kwargs: Any) -> Path:
        """Write a JSON summary to disk."""
        result = self.render(scene, **kwargs)
        output_path = Path(path)
        output_path.write_text(json.dumps(result.output, default=str, indent=2))
        return output_path

    def animate(self, scene: Any, path: str | Path, **kwargs: Any) -> Path:
        """Write timeline metadata to disk as a lightweight animation stand-in."""
        result = self.render(scene, **kwargs)
        output_path = Path(path)
        output_path.write_text(
            json.dumps(result.output.get("timeline", {}), default=str, indent=2)
        )
        return output_path


def _matplotlib_factory() -> VizBackend:
    from earth2studio.viz.backends.matplotlib import MatplotlibBackend

    return MatplotlibBackend()
