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
"""Agent-friendly summary: ANARI-SDK native viewer handoff backend.

Key APIs: `AnariBackend` registers the optional `anari` backend; `render`
returns a backend-neutral ANARI scene handoff payload; `show` creates an
`AnariViewerSession` that writes a native SDK-viewer descriptor and can launch a
configured ANARI-SDK viewer executable. The module does not import ANARI,
VisRTX, or any vendor device at import time.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from earth2studio.viz.assets import AssetSource, TextureSource
from earth2studio.viz.backends.base import (
    BackendCapabilities,
    RenderResult,
    VizDependencyError,
)
from earth2studio.viz.cache import viz_cache_root
from earth2studio.viz.layers import Layer
from earth2studio.viz.textures import TextureSequence

_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_LIBRARY = "helide"
_DEFAULT_DEVICE = "default"
_DEFAULT_VIEWER_NAMES = ("anariViewer", "anariViewer.exe")


@dataclass(frozen=True)
class AnariRuntimeStatus:
    """Availability summary for ANARI Python bindings and SDK viewer pieces."""

    anari: bool
    sdk_viewer: bool
    library: str
    device: str
    viewer_executable: str | None
    checked: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """Return whether ANARI Python bindings are importable."""
        return self.anari

    @property
    def native_viewer_ready(self) -> bool:
        """Return whether an ANARI-SDK native viewer executable is available."""
        return self.sdk_viewer

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable runtime status."""
        return {
            "anari": self.anari,
            "sdk_viewer": self.sdk_viewer,
            "ready": self.ready,
            "native_viewer_ready": self.native_viewer_ready,
            "library": self.library,
            "device": self.device,
            "viewer_executable": self.viewer_executable,
            "checked": self.checked,
            "metadata": dict(self.metadata),
        }


class AnariBackend:
    """Optional ANARI backend for SDK-viewer-native scientific rendering."""

    name = "anari"
    capabilities = BackendCapabilities(
        raster=True,
        points=True,
        vectors=True,
        terrain=True,
        images=True,
        meshes=True,
        volumes=True,
        texture_streaming=False,
        animation=True,
        interactive=True,
        export=True,
        metadata={
            "delivery": ("headless_descriptor", "sdk_native_viewer"),
            "viewer": "anari_sdk_interactive_viewer",
            "device_default": _DEFAULT_DEVICE,
            "library_default": _DEFAULT_LIBRARY,
            "vendor_device": None,
        },
    )

    def supports(self, scene: Any) -> bool:
        """Return whether this backend can describe the provided scene."""
        return all(getattr(layer, "kind", None) is not None for layer in scene.layers)

    def render(self, scene: Any, **backend_kwargs: Any) -> RenderResult:
        """Return a serializable ANARI-SDK viewer handoff payload."""
        require_renderer = bool(backend_kwargs.pop("require_renderer", False))
        require_viewer = bool(backend_kwargs.pop("require_viewer", False))
        width = int(backend_kwargs.pop("width", _DEFAULT_WIDTH))
        height = int(backend_kwargs.pop("height", _DEFAULT_HEIGHT))
        library = _library_name(backend_kwargs.pop("library", None))
        device = str(backend_kwargs.pop("device", _DEFAULT_DEVICE))
        viewer_executable = backend_kwargs.pop("viewer_executable", None)
        runtime = _runtime_status(
            require_renderer=require_renderer,
            require_viewer=require_viewer,
            library=library,
            device=device,
            viewer_executable=viewer_executable,
        )
        payload = _build_anari_payload(
            scene,
            runtime=runtime,
            width=width,
            height=height,
        )
        return RenderResult(
            backend=self.name,
            output=payload,
            metadata={"kwargs": dict(backend_kwargs), "runtime": runtime.as_dict()},
        )

    def show(
        self,
        scene: Any,
        *,
        streaming: bool = False,
        auto_flush: bool = True,
        **backend_kwargs: Any,
    ) -> "AnariViewerSession":
        """Create a native ANARI-SDK viewer session descriptor."""
        return AnariViewerSession(
            backend=self,
            scene=scene,
            streaming=streaming,
            auto_flush=auto_flush,
            **backend_kwargs,
        )

    def save(self, scene: Any, path: str | Path, **backend_kwargs: Any) -> Path:
        """Write the ANARI handoff payload to JSON."""
        result = self.render(scene, **backend_kwargs)
        output_path = Path(path)
        output_path.write_text(json.dumps(result.output, default=str, indent=2))
        return output_path

    def animate(self, scene: Any, path: str | Path, **backend_kwargs: Any) -> Path:
        """Export timeline metadata as the first ANARI animation artifact."""
        result = self.render(scene, **backend_kwargs)
        output_path = Path(path)
        output_path.write_text(
            json.dumps(result.output["timeline"], default=str, indent=2)
        )
        return output_path


class AnariViewerSession:
    """Backend-owned ANARI-SDK native viewer session synchronized by events."""

    def __init__(
        self,
        *,
        backend: AnariBackend,
        scene: Any,
        streaming: bool = False,
        auto_flush: bool = True,
        open_viewer: bool = True,
        output_dir: str | Path | None = None,
        title: str | None = None,
        viewer_args: Sequence[str] | None = None,
        **backend_kwargs: Any,
    ):
        self.backend = backend.name
        self.scene = scene
        self.streaming = streaming
        self.auto_flush = auto_flush
        self.open_viewer = open_viewer
        self.closed = False
        self.pending_events: list[Any] = []
        self.result: RenderResult | None = None
        self.payload: dict[str, Any] = {}
        self.process: subprocess.Popen[str] | None = None
        self._hold_depth = 0
        self._title = title
        self._backend = backend
        self._backend_kwargs = dict(backend_kwargs)
        self._viewer_args = tuple(viewer_args or ())
        self.output_dir = _session_output_dir(output_dir)
        self.descriptor_path = self.output_dir / "anari_scene.json"
        self.flush()
        if self.open_viewer and self.payload["runtime"]["sdk_viewer"]:
            self.launch()

    @property
    def output(self) -> dict[str, Any]:
        """Return the latest payload without forcing pending-event reconciliation."""
        if self.result is None and not self.closed:
            self.flush()
        return self.payload

    def update(self, event: Any) -> None:
        """Record a scene event and flush when automatic updates are enabled."""
        if self.closed:
            return
        self.pending_events.append(event)
        if self.auto_flush and self._hold_depth == 0:
            self.flush()

    def flush(self) -> RenderResult:
        """Refresh the native viewer handoff descriptor."""
        if self.closed and self.result is not None:
            return self.result
        kwargs = dict(self._backend_kwargs)
        self.result = self._backend.render(self.scene, **kwargs)
        self.payload = dict(self.result.output)
        if self._title is not None:
            self.payload["title"] = self._title
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.descriptor_path.write_text(
            json.dumps(self.payload, default=str, indent=2),
            encoding="utf-8",
        )
        self.pending_events.clear()
        return self.result

    def launch(self) -> subprocess.Popen[str] | None:
        """Launch the configured ANARI-SDK viewer executable if available."""
        executable = self.payload["runtime"].get("viewer_executable")
        if not executable:
            return None
        if self.process is not None and self.process.poll() is None:
            return self.process
        env = os.environ.copy()
        env["ANARI_LIBRARY"] = self.payload["runtime"]["library"]
        env["EARTH2STUDIO_ANARI_SCENE"] = str(self.descriptor_path)
        args = [executable, *self._viewer_args]
        self.process = subprocess.Popen(  # noqa: S603
            args,
            env=env,
            cwd=str(self.output_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self.process

    @contextmanager
    def hold(self) -> Iterator["AnariViewerSession"]:
        """Collect scene events until callers explicitly call `flush`."""
        self._hold_depth += 1
        try:
            yield self
        finally:
            self._hold_depth -= 1

    def close(self) -> None:
        """Detach this session from its scene and stop receiving updates."""
        if self.closed:
            return
        self.closed = True
        self.pending_events.clear()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        if hasattr(self.scene, "_detach_session"):
            self.scene._detach_session(self)


def _library_name(library: Any | None) -> str:
    if library is not None:
        return str(library)
    return os.environ.get("ANARI_LIBRARY", _DEFAULT_LIBRARY)


def _runtime_status(
    *,
    require_renderer: bool = False,
    require_viewer: bool = False,
    library: str,
    device: str,
    viewer_executable: Any | None = None,
) -> AnariRuntimeStatus:
    anari_available = importlib.util.find_spec("anari") is not None
    viewer_path = _resolve_viewer_executable(viewer_executable)
    if require_renderer and not anari_available:
        raise VizDependencyError("anari", "anari")
    if require_viewer and viewer_path is None:
        raise VizDependencyError("anari", "anariViewer")
    return AnariRuntimeStatus(
        anari=anari_available,
        sdk_viewer=viewer_path is not None,
        library=library,
        device=device,
        viewer_executable=None if viewer_path is None else str(viewer_path),
        metadata={
            "ANARI_LIBRARY": os.environ.get("ANARI_LIBRARY"),
            "ANARI_VIEWER": os.environ.get("ANARI_VIEWER"),
            "ANARI_VIEWER_EXECUTABLE": os.environ.get("ANARI_VIEWER_EXECUTABLE"),
        },
    )


def _resolve_viewer_executable(viewer_executable: Any | None = None) -> str | None:
    candidate = viewer_executable
    if candidate is None:
        candidate = os.environ.get("ANARI_VIEWER_EXECUTABLE") or os.environ.get(
            "ANARI_VIEWER"
        )
    if candidate is not None:
        path = Path(candidate)
        if path.exists():
            return str(path)
        return shutil.which(str(candidate))
    for name in _DEFAULT_VIEWER_NAMES:
        found = shutil.which(name)
        if found is not None:
            return found
    return None


def _session_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    root = viz_cache_root() / "anari_sessions"
    return root / f"session_{uuid.uuid4().hex}"


def _build_anari_payload(
    scene: Any,
    *,
    runtime: AnariRuntimeStatus,
    width: int,
    height: int,
) -> dict[str, Any]:
    summary = scene.summary()
    layers = [_layer_payload(index, layer) for index, layer in enumerate(scene.layers)]
    return {
        "title": summary["title"] or "Earth2 Studio ANARI",
        "mode": "native_viewer",
        "layout": {
            "viewport": "anari_sdk_interactive_viewer",
            "timeline": "viewer_control",
            "layers": "viewer_control",
            "camera": "orbit",
        },
        "render": {
            "renderer": "anari",
            "viewer_component": "anari_sdk_interactive_viewer",
            "library": runtime.library,
            "device": runtime.device,
            "resolution": (width, height),
        },
        "runtime": runtime.as_dict(),
        "camera": summary["camera"],
        "timeline": _timeline_payload(summary["timeline"], layers),
        "layers": layers,
        "scene": summary,
        "handoff": {
            "format": "earth2studio.viz.anari.session.v1",
            "purpose": "native_sdk_viewer_component",
            "notes": (
                "This payload is the Python-to-native bridge for an "
                "ANARI-SDK viewer component. Vendor-specific devices such as "
                "VisRTX are intentionally not selected by default."
            ),
        },
    }


def _layer_payload(index: int, layer: Layer) -> dict[str, Any]:
    data = getattr(layer, "data", None)
    return {
        "id": layer.id,
        "name": layer.name,
        "kind": layer.kind,
        "order": index,
        "visible": layer.visible,
        "style": layer.style.as_dict(),
        "projection": layer.projection.as_dict(),
        "time_extent": layer.time_extent,
        "source": _source_payload(data),
        "texture": _texture_payload(data),
        "anari_target": _anari_target(layer),
        "metadata": dict(layer.metadata),
    }


def _anari_target(layer: Layer) -> str:
    targets = {
        "raster": "textured_surface_or_sampler",
        "points": "sphere_or_glyph_geometry",
        "track": "curve_geometry",
        "vectors": "curve_cone_or_glyph_geometry",
        "terrain": "triangle_mesh_surface",
        "draped_raster": "textured_terrain_surface",
        "image": "image_sampler_surface",
        "geotiff": "image_sampler_surface",
        "mesh": "surface_geometry",
        "region_cube": "spatial_field_volume",
        "volume": "spatial_field_volume",
    }
    return targets.get(layer.kind, "scene_object")


def _timeline_payload(
    timeline: dict[str, Any],
    layers: list[dict[str, Any]],
) -> dict[str, Any]:
    frames = list(timeline.get("frames", ()))
    coverages = []
    for layer in layers:
        start, end = _coverage_from_extent(layer.get("time_extent"), frames)
        coverages.append(
            {
                "id": layer["id"],
                "name": layer["name"],
                "start": start,
                "end": end,
                "visible": layer["visible"],
            }
        )
    return {
        "frames": frames,
        "current": timeline.get("current"),
        "mode": timeline.get("mode"),
        "coverages": coverages,
    }


def _coverage_from_extent(
    extent: Any,
    frames: list[Any],
) -> tuple[int, int]:
    if not frames:
        return 0, 0
    if extent is None:
        return 0, len(frames) - 1
    try:
        start = frames.index(extent[0])
        end = frames.index(extent[1])
    except (ValueError, TypeError, IndexError):
        start, end = 0, len(frames) - 1
    return start, max(start, end)


def _texture_payload(data: Any) -> dict[str, Any] | None:
    if isinstance(data, TextureSequence):
        return {
            "type": "sequence",
            "frame_count": len(data.frames),
            "prefetch_policy": data.cache_policy.__dict__,
            "summary": data.as_dict(),
        }
    if isinstance(data, TextureSource):
        return {"type": "source", "summary": data.as_dict()}
    return None


def _source_payload(data: Any) -> dict[str, Any] | None:
    if isinstance(data, AssetSource):
        return data.as_dict()
    if hasattr(data, "as_dict"):
        return data.as_dict()
    if hasattr(data, "data") and hasattr(data, "shape_2d"):
        return {
            "data_type": type(data).__name__,
            "shape": data.shape_2d,
            "device": getattr(data, "device", None),
            "variable": getattr(data, "variable", None),
        }
    if hasattr(data, "table") and hasattr(data, "size"):
        return {
            "data_type": type(data).__name__,
            "size": data.size,
            "device": getattr(data, "device", None),
            "x": getattr(data, "x", None),
            "y": getattr(data, "y", None),
            "time": getattr(data, "time", None),
        }
    return None
