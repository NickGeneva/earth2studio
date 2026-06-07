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
"""Agent-friendly summary: OVRTX browser/notebook session backend.

Key APIs: `OvrTxBackend` registers the optional `ovrtx` backend; `show`
creates an `OvrTxSession` with browser and notebook representations; `render`
returns a backend-neutral globe payload plus generated OpenUSD session text.
The module does not import OVRTX, ovstream, or USD at import time. Runtime
validation is explicit through backend keyword arguments so tests and static
workflows stay CPU-only while RTX streaming can be wired in when installed.
"""

from __future__ import annotations

import html
import importlib.util
import json
import os
import uuid
import webbrowser
from collections.abc import Iterator
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
from earth2studio.viz.domains import default_texture_domain
from earth2studio.viz.layers import Layer
from earth2studio.viz.textures import TextureSequence

_CANONICAL_RENDER_PRODUCT = "/Session/Render/Viewport"
_CANONICAL_CAMERA = "/Session/Camera"
_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720


@dataclass(frozen=True)
class OvrTxRuntimeStatus:
    """Availability summary for optional OVRTX browser-streaming pieces."""

    ovrtx: bool
    ovstream: bool
    mode: str
    checked: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """Return whether a full browser-streaming runtime is importable."""
        return self.ovrtx and self.ovstream

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable runtime status."""
        return {
            "ovrtx": self.ovrtx,
            "ovstream": self.ovstream,
            "ready": self.ready,
            "mode": self.mode,
            "checked": self.checked,
            "metadata": dict(self.metadata),
        }


class OvrTxBackend:
    """Optional OVRTX backend for globe-oriented interactive sessions."""

    name = "ovrtx"
    capabilities = BackendCapabilities(
        raster=True,
        points=True,
        vectors=True,
        terrain=True,
        images=True,
        meshes=True,
        volumes=False,
        texture_streaming=True,
        animation=True,
        interactive=True,
        export=False,
        metadata={
            "delivery": ("browser", "notebook"),
            "transport": "ovstream_webrtc",
            "stage": "openusd_session",
            "camera": "orbit",
            "layout": "globe_timeline_layers",
        },
    )

    def supports(self, scene: Any) -> bool:
        """Return whether this backend can describe the provided scene."""
        unsupported = {"volume"}
        return all(
            getattr(layer, "kind", "") not in unsupported for layer in scene.layers
        )

    def render(self, scene: Any, **backend_kwargs: Any) -> RenderResult:
        """Return a serializable OVRTX globe payload and session OpenUSD text."""
        require_renderer = bool(backend_kwargs.pop("require_renderer", False))
        require_stream = bool(backend_kwargs.pop("require_stream", require_renderer))
        runtime = _runtime_status(
            require_renderer=require_renderer,
            require_stream=require_stream,
        )
        width = int(backend_kwargs.pop("width", _DEFAULT_WIDTH))
        height = int(backend_kwargs.pop("height", _DEFAULT_HEIGHT))
        payload = _build_globe_payload(
            scene,
            runtime=runtime,
            width=width,
            height=height,
            stream_host=backend_kwargs.pop("stream_host", "127.0.0.1"),
            signaling_port=backend_kwargs.pop("signaling_port", 49100),
        )
        payload["stage"] = _build_globe_usda(payload)
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
    ) -> "OvrTxSession":
        """Create a browser or notebook session shell for the scene."""
        return OvrTxSession(
            backend=self,
            scene=scene,
            streaming=streaming,
            auto_flush=auto_flush,
            **backend_kwargs,
        )

    def save(self, scene: Any, path: str | Path, **backend_kwargs: Any) -> Path:
        """Write the OVRTX payload to JSON for renderer integration debugging."""
        result = self.render(scene, **backend_kwargs)
        output_path = Path(path)
        output_path.write_text(json.dumps(result.output, default=str, indent=2))
        return output_path

    def animate(self, scene: Any, path: str | Path, **backend_kwargs: Any) -> Path:
        """Export timeline metadata to JSON until RTX movie output is available."""
        result = self.render(scene, **backend_kwargs)
        output_path = Path(path)
        output_path.write_text(
            json.dumps(result.output["timeline"], default=str, indent=2)
        )
        return output_path


class OvrTxSession:
    """Backend-owned browser/notebook session synchronized by scene events."""

    def __init__(
        self,
        *,
        backend: OvrTxBackend,
        scene: Any,
        streaming: bool = False,
        auto_flush: bool = True,
        viewer: str = "browser",
        open_browser: bool = True,
        output_dir: str | Path | None = None,
        title: str | None = None,
        **backend_kwargs: Any,
    ):
        self.backend = backend.name
        self.scene = scene
        self.streaming = streaming
        self.auto_flush = auto_flush
        self.viewer = _normalize_viewer(viewer)
        self.open_browser = open_browser
        self.closed = False
        self.pending_events: list[Any] = []
        self.result: RenderResult | None = None
        self.payload: dict[str, Any] = {}
        self.html: str = ""
        self._hold_depth = 0
        self._title = title
        self._backend = backend
        self._backend_kwargs = dict(backend_kwargs)
        self.output_dir = _session_output_dir(output_dir)
        self.html_path = self.output_dir / "index.html"
        self.url = self.html_path.as_uri()
        self.flush()
        if self.viewer == "browser" and self.open_browser:
            webbrowser.open(self.url)

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
        """Refresh the session payload and browser/notebook document."""
        if self.closed and self.result is not None:
            return self.result
        kwargs = dict(self._backend_kwargs)
        self.result = self._backend.render(self.scene, **kwargs)
        self.payload = dict(self.result.output)
        if self._title is not None:
            self.payload["title"] = self._title
        self.html = _render_viewer_html(self.payload)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.html_path.write_text(self.html, encoding="utf-8")
        self.pending_events.clear()
        return self.result

    @contextmanager
    def hold(self) -> Iterator["OvrTxSession"]:
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
        if hasattr(self.scene, "_detach_session"):
            self.scene._detach_session(self)

    def _repr_html_(self) -> str:
        """Return an inline notebook representation."""
        document = self.html or _render_viewer_html(self.payload)
        return (
            '<iframe title="Earth2 Studio OVRTX" '
            'style="width:100%;height:640px;border:0;display:block" '
            f'srcdoc="{html.escape(document, quote=True)}"></iframe>'
        )


def _normalize_viewer(viewer: str) -> str:
    normalized = viewer.lower()
    if normalized not in {"browser", "notebook"}:
        raise ValueError("OVRTX viewer must be 'browser' or 'notebook'")
    return normalized


def _session_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    root = viz_cache_root() / "ovrtx_sessions"
    return root / f"session_{uuid.uuid4().hex}"


def _runtime_status(
    *,
    require_renderer: bool = False,
    require_stream: bool = False,
) -> OvrTxRuntimeStatus:
    ovrtx_available = importlib.util.find_spec("ovrtx") is not None
    ovstream_available = importlib.util.find_spec("ovstream") is not None
    missing = []
    if require_renderer and not ovrtx_available:
        missing.append(("ovrtx", "ovrtx"))
    if require_stream and not ovstream_available:
        missing.append(("ovrtx", "ovstream"))
    if missing:
        backend, package = missing[0]
        raise VizDependencyError(backend, package)
    return OvrTxRuntimeStatus(
        ovrtx=ovrtx_available,
        ovstream=ovstream_available,
        mode="webrtc" if ovstream_available else "descriptor",
        metadata={
            "OVRTX_SKIP_USD_CHECK": os.environ.get("OVRTX_SKIP_USD_CHECK"),
            "OVRTX_BIN_PATH": os.environ.get("OVRTX_BIN_PATH"),
        },
    )


def _build_globe_payload(
    scene: Any,
    *,
    runtime: OvrTxRuntimeStatus,
    width: int,
    height: int,
    stream_host: str,
    signaling_port: int,
) -> dict[str, Any]:
    summary = scene.summary()
    texture_domain = default_texture_domain()
    layers = [_layer_payload(index, layer) for index, layer in enumerate(scene.layers)]
    default_texture = _default_texture_payload(scene)
    return {
        "title": summary["title"] or "Earth2 Studio OVRTX",
        "mode": "globe",
        "layout": {
            "viewport": "streamed_video",
            "timeline": "bottom",
            "layers": "upper_right",
            "camera": "orbit",
        },
        "render": {
            "renderer": "ovrtx",
            "transport": "ovstream_webrtc",
            "render_product": _CANONICAL_RENDER_PRODUCT,
            "camera": _CANONICAL_CAMERA,
            "resolution": (width, height),
        },
        "stream": {
            "host": stream_host,
            "signaling_port": signaling_port,
            "video_element_id": "remote-video",
            "input": "nvst_native",
            "state_messages": "json_data_channel",
        },
        "runtime": runtime.as_dict(),
        "camera": summary["camera"],
        "timeline": _timeline_payload(summary["timeline"], layers),
        "layers": layers,
        "default_texture": default_texture,
        "default_texture_domain": texture_domain.as_dict(),
        "scene": summary,
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
        "texture": _texture_payload(data),
        "source": _source_payload(data),
        "metadata": dict(layer.metadata),
    }


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


def _default_texture_payload(scene: Any) -> dict[str, Any]:
    for layer in scene.layers:
        metadata = getattr(layer, "metadata", {})
        texture_domain = metadata.get("texture_domain")
        if texture_domain:
            return {
                "layer_id": layer.id,
                "layer_name": layer.name,
                "domain": texture_domain,
                "source": _source_payload(layer.data),
            }
    source = default_texture_domain().source("global_base_color")
    return {
        "layer_id": None,
        "layer_name": source.name,
        "domain": default_texture_domain().as_dict(),
        "source": source.as_dict(),
    }


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


def _build_globe_usda(payload: dict[str, Any]) -> str:
    width, height = payload["render"]["resolution"]
    texture = payload["default_texture"]["source"]
    texture_uri = "" if texture is None else texture.get("uri", "")
    texture_comment = f"# defaultTexture = {texture_uri}\n" if texture_uri else ""
    return f"""#usda 1.0
(
    defaultPrim = "World"
)

{texture_comment}def Xform "World"
{{
    def Sphere "Earth"
    {{
        double radius = 6371000
        uniform token purpose = "render"
    }}
}}

def Xform "Session"
{{
    def Camera "Camera"
    {{
        float focalLength = 24
        double3 xformOp:translate = (0, 0, 22000000)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}

    def "Render"
    {{
        def RenderProduct "Viewport"
        {{
            rel camera = <{_CANONICAL_CAMERA}>
            int2 resolution = ({width}, {height})
            rel orderedVars = [</Session/Render/Vars/LdrColor>]
        }}

        def RenderVar "Vars/LdrColor"
        {{
            uniform string sourceName = "LdrColor"
        }}
    }}
}}
"""


def _render_viewer_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, default=str).replace("</", "<\\/")
    title = html.escape(str(payload["title"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: #0b1017;
      color: #eef4fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      overflow: hidden;
      background: #0b1017;
    }}
    .shell {{
      position: relative;
      width: 100vw;
      height: 100vh;
      min-height: 520px;
      background:
        linear-gradient(180deg, rgba(11, 16, 23, 0.16), rgba(11, 16, 23, 0.74)),
        #0b1017;
    }}
    .viewport {{
      position: absolute;
      inset: 0;
      overflow: hidden;
      background: #03070c;
    }}
    #remote-video {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #03070c;
    }}
    .stream-status {{
      position: absolute;
      inset: auto auto 168px 26px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-width: min(520px, calc(100vw - 360px));
      padding: 10px 12px;
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 8px;
      background: rgba(8, 12, 18, 0.78);
      backdrop-filter: blur(14px);
      box-shadow: 0 18px 60px rgba(0,0,0,0.24);
    }}
    .stream-status strong {{
      font-size: 12px;
      font-weight: 700;
    }}
    .stream-status span {{
      font-size: 12px;
      color: #b8c6d7;
      line-height: 1.35;
    }}
    .layers {{
      position: absolute;
      top: 20px;
      right: 24px;
      width: min(308px, calc(100vw - 48px));
      max-height: calc(100vh - 178px);
      overflow: auto;
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 8px;
      background: rgba(8, 12, 18, 0.76);
      backdrop-filter: blur(16px);
      box-shadow: 0 20px 70px rgba(0,0,0,0.28);
    }}
    .layers header {{
      position: sticky;
      top: 0;
      z-index: 1;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.12);
      background: rgba(8, 12, 18, 0.88);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .layer-list {{
      display: flex;
      flex-direction: column;
      padding: 8px;
      gap: 6px;
    }}
    .layer-row {{
      display: grid;
      grid-template-columns: 28px 1fr auto;
      gap: 10px;
      align-items: center;
      min-height: 36px;
      padding: 6px;
      border-radius: 6px;
      color: #eef4fb;
    }}
    .layer-row[aria-disabled="true"] {{
      opacity: 0.48;
    }}
    .layer-row input {{
      width: 18px;
      height: 18px;
      accent-color: #5fc4ff;
    }}
    .layer-name {{
      overflow-wrap: anywhere;
      font-size: 12px;
      line-height: 1.25;
    }}
    .layer-kind {{
      color: #9cafc5;
      font-size: 11px;
      text-transform: uppercase;
    }}
    .timeline {{
      position: absolute;
      left: 50%;
      bottom: 24px;
      width: min(680px, calc(100vw - 48px));
      transform: translateX(-50%);
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 8px;
      background: rgba(8, 12, 18, 0.78);
      backdrop-filter: blur(16px);
      box-shadow: 0 20px 70px rgba(0,0,0,0.28);
    }}
    .timeline-controls {{
      display: grid;
      grid-template-columns: 34px 1fr 138px;
      gap: 12px;
      align-items: center;
      padding: 12px 14px 8px;
    }}
    button {{
      height: 30px;
      border: 1px solid rgba(255,255,255,0.18);
      border-radius: 6px;
      background: rgba(255,255,255,0.08);
      color: #eef4fb;
      cursor: pointer;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: #5fc4ff;
    }}
    .time-label {{
      color: #b8c6d7;
      font-size: 12px;
      text-align: right;
      overflow-wrap: anywhere;
    }}
    .coverage {{
      position: relative;
      height: 16px;
      margin: 0 58px 12px 60px;
      border-radius: 4px;
      background: rgba(255,255,255,0.08);
      overflow: hidden;
    }}
    .coverage span {{
      position: absolute;
      top: 3px;
      height: 10px;
      min-width: 2px;
      border-radius: 999px;
      background: #5fc4ff;
      opacity: 0.82;
    }}
    @media (max-width: 720px) {{
      .stream-status {{
        inset: 18px 18px auto 18px;
        max-width: none;
      }}
      .layers {{
        top: auto;
        right: 18px;
        bottom: 132px;
        max-height: 180px;
      }}
      .timeline {{
        bottom: 16px;
      }}
      .timeline-controls {{
        grid-template-columns: 34px 1fr;
      }}
      .time-label {{
        grid-column: 1 / -1;
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="viewport" aria-label="OVRTX streamed viewport">
      <video id="remote-video" autoplay muted playsinline></video>
      <aside class="stream-status" id="stream-status"></aside>
    </section>
    <aside class="layers" aria-label="Layers">
      <header>Layers</header>
      <div class="layer-list" id="layer-list"></div>
    </aside>
    <section class="timeline" aria-label="Timeline">
      <div class="timeline-controls">
        <button id="play" type="button" aria-label="Play or pause">&#9654;</button>
        <input id="time" type="range" min="0" max="0" value="0" step="1">
        <div class="time-label" id="time-label"></div>
      </div>
      <div class="coverage" id="coverage"></div>
    </section>
  </main>
  <script type="application/json" id="payload">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const frames = payload.timeline.frames || [];
    const layers = payload.layers || [];
    const time = document.getElementById("time");
    const label = document.getElementById("time-label");
    const play = document.getElementById("play");
    const coverage = document.getElementById("coverage");
    const layerList = document.getElementById("layer-list");
    const status = document.getElementById("stream-status");
    let playing = false;
    let timer = null;

    function setStatus() {{
      const runtime = payload.runtime || {{}};
      const stream = payload.stream || {{}};
      const mode = runtime.ready ? "ready" : "waiting";
      status.innerHTML = `
        <strong>OVRTX ${{mode}}</strong>
        <span>${{runtime.ready
          ? `ovstream Direct target ${{stream.host}}:${{stream.signaling_port}}`
          : "Install/import ovrtx and ovstream, then attach the backend render loop to this session."}}</span>
      `;
    }}

    function renderLayers() {{
      layerList.innerHTML = "";
      for (const layer of layers) {{
        const row = document.createElement("label");
        row.className = "layer-row";
        row.setAttribute("aria-disabled", String(!layer.visible));
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = Boolean(layer.visible);
        checkbox.addEventListener("change", () => {{
          row.setAttribute("aria-disabled", String(!checkbox.checked));
        }});
        const name = document.createElement("div");
        name.className = "layer-name";
        name.textContent = layer.name || layer.id;
        const kind = document.createElement("div");
        kind.className = "layer-kind";
        kind.textContent = layer.kind || "layer";
        row.append(checkbox, name, kind);
        layerList.append(row);
      }}
    }}

    function renderCoverage() {{
      coverage.innerHTML = "";
      const count = Math.max(1, frames.length - 1);
      for (const item of payload.timeline.coverages || []) {{
        if (!item.visible) continue;
        const bar = document.createElement("span");
        const start = Math.max(0, Math.min(1, item.start / count));
        const end = Math.max(start + 0.01, Math.min(1, item.end / count));
        bar.style.left = `${{start * 100}}%`;
        bar.style.width = `${{(end - start) * 100}}%`;
        coverage.append(bar);
      }}
    }}

    function setFrame(index) {{
      const value = frames[index] ?? "static";
      label.textContent = String(value);
      time.value = String(index);
    }}

    function togglePlay() {{
      playing = !playing;
      play.textContent = playing ? "||" : "\\u25b6";
      if (timer) window.clearInterval(timer);
      if (playing) {{
        timer = window.setInterval(() => {{
          const max = Number(time.max);
          const next = Number(time.value) >= max ? 0 : Number(time.value) + 1;
          setFrame(next);
        }}, 800);
      }}
    }}

    time.max = String(Math.max(0, frames.length - 1));
    time.addEventListener("input", event => setFrame(Number(event.target.value)));
    play.addEventListener("click", togglePlay);
    setStatus();
    renderLayers();
    renderCoverage();
    setFrame(0);
  </script>
</body>
</html>
"""
