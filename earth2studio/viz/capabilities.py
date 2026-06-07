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
"""Agent-friendly summary: Command Center parity inventory for viz.

Key APIs: `VizCapability` records one application capability, its parity status,
and the package area that should own it; `default_capability_inventory` returns
the current Earth2 Studio viz parity map without adding new public scene APIs.
This module is intentionally internal planning scaffolding for implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CapabilityStatus = Literal["implemented", "partial", "missing", "deferred"]


@dataclass(frozen=True, kw_only=True)
class VizCapability:
    """One Command Center-inspired visualization capability."""

    id: str
    command_center_concept: str
    earth2studio_area: str
    status: CapabilityStatus
    public_api_required: bool
    summary: str
    missing: tuple[str, ...] = field(default_factory=tuple)
    next_steps: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        """Return a serializable capability summary."""
        return {
            "id": self.id,
            "command_center_concept": self.command_center_concept,
            "earth2studio_area": self.earth2studio_area,
            "status": self.status,
            "public_api_required": self.public_api_required,
            "summary": self.summary,
            "missing": list(self.missing),
            "next_steps": list(self.next_steps),
        }


def default_capability_inventory() -> tuple[VizCapability, ...]:
    """Return the current feature-parity inventory for `earth2studio.viz`."""
    return (
        VizCapability(
            id="scene-layer-registry",
            command_center_concept="FeaturesAPI",
            earth2studio_area="scene.py/layers.py",
            status="partial",
            public_api_required=False,
            summary="Scenes can add, get, remove, summarize, and hide/show layers.",
            missing=(
                "Layer reorder operations",
                "Layer filtering/query predicates",
                "Change events for backend delegates",
            ),
            next_steps=(
                "Add an internal scene event stream.",
                "Add backend-facing ordered layer diffs without new top-level helpers.",
            ),
        ),
        VizCapability(
            id="base-feature-metadata",
            command_center_concept="Base Feature",
            earth2studio_area="layers.py",
            status="implemented",
            public_api_required=False,
            summary="Layer carries id, name, kind, visibility, time extent, style, projection, and metadata.",
        ),
        VizCapability(
            id="image-feature-materials",
            command_center_concept="Image feature",
            earth2studio_area="layers.py/styles.py/assets.py",
            status="partial",
            public_api_required=False,
            summary="Image, GeoTIFF, texture sequence, alpha, gamma, scalar remapping, style, and projection intents exist.",
            missing=(
                "Alpha-source layering",
                "Flip U/V",
                "Longitudinal offset",
                "Affine texture transform",
            ),
            next_steps=(
                "Keep alpha-source masks separate from simple layer opacity.",
                "Represent UV, offset, and affine controls as backend/source transform metadata.",
                "Avoid top-level scene helpers for texture-coordinate transforms until backend behavior proves stable.",
            ),
        ),
        VizCapability(
            id="timeline-playback",
            command_center_concept="TimeManager",
            earth2studio_area="timeline.py/textures.py",
            status="partial",
            public_api_required=False,
            summary="Timeline frame inference and texture sequence timestamp selection exist.",
            missing=(
                "Playback rate",
                "Loop policy at scene timeline level",
                "UTC range to playback-time mapping",
                "Frame-change events for renderer sessions",
            ),
            next_steps=(
                "Add internal timeline state transitions for backend sessions.",
                "Route frame-change notifications to texture managers and renderer delegates.",
            ),
        ),
        VizCapability(
            id="dynamic-texture-streaming",
            command_center_concept="TimestampedSequence and dynamictexture",
            earth2studio_area="textures.py/base.py/backends/",
            status="partial",
            public_api_required=False,
            summary="Texture frames, texture sequences, cache policy, and texture manager protocol exist.",
            missing=(
                "Concrete OVRTX texture manager",
                "Async decode and upload queues",
                "CPU staging cache",
                "GPU residency cache",
                "Mosaic/tile and LOD-aware loaders",
            ),
            next_steps=(
                "Implement texture manager under the first local renderer backend.",
                "Keep resolve/prefetch/release/clear as the only backend contract.",
            ),
        ),
        VizCapability(
            id="default-global-textures",
            command_center_concept="Application default world textures",
            earth2studio_area="cache.py/domains.py",
            status="implemented",
            public_api_required=False,
            summary="Default global texture domain resolves readable assets under EARTH2STUDIO_CACHE/viz/v5/default_textures.",
        ),
        VizCapability(
            id="grid-projection-support",
            command_center_concept="latlong, tiled latlong, diamond, hpx, goes projections",
            earth2studio_area="grids.py/adapters/xarray.py/styles.py",
            status="partial",
            public_api_required=False,
            summary="GridSpec records regular lat/lon, curvilinear lat/lon, projected/native, HPX/HEALPix, diamond, GOES, and geohash-indexed grid intent.",
            missing=(
                "Concrete backend lowering for geohash cells",
                "Concrete backend lowering for HPX/HEALPix textures",
                "Tiled latlong mosaic loader integration",
            ),
            next_steps=(
                "Route GridSpec metadata into backend payload builders.",
                "Keep projection-specific rendering logic in backend adapters.",
            ),
        ),
        VizCapability(
            id="regional-terrain",
            command_center_concept="Local/regional terrain and projected overlays",
            earth2studio_area="regional.py/scene.py/layers.py",
            status="partial",
            public_api_required=False,
            summary="Region, terrain, draped raster, mesh, and region-cube intents exist.",
            missing=(
                "Tiled terrain mesh generation",
                "OpenUSD terrain export",
                "Renderer-backed local scene session",
                "Vertical datum transforms",
            ),
            next_steps=(
                "Add internal terrain tiling and mesh payload builders.",
                "Add backend/export support before adding more user calls.",
            ),
        ),
        VizCapability(
            id="vector-flow-objects",
            command_center_concept="Curves, markers, tracks, flow features",
            earth2studio_area="layers.py/scene.py/backends/",
            status="partial",
            public_api_required=False,
            summary="Point and vector layer intents exist; track layer class and quick helpers exist.",
            missing=(
                "Scene-level track adapter",
                "Streamline generation",
                "3D glyph instancing",
                "Backend flow-object lowering",
            ),
            next_steps=(
                "Build internal vector geometry payloads for renderers.",
                "Promote scene helpers only after examples show repeated need.",
            ),
        ),
        VizCapability(
            id="application-session",
            command_center_concept="globe_view delegates and renderer session",
            earth2studio_area="backends/",
            status="missing",
            public_api_required=False,
            summary="No local renderer application/session object exists yet.",
            missing=(
                "Session lifecycle",
                "Renderer delegate subscription to scene changes",
                "Camera synchronization",
                "Resource cleanup hooks",
                "Interactive picking/selection",
            ),
            next_steps=(
                "Prototype a backend-owned session object under OVRTX/OpenUSD paths.",
                "Keep `Scene.show(backend=...)` as the user entrypoint.",
            ),
        ),
        VizCapability(
            id="data-to-visual-payload",
            command_center_concept="DFM bridge",
            earth2studio_area="adapters/backends/",
            status="missing",
            public_api_required=False,
            summary="No bridge exists yet to encode xarray fields into optimized image/mesh/volume payloads.",
            missing=(
                "Stable colormap normalization to uint8/RGBA",
                "Texture compression path",
                "Volume payload conversion",
                "Forecast variable provenance on visual payloads",
            ),
            next_steps=(
                "Add backend-internal payload builders for raster-to-texture conversion.",
                "Reuse xarray labels and lexicon naming without direct IO object coupling.",
            ),
        ),
    )


def summarize_capability_inventory(
    inventory: tuple[VizCapability, ...] | None = None,
) -> dict[str, object]:
    """Return status counts and missing capability records."""
    capabilities = inventory or default_capability_inventory()
    counts = {status: 0 for status in ("implemented", "partial", "missing", "deferred")}
    for capability in capabilities:
        counts[capability.status] += 1
    return {
        "counts": counts,
        "missing_or_partial": [
            capability.as_dict()
            for capability in capabilities
            if capability.status in {"partial", "missing"}
        ],
    }
