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
"""Agent-friendly summary: external asset descriptors for viz layers.

Key APIs: `AssetSource` describes a path or in-memory object with optional CRS,
bounds, time, and metadata; `TextureSource` specializes image-like assets for
renderer texture managers; `MeshSource` specializes local 3D geometry assets.
These descriptors are intentionally dependency-free and do not load files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_GEOTIFF_SUFFIXES = {".tif", ".tiff", ".geotiff"}
_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tga", ".webp"}
_MESH_SUFFIXES = {
    ".glb",
    ".gltf",
    ".obj",
    ".ply",
    ".stl",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
}


@dataclass(frozen=True, kw_only=True)
class AssetSource:
    """Reference to an external or in-memory visualization asset."""

    uri: str | Path | None = None
    data: Any | None = None
    kind: str = "asset"
    name: str | None = None
    mime_type: str | None = None
    crs: str | None = None
    bounds: tuple[float, float, float, float] | None = None
    time: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the source has enough information to be resolved."""
        if self.uri is None and self.data is None:
            raise ValueError("AssetSource requires either uri or data")
        if self.bounds is not None and len(self.bounds) != 4:
            raise ValueError("AssetSource bounds must be a 4-tuple")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        kind: str | None = None,
        name: str | None = None,
        mime_type: str | None = None,
        crs: str | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        time: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AssetSource":
        """Create an asset source from a path or URI without opening it."""
        return cls(
            uri=path,
            kind=kind or infer_asset_kind(path),
            name=name or _display_name(path),
            mime_type=mime_type,
            crs=crs,
            bounds=bounds,
            time=time,
            metadata=dict(metadata or {}),
        )

    @property
    def key(self) -> str:
        """Return a stable cache key for this logical source."""
        if self.uri is not None:
            return str(self.uri)
        if self.name:
            return self.name
        return f"{self.kind}:{type(self.data).__module__}.{type(self.data).__name__}"

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable asset summary without embedding data payloads."""
        return {
            "uri": None if self.uri is None else str(self.uri),
            "data_type": None if self.data is None else type(self.data).__name__,
            "kind": self.kind,
            "name": self.name,
            "mime_type": self.mime_type,
            "crs": self.crs,
            "bounds": self.bounds,
            "time": self.time,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, kw_only=True)
class TextureSource(AssetSource):
    """Image-like source intended for backend texture staging and upload."""

    kind: str = "texture"
    codec: str | None = None
    channels: str = "rgba"
    tile_size: tuple[int, int] | None = None
    levels: int | None = None

    def __post_init__(self) -> None:
        """Validate texture metadata while leaving file loading to backends."""
        super().__post_init__()
        if self.tile_size is not None and (
            len(self.tile_size) != 2 or any(size <= 0 for size in self.tile_size)
        ):
            raise ValueError("TextureSource tile_size must contain positive sizes")
        if self.levels is not None and self.levels <= 0:
            raise ValueError("TextureSource levels must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable texture source summary."""
        payload = super().as_dict()
        payload.update(
            {
                "codec": self.codec,
                "channels": self.channels,
                "tile_size": self.tile_size,
                "levels": self.levels,
            }
        )
        return payload


@dataclass(frozen=True, kw_only=True)
class MeshSource(AssetSource):
    """Geometry source intended for renderer or OpenUSD mesh backends."""

    kind: str = "mesh"
    transform: tuple[float, ...] | None = None
    material: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable mesh source summary."""
        payload = super().as_dict()
        payload.update(
            {
                "transform": self.transform,
                "material": dict(self.material),
            }
        )
        return payload


def infer_asset_kind(uri: str | Path) -> str:
    """Infer a coarse asset kind from a path or URI suffix."""
    suffix = _clean_suffix(uri)
    if suffix in _GEOTIFF_SUFFIXES:
        return "geotiff"
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _MESH_SUFFIXES:
        return "mesh"
    return "asset"


def _clean_suffix(uri: str | Path) -> str:
    text = str(uri).split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]
    return Path(text).suffix.lower()


def _display_name(uri: str | Path) -> str:
    text = str(uri).split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]
    name = Path(text).name
    return name or str(uri)
