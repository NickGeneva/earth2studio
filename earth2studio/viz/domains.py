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
"""Agent-friendly summary: default texture domains for visualization scenes.

Key APIs: `TextureDomainAsset` describes one named default texture; `TextureDomain`
maps clear asset keys to cache paths and `TextureSource` descriptors; and
`default_texture_domain` provides the global default domain for base color,
topography, clouds, and boundary overlays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from earth2studio.viz.assets import TextureSource
from earth2studio.viz.cache import (
    DEFAULT_VIZ_CACHE_VERSION,
    readable_cache_filename,
    viz_cache_root,
)

GLOBAL_LATLON_BOUNDS = (-180.0, -90.0, 180.0, 90.0)
DEFAULT_TEXTURE_DOMAIN_NAME = "earth2studio_default_global"


@dataclass(frozen=True, kw_only=True)
class TextureDomainAsset:
    """Named default texture asset with a readable cache filename."""

    key: str
    filename: str
    role: str = "texture"
    kind: str = "image"
    codec: str | None = "ktx2"
    crs: str | None = "EPSG:4326"
    bounds: tuple[float, float, float, float] | None = GLOBAL_LATLON_BOUNDS
    channels: str = "rgba"
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def named(
        cls,
        key: str,
        *,
        suffix: str = ".ktx2",
        **kwargs: Any,
    ) -> "TextureDomainAsset":
        """Create a default asset with an unhashed filename derived from `key`."""
        return cls(
            key=key,
            filename=readable_cache_filename(key, suffix=suffix),
            **kwargs,
        )

    def source(
        self,
        *,
        cache_path: str | Path,
        domain: str,
        version: str,
    ) -> TextureSource:
        """Return a texture source pointing at this asset's cache path."""
        metadata = {
            "domain": domain,
            "version": version,
            "role": self.role,
            "default_texture": True,
            "optimized": self.codec == "ktx2",
            "clear_cache_name": True,
            "cache_policy": "readable_unhashed_filenames",
            "cache_root": "EARTH2STUDIO_CACHE",
            **self.metadata,
        }
        if self.description is not None:
            metadata["description"] = self.description
        return TextureSource(
            uri=Path(cache_path) / self.filename,
            kind=self.kind,
            name=self.key,
            crs=self.crs,
            bounds=self.bounds,
            metadata=metadata,
            codec=self.codec,
            channels=self.channels,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable asset-domain summary."""
        return {
            "key": self.key,
            "filename": self.filename,
            "role": self.role,
            "kind": self.kind,
            "codec": self.codec,
            "crs": self.crs,
            "bounds": self.bounds,
            "channels": self.channels,
            "description": self.description,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, kw_only=True)
class TextureDomain:
    """Collection of named default texture assets backed by a cache directory."""

    name: str = DEFAULT_TEXTURE_DOMAIN_NAME
    version: str = DEFAULT_VIZ_CACHE_VERSION
    cache_root: str | Path | None = None
    assets: tuple[TextureDomainAsset, ...] = field(default_factory=tuple)
    directory_name: str = "default_textures"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        """Return this domain's cache directory path."""
        return (
            viz_cache_root(version=self.version, cache_root=self.cache_root)
            / self.directory_name
        )

    def ensure_cache(self) -> Path:
        """Create the domain cache directory and return it."""
        root = self.root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def asset(self, key: str) -> TextureDomainAsset:
        """Return a domain asset by key."""
        for asset in self.assets:
            if asset.key == key:
                return asset
        known = ", ".join(sorted(asset.key for asset in self.assets))
        raise KeyError(f"Unknown texture domain asset {key!r}. Available: {known}")

    def source(self, key: str) -> TextureSource:
        """Return a texture source for a named default asset."""
        return self.asset(key).source(
            cache_path=self.ensure_cache(),
            domain=self.name,
            version=self.version,
        )

    def sources(self) -> dict[str, TextureSource]:
        """Return texture sources for all assets in this domain."""
        return {asset.key: self.source(asset.key) for asset in self.assets}

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable texture-domain summary."""
        return {
            "name": self.name,
            "version": self.version,
            "cache_path": str(self.root),
            "directory_name": self.directory_name,
            "assets": [asset.as_dict() for asset in self.assets],
            "metadata": dict(self.metadata),
        }


def default_texture_domain(
    *,
    cache_root: str | Path | None = None,
    version: str = DEFAULT_VIZ_CACHE_VERSION,
) -> TextureDomain:
    """Return the global default texture domain."""
    assets = (
        TextureDomainAsset.named(
            "global_base_color",
            role="base_color",
            description="Optimized global base color texture.",
        ),
        TextureDomainAsset.named(
            "global_topography",
            role="topography",
            description="Optimized global topography and terrain shading texture.",
        ),
        TextureDomainAsset.named(
            "global_clouds",
            role="clouds",
            description="Optimized global cloud texture overlay.",
        ),
        TextureDomainAsset.named(
            "global_boundaries",
            role="boundaries",
            description="Optimized global political and coastline boundary overlay.",
        ),
    )
    return TextureDomain(
        name=DEFAULT_TEXTURE_DOMAIN_NAME,
        version=version,
        cache_root=cache_root,
        assets=assets,
        metadata={
            "cache_policy": "readable_unhashed_filenames",
            "cache_root": "EARTH2STUDIO_CACHE",
        },
    )
