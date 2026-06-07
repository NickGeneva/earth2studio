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
"""Agent-friendly summary: common cache paths for visualization assets.

Key APIs: `common_cache_root` returns the package-wide Earth2 Studio cache root;
`viz_cache_root` returns the versioned visualization cache directory; and
`readable_cache_filename` creates clear, unhashed, filesystem-safe asset names.
The viz cache intentionally uses `EARTH2STUDIO_CACHE`, not data-source cache
overrides, because default textures are renderer assets.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_VIZ_CACHE_VERSION = "v5"


def common_cache_root(*, create: bool = True) -> Path:
    """Return the package-wide Earth2 Studio cache root."""
    default = Path.home() / ".cache" / "earth2studio"
    root = Path(os.environ.get("EARTH2STUDIO_CACHE", default)).expanduser()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def viz_cache_root(
    *,
    version: str = DEFAULT_VIZ_CACHE_VERSION,
    cache_root: str | Path | None = None,
    create: bool = True,
) -> Path:
    """Return the versioned visualization cache root."""
    base = (
        Path(cache_root).expanduser()
        if cache_root is not None
        else common_cache_root(create=create)
    )
    root = base / "viz" / version
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def readable_cache_filename(name: str, *, suffix: str | None = None) -> str:
    """Return a readable cache filename without hashing the asset name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    if not cleaned:
        raise ValueError("Cache filename name must contain a visible character")
    if suffix is None:
        return cleaned
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    stem = Path(cleaned).stem if Path(cleaned).suffix else cleaned
    return f"{stem}{normalized_suffix}"
