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
"""Agent-friendly summary: timeline-aware texture streaming primitives.

Key APIs: `TextureFrame` identifies one texture payload, tile, LOD, timestamp,
and cache key; `TextureSequence` selects frames by index or timestamp and
returns prefetch candidates; `TextureCachePolicy` describes backend-internal
CPU/GPU cache budgets, prefetch radius, async staging, and eviction strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from earth2studio.viz.assets import AssetSource


@dataclass(frozen=True, kw_only=True)
class TextureCachePolicy:
    """Backend-internal policy for staging and GPU texture residency."""

    memory_budget: int | str | None = "25%"
    max_cpu_frames: int | None = 8
    max_gpu_frames: int | None = 4
    prefetch_radius: int = 2
    tile_size: tuple[int, int] | None = (512, 512)
    eviction: str = "timeline_lru"
    decode_async: bool = True
    upload_async: bool = True
    pin_current_frame: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate cache sizing and prefetch choices."""
        if self.prefetch_radius < 0:
            raise ValueError("TextureCachePolicy prefetch_radius must be non-negative")
        if self.max_cpu_frames is not None and self.max_cpu_frames <= 0:
            raise ValueError("TextureCachePolicy max_cpu_frames must be positive")
        if self.max_gpu_frames is not None and self.max_gpu_frames <= 0:
            raise ValueError("TextureCachePolicy max_gpu_frames must be positive")
        if self.tile_size is not None and (
            len(self.tile_size) != 2 or any(size <= 0 for size in self.tile_size)
        ):
            raise ValueError("TextureCachePolicy tile_size must contain positive sizes")
        if self.eviction not in {"lru", "timeline_lru", "timeline_distance"}:
            raise ValueError("TextureCachePolicy eviction is not supported")

    def prefetch_indices(self, index: int, frame_count: int) -> tuple[int, ...]:
        """Return frame indices that should be prefetched around `index`."""
        if frame_count <= 0:
            return ()
        if index < 0 or index >= frame_count:
            raise IndexError("Texture prefetch index is out of range")
        start = max(0, index - self.prefetch_radius)
        stop = min(frame_count, index + self.prefetch_radius + 1)
        return tuple(range(start, stop))


@dataclass(frozen=True, kw_only=True)
class TextureFrame:
    """One logical texture frame, tile, or LOD entry."""

    source: Any
    key: str | None = None
    index: int | None = None
    timestamp: Any | None = None
    tile: str | None = None
    lod: int | None = None
    bounds: tuple[float, float, float, float] | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        """Return a cache key suitable for CPU staging and GPU residency maps."""
        parts = [self.key or _source_key(self.source)]
        if self.index is not None:
            parts.append(f"index:{self.index}")
        if self.timestamp is not None:
            parts.append(f"time:{self.timestamp}")
        if self.tile is not None:
            parts.append(f"tile:{self.tile}")
        if self.lod is not None:
            parts.append(f"lod:{self.lod}")
        return "|".join(parts)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable frame summary without embedding texture bytes."""
        return {
            "source": _source_summary(self.source),
            "key": self.key,
            "cache_key": self.cache_key,
            "index": self.index,
            "timestamp": self.timestamp,
            "tile": self.tile,
            "lod": self.lod,
            "bounds": self.bounds,
            "shape": self.shape,
            "dtype": self.dtype,
            "metadata": dict(self.metadata),
        }


@dataclass
class TextureSequence:
    """Ordered texture frames with timestamp selection and prefetch helpers."""

    frames: list[TextureFrame] = field(default_factory=list)
    name: str | None = None
    loop: bool = False
    cache_policy: TextureCachePolicy = field(default_factory=TextureCachePolicy)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def time_extent(self) -> tuple[Any, Any] | None:
        """Return first and last timestamp when frames carry time metadata."""
        timestamps = [
            frame.timestamp for frame in self.frames if frame.timestamp is not None
        ]
        if not timestamps:
            return None
        ordered = sorted(timestamps, key=_timestamp_key)
        return ordered[0], ordered[-1]

    def append(self, frame: TextureFrame) -> TextureFrame:
        """Append a texture frame and return it."""
        self.frames.append(frame)
        return frame

    def select(
        self,
        *,
        index: int | None = None,
        timestamp: Any | None = None,
    ) -> TextureFrame:
        """Select a frame by index, nearest previous timestamp, or first frame."""
        selected = self.index_for(index=index, timestamp=timestamp)
        return self.frames[selected]

    def index_for(
        self,
        *,
        index: int | None = None,
        timestamp: Any | None = None,
    ) -> int:
        """Return the selected frame index without materializing the frame."""
        if not self.frames:
            raise ValueError("TextureSequence has no frames")
        if index is not None:
            return self._normalize_index(index)
        if timestamp is not None:
            return self._index_for_timestamp(timestamp)
        return 0

    def prefetch_frames(
        self,
        *,
        index: int | None = None,
        timestamp: Any | None = None,
        policy: TextureCachePolicy | None = None,
    ) -> tuple[TextureFrame, ...]:
        """Return frames that a backend should stage around the active frame."""
        active_index = self.index_for(index=index, timestamp=timestamp)
        active_policy = policy or self.cache_policy
        indices = active_policy.prefetch_indices(active_index, len(self.frames))
        return tuple(self.frames[i] for i in indices)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable sequence summary."""
        return {
            "name": self.name,
            "loop": self.loop,
            "time_extent": self.time_extent,
            "frame_count": len(self.frames),
            "metadata": dict(self.metadata),
        }

    def _normalize_index(self, index: int) -> int:
        if self.loop:
            return index % len(self.frames)
        if index < 0 or index >= len(self.frames):
            raise IndexError("TextureSequence index is out of range")
        return index

    def _index_for_timestamp(self, timestamp: Any) -> int:
        timed = [
            (index, frame.timestamp)
            for index, frame in enumerate(self.frames)
            if frame.timestamp is not None
        ]
        if not timed:
            return 0
        ordered = sorted(timed, key=lambda item: _timestamp_key(item[1]))
        selected = ordered[0][0]
        for index, frame_time in ordered:
            if _timestamp_leq(frame_time, timestamp):
                selected = index
            else:
                break
        return selected


def _source_key(source: Any) -> str:
    if isinstance(source, AssetSource):
        return source.key
    key = getattr(source, "key", None)
    if key is not None:
        return str(key)
    return f"{type(source).__module__}.{type(source).__name__}"


def _source_summary(source: Any) -> Any:
    if hasattr(source, "as_dict"):
        return source.as_dict()
    return {"data_type": type(source).__name__}


def _timestamp_key(value: Any) -> str:
    return str(value)


def _timestamp_leq(left: Any, right: Any) -> bool:
    try:
        return bool(left <= right)
    except TypeError:
        return _timestamp_key(left) <= _timestamp_key(right)
