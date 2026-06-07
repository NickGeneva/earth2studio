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
"""Agent-friendly summary: tests for viz asset and texture primitives.

Key APIs under test: asset kind inference, source summaries, texture cache
policy validation, timestamped texture selection, prefetch windows, and scene
asset layer helpers without mocks or renderer dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from earth2studio.viz import (
    DEFAULT_VIZ_CACHE_VERSION,
    AssetSource,
    AssetSourceProtocol,
    MeshSource,
    Scene,
    TextureCachePolicy,
    TextureDomainAsset,
    TextureFrame,
    TextureManagerProtocol,
    TextureSequence,
    TextureSource,
    common_cache_root,
    default_texture_domain,
    infer_asset_kind,
    readable_cache_filename,
    viz_cache_root,
)


def test_asset_source_infers_common_path_kinds(tmp_path: Path) -> None:
    image = AssetSource.from_path(tmp_path / "clouds.png")
    geotiff = AssetSource.from_path("s3://bucket/local_dem.cog.tif?version=1")
    mesh = AssetSource.from_path("terrain.usdz")

    assert image.kind == "image"
    assert image.name == "clouds.png"
    assert geotiff.kind == "geotiff"
    assert mesh.kind == "mesh"
    assert infer_asset_kind("unknown.bin") == "asset"
    assert isinstance(image, AssetSourceProtocol)


def test_asset_sources_validate_and_summarize() -> None:
    asset = TextureSource(
        data=b"jpeg",
        kind="image",
        name="frame",
        bounds=(-180.0, -90.0, 180.0, 90.0),
        codec="jpeg",
        tile_size=(256, 256),
        levels=3,
    )
    mesh = MeshSource(uri="terrain.usd", crs="EPSG:32610", material={"roughness": 1})

    assert asset.key == "frame"
    assert asset.as_dict()["codec"] == "jpeg"
    assert mesh.as_dict()["material"]["roughness"] == 1

    with pytest.raises(ValueError, match="requires either uri or data"):
        AssetSource()
    with pytest.raises(ValueError, match="bounds"):
        AssetSource(data=object(), bounds=(0.0, 1.0, 2.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tile_size"):
        TextureSource(data=b"x", tile_size=(0, 256))
    with pytest.raises(ValueError, match="levels"):
        TextureSource(data=b"x", levels=0)


def test_texture_cache_policy_prefetch_indices() -> None:
    policy = TextureCachePolicy(prefetch_radius=2)

    assert policy.prefetch_indices(2, 5) == (0, 1, 2, 3, 4)
    assert policy.prefetch_indices(0, 5) == (0, 1, 2)
    assert policy.prefetch_indices(0, 0) == ()

    with pytest.raises(IndexError, match="out of range"):
        policy.prefetch_indices(5, 5)
    with pytest.raises(ValueError, match="prefetch_radius"):
        TextureCachePolicy(prefetch_radius=-1)
    with pytest.raises(ValueError, match="max_cpu_frames"):
        TextureCachePolicy(max_cpu_frames=0)
    with pytest.raises(ValueError, match="max_gpu_frames"):
        TextureCachePolicy(max_gpu_frames=0)
    with pytest.raises(ValueError, match="tile_size"):
        TextureCachePolicy(tile_size=(512, 0))
    with pytest.raises(ValueError, match="eviction"):
        TextureCachePolicy(eviction="oldest")


def test_texture_sequence_selects_frames_and_prefetches() -> None:
    times = pd.date_range("2026-06-07", periods=3, freq="h")
    source = TextureSource(uri="frames.jpg", kind="image")
    frames = [
        TextureFrame(source=source, key="layer", index=i, timestamp=time)
        for i, time in enumerate(times)
    ]
    sequence = TextureSequence(
        frames=frames,
        name="forecast",
        loop=True,
        cache_policy=TextureCachePolicy(prefetch_radius=1),
    )

    assert sequence.time_extent == (times[0], times[-1])
    assert sequence.select(index=4).index == 1
    assert sequence.select(timestamp=times[1] + pd.Timedelta(minutes=30)).index == 1
    assert sequence.select(timestamp=times[0] - pd.Timedelta(minutes=1)).index == 0
    assert [frame.index for frame in sequence.prefetch_frames(timestamp=times[1])] == [
        0,
        1,
        2,
    ]
    assert "time:" in frames[0].cache_key
    assert sequence.as_dict()["frame_count"] == 3


def test_texture_sequence_rejects_empty_and_out_of_range() -> None:
    source = TextureSource(uri="frames.jpg", kind="image")
    sequence = TextureSequence(
        frames=[TextureFrame(source=source, index=0)],
        loop=False,
    )

    assert sequence.select().index == 0
    with pytest.raises(IndexError, match="out of range"):
        sequence.select(index=1)
    with pytest.raises(ValueError, match="no frames"):
        TextureSequence().select(index=0)


def test_texture_sequence_cache_metadata_and_mixed_timestamps() -> None:
    class KeyedSource:
        key = "custom-source"

    keyed = KeyedSource()
    tiled = TextureFrame(
        source=keyed,
        index=0,
        tile="z0/x0/y0",
        lod=2,
        shape=(256, 256, 4),
        dtype="uint8",
        metadata={"role": "albedo"},
    )
    untimed = TextureSequence(frames=[tiled], name="Untimed")
    mixed = TextureSequence(
        frames=[
            TextureFrame(source=TextureSource(uri="early.png"), index=0, timestamp="b"),
            TextureFrame(source=TextureSource(uri="late.png"), index=1, timestamp=1),
        ]
    )

    assert "tile:z0/x0/y0" in tiled.cache_key
    assert "lod:2" in tiled.cache_key
    assert tiled.as_dict()["source"]["data_type"] == "KeyedSource"
    assert tiled.as_dict()["metadata"]["role"] == "albedo"
    assert untimed.time_extent is None
    assert untimed.select(index=0).index == 0
    assert untimed.select(timestamp=pd.Timestamp("2026-06-07")).index == 0
    assert mixed.select(timestamp=1).index == 1
    assert "builtins.object" in TextureFrame(source=object()).cache_key
    assert TextureFrame(source=TextureSource(uri="summary.png")).as_dict()["source"][
        "uri"
    ] == "summary.png"


def test_scene_adds_asset_layers_and_texture_sequence() -> None:
    time = pd.Timestamp("2026-06-07T00:00:00")
    sequence = TextureSequence(
        frames=[
            TextureFrame(
                source=TextureSource(uri="frame0.jpg"), index=0, timestamp=time
            )
        ],
        name="Satellite",
    )
    scene = Scene(title="Assets")
    image = scene.add_image(
        sequence,
        bounds=(-180.0, -90.0, 180.0, 90.0),
        alpha=0.5,
    )
    geotiff = scene.add_geotiff(
        "local_dem.tif",
        role="terrain",
        crs="EPSG:32610",
    )
    mesh = scene.add_mesh(
        "terrain.usd",
        crs="EPSG:32610",
        transform=[1.0, 0.0, 0.0, 1.0],
        material={"roughness": 1.0},
    )

    assert image.kind == "image"
    assert image.style.alpha == 0.5
    assert image.metadata["sequence"]["frame_count"] == 1
    assert geotiff.kind == "geotiff"
    assert geotiff.metadata["role"] == "terrain"
    assert geotiff.metadata["asset"]["uri"] == "local_dem.tif"
    assert mesh.kind == "mesh"
    assert mesh.metadata["streaming"] is False
    assert mesh.metadata["asset"]["transform"] == (1.0, 0.0, 0.0, 1.0)
    assert scene.timeline.current == time
    assert scene.render("summary").output["layers"][0]["kind"] == "image"

    with pytest.raises(ValueError, match="GeoTIFF role"):
        scene.add_geotiff("bad.tif", role="unknown")


def test_texture_manager_protocol_runtime_check() -> None:
    class RecordingTextureManager:
        policy = TextureCachePolicy()

        def __init__(self) -> None:
            self.frames: list[Any] = []
            self.released: list[str] = []

        def resolve(self, frame: Any, **kwargs: Any) -> str:
            return f"handle:{frame.cache_key}"

        def prefetch(self, frames: Iterable[Any], **kwargs: Any) -> None:
            self.frames.extend(frames)

        def release_layer(self, layer_id: str) -> None:
            self.released.append(layer_id)

        def clear(self) -> None:
            self.frames.clear()

    manager = RecordingTextureManager()
    frame = TextureFrame(source=TextureSource(uri="frame.jpg"), index=0)

    assert isinstance(manager, TextureManagerProtocol)
    assert manager.resolve(frame).startswith("handle:")
    manager.prefetch([frame])
    manager.release_layer("layer-001")
    manager.clear()
    assert manager.frames == []
    assert manager.released == ["layer-001"]


def test_viz_cache_root_uses_common_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path / "common"))
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", str(tmp_path / "data"))

    root = common_cache_root()
    viz_root = viz_cache_root()

    assert root == tmp_path / "common"
    assert viz_root == tmp_path / "common" / "viz" / DEFAULT_VIZ_CACHE_VERSION
    assert viz_root.is_dir()
    assert (
        readable_cache_filename("Global Base Color", suffix=".ktx2")
        == "Global_Base_Color.ktx2"
    )

    with pytest.raises(ValueError, match="visible character"):
        readable_cache_filename("   ")


def test_default_texture_domain_uses_readable_unhashed_paths(tmp_path: Path) -> None:
    domain = default_texture_domain(cache_root=tmp_path)
    source = domain.source("global_clouds")

    assert (
        source.uri
        == tmp_path / "viz" / "v5" / "default_textures" / "global_clouds.ktx2"
    )
    assert source.codec == "ktx2"
    assert source.bounds == (-180.0, -90.0, 180.0, 90.0)
    assert source.metadata["cache_policy"] == "readable_unhashed_filenames"
    assert source.metadata["clear_cache_name"] is True
    assert domain.ensure_cache().is_dir()
    assert "global_boundaries" in domain.sources()
    assert domain.as_dict()["cache_path"].endswith("default_textures")

    with pytest.raises(KeyError, match="Unknown texture domain asset"):
        domain.source("missing")


def test_scene_adds_default_texture_from_domain(tmp_path: Path) -> None:
    domain = default_texture_domain(cache_root=tmp_path)
    scene = Scene()
    layer = scene.add_default_texture("global_boundaries", domain=domain, alpha=0.25)

    assert layer.kind == "image"
    assert layer.name == "global_boundaries"
    assert layer.style.alpha == 0.25
    assert layer.metadata["asset"]["uri"].endswith(
        "viz\\v5\\default_textures\\global_boundaries.ktx2"
    ) or layer.metadata["asset"]["uri"].endswith(
        "viz/v5/default_textures/global_boundaries.ktx2"
    )
    assert layer.metadata["texture_domain"]["metadata"]["cache_policy"] == (
        "readable_unhashed_filenames"
    )


def test_texture_domain_asset_named_uses_visible_filename() -> None:
    asset = TextureDomainAsset.named(
        "regional default boundaries",
        suffix="png",
        role="boundaries",
        codec="png",
    )

    assert asset.filename == "regional_default_boundaries.png"
    assert asset.as_dict()["role"] == "boundaries"
