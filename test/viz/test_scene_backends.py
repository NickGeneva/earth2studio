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
"""Agent-friendly summary: tests for Scene and backend dispatch behavior.

Key APIs under test: `Scene` layer creation, summary backend rendering/saving,
backend registry, `plot`, and Matplotlib installed-or-missing behavior.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from earth2studio.viz import BackendProtocol, LayerProtocol, Scene, SceneProtocol, plot
from earth2studio.viz.backends.base import (
    SummaryBackend,
    VizDependencyError,
    available_backends,
    get_backend,
    register_backend,
)
from earth2studio.viz.backends.matplotlib import MatplotlibBackend
from earth2studio.viz.layers import Layer
from earth2studio.viz.regional import RegionSpec
from earth2studio.viz.styles import LayerStyle


def test_scene_adds_raster_points_and_summary(
    sample_dataarray: xr.DataArray,
    sample_frame: pd.DataFrame,
) -> None:
    scene = Scene(title="Weather")
    raster = scene.add_raster(
        sample_dataarray,
        variable="t2m",
        time=0,
        lead_time=0,
        colormap="turbo",
        alpha=0.75,
    )
    points = scene.add_points(sample_frame, color="red", size=8)

    assert raster.id == "raster-001"
    assert points.id == "points-002"
    assert raster.style.colormap == "turbo"
    assert raster.style.alpha == 0.75
    assert len(scene.timeline.frames) == 4

    result = scene.render("summary")
    assert result.backend == "summary"
    assert result.output["title"] == "Weather"
    assert [layer["kind"] for layer in result.output["layers"]] == ["raster", "points"]
    assert isinstance(scene, SceneProtocol)
    assert isinstance(raster, LayerProtocol)


def test_scene_layer_visibility_and_removal(sample_dataarray: xr.DataArray) -> None:
    scene = Scene()
    raster = scene.add_raster(sample_dataarray, variable="t2m", time=0, lead_time=0)
    raster.hide()

    assert scene.visible_layers == []
    assert scene.get_layer(raster.name) is raster

    raster.show()
    removed = scene.remove_layer(raster.id)
    assert removed is raster
    assert scene.layers == []


def test_scene_get_layer_rejects_missing() -> None:
    with pytest.raises(KeyError, match="was not found"):
        Scene().get_layer("missing")


def test_scene_rejects_duplicate_layer_ids() -> None:
    scene = Scene()
    layer = Layer(id="same", name="Layer", data=None)
    scene.add_layer(layer)

    with pytest.raises(ValueError, match="already exists"):
        scene.add_layer(Layer(id="same", name="Other", data=None))


def test_scene_adds_regional_layers(
    terrain_dataarray: xr.DataArray,
    cube_dataarray: xr.DataArray,
) -> None:
    region = RegionSpec.from_lonlat_bounds(
        name="region",
        west=-124.0,
        south=32.0,
        east=-118.0,
        north=38.0,
    )
    scene = Scene(region=region)
    terrain = scene.add_terrain(terrain_dataarray, vertical_exaggeration=2.0)
    draped = scene.add_draped_raster(terrain_dataarray, colormap="viridis")
    cube = scene.add_region_cube(
        cube_dataarray,
        variable="q850",
        vertical="height",
        mode="slices",
        levels=[100.0, 500.0],
    )

    assert terrain.kind == "terrain"
    assert terrain.projection.crs == region.crs
    assert terrain.metadata["vertical_exaggeration"] == 2.0
    assert draped.kind == "draped_raster"
    assert cube.kind == "region_cube"
    assert cube.metadata["levels"] == (100.0, 500.0)
    assert scene.summary()["region"]["name"] == "region"


def test_scene_adds_vector_dataset(sample_dataset: xr.Dataset) -> None:
    scene = Scene()
    vector = scene.add_vectors(
        sample_dataset,
        vector=("u10m", "v10m"),
        mode="streamlines",
        width=2.0,
    )

    assert vector.kind == "vectors"
    assert vector.metadata["mode"] == "streamlines"
    assert vector.metadata["vector"] == ("u10m", "v10m")
    assert "u" in vector.data
    assert vector.style.width == 2.0


def test_scene_adds_vector_dict() -> None:
    scene = Scene()
    vector = scene.add_vectors(
        {"x": [0.0], "y": [0.0], "u": [1.0], "v": [0.0]},
        mode="quiver",
    )

    assert vector.data["mode"] == "quiver"
    assert vector.data["u"] == [1.0]


def test_summary_backend_saves_json(
    tmp_path: Path, sample_dataarray: xr.DataArray
) -> None:
    scene = Scene(title="Save")
    scene.add_raster(sample_dataarray, variable="t2m", time=0, lead_time=0)
    path = scene.save(tmp_path / "scene.json", backend="summary")

    payload = json.loads(path.read_text())
    assert payload["title"] == "Save"
    assert payload["layers"][0]["kind"] == "raster"


def test_summary_backend_show_supports_and_animates(
    tmp_path: Path,
    sample_dataarray: xr.DataArray,
) -> None:
    scene = Scene(title="Animate")
    scene.add_raster(sample_dataarray, variable="t2m", time=0, lead_time=0)
    backend = SummaryBackend()

    assert isinstance(backend, BackendProtocol)
    assert backend.supports(scene)
    assert backend.show(scene).output["title"] == "Animate"

    path = backend.animate(scene, tmp_path / "timeline.json")
    payload = json.loads(path.read_text())
    assert "frames" in payload


def test_plot_uses_summary_backend(sample_dataarray: xr.DataArray) -> None:
    output = plot(
        sample_dataarray,
        variable="t2m",
        time=0,
        lead_time=0,
        backend="summary",
        title="Quick plot",
    )

    assert output["title"] == "Quick plot"
    assert output["layers"][0]["name"] == "t2m"


def test_backend_registry() -> None:
    register_backend("unit-summary", SummaryBackend, replace=True)

    assert "summary" in available_backends()
    assert get_backend("unit-summary").name == "summary"

    with pytest.raises(ValueError, match="already registered"):
        register_backend("summary", SummaryBackend)

    with pytest.raises(KeyError, match="Unknown visualization backend"):
        get_backend("does-not-exist")


def test_viz_dependency_error_message() -> None:
    error = VizDependencyError("backend", "package")

    assert error.backend == "backend"
    assert error.package == "package"
    assert "earth2studio[viz]" in str(error)


def test_matplotlib_backend_installed_or_missing(
    tmp_path: Path,
    sample_dataarray: xr.DataArray,
    sample_frame: pd.DataFrame,
) -> None:
    scene = Scene()
    scene.add_raster(
        sample_dataarray,
        variable="t2m",
        time=0,
        lead_time=0,
        gamma=0.8,
        input_range=(0.0, 100.0),
        output_range=(0.0, 1.0),
    )
    scene.add_points(sample_frame, color="red", size=4)
    scene.add_vectors({"x": [0.0], "y": [0.0], "u": [1.0], "v": [0.0]})
    backend = get_backend("matplotlib")

    if importlib.util.find_spec("matplotlib") is None:
        with pytest.raises(VizDependencyError):
            backend.render(scene)
    else:
        result = backend.render(scene)
        assert result.backend == "matplotlib"
        assert result.output is not None
        assert backend.show(scene) is not None
        saved = backend.save(scene, tmp_path / "figure.png")
        assert saved.exists()
        with pytest.raises(NotImplementedError, match="not implemented"):
            backend.animate(scene, tmp_path / "movie.gif")


def test_matplotlib_backend_supports_and_guard_paths(
    sample_dataarray: xr.DataArray,
    cube_dataarray: xr.DataArray,
) -> None:
    backend = MatplotlibBackend()
    scene = Scene()
    scene.add_raster(sample_dataarray, variable="t2m", time=0, lead_time=0)
    assert backend.supports(scene)

    scene.add_region_cube(cube_dataarray)
    assert not backend.supports(scene)


def test_layer_style_merge() -> None:
    style = LayerStyle(colormap="viridis", alpha=0.5, metadata={"a": 1})
    merged = style.merged(
        alpha=0.9,
        gamma=1.2,
        input_range=(0.0, 10.0),
        output_range=(0.1, 0.9),
        color="blue",
        metadata={"b": 2},
    )

    assert merged.colormap == "viridis"
    assert merged.alpha == 0.9
    assert merged.gamma == 1.2
    assert merged.input_range == (0.0, 10.0)
    assert merged.output_range == (0.1, 0.9)
    assert merged.color == "blue"
    assert merged.metadata == {"a": 1, "b": 2}
    assert merged.as_dict()["gamma"] == 1.2
