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
backend registry, `plot`, and Matplotlib/Cartopy installed-or-missing behavior.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from earth2studio.viz import (
    BackendProtocol,
    LayerProtocol,
    ProjectionSpec,
    Scene,
    SceneProtocol,
    plot,
)
from earth2studio.viz.backends import cartopy as cartopy_backend
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


class _FakeCRS:
    def __init__(self, kind: str, **kwargs: object):
        self.kind = kind
        self.kwargs = kwargs


class _FakeCCRS:
    def PlateCarree(self, **kwargs: object) -> _FakeCRS:
        return _FakeCRS("platecarree", **kwargs)

    def Robinson(self, **kwargs: object) -> _FakeCRS:
        return _FakeCRS("robinson", **kwargs)

    def Mollweide(self, **kwargs: object) -> _FakeCRS:
        return _FakeCRS("mollweide", **kwargs)

    def Orthographic(self, **kwargs: object) -> _FakeCRS:
        return _FakeCRS("orthographic", **kwargs)

    def LambertConformal(self, **kwargs: object) -> _FakeCRS:
        return _FakeCRS("lambert_conformal", **kwargs)

    def Globe(self, **kwargs: object) -> dict[str, object]:
        return dict(kwargs)


class _FakeFeature:
    def __init__(self, name: str):
        self.name = name
        self.scale: str | None = None

    def with_scale(self, scale: str) -> "_FakeFeature":
        feature = _FakeFeature(self.name)
        feature.scale = scale
        return feature


class _FakeCFeature:
    STATES = _FakeFeature("states")
    LAND = _FakeFeature("land")


class _FakeAxis:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.title: str | None = None
        self.axis_off = False

    def pcolormesh(self, *args: object, **kwargs: object) -> str:
        self.calls.append(("pcolormesh", dict(kwargs)))
        return "mesh"

    def scatter(self, *args: object, **kwargs: object) -> str:
        self.calls.append(("scatter", dict(kwargs)))
        return "points"

    def set_extent(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("set_extent", dict(kwargs)))

    def coastlines(self, **kwargs: object) -> None:
        self.calls.append(("coastlines", dict(kwargs)))

    def add_feature(self, feature: object, **kwargs: object) -> None:
        self.calls.append(("add_feature", {"feature": feature, **kwargs}))

    def gridlines(self, **kwargs: object) -> None:
        self.calls.append(("gridlines", dict(kwargs)))

    def set_title(self, title: str) -> None:
        self.title = title

    def set_axis_off(self) -> None:
        self.axis_off = True


class _FakeFigure:
    def __init__(self) -> None:
        self.title: str | None = None
        self.colorbars = 0

    def suptitle(self, title: str) -> None:
        self.title = title

    def colorbar(self, *args: object, **kwargs: object) -> None:
        self.colorbars += 1

    def savefig(self, path: str | Path) -> None:
        Path(path).write_text("saved")


class _FakePyplot:
    def subplots(
        self,
        nrows: int,
        ncols: int,
        **kwargs: object,
    ) -> tuple[_FakeFigure, list[list[_FakeAxis]]]:
        axes = [[_FakeAxis() for _ in range(ncols)] for _ in range(nrows)]
        return _FakeFigure(), axes


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
    assert raster.projection.metadata["grid"]["kind"] == "regular_latlon"
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
    assert backend.show(scene, tag="unit").metadata["kwargs"] == {"tag": "unit"}
    with pytest.raises(NotImplementedError, match="streaming sessions"):
        scene.show(streaming=True)

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
    assert "cartopy" in available_backends()
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
        with pytest.raises(NotImplementedError, match="streaming sessions"):
            backend.show(scene, streaming=True)
        saved = backend.save(scene, tmp_path / "figure.png")
        assert saved.exists()
        with pytest.raises(NotImplementedError, match="not implemented"):
            backend.animate(scene, tmp_path / "movie.gif")


def test_cartopy_backend_installed_or_missing(
    tmp_path: Path,
    sample_dataarray: xr.DataArray,
    sample_frame: pd.DataFrame,
) -> None:
    projection = ProjectionSpec(
        kind="robinson",
        metadata={"coastlines": False, "gridlines": False},
    )
    scene = Scene(title="Cartopy")
    scene.add_raster(
        sample_dataarray,
        variable="t2m",
        time=0,
        name="t2m",
        projection=projection,
    )
    scene.add_points(
        sample_frame,
        color="red",
        size=4,
        projection=projection,
    )
    backend = get_backend("cartopy")

    missing = next(
        (
            package
            for package in ("cartopy", "matplotlib")
            if importlib.util.find_spec(package) is None
        ),
        None,
    )
    if missing is not None:
        with pytest.raises(VizDependencyError) as error:
            backend.render(scene)
        assert error.value.package == missing
    else:
        result = backend.render(scene)
        assert result.backend == "cartopy"
        assert result.output is not None
        assert result.metadata["axes"].shape == (2, 2)
        saved = backend.save(scene, tmp_path / "cartopy.png")
        assert saved.exists()
        with pytest.raises(NotImplementedError, match="not implemented"):
            backend.animate(scene, tmp_path / "cartopy.gif")


@pytest.mark.parametrize(
    ("projection", "expected_kind"),
    [
        (ProjectionSpec(kind="plate_carree"), "platecarree"),
        (ProjectionSpec(kind="robinson"), "robinson"),
        (ProjectionSpec(kind="mollweide"), "mollweide"),
        (
            ProjectionSpec(
                kind="orthographic",
                metadata={"central_longitude": 300.0, "central_latitude": 10.0},
            ),
            "orthographic",
        ),
        (ProjectionSpec(kind="unknown"), "platecarree"),
    ],
)
def test_cartopy_projection_lowering(
    projection: ProjectionSpec,
    expected_kind: str,
) -> None:
    lowered = cartopy_backend._cartopy_projection(projection, _FakeCCRS())

    assert lowered.kind == expected_kind


def test_cartopy_backend_render_with_lightweight_plot_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_dataarray: xr.DataArray,
    sample_frame: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        cartopy_backend,
        "_cartopy",
        lambda: (_FakeCCRS(), _FakeCFeature()),
    )
    monkeypatch.setattr(cartopy_backend, "_pyplot", _FakePyplot)
    projection = ProjectionSpec(
        kind="lambert_conformal",
        metadata={
            "central_longitude": 262.5,
            "central_latitude": 38.5,
            "standard_parallels": (38.5, 38.5),
            "globe_semimajor_axis": 6371229,
            "globe_semiminor_axis": 6371229,
            "extent": (-110.0, -85.0, 30.0, 47.0),
            "states": True,
            "land": True,
            "gridline_labels": True,
        },
    )
    scene = Scene(title="Projected")
    scene.add_raster(
        sample_dataarray,
        variable="t2m",
        time=0,
        name="t2m",
        projection=projection,
    )
    scene.add_points(
        sample_frame,
        color="red",
        size=4,
        projection=projection,
    )
    backend = cartopy_backend.CartopyBackend()

    assert backend.supports(scene)
    result = backend.render(scene, colorbar=True)

    axes = result.metadata["axes"]
    assert result.backend == "cartopy"
    assert result.output.title == "Projected"
    assert result.output.colorbars == 2
    assert axes[0][0].title == "t2m | lead_time=0 h"
    assert axes[1][0].title == "Points"
    assert axes[1][1].axis_off
    assert axes[0][0].calls[0][0] == "pcolormesh"
    assert any(name == "scatter" for name, _ in axes[1][0].calls)
    assert any(name == "add_feature" for name, _ in axes[0][0].calls)

    shown = backend.show(scene)
    assert shown.title == "Projected"
    with pytest.raises(NotImplementedError, match="streaming sessions"):
        backend.show(scene, streaming=True)
    saved = backend.save(scene, tmp_path / "cartopy.txt")
    assert saved.read_text() == "saved"


def test_matplotlib_backend_renders_raster_sequences_as_layer_rows(
    sample_dataarray: xr.DataArray,
) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib not installed")

    scene = Scene(title="Layer rows")
    scene.add_raster(
        sample_dataarray,
        variable="t2m",
        time=0,
        name="t2m",
        colormap="viridis",
    )
    scene.add_raster(
        sample_dataarray,
        variable="u10m",
        time=0,
        name="u10m",
        colormap="plasma",
    )

    result = scene.render("matplotlib")

    assert result.backend == "matplotlib"
    assert result.output is not None
    axes = result.metadata["axes"]
    assert axes.shape == (2, 2)
    assert axes[0][0].get_title() == "t2m | lead_time=0 h"
    assert axes[1][1].get_title() == "u10m | lead_time=6 h"


def test_matplotlib_backend_renders_multiple_rasters_as_layer_rows(
    sample_dataarray: xr.DataArray,
) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib not installed")

    scene = Scene(title="Raster rows")
    scene.add_raster(
        sample_dataarray,
        variable="t2m",
        time=0,
        lead_time=0,
        name="t2m",
    )
    scene.add_raster(
        sample_dataarray,
        variable="u10m",
        time=0,
        lead_time=0,
        name="u10m",
    )

    result = scene.render("matplotlib")

    axes = result.metadata["axes"]
    assert axes.shape == (2, 1)
    assert axes[0][0].get_title() == "t2m"
    assert axes[1][0].get_title() == "u10m"


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
