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
import sys
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from earth2studio.viz import (
    BackendProtocol,
    LayerProtocol,
    ProjectionSpec,
    Scene,
    SceneEventProtocol,
    SceneProtocol,
    SceneSessionProtocol,
    plot,
)
from earth2studio.viz.assets import TextureSource
from earth2studio.viz.backends import cartopy as cartopy_backend
from earth2studio.viz.backends.anari import AnariBackend
from earth2studio.viz.backends.base import (
    SummaryBackend,
    VizDependencyError,
    available_backends,
    get_backend,
    register_backend,
)
from earth2studio.viz.backends.matplotlib import MatplotlibBackend
from earth2studio.viz.backends.ovrtx import OvrTxBackend
from earth2studio.viz.layers import Layer
from earth2studio.viz.regional import RegionSpec
from earth2studio.viz.styles import LayerStyle
from earth2studio.viz.textures import TextureFrame, TextureSequence


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


def _field(
    data: xr.DataArray,
    variable: str = "t2m",
    *,
    time: int | None = 0,
    lead_time: int | None = None,
) -> xr.DataArray:
    selected = data.sel(variable=variable)
    if time is not None:
        selected = selected.isel(time=time)
    if lead_time is not None:
        selected = selected.isel(lead_time=lead_time)
    return selected


def test_scene_adds_raster_points_and_summary(
    sample_dataarray: xr.DataArray,
    sample_frame: pd.DataFrame,
) -> None:
    scene = Scene(title="Weather")
    raster = scene.add_raster(
        _field(sample_dataarray, lead_time=0),
        name="t2m",
        colormap="turbo",
        alpha=0.75,
    )
    points = scene.add_points(sample_frame, color="red", size=8)

    assert raster.id == "raster-001"
    assert points.id == "points-002"
    assert raster.style.colormap == "turbo"
    assert raster.style.alpha == 0.75
    assert raster.projection.metadata["grid"]["kind"] == "regular_latlon"
    assert len(scene.timeline.frames) == 3

    result = scene.render("summary")
    assert result.backend == "summary"
    assert result.output["title"] == "Weather"
    assert [layer["kind"] for layer in result.output["layers"]] == ["raster", "points"]
    assert isinstance(scene, SceneProtocol)
    assert isinstance(raster, LayerProtocol)


def test_scene_layer_visibility_and_removal(sample_dataarray: xr.DataArray) -> None:
    scene = Scene()
    raster = scene.add_raster(_field(sample_dataarray, lead_time=0), name="t2m")
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
    scene.add_raster(_field(sample_dataarray, lead_time=0), name="t2m")
    path = scene.save(tmp_path / "scene.json", backend="summary")

    payload = json.loads(path.read_text())
    assert payload["title"] == "Save"
    assert payload["layers"][0]["kind"] == "raster"


def test_summary_backend_show_supports_and_animates(
    tmp_path: Path,
    sample_dataarray: xr.DataArray,
    sample_frame: pd.DataFrame,
) -> None:
    scene = Scene(title="Animate")
    raster = scene.add_raster(_field(sample_dataarray, lead_time=0), name="t2m")
    points = scene.add_points(sample_frame.iloc[:1].copy())
    backend = SummaryBackend()

    assert isinstance(backend, BackendProtocol)
    assert backend.supports(scene)
    assert backend.show(scene).output["title"] == "Animate"
    assert backend.show(scene, tag="unit").metadata["kwargs"] == {"tag": "unit"}

    session = scene.show(streaming=True, auto_flush=False, tag="stream")
    assert isinstance(session, SceneSessionProtocol)
    assert session.result.metadata["kwargs"] == {"tag": "stream"}

    raster.update(sample_dataarray.sel(variable="t2m").isel(time=0, lead_time=1))
    points.append(sample_frame.iloc[1:].copy())
    assert session.pending_events
    assert isinstance(session.pending_events[-1], SceneEventProtocol)
    assert len(points.data.table) == 3
    assert session.output["layers"][0]["name"] == "t2m"
    assert session.pending_events

    result = session.flush()
    assert result.output["layers"][0]["name"] == "t2m"
    assert session.pending_events == []

    with session.hold():
        raster.append(
            sample_dataarray.sel(variable="t2m").isel(time=1, lead_time=1),
            time=pd.Timestamp("2026-06-07T18:00:00"),
        )
    assert session.pending_events
    assert raster.data.frame_count == 2
    session.flush()
    raster.hide()
    assert session.pending_events
    session.close()
    assert session.closed

    auto_session = scene.show(streaming=True)
    raster.show()
    assert auto_session.pending_events == []
    auto_session.close()

    path = backend.animate(scene, tmp_path / "timeline.json")
    payload = json.loads(path.read_text())
    assert "frames" in payload


def test_scene_streaming_updates_cover_raster_points_and_textures(
    sample_dataarray: xr.DataArray,
    sample_frame: pd.DataFrame,
) -> None:
    scene = Scene(title="Streaming payloads")
    raster = scene.add_raster(
        _field(sample_dataarray, lead_time=0),
        name="t2m",
    )
    sequence = scene.add_raster(
        _field(sample_dataarray, lead_time=None),
        name="t2m lead times",
    )
    points = scene.add_points(sample_frame.iloc[:1].copy(), name="Stations")
    image = scene.add_image(TextureSource(uri="first.png", time=pd.Timestamp("2026-06-07")))
    session = scene.show(streaming=True, auto_flush=False)

    next_frame = sample_dataarray.sel(variable="t2m").isel(time=1, lead_time=1)
    raster.update(next_frame, quality="analysis")
    raster.append(
        sample_dataarray.sel(variable="t2m").isel(time=0, lead_time=1),
        time=pd.Timestamp("2026-06-07T12:00:00"),
    )
    raster.append(
        sample_dataarray.sel(variable="t2m").isel(time=1, lead_time=0),
        time=pd.Timestamp("2026-06-07T18:00:00"),
    )
    points.append(sample_frame.iloc[1:].copy())

    first_texture = TextureFrame(
        source=TextureSource(uri="frame0.png"),
        index=0,
        timestamp=pd.Timestamp("2026-06-07T00:00:00"),
    )
    image.update(first_texture, role="satellite")
    image.append(
        TextureFrame(
            source=TextureSource(uri="frame1.png"),
            index=1,
            timestamp=pd.Timestamp("2026-06-07T01:00:00"),
        )
    )
    image.append("frame2.png", time=pd.Timestamp("2026-06-07T02:00:00"))
    image.append(
        TextureSequence(
            frames=[
                TextureFrame(
                    source=TextureSource(uri="frame3.png"),
                    index=3,
                    timestamp=pd.Timestamp("2026-06-07T03:00:00"),
                )
            ]
        )
    )

    incoming_sequence = sample_dataarray.sel(variable="t2m").isel(time=1)
    sequence.append(incoming_sequence)
    with pytest.raises(ValueError, match="frame dimensions must match"):
        sequence.append(sample_dataarray.sel(variable="t2m").isel(lead_time=1))

    assert session.pending_events
    result = session.flush()
    assert result.output["layers"][0]["metadata"]["quality"] == "analysis"
    assert raster.data.frame_count == 3
    assert points.data.size == 3
    assert image.data.as_dict()["frame_count"] == 4
    assert image.metadata["role"] == "satellite"
    assert scene.timeline.range() is not None
    session.close()


def test_scene_asset_variants_region_cube_and_vector_payloads(
    tmp_path: Path,
    cube_dataarray: xr.DataArray,
    sample_dataset: xr.Dataset,
) -> None:
    region = RegionSpec.from_lonlat_bounds(
        name="local",
        west=-123.0,
        south=36.0,
        east=-121.0,
        north=38.0,
        target_crs="EPSG:4326",
    )
    scene = Scene(title="Asset variants", region=region)
    image = scene.add_image(
        TextureSource(
            uri="source.png",
            name="source",
            crs="EPSG:4326",
            bounds=(-123.0, 36.0, -121.0, 38.0),
            time=pd.Timestamp("2026-06-07T00:00:00"),
            codec="png",
        ),
        name="renamed",
        mime_type="image/png",
    )
    geotiff = scene.add_geotiff(
        TextureSource(uri="terrain.tif", name="terrain", crs="EPSG:32610"),
        name="Geo",
        time=pd.Timestamp("2026-06-07T01:00:00"),
    )
    mesh_from_asset = scene.add_mesh(
        TextureSource(uri="mesh-source.usd", name="mesh source"),
        transform=(1.0, 0.0, 0.0, 1.0),
    )
    mesh_from_object = scene.add_mesh(
        {"vertices": [(0.0, 0.0, 0.0)]},
        name="Inline mesh",
        material={"roughness": 0.5},
    )
    cube = scene.add_region_cube(
        cube_dataarray.to_dataset(name="q850")["q850"],
        vertical="z",
        levels=[100.0],
    )
    vector = scene.add_vectors(
        sample_dataset,
        vector=("u10m", "v10m", "t2m"),
        mode="glyphs",
    )

    assert image.metadata["asset"]["codec"] == "png"
    assert geotiff.metadata["asset"]["time"] == pd.Timestamp("2026-06-07T01:00:00")
    assert mesh_from_asset.metadata["asset"]["uri"] == "mesh-source.usd"
    assert mesh_from_object.metadata["asset"]["material"]["roughness"] == 0.5
    assert cube.metadata["vertical"] == "z"
    assert cube.metadata["levels"] == (100.0,)
    assert vector.data["w"].name == "t2m"

    saved = scene.save(tmp_path / "scene.json")
    animated = scene.animate(tmp_path / "timeline.json")
    assert json.loads(saved.read_text())["title"] == "Asset variants"
    assert "frames" in json.loads(animated.read_text())


def test_plot_uses_summary_backend(sample_dataarray: xr.DataArray) -> None:
    output = plot(
        _field(sample_dataarray, lead_time=0),
        backend="summary",
        title="Quick plot",
        name="t2m",
    )

    assert output["title"] == "Quick plot"
    assert output["layers"][0]["name"] == "t2m"


def test_backend_registry() -> None:
    register_backend("unit-summary", SummaryBackend, replace=True)

    assert "summary" in available_backends()
    assert "anari" in available_backends()
    assert "cartopy" in available_backends()
    assert "ovrtx" in available_backends()
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
        _field(sample_dataarray, lead_time=0),
        name="t2m",
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
        session = backend.show(scene, streaming=True, auto_flush=False)
        assert session.output is not None
        session.close()
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
        _field(sample_dataarray, lead_time=None),
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
        _field(sample_dataarray, lead_time=None),
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
    assert axes[0][0].title == "t2m | time=2026-06-07T00:00:00"
    assert axes[1][0].title == "Points"
    assert axes[1][1].axis_off
    assert axes[0][0].calls[0][0] == "pcolormesh"
    assert any(name == "scatter" for name, _ in axes[1][0].calls)
    assert any(name == "add_feature" for name, _ in axes[0][0].calls)

    shown = backend.show(scene)
    assert shown.title == "Projected"
    session = backend.show(scene, streaming=True, auto_flush=False)
    assert session.output.title == "Projected"
    session.close()
    saved = backend.save(scene, tmp_path / "cartopy.txt")
    assert saved.read_text() == "saved"


def test_ovrtx_backend_browser_session_and_dependency_gate(
    tmp_path: Path,
    sample_dataarray: xr.DataArray,
    sample_frame: pd.DataFrame,
) -> None:
    scene = Scene(title="OVRTX globe")
    scene.add_default_texture()
    raster = scene.add_raster(
        _field(sample_dataarray, lead_time=None),
        name="t2m",
        alpha=0.8,
    )
    scene.add_points(sample_frame, name="Stations")
    backend = OvrTxBackend()

    assert backend.supports(scene)
    assert not backend.supports(
        Scene(layers=[Layer(id="volume", name="Volume", data=None, kind="volume")])
    )

    session = scene.show(
        "ovrtx",
        streaming=True,
        auto_flush=False,
        open_browser=False,
        output_dir=tmp_path / "ovrtx",
        width=640,
        height=360,
    )

    assert session.backend == "ovrtx"
    assert session.url.startswith("file:")
    assert session.html_path.exists()
    assert session.payload["layout"] == {
        "viewport": "streamed_video",
        "timeline": "bottom",
        "layers": "upper_right",
        "camera": "orbit",
    }
    assert session.payload["render"]["transport"] == "ovstream_webrtc"
    assert session.payload["render"]["resolution"] == (640, 360)
    assert session.payload["stream"]["input"] == "nvst_native"
    assert session.payload["default_texture"]["source"]["name"] == "global_base_color"
    assert 'def RenderProduct "Viewport"' in session.payload["stage"]
    assert "rel camera = </Session/Camera>" in session.payload["stage"]
    assert "remote-video" in session.html
    assert "layer-list" in session.html
    assert "coverage" in session.html
    assert "_repr_html_" not in session._repr_html_()
    assert session.output["title"] == "OVRTX globe"

    raster.hide()
    assert session.pending_events
    assert session.payload["layers"][1]["visible"] is True
    session.flush()
    assert session.payload["layers"][1]["visible"] is False
    with session.hold():
        raster.show()
    assert session.pending_events
    assert session.flush().output["layers"][1]["visible"] is True
    session.close()
    session.update(object())
    assert session.pending_events == []
    assert session.flush().output["title"] == "OVRTX globe"
    session.close()

    empty_payload = backend.render(Scene()).output
    assert empty_payload["timeline"]["frames"] == []
    assert empty_payload["default_texture"]["layer_id"] is None

    saved = backend.save(scene, tmp_path / "ovrtx.json")
    assert json.loads(saved.read_text())["render"]["renderer"] == "ovrtx"
    animated = backend.animate(scene, tmp_path / "ovrtx_timeline.json")
    assert "frames" in json.loads(animated.read_text())

    notebook = backend.show(
        scene,
        viewer="notebook",
        open_browser=True,
        output_dir=tmp_path / "ovrtx-notebook",
        title="Notebook globe",
    )
    assert notebook.viewer == "notebook"
    assert notebook.payload["title"] == "Notebook globe"
    notebook.close()

    with pytest.raises(ValueError, match="browser"):
        backend.show(scene, viewer="desktop", open_browser=False)

    missing = next(
        (
            package
            for package in ("ovrtx", "ovstream")
            if importlib.util.find_spec(package) is None
        ),
        None,
    )
    if missing is None:
        assert backend.render(scene, require_renderer=True).metadata["runtime"]["ready"]
    else:
        with pytest.raises(VizDependencyError) as error:
            backend.render(scene, require_renderer=True)
        assert error.value.package == missing


def test_anari_backend_sdk_viewer_handoff_and_dependency_gate(
    tmp_path: Path,
    sample_dataarray: xr.DataArray,
    sample_frame: pd.DataFrame,
) -> None:
    scene = Scene(title="ANARI native")
    raster = scene.add_raster(
        _field(sample_dataarray, lead_time=None),
        name="t2m",
        alpha=0.8,
    )
    scene.add_points(sample_frame, name="Stations")
    backend = AnariBackend()

    assert backend.supports(scene)
    session = scene.show(
        "anari",
        streaming=True,
        auto_flush=False,
        open_viewer=False,
        output_dir=tmp_path / "anari",
        width=640,
        height=360,
        library="helide",
    )

    assert session.backend == "anari"
    assert session.descriptor_path.exists()
    assert session.payload["layout"] == {
        "viewport": "anari_sdk_interactive_viewer",
        "timeline": "viewer_control",
        "layers": "viewer_control",
        "camera": "orbit",
    }
    assert session.payload["render"]["renderer"] == "anari"
    assert session.payload["render"]["viewer_component"] == (
        "anari_sdk_interactive_viewer"
    )
    assert session.payload["render"]["library"] == "helide"
    assert session.payload["render"]["resolution"] == (640, 360)
    assert session.payload["runtime"]["library"] == "helide"
    assert isinstance(session.payload["runtime"]["sdk_viewer"], bool)
    assert session.payload["handoff"]["format"] == (
        "earth2studio.viz.anari.session.v1"
    )
    assert session.payload["layers"][0]["anari_target"] == (
        "textured_surface_or_sampler"
    )
    assert session.payload["layers"][1]["anari_target"] == "sphere_or_glyph_geometry"

    raster.hide()
    assert session.pending_events
    assert session.payload["layers"][0]["visible"] is True
    session.flush()
    assert session.payload["layers"][0]["visible"] is False
    with session.hold():
        raster.show()
    assert session.pending_events
    assert session.flush().output["layers"][0]["visible"] is True
    session.close()
    session.update(object())
    assert session.pending_events == []
    assert session.flush().output["title"] == "ANARI native"
    session.close()

    saved = backend.save(scene, tmp_path / "anari.json")
    assert json.loads(saved.read_text())["render"]["renderer"] == "anari"
    animated = backend.animate(scene, tmp_path / "anari_timeline.json")
    assert "frames" in json.loads(animated.read_text())

    if importlib.util.find_spec("anari") is None:
        with pytest.raises(VizDependencyError) as error:
            backend.render(scene, require_renderer=True)
        assert error.value.package == "anari"
    else:
        assert backend.render(scene, require_renderer=True).metadata["runtime"]["ready"]

    with pytest.raises(VizDependencyError) as error:
        backend.render(
            scene,
            require_viewer=True,
            viewer_executable=tmp_path / "missing-anari-viewer",
        )
    assert error.value.package == "anariViewer"

    launched = backend.show(
        scene,
        open_viewer=True,
        output_dir=tmp_path / "anari-launched",
        viewer_executable=sys.executable,
        viewer_args=("-c", "import time; time.sleep(30)"),
        title="Launched ANARI",
    )
    assert launched.payload["title"] == "Launched ANARI"
    assert launched.process is not None
    assert launched.process.poll() is None
    assert launched.launch() is launched.process
    launched.result = None
    assert launched.output["title"] == "Launched ANARI"
    process = launched.process
    launched.close()
    process.wait(timeout=5)


def test_matplotlib_backend_renders_raster_sequences_as_layer_rows(
    sample_dataarray: xr.DataArray,
) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib not installed")

    scene = Scene(title="Layer rows")
    scene.add_raster(
        _field(sample_dataarray, lead_time=None),
        name="t2m",
        colormap="viridis",
    )
    scene.add_raster(
        _field(sample_dataarray, "u10m", lead_time=None),
        name="u10m",
        colormap="plasma",
    )

    result = scene.render("matplotlib")

    assert result.backend == "matplotlib"
    assert result.output is not None
    axes = result.metadata["axes"]
    assert axes.shape == (2, 2)
    assert axes[0][0].get_title() == "t2m | time=2026-06-07T00:00:00"
    assert axes[1][1].get_title() == "u10m | time=2026-06-07T06:00:00"


def test_matplotlib_backend_renders_multiple_rasters_as_layer_rows(
    sample_dataarray: xr.DataArray,
) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib not installed")

    scene = Scene(title="Raster rows")
    scene.add_raster(
        _field(sample_dataarray, lead_time=0),
        name="t2m",
    )
    scene.add_raster(
        _field(sample_dataarray, "u10m", lead_time=0),
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
    scene.add_raster(_field(sample_dataarray, lead_time=0), name="t2m")
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
