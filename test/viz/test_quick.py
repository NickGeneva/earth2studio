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
"""Agent-friendly summary: tests for quick viz plotting helpers.

Key APIs under test: `plot_raster_grid`, `save_raster_grid`, `plot_series`,
`save_series`, `plot_points`, `save_points`, `plot_tracks`, and `save_tracks`
using real xarray and pandas data instead of mocks.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from earth2studio import viz


def test_plot_and_save_raster_grid(
    tmp_path: Path,
    sample_dataarray: xr.DataArray,
) -> None:
    panels = [
        viz.raster_panel(
            sample_dataarray,
            variable="t2m",
            time=0,
            lead_time=0,
            title="t2m",
            colormap="viridis",
        ),
        viz.raster_panel(
            sample_dataarray,
            variable="u10m",
            time=0,
            lead_time=0,
            title="u10m",
            colorbar_label="u10m",
        ),
    ]

    fig = viz.plot_raster_grid(panels, ncols=2, title="fields")
    assert len(fig.axes) == 4

    path = viz.save_raster_grid(panels, tmp_path / "grid.png", ncols=2)
    assert path.exists()


def test_raster_dataarray_constructor() -> None:
    data = viz.raster_dataarray(
        [[1.0, 2.0], [3.0, 4.0]],
        lat=[10.0, 11.0],
        lon=[20.0, 21.0],
        name="t2m",
        attrs={"units": "K"},
    )

    assert data.dims == ("lat", "lon")
    assert data.name == "t2m"
    assert data.attrs["units"] == "K"


def test_raster_dataarray_constructor_with_curvilinear_coords() -> None:
    lon, lat = np.meshgrid([20.0, 21.0], [10.0, 11.0])

    data = viz.raster_dataarray(
        [[1.0, 2.0], [3.0, 4.0]],
        lat=lat,
        lon=lon,
        name="refc",
    )

    assert data.dims == ("y", "x")
    assert data.coords["lat"].dims == ("y", "x")
    assert data.coords["lon"].dims == ("y", "x")


def test_plot_raster_grid_rejects_bad_layout(sample_dataarray: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="At least one panel"):
        viz.plot_raster_grid([])

    with pytest.raises(ValueError, match="ncols"):
        viz.plot_raster_grid([sample_dataarray], ncols=0)

    with pytest.raises(NotImplementedError, match="matplotlib"):
        viz.plot_raster_grid([sample_dataarray], backend="summary")


def test_plot_and_save_series(tmp_path: Path) -> None:
    panel = viz.series_panel(
        pd.date_range("2026-06-07", periods=3, freq="h"),
        [0.0, 1.0, 0.5],
        label="soi",
    )

    fig = viz.plot_series([panel], title="SOI", ylabel="index")
    assert fig.axes[0].get_ylabel() == "index"

    path = viz.save_series([panel], tmp_path / "series.png", title="SOI")
    assert path.exists()


def test_plot_and_save_points(
    tmp_path: Path,
    sample_frame: pd.DataFrame,
) -> None:
    fig = viz.plot_points(sample_frame, fields=["temperature"], title="stations")
    assert fig.axes[0].get_title() == "stations"

    path = viz.save_points(
        sample_frame,
        tmp_path / "points.png",
        fields=["temperature"],
        title="stations",
    )
    assert path.exists()


def test_plot_and_save_point_sets(
    tmp_path: Path,
    sample_frame: pd.DataFrame,
) -> None:
    panels = [
        viz.point_panel(
            sample_frame,
            lat="latitude",
            lon="longitude",
            fields=(),
            title="stations",
            color="tab:blue",
        ),
        viz.point_panel(
            sample_frame,
            lat="latitude",
            lon="longitude",
            fields=["temperature"],
            title="temperature",
        ),
    ]

    fig = viz.plot_point_sets(panels, ncols=2, title="points")
    assert len(fig.axes) == 3

    path = viz.save_point_sets(panels, tmp_path / "point_sets.png", ncols=2)
    assert path.exists()


def test_plot_and_save_tracks(tmp_path: Path) -> None:
    tracks = pd.DataFrame(
        {
            "track": ["a", "a", "b", "b"],
            "lat": [10.0, 11.0, 20.0, 21.0],
            "lon": [-50.0, -51.0, -60.0, -61.0],
        }
    )
    panel = viz.track_panel(
        tracks,
        group="track",
        label="forecast",
        color="blue",
    )

    fig = viz.plot_tracks([panel], title="tracks")
    assert fig.axes[0].get_title() == "tracks"

    path = viz.save_tracks([panel], tmp_path / "tracks.png", title="tracks")
    assert path.exists()


def test_plot_tracks_accepts_mapping() -> None:
    tracks = pd.DataFrame({"lat": [10.0, 11.0], "lon": [-50.0, -51.0]})

    fig = viz.plot_tracks({"analysis": tracks})

    assert fig.axes[0].get_xlabel() == "longitude"
