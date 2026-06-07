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
"""Agent-friendly summary: tests for timeline, camera, and regional metadata.

Key APIs under test: `Timeline`, xarray/dataframe frame inference, `Camera`,
and `RegionSpec`.
"""

import pandas as pd
import pytest
import xarray as xr

from earth2studio.viz.camera import Camera
from earth2studio.viz.regional import RegionSpec
from earth2studio.viz.timeline import (
    Timeline,
    infer_frames_from_dataframe,
    infer_frames_from_xarray,
)


def test_timeline_dedupes_and_sets_frames() -> None:
    first = pd.Timestamp("2026-06-07T00:00:00")
    second = pd.Timestamp("2026-06-07T06:00:00")
    timeline = Timeline([first, first, second])

    assert timeline.frames == [first, second]
    assert timeline.current == first
    assert timeline.range() == (first, second)

    timeline.set(time=second)
    assert timeline.current == second

    timeline.set(index=0)
    assert timeline.current == first


def test_empty_timeline_behaviors() -> None:
    timeline = Timeline()

    assert timeline.current is None
    assert timeline.range() is None
    assert timeline.set() is timeline


def test_timeline_rejects_missing_frame() -> None:
    timeline = Timeline([pd.Timestamp("2026-06-07")])

    with pytest.raises(KeyError, match="was not found"):
        timeline.set(time=pd.Timestamp("2026-06-08"))


def test_timeline_rejects_bad_index() -> None:
    with pytest.raises(IndexError, match="out of range"):
        Timeline([1], current_index=2)


def test_infer_valid_time_from_xarray(sample_dataarray: xr.DataArray) -> None:
    frames = infer_frames_from_xarray(sample_dataarray, valid_time=True)

    assert len(frames) == 4
    assert frames[0] == pd.Timestamp("2026-06-07T00:00:00")
    assert frames[-1] == pd.Timestamp("2026-06-07T12:00:00")


def test_infer_lead_time_only_from_xarray() -> None:
    data = xr.DataArray(
        [1.0, 2.0],
        dims=("lead_time",),
        coords={"lead_time": pd.to_timedelta([0, 6], unit="h")},
    )

    assert infer_frames_from_xarray(data) == list(data.coords["lead_time"].values)


def test_infer_frames_from_timeless_xarray() -> None:
    data = xr.DataArray([1.0, 2.0], dims=("x",))

    assert infer_frames_from_xarray(data) == []


def test_infer_frames_from_dataframe(sample_frame: pd.DataFrame) -> None:
    frames = infer_frames_from_dataframe(sample_frame)

    assert len(frames) == 3
    assert frames[0] == pd.Timestamp("2026-06-07T00:00:00")


def test_infer_frames_from_dataframe_edge_cases(sample_frame: pd.DataFrame) -> None:
    no_time = sample_frame.drop(columns=["valid_time"])

    assert infer_frames_from_dataframe(no_time) == []

    with pytest.raises(KeyError, match="Time column"):
        infer_frames_from_dataframe(sample_frame, time="missing")


def test_camera_set_orbit_and_summary() -> None:
    camera = Camera().set(lon=-100.0, lat=40.0, distance=10.0, custom="value")
    camera.orbit(target="region", azimuth=225.0, elevation=35.0, distance=1000.0)

    summary = camera.as_dict()
    assert summary["lon"] == -100.0
    assert summary["target"] == "region"
    assert summary["heading"] == 225.0
    assert summary["pitch"] == 35.0
    assert summary["metadata"] == {"custom": "value"}


def test_region_spec_from_lonlat_bounds() -> None:
    region = RegionSpec.from_lonlat_bounds(
        name="coastal",
        west=-124.0,
        south=32.0,
        east=-118.0,
        north=38.0,
        vertical_datum="EGM96",
    )

    assert region.crs == "EPSG:32610"
    assert region.source_crs == "EPSG:4326"
    assert region.width == 6.0
    assert region.height == 6.0
    assert region.contains_xy(-120.0, 35.0)
    assert not region.contains_xy(-130.0, 35.0)
    assert region.as_dict()["vertical_datum"] == "EGM96"


def test_region_spec_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="east must be greater"):
        RegionSpec.from_lonlat_bounds(
            name="bad", west=1.0, south=0.0, east=0.0, north=1.0
        )

    with pytest.raises(ValueError, match="xmax > xmin"):
        RegionSpec(name="bad", crs="EPSG:4326", bounds=(1.0, 0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="ymax > ymin"):
        RegionSpec(name="bad", crs="EPSG:4326", bounds=(0.0, 1.0, 1.0, 0.0))

    with pytest.raises(ValueError, match="origin"):
        RegionSpec(
            name="bad",
            crs="EPSG:4326",
            bounds=(0.0, 0.0, 1.0, 1.0),
            origin=(0.0, 0.0),
        )
