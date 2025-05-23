# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.io import KVBackend
from earth2studio.utils.coords import convert_multidim_to_singledim, split_coords


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_kv_fields(time: list[np.datetime64], variable: list[str], device: str) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Test Memory Store
    z = KVBackend(device=device)
    assert isinstance(z.root, dict)

    array_name = "fields"
    z.add_array(total_coords, array_name)
    for dim in total_coords:
        assert dim in z.coords
        assert z.coords[dim].shape == total_coords[dim].shape

    # Test __contains__
    assert array_name in z

    # Test __getitem__
    shape = tuple([len(dim) for dim in total_coords.values()])
    assert z[array_name].shape == shape

    # Test __len__
    assert len(z) == 5

    # Test __iter__
    for array in z:
        assert array in ["fields", "time", "variable", "lat", "lon"]

    # Test add_array with torch.Tensor
    z.add_array(
        total_coords,
        "dummy_1",
        data=torch.randn(shape, device=device, dtype=torch.float32),
    )

    assert "dummy_1" in z
    assert z["dummy_1"].shape == shape

    # Test writing

    # Test full write
    x = torch.randn(shape, device=device, dtype=torch.float32)
    z.write(x, total_coords, "fields_1")
    assert "fields_1" in z
    assert z["fields_1"].shape == x.shape

    partial_coords = OrderedDict(
        {
            "time": np.asarray(time)[:1],
            "variable": np.asarray(variable)[:1],
            "lat": total_coords["lat"],
            "lon": total_coords["lon"][:180],
        }
    )
    partial_data = torch.randn((1, 1, 180, 180), device=device)
    z.write(partial_data, partial_coords, array_name)
    assert torch.allclose(z[array_name][0, 0, :, :180], partial_data)

    xx, _ = z.read(partial_coords, array_name, device=device)
    assert torch.allclose(partial_data, xx)

    # test to xarray
    ds = z.to_xarray()
    assert isinstance(ds, xr.Dataset)


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_kv_variable(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Remove var names
    coords = total_coords.copy()
    var_names = coords.pop("variable")

    # Test Memory Store
    z = KVBackend(device=device)
    assert isinstance(z.root, dict)

    z.add_array(coords, var_names)
    for dim in coords:
        assert z.coords[dim].shape == coords[dim].shape

    for var_name in var_names:
        assert var_name in z
        assert z[var_name].shape == tuple([len(values) for values in coords.values()])

    # Test writing
    partial_coords = OrderedDict(
        {
            "time": np.asarray(time)[:1],
            "variable": np.asarray(variable)[:1],
            "lat": total_coords["lat"],
            "lon": total_coords["lon"][:180],
        }
    )
    partial_data = torch.randn((1, 1, 180, 180), device=device)

    z.write(*split_coords(partial_data, partial_coords, "variable"))
    assert torch.allclose(z[variable[0]][0, :, :180], partial_data)

    # test to xarray
    ds = z.to_xarray()
    assert isinstance(ds, xr.Dataset)


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_kv_exceptions(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Test Memory Store
    z = KVBackend(device=device)
    assert isinstance(z.root, dict)

    # Test mismatch between len(array_names) and len(data)
    shape = tuple([len(values) for values in total_coords.values()])
    array_name = "fields"
    dummy = torch.randn(shape, device=device, dtype=torch.float32)
    with pytest.raises(ValueError):
        z.add_array(total_coords, array_name, data=[dummy] * 2)

    # Test trying to add the same array twice.
    z.add_array(
        total_coords,
        ["dummy_1"],
        data=[dummy],
    )
    with pytest.raises(AssertionError):
        z.add_array(
            total_coords,
            ["dummy_1"],
            data=[dummy],
        )

    # Try to write with bad coords
    bad_coords = {"ensemble": np.arange(0)} | total_coords
    bad_shape = (1,) + shape
    dummy = torch.randn(bad_shape, device=device, dtype=torch.float32)
    with pytest.raises(AssertionError):
        z.write(dummy, bad_coords, "dummy_1")

    # Try to write with too many array names
    with pytest.raises(ValueError):
        z.write([dummy, dummy], bad_coords, "dummy_1")


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu"])
def test_kv_fields_multidim(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    lat = np.linspace(-90, 90, 180)
    lon = np.linspace(0, 360, 360, endpoint=False)
    LON, LAT = np.meshgrid(lon, lat)

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": LAT,
            "lon": LON,
        }
    )

    adjusted_coords, _ = convert_multidim_to_singledim(total_coords)

    # Test Memory Store
    z = KVBackend(device=device)
    assert isinstance(z.root, dict)

    array_name = "fields"
    z.add_array(total_coords, array_name)
    for dim in adjusted_coords:
        assert dim in z.coords
        assert z.coords[dim].shape == adjusted_coords[dim].shape

    # Test __contains__
    assert array_name in z

    # Test __getitem__
    shape = tuple([len(dim) for dim in adjusted_coords.values()])
    assert z[array_name].shape == shape

    # Test __len__
    assert len(z) == (7)

    # Test __iter__
    for array in z:
        assert array in ["fields", "time", "variable", "lat", "lon", "ilat", "ilon"]

    for array in ["fields", "time", "variable", "lat", "lon", "ilat", "ilon"]:
        assert array in z

    # Test add_array with torch.Tensor
    z.add_array(
        total_coords,
        "dummy_1",
        data=torch.randn(shape, device=device, dtype=torch.float32),
    )

    assert "dummy_1" in z
    assert z["dummy_1"].shape == shape

    # Test writing

    # Test full write
    x = torch.randn(shape, device=device, dtype=torch.float32)
    z.write(x, adjusted_coords, array_name)

    xx, _ = z.read(adjusted_coords, array_name, device=device)
    assert torch.allclose(x, xx)

    # Test separate write
    z.write(x, total_coords, "fields_1")
    assert "fields_1" in z
    assert z["fields_1"].shape == x.shape

    xx, _ = z.read(total_coords, "fields_1", device=device)
    assert torch.allclose(x, xx)
