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

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.io import ZarrBackend
from earth2studio.models.px import Persistence
from earth2studio.utils.checkpoint import (
    capture_rng_state,
    default_restart_path,
    get_last_step,
    init_restart,
    load_restart,
    restore_rng_state,
    save_restart,
    validate_restart_compatibility,
)


def _state_coords() -> OrderedDict[str, np.ndarray]:
    return OrderedDict(
        {
            "time": np.array([np.datetime64("2024-01-01")]),
            "lead_time": np.array(
                [np.timedelta64(0, "h"), np.timedelta64(6, "h")]
            ),
            "variable": np.array(["t2m", "u10m"]),
            "lat": np.arange(2),
        }
    )


def test_default_restart_path_uses_earth2studio_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    path = default_restart_path("forecast")

    assert path == tmp_path / "restart" / "forecast.zarr"


def test_save_load_restart_and_last_step():
    io = ZarrBackend()
    coords = _state_coords()
    init_restart(io, coords, np.arange(3))

    step0_coords = coords.copy()
    step0_coords["lead_time"] = coords["lead_time"][:1]
    x0 = torch.randn(1, 1, 2, 2)
    save_restart(io, x0, step0_coords, step=0)

    selector = OrderedDict({"time": coords["time"]})
    assert get_last_step(io, selector) == 0

    step1_coords = coords.copy()
    step1_coords["lead_time"] = coords["lead_time"][1:]
    x1 = torch.randn(1, 1, 2, 2)
    rng = capture_rng_state()
    save_restart(io, x1, step1_coords, step=1, rng=rng)

    assert get_last_step(io, selector) == 1

    state = load_restart(io, step1_coords, step=1)
    assert state.step == 1
    assert state.rng is not None
    assert torch.allclose(state.x, x1)
    assert np.array_equal(state.coords["lead_time"], step1_coords["lead_time"])


def test_get_last_step_requires_all_selected_cells():
    io = ZarrBackend()
    coords = _state_coords()
    coords["time"] = np.array(
        [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]
    )
    init_restart(io, coords, np.arange(3))

    step0_coords = coords.copy()
    step0_coords["lead_time"] = coords["lead_time"][:1]
    x0 = torch.randn(2, 1, 2, 2)
    save_restart(io, x0, step0_coords, step=0)

    step1_coords = coords.copy()
    step1_coords["time"] = coords["time"][:1]
    step1_coords["lead_time"] = coords["lead_time"][1:]
    x1 = torch.randn(1, 1, 2, 2)
    save_restart(io, x1, step1_coords, step=1)

    assert get_last_step(io, OrderedDict({"time": coords["time"]})) == 0
    assert get_last_step(io, OrderedDict({"time": coords["time"][:1]})) == 1


def test_get_last_step_rejects_unknown_selector_values():
    io = ZarrBackend()
    coords = _state_coords()
    init_restart(io, coords, np.arange(3))

    selector = OrderedDict({"time": np.array([np.datetime64("2024-01-02")])})
    with pytest.raises(ValueError, match="not in progress store"):
        get_last_step(io, selector)


def test_save_restart_requires_initialized_store():
    io = ZarrBackend()
    coords = _state_coords()
    step0_coords = coords.copy()
    step0_coords["lead_time"] = coords["lead_time"][:1]

    with pytest.raises(RuntimeError, match="init_restart"):
        save_restart(io, torch.randn(1, 1, 2, 2), step0_coords, step=0)


def test_rng_capture_restore_roundtrip():
    torch.manual_seed(123)
    np.random.seed(123)

    state = capture_rng_state()
    torch_expected = torch.randn(3)
    numpy_expected = np.random.randn(3)

    restore_rng_state(state)

    assert torch.allclose(torch.randn(3), torch_expected)
    assert np.allclose(np.random.randn(3), numpy_expected)


def test_validate_restart_compatibility():
    coords = _state_coords()
    state_coords = coords.copy()
    state_coords["batch"] = np.arange(1)
    state_coords.move_to_end("batch", last=False)
    x = torch.randn(1, 1, 2, 2, 2)
    model = Persistence(
        ["t2m", "u10m"], OrderedDict({"lat": np.arange(2)}), history=2
    )
    state = load_restart_from_memory(x, state_coords, step=0)

    validate_restart_compatibility(state, model)


def test_validate_restart_shape_failure():
    coords = _state_coords()
    model = Persistence(["t2m", "u10m"], OrderedDict({"lat": np.arange(2)}))
    state = load_restart_from_memory(torch.randn(1, 2, 2), coords, step=0)

    with pytest.raises(ValueError, match="Tensor has"):
        validate_restart_compatibility(state, model)


def load_restart_from_memory(x, coords, step):
    from earth2studio.utils.checkpoint import RestartState

    return RestartState(x=x, coords=coords, step=step)
