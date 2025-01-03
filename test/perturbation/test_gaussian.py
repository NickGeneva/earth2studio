# SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import pytest
import torch

from earth2studio.perturbation import Gaussian


@pytest.mark.parametrize(
    "x, coords",
    [
        [
            torch.randn(2, 16, 16, 16),
            OrderedDict([("a", []), ("variable", []), ("lat", []), ("lon", [])]),
        ],
        [
            torch.randn(2, 32, 16),
            OrderedDict([("variable", []), ("lat", []), ("lon", [])]),
        ],
    ],
)
@pytest.mark.parametrize(
    "amplitude",
    [1.0, 0.05, "tensor"],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_gaussian(x, coords, amplitude, device):

    x = x.to(device)

    if amplitude == "tensor":
        amplitude = torch.randn(
            [x.shape[list(coords).index("variable")], 1, 1], device=device
        )
    prtb = Gaussian(amplitude)
    xout, coords = prtb(x, coords)
    dx = xout - x

    assert dx.shape == x.shape
    assert torch.allclose(
        torch.mean(dx), torch.Tensor([0]).to(device), rtol=1e-2, atol=1e-1
    )

    if not isinstance(amplitude, torch.Tensor):
        assert torch.allclose(
            torch.std(dx), torch.Tensor([amplitude]).to(device), rtol=1e-2, atol=1e-1
        )
    assert dx.device == x.device
