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

import cupy as cp
import numpy as np
import torch
import xarray as xr


@xr.register_dataarray_accessor("e2s")
class Earth2StudioDataArrayAccessor:
    """Earth2Studio custom Xarray methods for DataArrays."""

    def __init__(self, da: xr.DataArray):
        self.da = da

    @property
    def is_cupy(self) -> bool:
        """
        Check to see if the underlying array is a cupy array.

        Returns
        -------
        bool
            Whether the underlying data is a cupy array.
        """
        return self.da.data.is_cupy

    @property
    def device(self) -> torch.device:
        """Get device of current of underlying data array

        Returns
        -------
        bool
            Whether the underlying data is a cupy array.
        """
        if isinstance(self.da.data, np.ndarray):
            return torch.device("cpu")
        elif hasattr(self.da.data, "__cuda_array_interface__"):
            return torch.device(f"cuda:{cp.cuda.Device().id}")

    def to(self, device: torch.device | str) -> xr.DataArray:
        """Get device of current of underlying data array

        Returns
        -------
        bool
            Whether the underlying data is a cupy array.
        """
        device = torch.device(device)
        if device.type == "cpu":
            if isinstance(self.da.data, cp.ndarray):
                return xr.DataArray(
                    cp.asnumpy(self.da.data),
                    dims=self.da.dims,
                    coords=self.da.coords,
                    attrs=self.da.attrs,
                )
            return self.da
        elif device.type == "cuda":
            device_id = device.index if device.index is not None else 0
            with cp.cuda.Device(device_id):
                if isinstance(self.da.data, np.ndarray):
                    return xr.DataArray(
                        cp.asarray(self.da.data),
                        dims=self.da.dims,
                        coords=self.da.coords,
                        attrs=self.da.attrs,
                    )
                elif isinstance(self.da.data, cp.ndarray):
                    return self.da

    def numpy(self) -> xr.DataArray:
        """Returns numpy based data array"""
        return self.to("cpu")

    def cpu(self) -> xr.DataArray:
        """Returns numpy based data array"""
        return self.to("cpu")

    def torch(self) -> torch.Tensor:
        """Returns the data arrays in dataset as dictionary of PyTorch tensors.
        This is zero-copy.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of tensors
        """
        return torch.as_tensor(self.da.data)

    def batch(self, target_coordinates: xr.Dataset) -> xr.DataArray:
        """Batches or stacks a data array based on input target coordinates

        Parameters
        ----------
        target_coordinates : xr.Dataset
            Empty data array with dims and coordinate systems that represents the
            required batched shape.

        Returns
        -------
        xr.DataArray
            Xarray with required leading dimensions stacked into a batch dimension
        """

        # Get coordinate system from coords dataset
        if "batch" not in target_coordinates.dims[0]:
            return self.da

        coords = tuple(target_coordinates.dims)
        batch_coords = [c for c in self.da.dims if c not in coords]

        if not batch_coords:
            return self.da.expand_dims(dim={"batch": 1}, axis=0)

        # Stack the batch coordinates into a single dimension
        return self.da.stack(**{"batch": batch_coords}).transpose("batch", ...)

    def unbatch(self) -> xr.DataArray:
        """Calls unstack on a batched data array

        Returns
        -------
        _type_
            _description_
        """
        return self.da.unstack()
