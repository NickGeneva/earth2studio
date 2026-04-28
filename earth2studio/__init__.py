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

__version__ = "0.14.0rc0"

# Import netCDF4 early to ensure its bundled HDF5 library is loaded before h5py's.
# Both packages bundle different builds of libhdf5; whichever loads first wins.
# If h5py loads first (e.g., via torch/onnx), netCDF4 read/write operations fail
# TODO: Remove when netCDF4 update is compatable with h5py....
import netCDF4 as _netCDF4  # noqa: F401, E402

# Deprecation warnings
# import sys
# import warnings
