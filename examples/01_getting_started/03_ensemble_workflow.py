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

# %%
"""
Running Ensemble Inference
==========================

Simple ensemble inference workflow.

This example will demonstrate how to run a simple inference workflow to generate a
ensemble forecast using one of the built in models of Earth-2 Inference
Studio.

In this example you will learn:

- How to instantiate a built in prognostic model
- Creating a data source and IO object
- Select a perturbation method
- Running a simple built in workflow for ensembling
- Post-processing results
"""

# /// script
# dependencies = [
#   "torch==2.11.0", # Match lock file to avoid torch-harmonics issue
#   "earth2studio[fcn,perturbation,viz] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "scipy>=1.15.2",
# ]
#
# [tool.uv]
# no-build-isolation-package = ["torch-harmonics"]
# ///

# %%
# Set Up
# ------
# All workflows inside Earth2Studio require constructed components to be
# handed to them. In this example, we will use the built in ensemble workflow
# :py:meth:`earth2studio.run.ensemble`.

# %%
# .. literalinclude:: ../../earth2studio/run.py
#    :language: python
#    :start-after: # sphinx - ensemble start
#    :end-before: # sphinx - ensemble end

# %%
# We need the following:
#
# - Prognostic Model: Use the built in FourCastNet model :py:class:`earth2studio.models.px.FCN`.
# - Perturbation Method: Use the Spherical Gaussian Method :py:class:`earth2studio.perturbation.SphericalGaussian`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np
import xarray as xr

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN
from earth2studio.perturbation import SphericalGaussian
from earth2studio.run import ensemble

# Load the default model package which downloads the check point from NGC
package = FCN.load_default_package()
model = FCN.load_model(package)

# Instantiate the pertubation method
sg = SphericalGaussian(noise_amplitude=0.15)

# Create the data source
data = GFS()

# Create the IO handler, store in memory
chunks = {"ensemble": 1, "time": 1, "lead_time": 1}
io = ZarrBackend(
    file_name="outputs/03_ensemble_sg.zarr",
    chunks=chunks,
    backend_kwargs={"overwrite": True},
)

# %%
# Execute the Workflow
# --------------------
# With all components initialized, running the workflow is a single line of Python code.
# Workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.
#
# For the forecast we will predict for 10 steps (for FCN, this is 60 hours) with 8 ensemble
# members which will be ran in 2 batches with batch size 4.

# %%

nsteps = 10
nensemble = 8
batch_size = 2
io = ensemble(
    ["2024-01-01"],
    nsteps,
    nensemble,
    model,
    data,
    io,
    sg,
    batch_size=batch_size,
    output_coords={"variable": np.array(["t2m", "tcwv"])},
)

# %%
# Post Processing
# ---------------
# The last step is to post process our results. The viz module accepts xarray-native
# fields and owns the static plotting backend.
#
# Notice that the Zarr IO function has additional APIs to interact with the stored data.

# %%
from earth2studio import viz

forecast = "2024-01-01"
variable = "tcwv"
lead_steps = [0, 2, 4, 6, 8]
dataset = xr.open_zarr("outputs/03_ensemble_sg.zarr")
field = dataset[variable].isel(time=0, lead_time=lead_steps)

scene = viz.Scene(title=f"{forecast} {variable} ensemble")
projection = viz.ProjectionSpec(kind="robinson")
scene.add_raster(
    field.sel(ensemble=0),
    name="Member 0",
    colormap="Blues",
    projection=projection,
)
scene.add_raster(
    field.sel(ensemble=1),
    name="Member 1",
    colormap="Blues",
    projection=projection,
)
scene.add_raster(
    field.std(dim="ensemble"),
    name="Ensemble std",
    colormap="Blues",
    projection=projection,
)
scene.save(
    f"outputs/03_{forecast}_{variable}_ensemble.jpg",
    backend="cartopy",
)
