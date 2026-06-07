# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
"""
HealDA Global Data Assimilation
================================

Producing a global weather analysis from satellite and in-situ observations.

This example demonstrates how to use the HealDA data assimilation model to produce
a global weather analysis on a HEALPix grid from sparse in-situ (conventional) and
satellite radiance observations sourced from the NOAA UFS replay archive.
Three runs are compared: conventional observations only, satellite observations only,
and both combined to illustrate the impact of each observation type.

In this example you will learn:

- How to load and initialise the HealDA data assimilation model
- Fetching UFS conventional and satellite observation DataFrames
- Running the model with different observation combinations
- Comparing the assimilated global fields against ERA5 data
"""

# /// script
# dependencies = [
#   "earth2studio[da-healda,viz] @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
# Set Up
# ------
# This example requires the following components:
#
# - Assimilation Model: HealDA :py:class:`earth2studio.models.da.HealDA`.
# - Datasource (conv): UFS conventional observations
#   :py:class:`earth2studio.data.UFSObsConv`.
# - Datasource (sat): UFS satellite observations
#   :py:class:`earth2studio.data.UFSObsSat`.
#
# HealDA is a stateless neural-network-based data assimilation model that ingests
# conventional (radiosonde, surface station, GPS-RO, etc.) and satellite radiance
# observations and produces a single global weather analysis on a HEALPix level-6
# grid.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from datetime import timedelta

import numpy as np
import torch
import xarray as xr
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

from earth2studio.data import NCAR_ERA5, UFSObsConv, UFSObsSat, fetch_dataframe
from earth2studio.models.da import HealDA

# Load the default model package (downloads checkpoint from HuggingFace)
# Setting lat_lon=True regrids the native HEALPix output to a regular lat-lon grid.
package = HealDA.load_default_package()
model = HealDA.load_model(package, lat_lon=True)
model = model.to("cuda:0")

# %%
# Fetch Observations
# ------------------
# Pull conventional and satellite observation DataFrames for the analysis time.
# The UFS data sources return pandas DataFrames that match the schemas expected by
# :py:meth:`HealDA.input_coords`.  We use
# :py:func:`earth2studio.data.fetch_dataframe` which attaches ``request_time``
# metadata required by the model.  The time_tolerance parameter defines a time
# window around the analysis time so that observations will be retrieved for.

# %%
analysis_time = np.array([np.datetime64("2024-01-01T00:00")])

conv_source = UFSObsConv(time_tolerance=(timedelta(hours=-21), timedelta(hours=3)))
conv_schema, sat_schema = model.input_coords()
conv_df = fetch_dataframe(
    conv_source,
    time=analysis_time,
    variable=np.array(conv_schema["variable"]),
    fields=np.array(list(conv_schema.keys())),
)
logger.info(f"Fetched {len(conv_df)} conventional observations")

sat_source = UFSObsSat(time_tolerance=(timedelta(hours=-21), timedelta(hours=3)))
sat_df = fetch_dataframe(
    sat_source,
    time=analysis_time,
    variable=np.array(sat_schema["variable"]),
    fields=np.array(list(sat_schema.keys())),
)
logger.info(f"Fetched {len(sat_df)} satellite observations")

# %%
# Observation Locations
# ---------------------
# Plot the spatial distribution of conventional and satellite observations to
# visualise their coverage before running the assimilation. There are 12-14 million
# observations typically for the model's 24-hour time window.

# %%
from earth2studio import viz

conv_plot = conv_df.iloc[::10]
sat_plot = sat_df.iloc[::10]
scene = viz.Scene(title=f"Observation Locations {str(analysis_time[0])[:16]} UTC")
scene.add_points(
    conv_plot,
    lat="lat",
    lon="lon",
    fields=(),
    name=f"Conventional obs (n={len(conv_df):,})",
    color="tab:blue",
    size=0.1,
    alpha=0.3,
)
scene.add_points(
    sat_plot,
    lat="lat",
    lon="lon",
    fields=(),
    name=f"Satellite obs (n={len(sat_df):,})",
    color="tab:orange",
    size=0.1,
    alpha=0.3,
)
scene.save(
    "outputs/22_healda_obs_locations.jpg",
    backend="matplotlib",
    figsize=(16, 4),
)
# %%
# DA models can be called directly for stateless inference or via
# :py:meth:`~earth2studio.models.da.HealDA.create_generator` for stateful (iterative)
# assimilation workflows.  Here we use the direct call API to invoke the model.
#
# HealDA is designed to work with the (-21, 3) hour observation window from the UFS
# replay archive using both conventional and satellite observations. However, the DA
# model interface is flexible enough to accept different time windows and observation
# sources. Below we test three configurations - conventional only, satellite only, and
# proper combined - to illustrate the impact each observation type has on the analysis.

# %%
torch.manual_seed(42)
result_both = model(conv_obs=conv_df, sat_obs=sat_df)
logger.info(f"Combined analysis shape: {result_both.shape}")

# %%
torch.manual_seed(42)
result_sat = model(sat_obs=sat_df)
logger.info(f"Sat-only analysis shape: {result_sat.shape}")

# %%
torch.manual_seed(42)
result_conv = model(conv_obs=conv_df)
logger.info(f"Conv-only analysis shape: {result_conv.shape}")

# %%
# Post Processing
# ---------------
# Because we loaded the model with ``lat_lon=True`` the output is already on a
# regular equiangular lat-lon grid, so no manual regridding is needed.
# Compare the three runs for surface temperature (t2m) and geopotential 500 hPa
# (z500). Each row shows a different observation configuration.

# %%
plot_vars = ["t2m", "z500"]
titles = ["Conv + Sat", "Sat only", "Conv only"]
results = [result_both, result_sat, result_conv]

lat = results[0].coords["lat"].values
lon = results[0].coords["lon"].values
cmaps = ["Spectral_r", "PRGn"]


def to_numpy(arr):
    """CuPy / Numpy helper function"""
    return arr.get() if hasattr(arr, "get") else arr


scene = viz.Scene(title=f"HealDA Analysis {str(analysis_time[0])[:16]} UTC")
for title, da in zip(titles, results):
    for var, cmap in zip(plot_vars, cmaps):
        field = to_numpy(da.sel(variable=var).data[0])
        scene.add_raster(
            xr.DataArray(
                field,
                dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
                name=var,
            ),
            name=f"{title} - {var}",
            colormap=cmap,
        )

scene.save(
    "outputs/22_healda_analysis.jpg",
    backend="matplotlib",
    figsize=(14, 8),
)
# %%
# HealDA vs ERA5
# --------------
# Next fetch ERA5 reanalysis at 0.25° resolution from the NCAR archive to compare
# against the assimilated fields. HealDA outputs standard Earth2Studio variable
# names so we can query ERA5 with the same identifiers. We expect the runs that
# are missing an observation source to show larger errors, while the combined run
# yields the most accurate global prediction.

# %%
era5_ds = NCAR_ERA5()
era5_da = era5_ds(analysis_time, plot_vars)
era5_interp = era5_da.interp(lat=lat, lon=lon, method="nearest")

# %%
diff_titles = ["Conv+Sat - ERA5", "Sat - ERA5", "Conv - ERA5"]
diff_results = [result_both, result_sat, result_conv]
for title, da_pred in zip(diff_titles, diff_results):
    for var in plot_vars:
        field_pred = to_numpy(da_pred.sel(variable=var).data[0])
        field_era5 = to_numpy(era5_interp.sel(variable=var).data[0])
        mae = float(np.abs(field_pred - field_era5).mean())
        logger.info(f"{title} | {var} MAE: {mae:.4f}")

# %%
diff_ranges = {"t2m": (-10, 10), "z500": (-500, 500)}
scene = viz.Scene(title=f"HealDA Analysis Error {str(analysis_time[0])[:16]} UTC")
for title, da_pred in zip(diff_titles, diff_results):
    for var in plot_vars:
        field_pred = to_numpy(da_pred.sel(variable=var).data[0])
        field_era5 = to_numpy(era5_interp.sel(variable=var).data[0])
        diff = field_pred - field_era5
        scene.add_raster(
            xr.DataArray(
                diff,
                dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
                name=f"{var}_diff",
            ),
            name=f"{title} - {var}",
            colormap="RdBu_r",
            vmin=diff_ranges[var][0],
            vmax=diff_ranges[var][1],
        )

scene.save(
    "outputs/22_healda_differences.jpg",
    backend="matplotlib",
    figsize=(14, 8),
)
