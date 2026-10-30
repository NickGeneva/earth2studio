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

import warnings
from collections.abc import Callable

import numpy as np

from .base import LexiconType

LEVELS = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]

MDL_LEVELS = np.arange(1, 138)

# Number of trailing hourly samples summed together for each accumulated
# variable, e.g. "tp06" is the trailing 6-hour sum of hourly "tp" and
# "ttr03" is the trailing 3-hour sum of hourly "ttr".
ACCUMULATION_HOURS = {
    "cp06": 6,
    "ro06": 6,
    "sf06": 6,
    "ssrd06": 6,
    "strd06": 6,
    "tp06": 6,
    "tp12": 12,
    "tp24": 24,
    "e06": 6,
    "ttr03": 3,
}


class ARCO_ERA5Lexicon(metaclass=LexiconType):
    """ARCO ERA5 Lexicon
    ARCO specified <Variable ID>::<Level>. Model-level identifiers end in ``k``.

    Note
    ----
    Maps established Earth2Studio variables to the Google analysis-ready stores.
    Snowfall and runoff amounts use metres of water equivalent.
    """

    VOCAB = {
        "u10m": "10m_u_component_of_wind::",
        "v10m": "10m_v_component_of_wind::",
        "u100m": "100m_u_component_of_wind::",
        "v100m": "100m_v_component_of_wind::",
        "t2m": "2m_temperature::",
        "d2m": "2m_dewpoint_temperature::",
        "sp": "surface_pressure::",
        "sst": "sea_surface_temperature::",
        "msl": "mean_sea_level_pressure::",
        "tcwv": "total_column_water_vapour::",
        "tp": "total_precipitation::",
        "lsm": "land_sea_mask::",
        "z": "geopotential_at_surface::",
        "mtdwswrf": "mean_top_downward_short_wave_radiation_flux::",
        "ttr": "top_net_thermal_radiation::",
        "skt": "skin_temperature::",
        "sic": "sea_ice_cover::",
        # AIFS/AIFS ENS and AIFS2 aliases backed by ARCO_ERA5 single-level fields.
        "cdww": "coefficient_of_drag_with_waves::",
        "cp06": "convective_precipitation::",
        "cos_mwd": "mean_wave_direction::",
        "hcc": "high_cloud_cover::",
        "lcc": "low_cloud_cover::",
        "mcc": "medium_cloud_cover::",
        "mwp": "mean_wave_period::",
        "ro": "runoff::",
        "ro06": "runoff::",
        "sd": "snow_depth::",
        "sdor": "standard_deviation_of_orography::",
        "sf": "snowfall::",
        "sf06": "snowfall::",
        "sin_mwd": "mean_wave_direction::",
        "slor": "slope_of_sub_gridscale_orography::",
        "ssrd06": "surface_solar_radiation_downwards::",
        "stl1": "soil_temperature_level_1::",
        "stl2": "soil_temperature_level_2::",
        "strd06": "surface_thermal_radiation_downwards::",
        "swh": "significant_height_of_combined_wind_waves_and_swell::",
        "swvl1": "volumetric_soil_water_layer_1::",
        "swvl2": "volumetric_soil_water_layer_2::",
        "tcc": "total_cloud_cover::",
        "tcw": "total_column_water::",
        "tp06": "total_precipitation::",
        "ttr03": "top_net_thermal_radiation::",
        "wmb": "model_bathymetry::",
        # Variables already supported by the base vocabulary or other lexicons.
        "fg10m": "10m_wind_gust_since_previous_post_processing::",
        "blh": "boundary_layer_height::",
        "cape": "convective_available_potential_energy::",
        "cp": "convective_precipitation::",
        "ewss": "eastward_turbulent_surface_stress::",
        "fsr": "forecast_surface_roughness::",
        "zust": "friction_velocity::",
        "i10fg": "instantaneous_10m_wind_gust::",
        "iews": "instantaneous_eastward_turbulent_surface_stress::",
        "ie": "instantaneous_moisture_flux::",
        "inss": "instantaneous_northward_turbulent_surface_stress::",
        "lsp": "large_scale_precipitation::",
        "msdwlwrf": "mean_surface_downward_long_wave_radiation_flux::",
        "msdwswrf": "mean_surface_downward_short_wave_radiation_flux::",
        "mslhf": "mean_surface_latent_heat_flux::",
        "msshf": "mean_surface_sensible_heat_flux::",
        "mwd": "mean_wave_direction::",
        "nsss": "northward_turbulent_surface_stress::",
        "ptype": "precipitation_type::",
        "asn": "snow_albedo::",
        "smlt": "snowmelt::",
        "stl3": "soil_temperature_level_3::",
        "stl4": "soil_temperature_level_4::",
        "slhf": "surface_latent_heat_flux::",
        "ssr": "surface_net_solar_radiation::",
        "str": "surface_net_thermal_radiation::",
        "ssrd": "surface_solar_radiation_downwards::",
        "strd": "surface_thermal_radiation_downwards::",
        "tisr": "toa_incident_solar_radiation::",
        "tsr": "top_net_solar_radiation::",
        "tciw": "total_column_cloud_ice_water::",
        "tclw": "total_column_cloud_liquid_water::",
        "tco3": "total_column_ozone::",
        "fdir": "total_sky_direct_solar_radiation_at_surface::",
        "swvl3": "volumetric_soil_water_layer_3::",
        "swvl4": "volumetric_soil_water_layer_4::",
        # Existing accumulation and flux aliases; ARCO amounts are hourly.
        "tp01": "total_precipitation::",
        "tp1h": "total_precipitation::",
        "tp12": "total_precipitation::",
        "tp24": "total_precipitation::",
        "sf1h": "snowfall::",
        "ssrd1h": "surface_solar_radiation_downwards::",
        "ttr1h": "top_net_thermal_radiation::",
        "e06": "evaporation::",
        "rowe": "runoff::",
        "tpf": "mean_total_precipitation_rate::",
        "rsds": "mean_surface_downward_short_wave_radiation_flux::",
        "rlut": "mean_top_net_long_wave_radiation_flux::",
        "mtuwlwrf": "mean_top_net_long_wave_radiation_flux::",
    }
    VOCAB.update({f"u{level}": f"u_component_of_wind::{level}" for level in LEVELS})
    VOCAB.update({f"v{level}": f"v_component_of_wind::{level}" for level in LEVELS})
    VOCAB.update({f"w{level}": f"vertical_velocity::{level}" for level in LEVELS})
    VOCAB.update({f"z{level}": f"geopotential::{level}" for level in LEVELS})
    VOCAB.update({f"t{level}": f"temperature::{level}" for level in LEVELS})
    VOCAB.update({f"q{level}": f"specific_humidity::{level}" for level in LEVELS})
    VOCAB.update(
        {
            f"clwc{level}": f"specific_cloud_liquid_water_content::{level}"
            for level in LEVELS
        }
    )
    VOCAB.update(
        {
            f"ciwc{level}": f"specific_cloud_ice_water_content::{level}"
            for level in LEVELS
        }
    )

    VOCAB.update(
        {
            f"pv{level}": f"potential_vorticity::{level}"
            for level in [
                50,
                100,
                150,
                200,
                250,
                300,
                400,
                500,
                600,
                700,
                850,
                925,
                1000,
            ]
        }
    )

    # Model levels are stored in a separate Zarr group, but use the same base names
    VOCAB.update(
        {f"u{level}k": f"u_component_of_wind::{level}" for level in MDL_LEVELS}
    )
    VOCAB.update(
        {f"v{level}k": f"v_component_of_wind::{level}" for level in MDL_LEVELS}
    )
    VOCAB.update({f"t{level}k": f"temperature::{level}" for level in MDL_LEVELS})
    VOCAB.update({f"q{level}k": f"specific_humidity::{level}" for level in MDL_LEVELS})
    VOCAB.update(
        {
            f"clwc{level}k": f"specific_cloud_liquid_water_content::{level}"
            for level in MDL_LEVELS
        }
    )
    VOCAB.update(
        {
            f"ciwc{level}k": f"specific_cloud_ice_water_content::{level}"
            for level in MDL_LEVELS
        }
    )
    VOCAB.update(
        {
            f"crwc{level}k": f"specific_rain_water_content::{level}"
            for level in MDL_LEVELS
        }
    )
    VOCAB.update(
        {
            f"cswc{level}k": f"specific_snow_water_content::{level}"
            for level in MDL_LEVELS
        }
    )

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in ARCO_ERA5 vocabulary."""
        arco_key = cls.VOCAB[val]

        if val == "cos_mwd":

            def mod(x: np.array) -> np.array:
                return np.cos(np.deg2rad(x))

        elif val == "sin_mwd":

            def mod(x: np.array) -> np.array:
                return np.sin(np.deg2rad(x))

        elif val in {"rlut", "mtuwlwrf", "mslhf", "msshf"}:

            def mod(x: np.array) -> np.array:
                # ERA5 fluxes are positive downward; these aliases are upward.
                return -x

        else:

            def mod(x: np.array) -> np.array:
                """Modify name (if necessary)."""
                return x

        return arco_key, mod


def __getattr__(name: str) -> type[ARCO_ERA5Lexicon]:
    """Return deprecated lexicon aliases."""
    if name == "ARCOLexicon":
        warnings.warn(
            "ARCOLexicon has been renamed to ARCO_ERA5Lexicon and will be removed "
            "in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return ARCO_ERA5Lexicon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
