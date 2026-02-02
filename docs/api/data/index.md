# Data Sources

Earth2Studio provides multiple data source interfaces for accessing weather and climate
data. Each interface is designed for a specific use case.

## Overview

| Interface | Description | Returns | Use Case |
|-----------|-------------|---------|----------|
| [DataSource](datasources.md) | Analysis/reanalysis data indexed by time | `xr.DataArray` | ERA5, GFS analysis, satellite data |
| [ForecastSource](forecastsources.md) | Forecast data with initialization time and lead time | `xr.DataArray` | GFS forecasts, ECMWF IFS, ensemble predictions |
| [DataFrameSource](dataframesources.md) | Sparse/tabular observational data | `pd.DataFrame` | Weather station observations, point measurements |
| [ForecastFrameSource](dataframesources.md) | Sparse/tabular forecast data with lead time | `pd.DataFrame` | Point forecast data |

## Available Data Sources

### Analysis & Reanalysis Data

| Source | Description | Resolution | Coverage |
|--------|-------------|------------|----------|
| `ARCO` | Google's Analysis-Ready Cloud-Optimized ERA5 | 0.25° | Global |
| `CDS` | Copernicus Climate Data Store | Various | Global |
| `GFS` | NOAA Global Forecast System analysis | 0.25° | Global |
| `HRRR` | High-Resolution Rapid Refresh | 3km | CONUS |
| `IFS` | ECMWF Integrated Forecasting System | 0.25° | Global |
| `NCAR_ERA5` | NCAR Research Data Archive ERA5 | 0.25° | Global |
| `WB2ERA5` | WeatherBench2 ERA5 subset | Various | Global |

### Forecast Data

| Source | Description | Resolution | Lead Times |
|--------|-------------|------------|------------|
| `GFS_FX` | GFS forecast data | 0.25° | Up to 384h |
| `HRRR_FX` | HRRR forecast data | 3km | Up to 48h |
| `GEFS_FX` | Global Ensemble Forecast System | 0.25° | Up to 384h |
| `IFS_FX` | ECMWF IFS forecasts | 0.25° | Up to 240h |
| `AIFS_FX` | ECMWF AIFS ML forecasts | 0.25° | Up to 240h |

### Observational Data

| Source | Description | Type |
|--------|-------------|------|
| `ISD` | Integrated Surface Database | Weather stations |
| `GOES` | NOAA GOES satellite imagery | Satellite |
| `MRMS` | Multi-Radar Multi-Sensor | Radar composite |

---

## Base Interfaces

### DataSource

The primary interface for accessing analysis/reanalysis data indexed by time.

::: earth2studio.data.base.DataSource
    options:
      show_root_heading: true
      members: false

### ForecastSource

Interface for accessing forecast data indexed by both initialization time and lead time.

::: earth2studio.data.base.ForecastSource
    options:
      show_root_heading: true
      members: false

### DataFrameSource

Interface for sparse/tabular data such as weather station observations.

::: earth2studio.data.base.DataFrameSource
    options:
      show_root_heading: true
      members: false

### ForecastFrameSource

Interface for sparse/tabular forecast data indexed by initialization time and lead time.

::: earth2studio.data.base.ForecastFrameSource
    options:
      show_root_heading: true
      members: false
