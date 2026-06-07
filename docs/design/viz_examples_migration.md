# Earth2 Studio Viz Examples Migration

## Scope

This note inventories example and recipe plotting code that should migrate from
manual Matplotlib/CartoPy implementations to `earth2studio.viz`.

The migration rule is intentionally narrow:

- Keep examples xarray-native or dataframe-native.
- Open model output stores with xarray first, for example `xr.open_zarr(...)`,
  `xr.open_dataset(...)`, or an existing xarray-backed IO representation.
- Pass `xr.DataArray`, `xr.Dataset`, pandas data frames, or cuDF data frames into
  `viz`.
- Do not make `viz` accept Earth2 Studio IO backend objects directly in this
  phase.
- Use Earth2 Studio lexicon variable names such as `t2m`, `u10m`, `v10m`,
  `ws10m`, `msl`, `tcwv`, `q850`, `z500`, `tp`, and `refc`.

## Example Migration Status

The current example tree has been migrated away from manual Matplotlib raster
grids and `raster_panel` descriptors for raster and point examples. Examples now
prefer:

- `Scene.add_raster(...)` for xarray-backed fields.
- `Scene.add_points(...)` for dataframe-backed point observations.
- xarray `.sel(...)`, `.isel(...)`, `.mean(...)`, `.std(...)`, and explicit
  `xr.DataArray(...)` construction before calling `viz`.
- Backend-owned raster layout for layer rows and `time` or `lead_time` columns.
- Projected geographic plots use `backend="cartopy"` with layer-level
  `ProjectionSpec` metadata instead of bespoke Cartopy subplot construction.

The remaining example quick helpers are intentionally limited to APIs that do
not yet have a scene-layer equivalent:

- `examples/02_medium_range/05_cyclone_tracking.py` uses `save_tracks` /
  `track_panel`.
- `examples/06_seasonal/01_seasonal_statistics.py` uses `save_series` /
  `series_panel`.

The migrated raster and point examples cover:

| Area | Files |
| --- | --- |
| Getting started | `examples/01_getting_started/01_deterministic_workflow.py`, `02_diagnostic_workflow.py`, `03_ensemble_workflow.py` |
| Medium range | `examples/02_medium_range/01_ensemble_workflow_extend.py`, `02_model_perturbation_hook.py`, `03_huge_ensembles.py`, `04_temporal_interpolation.py`, `06_atlas_inference.py` |
| Downscaling | `examples/03_downscaling/01_corrdiff_inference.py`, `02_cbottle_super_resolution.py`, `03_ensemble_downscaling.py` |
| Nowcasting | `examples/04_nowcasting/01_stormcast_example.py`, `02_stormcast_ensemble_example.py`, `03_stormscope_goes_example.py` |
| Data assimilation | `examples/05_data_assimilation/01_stormcast_sda.py`, `02_healda.py` |
| Seasonal | `examples/06_seasonal/02_dlesym_example.py` |
| Misc | `examples/07_misc/01_distributed_manager.py`, `02_cbottle_generation.py`, `03_io_performance.py`, `04_local_datasource.py`, `05_cbottle_tc_guidance.py` |
| Extend | `examples/08_extend/01_custom_prognostic.py`, `02_custom_diagnostic.py`, `03_custom_datasource.py` |

Recipe plotting code that should be evaluated after the examples:

- `recipes/eval/src/report/plotting.py`
- `recipes/hens/hens_notebook.py`
- `recipes/hens/src/plot/fork_n_spoon.py`
- `recipes/tc_tracking/plot_tracks_n_fields.py`
- `recipes/tc_tracking/src/plt/plotting_helpers.py`

## Migration Order

1. Keep raster examples on `Scene.add_raster` and xarray-native selection.
2. Keep point examples on `Scene.add_points` and dataframe-native columns.
3. Replace vector examples after `VectorLayer` supports quiver/barb lowering.
4. Replace track and cyclone examples after `TrackLayer` supports grouped line
   rendering.
5. Replace scalar diagnostic examples after a scene-level line layer exists.
6. Replace report-generation recipes after the example API stabilizes.

## Canonical Loading Pattern

Prefer this pattern in examples:

```python
import xarray as xr
from earth2studio import viz

ds = xr.open_zarr("outputs/forecast.zarr")
scene = viz.Scene(title="t2m forecast")
projection = viz.ProjectionSpec(kind="robinson")
scene.add_raster(
    ds["t2m"].isel(time=0, lead_time=[0, 2, 4]),
    colormap="turbo",
    projection=projection,
)
scene.save("outputs/t2m_forecast.jpg", backend="cartopy")
```

When an example already has an Earth2 Studio IO backend in memory, convert or
open it as xarray before calling `viz`. Avoid teaching `viz` about concrete IO
backend classes.

cBottle native `hpx` arrays can be passed directly to `Scene.add_raster`; the
adapter renders them as native HEALPix heatmap mosaics when the Matplotlib
backend is used. That is separate from future Cartopy/earth2grid geographic
reprojection support.

## API Gaps To Close Before Full Migration

- Cartopy animation and richer map styling, for example `contourf` lowering and
  country/state feature presets beyond the current metadata controls.
- `TrackLayer`: grouped line rendering for cyclone tracks and object paths.
- `LineLayer`: scalar time-series plots for statistics examples.
- `VectorLayer`: quiver/barb rendering from `u10m`/`v10m` and derived `ws10m`.
- Derived-variable helpers that keep lexicon vocabulary clear, for example
  deriving `ws10m` from `u10m` and `v10m` when only components are present.
