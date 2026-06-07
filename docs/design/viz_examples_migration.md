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

## Example Inventory

The current example tree has 27 Python examples with manual plotting code:

| Area | Files |
| --- | --- |
| Getting started | `examples/01_getting_started/01_deterministic_workflow.py`, `02_diagnostic_workflow.py`, `03_ensemble_workflow.py` |
| Medium range | `examples/02_medium_range/01_ensemble_workflow_extend.py`, `02_model_perturbation_hook.py`, `03_huge_ensembles.py`, `04_temporal_interpolation.py`, `05_cyclone_tracking.py`, `06_atlas_inference.py` |
| Downscaling | `examples/03_downscaling/01_corrdiff_inference.py`, `02_cbottle_super_resolution.py`, `03_ensemble_downscaling.py` |
| Nowcasting | `examples/04_nowcasting/01_stormcast_example.py`, `02_stormcast_ensemble_example.py`, `03_stormscope_goes_example.py` |
| Data assimilation | `examples/05_data_assimilation/01_stormcast_sda.py`, `02_healda.py` |
| Seasonal | `examples/06_seasonal/01_seasonal_statistics.py`, `02_dlesym_example.py` |
| Misc | `examples/07_misc/01_distributed_manager.py`, `02_cbottle_generation.py`, `03_io_performance.py`, `04_local_datasource.py`, `05_cbottle_tc_guidance.py` |
| Extend | `examples/08_extend/01_custom_prognostic.py`, `02_custom_diagnostic.py`, `03_custom_datasource.py` |

Recipe plotting code that should be evaluated after the examples:

- `recipes/eval/src/report/plotting.py`
- `recipes/hens/hens_notebook.py`
- `recipes/hens/src/plot/fork_n_spoon.py`
- `recipes/tc_tracking/plot_tracks_n_fields.py`
- `recipes/tc_tracking/src/plt/plotting_helpers.py`

## Migration Order

1. Replace single-field global map examples with `viz.plot` or
   `Scene.add_raster`.
2. Replace multi-panel examples after adding a simple `viz.grid` or
   `SceneGrid` helper.
3. Replace ensemble mean/std maps after standardizing xarray reduction examples.
4. Replace vector examples after `VectorLayer` supports quiver/barb lowering.
5. Replace track and cyclone examples after `TrackLayer` supports grouped line
   rendering.
6. Replace report-generation recipes only after the example API stabilizes.

## Canonical Loading Pattern

Prefer this pattern in examples:

```python
import xarray as xr
from earth2studio import viz

ds = xr.open_zarr("outputs/forecast.zarr")
viz.plot(ds, variable="t2m", time=0, lead_time=4, colormap="turbo")
```

When an example already has an Earth2 Studio IO backend in memory, convert or
open it as xarray before calling `viz`. Avoid teaching `viz` about concrete IO
backend classes.

cBottle native `hpx` arrays can be passed directly to `viz.raster_panel` or
`viz.plot_raster_grid`; the adapter renders them as native HEALPix heatmap
mosaics. That is separate from future Cartopy/earth2grid geographic
reprojection support.

## API Gaps To Close Before Full Migration

- `CartopyBackend`: projection-aware maps, coastlines, state/country features,
  and `pcolormesh`/`contourf` choices.
- `SceneGrid`: small multiples for lead-time panels, ensemble panels, and
  comparison rows.
- `TrackLayer`: grouped line rendering for cyclone tracks and object paths.
- `LineLayer`: scalar time-series plots for statistics examples.
- `VectorLayer`: quiver/barb rendering from `u10m`/`v10m` and derived `ws10m`.
- Derived-variable helpers that keep lexicon vocabulary clear, for example
  deriving `ws10m` from `u10m` and `v10m` when only components are present.
