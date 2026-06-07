(viz_userguide)=

# Visualization

The {mod}`earth2studio.viz` module provides a Python-first visualization layer
for Earth2 Studio data. It is built around xarray data arrays and datasets for
dense weather fields, pandas or cuDF-like data frames for sparse observations,
and backend-neutral scene objects that can later route to static plots, local
3D renderers, or scene exporters.

The initial implementation focuses on a small set of stable concepts:

- {class}`earth2studio.viz.Scene` stores ordered layers, a timeline, a camera,
  and optional regional metadata.
- Raster layers accept {class}`xarray.DataArray` or {class}`xarray.Dataset`
  inputs and preserve coordinate labels.
- Point layers accept pandas or cuDF-like tables with `lat/lon`, `x/y`, and
  optional time columns.
- Regional terrain layers preserve local CRS, vertical datum, and topography
  metadata through {class}`earth2studio.viz.RegionSpec`.
- Backends are selected by name and imported lazily. Built-in backends include
  `summary`, `matplotlib`, and `cartopy`.

Visualization variable names should follow the Earth2 Studio lexicon vocabulary
used elsewhere in the package. Common examples are `t2m`, `u10m`, `v10m`,
`ws10m`, `msl`, `tcwv`, `q850`, `z500`, `tp`, and `refc`. For derived or
domain-specific regional data, name the xarray variable explicitly before
passing it into `viz`.

## Quick Raster Plot

```python
from earth2studio import viz

fig = viz.plot(
    forecast,
    variable="t2m",
    time="2026-06-07T00:00:00",
    lead_time="6h",
    backend="matplotlib",
    colormap="turbo",
)
```

Use `backend="summary"` when testing or inspecting scene metadata without a
plotting dependency:

```python
summary = viz.plot(
    forecast,
    variable="t2m",
    time=0,
    lead_time=0,
    backend="summary",
)
```

## Layer Time Series

Example scripts often start from model output stores. Open those stores with
xarray, select the variable and timesteps you want to visualize, then add those
arrays directly as scene layers. For geographic rasters, attach a
{class}`earth2studio.viz.ProjectionSpec` and use the Cartopy backend; each
raster layer becomes a row and each `time` or `lead_time` frame becomes a
column:

```python
import xarray as xr
from earth2studio import viz

ds = xr.open_zarr("outputs/forecast.zarr")
field = ds["tcwv"].isel(time=0, lead_time=[0, 2, 4, 6])

scene = viz.Scene(title="tcwv ensemble")
projection = viz.ProjectionSpec(kind="robinson")
scene.add_raster(
    field.sel(ensemble=0),
    name="Member 0",
    colormap="Blues",
    projection=projection,
)
scene.add_raster(
    field.std(dim="ensemble"),
    name="Ensemble std",
    colormap="Blues",
    projection=projection,
)
scene.save("outputs/tcwv_ensemble.jpg", backend="cartopy")
```

Any additional non-spatial dimensions must be explicitly selected or reduced in
xarray before the layer call. This keeps the layer API simple while making
hidden ambiguity fail loudly.

Quick helper functions remain available for small one-off artifacts where a
scene would be unnecessary:

- {func}`earth2studio.viz.save_series` for one-dimensional diagnostics such as
  `soi`.
- {func}`earth2studio.viz.save_points` and
  {func}`earth2studio.viz.save_point_sets` for pandas or cuDF-like observation
  tables.
- {func}`earth2studio.viz.save_tracks` for dataframe-backed grouped paths such
  as tropical cyclone tracks.

## Multi-Layer Scene

```python
scene = viz.Scene(title="Forecast and stations")
scene.add_raster(forecast, variable="tcwv", time=0, lead_time=0, colormap="magma")
scene.add_points(
    stations,
    lat="latitude",
    lon="longitude",
    time="valid_time",
    color="temperature",
    size=8,
)

scene.camera.set(lon=-98.0, lat=38.0, distance=2.4)
scene.show(backend="matplotlib")
```

`show` keeps shared options explicit and sends every renderer-specific option as
a keyword argument. The core call shape is:

```python
scene.show(
    backend="cartopy",
    streaming=False,
    auto_flush=True,
    figsize=(12, 6),
    colorbar=True,
)
```

`backend` selects the renderer, `streaming` requests a persistent session,
`auto_flush` controls whether layer changes immediately update the backend, and
remaining keyword arguments are validated by the selected backend.

## Streaming Sessions

When `streaming=True`, `show` returns a backend-owned session. Updating layer
state and flushing backend state are separate operations:

```python
scene = viz.Scene(title="Streaming forecast")
layer = scene.add_raster(forecast.sel(ensemble=0), name="t2m", colormap="turbo")

session = scene.show(backend="cartopy", streaming=True, auto_flush=False)

layer.update(next_frame)      # mutate scene/layer state only
layer.append(later_frame)     # extend the layer's time series
session.flush()               # redraw or reconcile the backend once
session.close()
```

The default is `auto_flush=True`, which keeps notebooks and small scripts
simple. Use `auto_flush=False` or `session.hold()` when a model loop appends
several frames, textures, or point batches and the backend should update once.
The built-in `summary`, `matplotlib`, and `cartopy` sessions implement rough
streaming by redrawing from the current scene. The `ovrtx` session uses the same
event path to keep a browser/notebook globe payload synchronized with layer and
timeline changes while the renderer-owned texture upload and WebRTC frame loop
are attached under the backend.

```python
scene = viz.Scene(title="OVRTX globe")
scene.add_default_texture()
layer = scene.add_raster(forecast.sel(ensemble=0), name="t2m", colormap="turbo")

session = scene.show(
    backend="ovrtx",
    streaming=True,
    open_browser=True,
)

layer.append(next_valid_time_frame, time=next_valid_time)
```

The OVRTX browser surface follows the Command Center layout: streamed viewport
pixels in the main view, a layer list in the upper right, a timeline along the
bottom, and orbit-camera state owned by the renderer path. In notebooks, the
returned session exposes `_repr_html_()` so the same session document can be
embedded inline.

## Regional Terrain Scene

Regional scenes use {class}`earth2studio.viz.RegionSpec` to keep local
coordinate-system and vertical metadata attached to the visualization.

```python
region = viz.RegionSpec.from_lonlat_bounds(
    name="coastal-state",
    west=-124.8,
    south=32.4,
    east=-113.8,
    north=42.1,
    target_crs="auto_utm",
    vertical_datum="EGM96",
)

scene = viz.Scene(region=region)
scene.add_terrain(dem, name="Terrain", vertical_exaggeration=1.5)
scene.add_draped_raster(forecast, variable="ws10m", time=0, colormap="turbo")
scene.add_region_cube(regional_cube, variable="q850", vertical="height", mode="slices")
```

The same scene description can later route to OpenUSD or OVRTX-style local
renderers as those backends mature. The core layer API does not require users to
rewrite xarray or dataframe ingestion code when a backend changes.

## Grids and Projections

Raster layers infer a grid descriptor from xarray dimensions, coordinates, and
attributes:

```python
scene = viz.Scene()
layer = scene.add_raster(forecast, variable="t2m")
grid = layer.projection.metadata["grid"]
```

The descriptor makes the spatial representation visible to backends and tests.
Current grid intents include regular lat/lon, curvilinear lat/lon, projected
x/y grids, model-native grids, cubed-sphere face stacks, HPX/HEALPix,
diamond-style globe grids, GOES, and geohash-indexed data.

Native indexed grids can still be plotted without hand-written Matplotlib code.
For example, cBottle can return `hpx` instead of lat/lon coordinates; the layer
adapter tiles that HEALPix vector into a native heatmap:

```python
cbottle_ds.lat_lon = False
sample = cbottle_ds([timestamp], ["tcwv"])

scene = viz.Scene(title="Native HEALPix heatmap")
scene.add_raster(sample.sel(variable="tcwv").isel(time=0), colormap="cubehelix")
scene.save("outputs/tcwv_hpx_heatmap.png", backend="matplotlib")
```

The heatmap is a native grid diagnostic, not a map reprojection. Use lat/lon
output or a future backend-specific payload builder when geographic coastlines
and projected overlays are required.

Static Cartopy rendering is strongest for geographic `lat`/`lon` rasters and
dataframe point layers. Static Matplotlib rendering remains useful for native
model-grid diagnostics that should not be interpreted as geographic maps. For
geohash trigger tables, provide decoded `lat`/`lon` or `x`/`y` columns for
static plotting today; future renderer payload builders can lower geohash cells
into polygons or instanced geometry without changing the scene API.

## External Assets

The core forecast path should stay xarray-native, but the scene model can also
carry asset-native layers for imagery, GeoTIFFs, and meshes:

```python
from earth2studio import viz

scene = viz.Scene(title="Regional asset scene")
scene.add_image(
    "blue_marble.jpg",
    bounds=(-180.0, -90.0, 180.0, 90.0),
)
scene.add_geotiff(
    "local_dem.cog.tif",
    role="terrain",
    crs="EPSG:32610",
)
scene.add_mesh(
    "terrain.usd",
    crs="EPSG:32610",
)

summary = scene.render("summary").output
```

These calls create layer intent without opening files. Backends decide how to
decode, tile, stream, upload, cache, or export the assets. For immediate static
plots, open GeoTIFF-like rasters with xarray/rioxarray first and pass the
resulting array to `add_raster`, `add_terrain`, or `add_draped_raster`.

Time-varying image textures can be represented explicitly:

```python
from earth2studio import viz

valid_times = ["2026-06-07T00:00:00", "2026-06-07T01:00:00"]
sequence = viz.TextureSequence(name="Satellite")
for index, timestamp in enumerate(valid_times):
    sequence.append(
        viz.TextureFrame(
            source=viz.TextureSource(uri=f"frames/{index:03d}.jpg"),
            index=index,
            timestamp=timestamp,
        )
    )

scene = viz.Scene()
scene.add_image(sequence)
```

Interactive backends can use the sequence and cache policy to prefetch nearby
frames while keeping the public scene API unchanged.

## Default Texture Domain

Global scenes can use a default texture domain for common base assets such as
global base color, topography, cloud overlays, and boundary overlays:

```python
from earth2studio import viz

scene = viz.Scene(title="Global defaults")
scene.add_default_texture("global_base_color")
scene.add_default_texture("global_boundaries", alpha=0.6)
```

The default domain stores assets under the common Earth2 Studio cache root:

```text
${EARTH2STUDIO_CACHE:-~/.cache/earth2studio}/viz/v5/default_textures/
```

Filenames are readable and unhashed, for example `global_clouds.ktx2` and
`global_boundaries.ktx2`, so the cache can be inspected and pre-populated by
users or deployment tooling. This default texture cache intentionally uses
`EARTH2STUDIO_CACHE` rather than `EARTH2STUDIO_DATA_CACHE` because these are
renderer assets, not datasource fetches.

## Vector and Flow Intent

Vector layers store semantic intent rather than renderer-specific geometry.
Backends can lower the same layer into Matplotlib quivers, CartoPy barbs,
OpenUSD curves, OVRTX glyphs, or ANARI geometry.

```python
scene.add_vectors(
    forecast_dataset,
    vector=("u10m", "v10m"),
    mode="streamlines",
    width=2.0,
)
```

The first supported static mode is quiver-style vector display. Streamlines,
glyph instancing, and 3D flow objects are represented in the layer metadata so
backend implementations can add support without changing user code.

## Backend Registry

Backends are registered by name:

```python
from earth2studio.viz import available_backends, get_backend

available_backends()
backend = get_backend("summary")
```

The built-in `summary` backend has no optional dependencies and returns a
serializable scene dictionary. The `matplotlib`, `cartopy`, `ovrtx`, and
`anari` backends import their plotting or renderer dependencies only when those
runtime paths are requested. `scene.show(backend="ovrtx", streaming=True)` can
create the browser/notebook session payload without importing OVRTX; pass
`require_renderer=True` when a script should fail early unless `ovrtx` and
`ovstream` are importable.

`scene.show(backend="anari", streaming=True)` creates an ANARI-SDK native viewer
handoff descriptor. This path is intentionally SDK-oriented: it defaults to the
SDK sample `helide` library, can launch a configured ANARI-SDK viewer executable,
and does not select NVIDIA VisRTX unless a user or downstream integration opts
into that device explicitly.

## Scalar Styling

Portable scalar controls live on the layer style:

```python
scene.add_raster(
    forecast,
    variable="t2m",
    colormap="turbo",
    alpha=0.75,
    gamma=0.9,
    input_range=(250.0, 320.0),
    output_range=(0.0, 1.0),
)
```

Use these for opacity, gamma correction, and scalar remapping. Renderer-specific
texture-coordinate details such as UV flips, longitudinal offsets, and affine
texture transforms are kept as backend/source metadata rather than promoted to
scene-level controls.

## Extension Protocols

The extension surface is intentionally small. Downstream packages should target
three structural protocols:

- {class}`earth2studio.viz.LayerProtocol`: a layer has `id`, `name`, `kind`,
  `visible`, `data`, `metadata`, and `summary()`.
- {class}`earth2studio.viz.SceneProtocol`: a scene has `layers`,
  `visible_layers`, `metadata`, and `summary()`.
- {class}`earth2studio.viz.SceneEventProtocol`: backend sessions receive
  scene, layer, timeline, and visibility mutation events.
- {class}`earth2studio.viz.SceneSessionProtocol`: a streaming session has
  `update`, `flush`, `hold`, and `close`.
- {class}`earth2studio.viz.BackendProtocol`: a backend has `supports`,
  `render`, `show`, `save`, and `animate`.

These protocols are designed for renderer packages and exporters. They do not
require Earth2 Studio data sources, model classes, or IO backend objects.

## Loading From Stores

Keep visualization xarray-native or dataframe-native. For model output stores,
load or lazily open the data with xarray first, then pass the resulting object to
`viz`:

```python
import xarray as xr
from earth2studio import viz

ds = xr.open_zarr("outputs/forecast.zarr")
viz.plot(ds, variable="t2m", time=0, lead_time=4)
```

The visualization API does not currently accept Earth2 Studio IO backend objects
directly. That boundary keeps layer APIs small and lets future high-performance
paths be added without changing the user-facing layer model.
