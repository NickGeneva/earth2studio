# Earth2 Studio Visualization Submodule Proposal

## Context

This proposal sketches an `earth2studio.viz` submodule for visualization APIs
that work directly with Earth2 Studio data structures while leaving rendering
backends replaceable.

The local Earth2 Command Center checkout described in the request was not
materialized on this VM. The closest matching reference found was the public
`NVIDIA-Omniverse-blueprints/earth2-weather-analytics` repository, cloned into
`work/earth2-weather-analytics` for inspection. The relevant Command Center code
lives under `earth-2-command-center/source/extensions`.

This revision incorporates the latest direction:

- Skip the web server/widget backend for now.
- Use one consolidated visualization dependency surface instead of splitting
  backend extras prematurely.
- Keep the API shape close to Earth2 Command Center while making Earth2 Studio
  the source of truth for xarray, pandas, and cuDF data.
- Make backend routing explicit so the same scene can target CartoPy,
  Matplotlib, OVRTX, OpenUSD, ANARI, or future exporters without changing user
  code.
- Keep public APIs xarray-native and dataframe-native. Users should open Zarr,
  NetCDF, or other stores with xarray first, then pass the xarray object to
  `viz`.
- Use Earth2 Studio lexicon variable names in examples: `t2m`, `u10m`, `v10m`,
  `ws10m`, `msl`, `tcwv`, `q850`, `z500`, `tp`, and `refc`.

## What Earth2 Studio Already Provides

Earth2 Studio is already very close to the right data model for a visualization
layer:

- `DataSource` and `ForecastSource` return `xr.DataArray` values with `time`,
  `variable`, optional `lead_time`, and spatial coordinates.
- `DataFrameSource` and `ForecastFrameSource` return `pd.DataFrame` values for
  sparse observations, station data, tracks, or tabular geospatial products.
- `fetch_dataframe` already promotes pandas data frames to `cudf.DataFrame` on
  CUDA devices when `cudf` is available.
- `fetch_data(..., legacy=False)` can return `xr.DataArray` directly, including
  CuPy-backed arrays for CUDA devices when `cupy` is available.
- `CoordSystem` is the existing internal coordinate representation for tensor
  workflows.
- `XarrayBackend`, `ZarrBackend`, and `NetCDF4Backend` expose stored forecast
  outputs through named arrays and coordinate metadata.

That means `viz` should consume, adapt, and normalize existing Earth2 Studio
objects rather than invent a new data container.

## Dependency Inventory

The current checkout already contains some visualization dependencies in
non-runtime contexts:

- `cartopy` and `matplotlib` are already listed in the docs dependency group in
  `pyproject.toml`.
- `bokeh==3.8.2` already appears in `requirements.txt`.
- No current `pyproject.toml` or `requirements.txt` entries were found for
  `plotly`, `ovrtx`, `anari`, `usd-core`, `holoviews`, `hvplot`, `datashader`,
  or `geoviews`.

The proposed packaging model is one large, fixed `viz` dependency group for now.
Do not split `viz-cartopy`, `viz-gpu`, `viz-web`, etc. until actual adoption
patterns justify it.

```toml
[project.optional-dependencies]
viz = [
    "matplotlib>=3.8",
    "cartopy>=0.23",
    "pillow>=10",
    "pyproj>=3.6",
    "shapely>=2.0",
    "geopandas>=0.14",
    "rasterio>=1.3",
    "rioxarray>=0.15",
    "bokeh>=3.8",
    "holoviews>=1.20",
    "hvplot>=0.11",
    "datashader>=0.17",
    "geoviews>=1.14",
    "plotly>=6",
    "usd-core>=26.5",
    "ovrtx==0.3.0.312915",
    "ovstream",
    "anari==1.0.1",
]
```

Notes:

- `ovrtx` is the NVIDIA Omniverse RTX package name. PyPI currently lists
  `0.3.0.312915` uploaded May 18, 2026, with Python >=3.10. NVIDIA's package
  index provides platform wheels.
- `anari` is the current PyPI package name for SDK-style Python bindings. The
  native interactive viewer path should still target a Khronos ANARI-SDK build
  with the viewer component enabled; the Python wheel alone is not the viewer.
- `usd-core` provides OpenUSD Python libraries. It is the right dependency
  anchor for scene export, terrain mesh packaging, OpenUSD payloads, and
  renderer handoff.
- RAPIDS packages such as `cudf` should be consumed when already present in the
  user's Earth2 Studio CUDA environment. They are broader runtime choices than
  visualization-specific dependencies and should not be selected independently
  by `viz` unless the main Earth2 Studio CUDA packaging strategy does so.
- FastAPI, Uvicorn, WebGL, and browser-server dependencies are intentionally
  omitted from this phase.

## Earth2 Command Center Alignment

Earth2 Command Center has a useful interaction model, even though its
implementation is Omniverse Kit-specific.

| Earth2 Command Center concept | Observed behavior | Earth2 Studio `viz` counterpart |
| --- | --- | --- |
| `FeaturesAPI` | Creates, stores, removes, reorders, filters, and emits change events for visual features. | `Scene` as an ordered layer registry with backend-neutral change points. |
| Base `Feature` | Common `id`, `name`, `active`, `time_coverage`, and `meta`. | Base `Layer` with `id`, `name`, `visible`, `time_extent`, and `metadata`. |
| `Image` feature | Projection, image/alpha sources, colormap, remapping, flips, longitudinal offset, affine transform. | `RasterLayer` plus `ProjectionSpec`, `LayerStyle`, and backend-side image encoding. |
| `Curves`, markers, tracks | Point/curve geometry with projection, color, width, and periodic metadata. | `PointLayer`, `TrackLayer`, and `VectorLayer` backed by pandas or cuDF frames. |
| `TimeManager` | Maps UTC time ranges to playback time and feature time coverage. | `Timeline` inferred from xarray `time`, `lead_time`, valid time, or dataframe time columns. |
| `TimestampedSequence` | Chooses the current texture/object for the active timeline frame. | `FrameSet` or timeline-aware data view that selects array/table slices lazily. |
| DFM bridge | Converts xarray-like weather tensors to uint8 images, registers timestamped textures, and updates image features. | Keep numeric xarray data until backend dispatch, then encode to image, mesh, texture, or volume only at the boundary. |
| `globe_view` delegates | Observe feature changes and update renderer state, timeline, camera, and visibility. | Backend adapters implementing a common `VizBackend` protocol. |
| Camera controls | Interactive globe camera state lives in renderer/view delegates. | Backend-neutral `Camera` with lon/lat/distance/heading/pitch/roll and projection mode. |

The key thing to port is the API shape: scene/layer registry, typed layers,
timeline-aware data, backend delegates, and camera controls. The Omniverse
implementation details should stay out of Earth2 Studio core.

### Parity Gap Inventory

The current implementation should avoid adding more user-facing methods just to
mirror every Command Center feature one-for-one. Missing capability should land
inside existing package areas: scene internals, layer metadata, adapters,
backend delegates, texture managers, and exporters.

| Capability | Current status | Missing for parity | Owning area |
| --- | --- | --- | --- |
| Scene/layer registry | Partial | Reordering and filtering helpers. Internal change events now notify streaming sessions. | `scene.py`, backend delegates |
| Base feature metadata | Implemented | None for the first pass. | `layers.py` |
| Image feature material controls | Partial | Alpha opacity, gamma, and scalar remapping are portable style controls. Alpha-source masks, flip U/V, longitudinal offset, and affine texture transforms remain backend/source-transform details. | `styles.py`, `assets.py`, backend material lowering |
| Timeline playback | Partial | Playback rate, loop policy, and UTC-to-playback mapping. Internal frame-change events now reach renderer sessions. | `timeline.py`, backend sessions |
| Dynamic texture streaming | Partial | Concrete OVRTX texture manager, async decode/upload queues, CPU staging cache, GPU residency cache, mosaic/tile/LOD loaders. | `textures.py`, `base.py`, renderer backend |
| Default global textures | Implemented | Actual packaged/pre-populated optimized assets. | `domains.py`, deployment packaging |
| Grid/projection support | Partial | Regular lat/lon, curvilinear lat/lon, projected/native, cubed-sphere, HPX/HEALPix, diamond, GOES, and geohash-indexed grid intent are represented. Native HEALPix/cubed face stacks can render as heatmap mosaics; backend payload builders still need concrete geographic lowering for HPX/geohash/tiled mosaics. | `grids.py`, `native.py`, adapters, backend payload builders |
| Regional terrain | Partial | Tiled terrain mesh generation, OpenUSD export, renderer-backed local scene session, vertical datum transforms. | `regional.py`, terrain builders, exporters |
| Vector/flow objects | Partial | Scene-level track adapter, streamline generation, 3D glyph instancing, backend flow-object lowering. | `layers.py`, vector payload builders |
| Application session | Partial | Summary, Matplotlib, and Cartopy sessions redraw from scene events. The initial OVRTX session now emits a browser/notebook globe payload with layer and timeline controls. Missing concrete OVRTX render loop, ovstream server ownership, camera sync, cleanup hooks, picking, and selection. | `backends/` |
| Data-to-visual payload bridge | Missing | Stable xarray-to-texture encoding, texture compression, volume payload conversion, forecast provenance on generated payloads. | `adapters/`, backend payload builders |

This gap inventory is also encoded in `earth2studio.viz.capabilities` as
internal scaffolding. It is intentionally not exported from `earth2studio.viz`
because it is an implementation planning tool, not a user plotting API.

### Texture Streaming Lessons

Command Center does a large amount of useful work below the feature API:

- `GenericTimestampedSequence` maps timeline time to the nearest current
  object and invokes update hooks.
- `TimestampedSequence` and `MosaicTimestampedSequence` keep feature sources as
  stable `dynamic://...` URLs while loading the current JPEG or tile set into
  dynamic texture objects.
- `TimestampedJPEGSequence` handles path-backed and byte-buffer-backed JPEGs,
  validates texture objects lazily, and toggles texture residency when image
  features are shown or hidden.
- The application setup path sizes the dynamic texture cache from a host-memory
  budget and can switch synchronous texture updates off for async staging.
- Metadata sequence loading converts timestamp-keyed image dictionaries into
  single-texture, mosaic, diamond, or HPX texture sequences without changing the
  public image feature fields.

Earth2 Studio should keep the same separation. User APIs should create layers
from xarray objects, dataframes, images, GeoTIFFs, meshes, or texture sequences.
Renderer backends should own decode, staging, prefetch, upload, eviction, and
resource lifetime.

## Ecosystem Mapping

The goal is not to out-Tableau Tableau or out-Cesium Cesium. The goal is to
make Earth2 Studio outputs visualizable before users are forced into export,
ETL, dashboard schema work, or another runtime.

| System | Advantages | Disadvantages for Earth2 Studio users | Proposed `earth2studio.viz` mapping |
| --- | --- | --- | --- |
| OVRTX / Omniverse RTX | NVIDIA-native RTX rendering, Python/C library path, strong fit for physically accurate GPU rendering and Omniverse interoperability. | New package surface; likely RTX/GPU and packaging constraints; not a full weather-specific API by itself. | High-fidelity local renderer backend for globe, raster texture, volume, and future USD-oriented outputs. |
| OpenUSD + RTX Scientific / IndeX | OpenUSD provides a scene graph and asset interchange target; RTX Scientific / IndeX provides NVIDIA volume rendering and compositing with surface geometry. | Requires explicit conversion from xarray/dataframes into meshes, textures, OpenVDB volumes, or USD payloads. | Primary route for regional digital twin packages: tiled terrain meshes, draped weather textures, local assets, and volume fields. |
| ANARI | Open standard for portable scientific rendering across engines; scene representation maps well to geometry, volumes, cameras, and materials. | Python binding/package maturity is uneven; current PyPI binding is CUDA/Linux-oriented. | Renderer-neutral scientific backend target so Earth2 Studio layers can route to multiple future engines. |
| CartoPy + Matplotlib + xarray.plot | Familiar Python stack, publication-quality static maps, strong projection support, already partially present in Earth2 Studio docs tooling. | Static-first; limited interactive timeline/globe experience; repeated boilerplate across recipes. | Default static backend for quick plots, animations, regression tests, and documentation examples. |
| HoloViz stack | Works with pandas, xarray, Dask, GeoPandas, Bokeh; Datashader handles very large data; good notebook ergonomics. | Larger dependency stack; interaction model is notebook/dashboard-oriented rather than Earth2-specific. | Optional Python-native exploratory backend/export path for tabular and gridded data without losing labels. |
| Plotly + Dash | Interactive Python figures, map support, easy sharing in notebooks and dashboards. | Geospatial/scientific semantics still need adapter code; Dash/server path is intentionally deferred. | Future figure backend/exporter for users who want interactive HTML artifacts from the same scene model. |
| deck.gl + kepler.gl | Mature layer model for large geospatial data, point clouds, trips, tiles, H3/S2, and time playback. | JavaScript/web-oriented; requires tabular/tile packaging and front-end runtime. | Reference model for layer capabilities and possible later scene export, but not an initial runtime backend. |
| CesiumJS | Strong 3D globe, imagery, terrain, 3D Tiles, entities, camera controls, and time-dynamic geospatial data. | Browser/JavaScript stack and geospatial asset pipeline; not directly xarray/cuDF-native. | Reference model for globe camera/timeline semantics and a possible later exporter, not a current backend. |
| Google Earth Engine + Looker Studio | Earth Engine provides cloud-scale geospatial processing and visualization; Looker Studio provides connectors, dashboards, geo charts, and sharing. | Data must move into Google's service and/or dashboard abstractions; weather model tensors and GPU data lose native Python semantics. | Keep Earth2 Studio analysis local and Python-native, with future export hooks for publishing selected outputs. |
| Azure Maps + Power BI | Enterprise map layers, markers, heat maps, 3D columns, filled maps, reference/tile layers, and BI sharing. | BI-first model; cloud/service dependencies; weak fit for forecast tensors, lead time, and scientific volume data. | Use similar layer taxonomy for points, heat maps, filled maps, routes, and reference layers while preserving forecast semantics. |
| Tableau | Mature BI workflow, geocoding, spatial files, dashboards, sharing, and non-programmer ergonomics. | Export/ETL heavy for xarray forecasts; not designed around lead time, valid time, GPU arrays, or model provenance. | Provide "first visualization in package" so Tableau becomes an optional presentation/export destination rather than the first stop. |

## Design Goals

1. Make the first API natural for Earth2 Studio users:

   ```python
   from earth2studio import viz

   scene = viz.Scene()
   scene.add_raster(forecast_da, variable="t2m", colormap="turbo")
   scene.add_points(obs_df, lat="lat", lon="lon", color="temperature")
   scene.show(backend="cartopy")
   ```

2. Support quick static plots and local interactive renderer sessions without a
   web server backend:

   ```python
   viz.plot(da, variable="tcwv", time=0, lead_time=6)

   scene = viz.Scene()
   scene.add_raster(forecast_da, variable="msl", colormap="viridis")
   scene.camera.set(lon=-98.0, lat=38.0, distance=2.4)
   scene.show(backend="ovrtx")
   ```

3. Use xarray labels and dataframe columns for selection whenever possible.

4. Preserve device locality until a backend needs materialization. CPU plotting
   can pull arrays to NumPy; GPU backends should accept CuPy, torch, DLPack, or
   CUDA array interface data when supported.

5. Make timeline and camera first-class concepts so static plots, notebooks,
   OVRTX, ANARI, and later exporters can share the same scene description.

6. Keep backend-specific objects out of the top-level API. Users should choose
   `backend="cartopy"` or `backend="ovrtx"` without rewriting layer creation.

7. Make large texture streaming a backend-internal capability. The public layer
   API should not change when a backend starts using tiled GeoTIFF reads, JPEG
   decode queues, CPU staging buffers, GPU texture caches, or timeline/camera
   prefetch.

8. Let simple layer calls express common time-series products. If a raster has
   selected `time` or `lead_time` frames, `Scene.add_raster` should preserve
   those frames as a raster sequence. The Matplotlib backend can then apply an
   opinionated default layout with layers as rows and frames as columns, without
   requiring examples to build bespoke subplot grids.

## Proposed Package Layout

```text
earth2studio/viz/
  __init__.py
  api.py
  assets.py
  textures.py
  quick.py
  scene.py
  layers.py
  timeline.py
  camera.py
  base.py
  grids.py
  selection.py
  styles.py
  regional.py
  terrain.py
  volume.py
  tiling.py
  adapters/
    __init__.py
    xarray.py
    dataframe.py
  backends/
    __init__.py
    base.py
    matplotlib.py
    cartopy.py
    ovrtx.py
    anari.py
    openusd.py
  exporters/
    __init__.py
    openusd.py
```

No `web.py` backend is included in this phase.

## Core API Model

### Scene

`Scene` is the user-facing registry of layers and view state.

Responsibilities:

- Hold ordered `Layer` objects.
- Hold a `Timeline` and `Camera`.
- Provide `add_raster`, `add_contour`, `add_points`, `add_tracks`,
  `add_vectors`, `add_terrain`, `add_draped_raster`, `add_region_cube`, and
  `add_volume`.
- Resolve data adapters lazily.
- Dispatch to a backend with `show`, `render`, `save`, or `animate`.

### Layer

`Layer` should be a small dataclass-like object with backend-neutral metadata:

```python
Layer(
    id="layer-001",
    name="2m temperature",
    kind="raster",
    data=...,
    visible=True,
    time_extent=(...),
    style=LayerStyle(
        colormap="turbo",
        vmin=250,
        vmax=320,
        alpha=0.85,
        gamma=0.9,
        input_range=(250, 320),
        output_range=(0, 1),
    ),
    projection=ProjectionSpec(kind="latlon"),
    metadata={...},
)
```

Initial layer types:

- `RasterLayer`: dense gridded `xr.DataArray` or selected variable from
  `xr.Dataset`; supports regular `lat/lon`, `y/x` with 2D lat/lon coords,
  future curvilinear grids, and selected `time` or `lead_time` raster
  sequences. Extra non-spatial dimensions such as `ensemble` should be selected
  or reduced with xarray before rendering.
- `ContourLayer`: derived from raster data but rendered as isolines or filled
  contours.
- `PointLayer`: pandas or cuDF points with lat/lon/time columns and scalar
  fields.
- `TrackLayer`: dataframe or list-of-dataframes with time-ordered paths.
- `VectorLayer`: paired vector components from dense fields or sparse rows.
- `TerrainLayer`: local or regional elevation/bathymetry surface from DEM, DSM,
  GeoTIFF, xarray, or NumPy-like height fields.
- `DrapedRasterLayer`: weather, imagery, land-use, hazard, or analysis rasters
  projected onto terrain or a flat regional plane.
- `ImageLayer`: external image, in-memory image, or timestamped texture
  sequence intended for a globe, map, terrain surface, or local plane.
- `GeoTiffLayer`: GeoTIFF or Cloud Optimized GeoTIFF intent with CRS/bounds
  metadata and a role such as raster, terrain, texture, or draped raster.
- `MeshLayer`: external or in-memory local 3D geometry, including OpenUSD,
  glTF/GLB, OBJ, STL, PLY, or backend-native mesh payloads.
- `RegionCubeLayer`: bounded local 3D box with `x/y/z/time` or
  `lon/lat/height/time` coordinates for regional atmospheric, ocean, sensor, or
  scenario data.
- `VolumeLayer`: volumetric scalar fields represented as slice stacks, transfer
  functions, or OpenUSD `UsdVol` / OpenVDB assets when a renderer supports them.

### Grid and Projection Support

Command Center uses image projection strings such as `latlong`, tiled
`latlong_u_v`, `diamond`, `hpx`, and `goes`, plus affine mappings for
rectangular lat/lon subsets. Earth2 Studio should not expose those as a large
scene API. Instead, each layer should carry a backend-neutral `GridSpec` through
its adapter and projection metadata.

Supported grid intents:

- `regular_latlon`: 1D latitude and longitude coordinates.
- `curvilinear_latlon`: 2D latitude/longitude coordinate arrays.
- `projected`: x/y coordinates with CRS or grid mapping metadata.
- `native`: model-native grids that need backend-specific lowering.
- `cubed_sphere`: six-face cubed-sphere arrays, typically `(face, height, width)`.
- `healpix`: HPX/HEALPix-style indexed or tiled spherical grids.
- `diamond`: Command Center ICON/diamond-style globe texture grids.
- `goes`: geostationary satellite projection intent.
- `geohash`: geohash-indexed regions or trigger cells.

For xarray raster layers, the adapter infers the grid descriptor and stores it
in `RasterView.grid`; `Scene` mirrors that into `ProjectionSpec.metadata["grid"]`
so it is visible in summaries and backend routing. For sparse dataframe trigger
data, static plotting should still use decoded lat/lon or x/y columns today,
while future backend payload builders can lower geohash cells into polygons or
instanced geometry.

For immediate static diagnostics, `native_grid_heatmap` lowers cBottle-style
`hpx` vectors, HEALPix PAD_XY face stacks, cubed-sphere arrays, and diamond face
stacks into a 2D heatmap mosaic. This is deliberately not a geographic
reprojection; it is a native-grid view that lets examples and tests inspect the
collected model values without carrying bespoke Matplotlib code. A later
CartoPy/earth2grid or renderer payload builder can turn the same `GridSpec` into
lat/lon textures, projected meshes, or globe-native tiles.

The rule is: every grid should be representable, but only backends that know how
to lower that grid should claim full rendering support. This avoids silently
pretending that all grids are regular lat/lon while keeping the public layer API
stable.

### Timeline

`Timeline` should infer temporal coverage from:

- xarray `time` coordinate.
- xarray `lead_time` coordinate, optionally converted to valid time with
  `time + lead_time`.
- dataframe time columns.
- explicit user-provided frame coordinates.

Recommended methods:

- `timeline.frames`
- `timeline.current`
- `timeline.set(time=..., lead_time=...)`
- `timeline.use_valid_time()`
- `timeline.play(rate=...)`
- `timeline.range()`

### Camera

`Camera` should be backend-neutral:

```python
Camera(
    lon=0.0,
    lat=20.0,
    distance=2.5,
    heading=0.0,
    pitch=0.0,
    roll=0.0,
    projection="globe",
)
```

CartoPy can translate this into projection parameters. OVRTX, OpenUSD, or ANARI
can map it to renderer camera state.

## Local Regional Scenes and Terrain Cubes

Regional visualization should be a first-class mode, not an accidental subset of
global weather plotting. This covers local digital twins, coastal-state or
province-scale views, terrain boxes, ocean/atmosphere cubes, mission-planning
style overlays, and other bounded 3D scenes where users orbit, pan, and inspect
data in a local coordinate frame.

### RegionSpec

Add a `RegionSpec` object that defines the spatial contract:

```python
RegionSpec(
    name="coastal-region",
    crs="EPSG:26910",
    bounds=(xmin, ymin, xmax, ymax),
    vertical_datum="EGM96",
    z_units="m",
    origin=(x0, y0, z0),
    local_frame="enu",
)
```

Responsibilities:

- Convert source lon/lat data into a local projected coordinate frame.
- Preserve CRS, vertical datum, units, and origin metadata for export.
- Support local orbit/fly cameras without global-globe assumptions.
- Provide a consistent bounding box for terrain, volume, assets, and overlays.

### TerrainLayer

`TerrainLayer` should accept DEM/DSM/bathymetry/topography inputs as:

- `xr.DataArray` with `x/y` or `lat/lon` coordinates.
- GeoTIFF or Cloud Optimized GeoTIFF paths.
- NumPy/CuPy arrays with explicit transform/CRS metadata.

The first NVIDIA-oriented representation should be a tiled OpenUSD mesh:

- Convert DEM samples into `UsdGeomMesh` terrain tiles.
- Use `subdivisionScheme="none"` for explicit polygonal terrain meshes.
- Author normals and optional material/texture coordinates.
- Use tile payloads/references so a regional scene can load progressively.
- Support LOD/downsampling for large regions.

Static fallbacks can generate hillshade, slope, contour, or color-relief maps
through Matplotlib/CartoPy.

### DrapedRasterLayer

`DrapedRasterLayer` maps 2D data onto a terrain surface:

- Forecast variables at surface level, such as temperature, wind speed, smoke,
  precipitation, soil moisture, or hazard indices.
- Orthophotos, land-use maps, road masks, or segmentation outputs.
- Time-indexed texture sequences for forecast playback.

For OVRTX/OpenUSD, the raster should become one or more texture assets bound to
terrain tiles or regional planes. For CartoPy/Matplotlib, it renders as a map
image over a projected axis.

### RegionCubeLayer

`RegionCubeLayer` represents bounded 3D fields:

- Shape examples: `(time, z, y, x)`, `(lead_time, level, lat, lon)`, or a
  dataframe of sparse 3D points with `x/y/z/time`.
- Supported views: horizontal slices, vertical cross sections, stacked planes,
  isosurfaces where supported, and full transfer-function volume rendering.
- Coordinate options: height above ground, height above ellipsoid, pressure
  level, depth below sea level, or model-level index with metadata.

The MVP should expose slice-stack rendering before full volume rendering:

```python
scene = viz.Scene(region=region)
scene.add_terrain(dem, texture=ortho)
scene.add_region_cube(
    cube,
    variable="q850",
    vertical="height",
    mode="slices",
    levels=[250, 500, 1000, 1500],
    colormap="viridis",
)
scene.show(backend="ovrtx")
```

Full volume rendering should use OpenUSD `UsdVol.Volume` with OpenVDB assets
when the runtime supports it. NVIDIA RTX Scientific / IndeX is the best-mapped
NVIDIA path for large scalar volumes, transfer functions, cross sections, and
depth-correct compositing with surface terrain.

### Regional API Example

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
scene.add_terrain(dem_da, name="Terrain", vertical_exaggeration=1.5)
scene.add_draped_raster(
    forecast_da,
    variable="ws10m",
    time="2026-06-07T18:00:00",
    colormap="turbo",
    alpha=0.7,
)
scene.add_points(units_or_sensors, x="x", y="y", z="altitude", color="status")
scene.add_region_cube(regional_cube, variable="q850", vertical="height", mode="slices")

scene.camera.orbit(target="region", azimuth=225, elevation=35, distance=140_000)
scene.show(backend="ovrtx")
scene.save("outputs/coastal_state.usd", backend="openusd")
```

## Texture Streaming and External Assets

The current Python-facing API should stay xarray-native and dataframe-native
for Earth2 Studio model outputs. External assets are a second, explicit path for
data that is already an image, GeoTIFF, texture sequence, or mesh:

```python
scene = viz.Scene()
scene.add_raster(forecast_ds, variable="t2m")
scene.add_image("blue_marble.jpg", bounds=(-180.0, -90.0, 180.0, 90.0))
scene.add_geotiff("local_dem.cog.tif", role="terrain", crs="EPSG:32610")
scene.add_mesh("terrain.usd", crs="EPSG:32610")
```

These methods create intent layers only. They do not open files or couple the
scene to rasterio, GDAL, OpenUSD, OVRTX, or ANARI at layer creation time.

### Default Texture Domain

Global scenes need a default texture domain so common assets exist without
wiring layer creation to a datasource. The default domain should include, at
minimum:

- `global_base_color`
- `global_topography`
- `global_clouds`
- `global_boundaries`

These are renderer assets, not model outputs, so their cache should live under
the common Earth2 Studio cache root instead of the datasource cache:

```text
${EARTH2STUDIO_CACHE:-~/.cache/earth2studio}/viz/v5/default_textures/
```

The default cache uses clear unhashed filenames such as
`global_base_color.ktx2`, `global_clouds.ktx2`, and
`global_boundaries.ktx2`. This makes pre-population, inspection, replacement,
and deployment packaging straightforward. Backends can still keep their own
GPU-resident caches underneath the texture manager, but the source cache for
default assets should remain human-readable.

### Internal Streaming Contract

Backend implementations should lower layers into a small set of internal
descriptors:

- `AssetSource`: path, URI, or in-memory object with `kind`, CRS, bounds, time,
  MIME type, and metadata.
- `TextureSource`: image-like asset metadata, including codec, channels,
  optional tile size, and optional level count.
- `MeshSource`: geometry asset metadata, transform, CRS, and material hints.
- `TextureFrame`: one timeline frame, tile, LOD, or viewport texture with a
  stable cache key.
- `TextureSequence`: ordered `TextureFrame` values with timestamp selection and
  prefetch candidate calculation.
- `TextureCachePolicy`: backend-owned budgets and behavior for CPU staging, GPU
  residency, async decode, async upload, current-frame pinning, and eviction.

The backend should then provide a texture manager, matching
`TextureManagerProtocol`, with four operations: resolve the active frame,
prefetch candidate frames, release one layer, and clear all resources.

### Cache and Prefetch Strategy

Recommended backend cache hierarchy:

1. Source cache: file handles, HTTP handles, object-store readers, rasterio
   datasets, OpenUSD stages, or application-provided byte buffers.
2. CPU staging cache: decoded JPEG/PNG/WebP frames, GeoTIFF windows, COG tiles,
   rescaled uint8/RGBA textures, or mesh payloads ready for backend upload.
3. GPU cache: OVRTX, ANARI, CUDA, OpenGL/Vulkan, or renderer-native texture and
   buffer handles.

Prefetch should be driven by both timeline and camera:

- Pin the current timeline frame while it is visible.
- Prefetch `n` previous and next frames using `TextureCachePolicy`.
- For tiled globe or regional terrain views, prefetch visible tiles and
  neighboring LODs from the camera frustum.
- Cancel or deprioritize stale requests when timeline scrubbing jumps.
- Draw the previous ready frame or a backend placeholder when a requested frame
  is still decoding or uploading.

Cache keys should include layer id, variable or asset URI, time/lead-time,
tile id, LOD, style hash, projection transform, and any resampling/remapping
metadata. Hidden layers should allow GPU handles to be deactivated or evicted;
scene/backend shutdown should release all backend-owned resources.

### Material Controls

The first-class material controls should be the ones that are portable across
static plots, OVRTX-style texture rendering, ANARI, and future exporters:

- `alpha`: layer opacity.
- `gamma`: scalar/texture power normalization.
- `input_range`: input scalar range before remapping.
- `output_range`: output scalar range after remapping.
- `colormap`, `vmin`, and `vmax`: existing scalar-to-color controls.

These controls belong in `LayerStyle` because they apply to xarray rasters,
image textures, draped rasters, terrain color ramps, and static plots.

Texture-coordinate controls should not become broad scene-level APIs yet:

- flip U/V
- longitudinal offset
- affine texture transforms
- projection-specific tile transforms

Those should live as source-transform metadata or backend material-lowering
details. They matter for renderer correctness, but exposing them too early at
the scene level would make the API feel like a direct clone of one renderer's
material system.

### Data-To-Visual Payload Bridge

The data-to-visual payload bridge is not a new user API. It is the internal
lowering layer that turns semantic Earth2 Studio data into renderer-ready
payloads.

User-facing layer creation should remain simple:

```python
scene.add_raster(forecast, variable="t2m", time=0, colormap="turbo")
scene.add_draped_raster(wind_speed, variable="ws10m")
scene.add_region_cube(cube, variable="q850")
```

The bridge is what a backend uses after that:

```text
xarray / dataframe layer
  -> selected labeled view
  -> styled visual payload
  -> backend resource handle
```

Concrete examples:

- Raster field to texture: select `time`, `lead_time`, and variable; compute or
  reuse stable `vmin/vmax`; apply colormap or channel packing; produce an
  RGBA/float texture frame with provenance.
- Forecast timeline to texture sequence: repeat raster-to-texture conversion
  per frame; generate stable cache keys; feed `TextureSequence` and
  `TextureManagerProtocol`.
- DEM to terrain mesh: turn a labeled height field into tiled vertices,
  indices, normals, texture coordinates, and vertical metadata.
- Regional cube to volume payload: turn `x/y/z/time` or
  `lon/lat/height/time` data into slice textures, volume bricks, or OpenVDB
  references depending on backend capability.
- Dataframe to geometry payload: turn pandas/cuDF points, tracks, vectors, or
  streamline seeds into backend geometry, curves, glyphs, or instance buffers.

The bridge should preserve xarray labels, lexicon names, units, variable
metadata, forecast initialization time, lead time, valid time, CRS, bounds, and
style hashes on generated payloads. That metadata is important for cache keys,
debugging, reproducibility, and renderer-independent export.

Recommended ownership:

- `adapters/` owns labeled data views: xarray/dataframe selection, coordinates,
  dimensions, and device locality.
- `payloads.py` or `backends/<backend>/payloads.py` should own visual payload
  builders: raster-to-texture, DEM-to-mesh, cube-to-volume, dataframe-to-geometry.
- `textures.py` owns timeline texture frames, texture sequences, and cache
  policy.
- Backend texture managers own decode, staging, GPU upload, residency, and
  eviction.

This keeps IO out of the visualization API. Zarr, NetCDF, GeoTIFF, or other
stores should still be opened by the caller or an asset helper first. The bridge
starts from xarray/dataframe/asset descriptors and stops at backend-ready
payloads.

### Real Texture Manager Expansion

The current code intentionally has only the expansion contract:
`TextureManagerProtocol.resolve`, `prefetch`, `release_layer`, and `clear`.
A real backend texture manager would be moderately difficult, but the existing
structure is ready for it.

Difficulty by stage:

- Low: static asset resolver. Map `TextureSource` and `TextureFrame` to local
  paths, readable default-domain cache entries, or in-memory buffers.
- Medium: CPU staging cache. Decode image bytes, read GeoTIFF windows, rescale
  xarray frames, and keep a bounded LRU/timeline-distance cache.
- Medium: timeline prefetch. Use `TextureSequence.prefetch_frames(...)` plus
  scene timeline changes to stage nearby frames and cancel stale work.
- High: GPU residency manager. Allocate backend-native texture handles, upload
  asynchronously, use fences/readiness states, pin visible frames, and evict old
  handles safely.
- High: tiled/LOD streaming. Combine camera frustum, globe/regional tile
  selection, COG/raster windows, mosaic textures, and neighbor/LOD prefetch.
- Highest: multi-backend parity. OVRTX, ANARI, OpenUSD, and static plot
  backends have different resource lifetimes and upload semantics, so the
  manager should be backend-owned rather than global.

The first real implementation should live under the first interactive renderer
backend, not in public `Scene`. A good first target is:

```text
earth2studio/viz/backends/ovrtx/
  textures.py      # concrete texture manager
  payloads.py      # raster/cube/asset lowering
  session.py       # renderer session, scene subscriptions, cleanup
```

The existing public API should not need to change. The backend can inspect
`RasterLayer`, `ImageLayer`, `GeoTiffLayer`, `TerrainLayer`,
`DrapedRasterLayer`, `RegionCubeLayer`, `TextureSequence`, `TextureSource`,
`TextureCachePolicy`, and `TextureDomain` to build and stream resources.

### GeoTIFFs, Images, and Meshes

For Earth2 Studio stores, users should still open with xarray and pass the
result to `add_raster`, `add_terrain`, or `add_draped_raster`. For asset-native
inputs:

- Images use `add_image(...)` and become `TextureSource` or `TextureSequence`
  descriptors. Backends decide whether to load from path, URL, bytes, or GPU
  array.
- GeoTIFF and Cloud Optimized GeoTIFF inputs use `add_geotiff(...)`, or users
  can open them through xarray/rioxarray first and use the normal raster or
  terrain APIs. OVRTX/OpenUSD backends should use windowed/tiled reads when the
  file is too large for one texture.
- Mesh inputs use `add_mesh(...)`. The OpenUSD backend should pass USD assets
  through directly when possible; other mesh formats can be lowered later by a
  mesh adapter without changing `Scene`.
- Xarray fields with CuPy, torch, or CUDA-array-backed data should preserve
  device locality until the backend materializes a texture or buffer. Static
  Matplotlib/CartoPy backends can still materialize NumPy arrays.

## Data Adapters

The adapters turn Earth2 Studio objects into layer-ready views:

- `XarrayAdapter`: accepts `xr.DataArray` or `xr.Dataset`, performs variable and
  coordinate selection, extracts time axes, and identifies spatial dimensions.
- `DataFrameAdapter`: accepts `pd.DataFrame` and `cudf.DataFrame`, infers or
  accepts lat/lon/time/value columns, and preserves GPU data until rendering.

Do not add public `IOAdapter` APIs in this first phase. Model output stores
should be opened by the caller with xarray, for example
`xr.open_zarr(...)` or `xr.open_dataset(...)`, and then passed to `viz`. Data
sources should be evaluated by the caller before visualization. This keeps the
viz layer focused on a small, stable data contract and avoids coupling renderer
APIs to Earth2 Studio model or IO lifecycles.

The canonical internal data views should be lightweight:

- `RasterView`: `data`, `dims`, `coords`, `time_index`, `variable`, `device`,
  `attrs`.
- `FrameView`: `table`, `lat`, `lon`, `time`, `fields`, `device`, `attrs`.
- `VolumeView`: `data`, `coords`, `vertical`, `time_index`, `device`, `attrs`.
- `AssetSource` and `TextureSequence`: asset-native descriptors for inputs that
  are already image, GeoTIFF, texture, or mesh assets.

## Backend Protocol

Backends should implement a protocol rather than inherit heavy base classes:

```python
class VizBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def supports(self, scene: Scene) -> bool: ...
    def render(self, scene: Scene, **backend_kwargs) -> RenderResult: ...
    def show(
        self,
        scene: Scene,
        *,
        streaming: bool = False,
        auto_flush: bool = True,
        **backend_kwargs,
    ) -> Any: ...
    def save(self, scene: Scene, path: str | Path, **backend_kwargs) -> Path: ...
    def animate(self, scene: Scene, path: str | Path, **backend_kwargs) -> Path: ...
```

Backend-specific controls should remain keyword-only. The shared front-door
arguments are limited to the scene operation itself, for example `backend`,
`streaming`, and `auto_flush` for `Scene.show(...)`; renderer launch details,
cache sizes, ports, devices, and layout preferences stay in `**backend_kwargs`
and are validated by the selected backend.

Streaming backends should return a duck-typed session:

```python
class SceneSessionProtocol(Protocol):
    backend: str
    scene: Scene
    auto_flush: bool
    closed: bool

    def update(self, event: SceneEventProtocol) -> None: ...
    def flush(self) -> Any: ...
    def hold(self) -> ContextManager[SceneSessionProtocol]: ...
    def close(self) -> None: ...
```

`Layer.update(...)` and `Layer.append(...)` mutate layer state. `auto_flush`
decides whether pending scene events immediately reconcile with the backend.
Manual loops can pass `auto_flush=False` and call `session.flush()` explicitly.
This preserves the Command Center delegate model without adding a second render
API beside `show`.

There is intentionally no `serve()` method in this phase.

Backends that support interactive or high-fidelity rendering should also own a
texture manager. That manager should be backend-specific, but conform to a tiny
protocol: `resolve(frame)`, `prefetch(frames)`, `release_layer(layer_id)`, and
`clear()`. This keeps the Command Center-style dynamic texture machinery below
the layer API.

Recommended initial backends:

- `matplotlib`: simple 2D images, smoke tests, non-geospatial fallback.
- `cartopy`: geospatial static plots, projections, coastlines, and publication
  figures.
- `ovrtx`: high-fidelity RTX renderer path with an initial browser/notebook
  session payload and future ovstream frame loop when the package and runtime
  are available.
- `openusd`: scene export and renderer handoff path for regional terrain,
  textured surfaces, local assets, cameras, and volume references.
- `anari`: portable scientific renderer path with a headless descriptor and
  ANARI-SDK native viewer handoff first. The initial route should use the SDK
  viewer component and sample `helide`/environment-selected libraries rather
  than selecting VisRTX by default.

`register_backend(name, factory)` should allow downstream teams to add plugins
without modifying Earth2 Studio.

## Backend Routing Strategy

`viz` should select a backend through capability negotiation, not hard-coded
if/else branches in user APIs.

| Scene content | Preferred route | Fallback route |
| --- | --- | --- |
| 2D non-geospatial raster | `matplotlib` | `cartopy` if projection metadata exists |
| Lat/lon raster or contour | `cartopy` | `matplotlib` with raw axes |
| Raster timeline animation | `cartopy.animate` | `matplotlib.animate` |
| Points/tracks on a map | `cartopy` | `matplotlib` raw axes |
| Regional terrain / DEM | `openusd` then `ovrtx` | `cartopy` hillshade or contour map |
| Draped regional raster | `openusd` textures then `ovrtx` | `cartopy` projected raster |
| Image / GeoTIFF texture asset | `ovrtx` or `openusd` via texture manager | user-opened xarray raster or static image summary |
| Local globe raster | `ovrtx` | exported imagery |
| Regional 3D cube | `ovrtx` or `anari` | slice stacks via `cartopy` |
| OpenUSD / OpenVDB volume | `ovrtx` with RTX Scientific / IndeX when available | slice stacks or cross sections |
| Meshes / vector glyphs | `openusd` then `ovrtx`, or `anari` | downsampled 2D vectors via `cartopy` |
| BI/dashboard publication | future exporter | user-managed Tableau/Power BI/Looker workflow |

This keeps backend changes behind a stable scene model. The same `Scene` can be
shown as a static CartoPy map, a local OVRTX regional scene, or a future
exported Cesium/deck.gl scene without changing the data ingestion code.

## Example APIs

Quick plot:

```python
fig = viz.plot(
    da,
    variable="t2m",
    time="2025-01-01T00:00:00",
    lead_time="24h",
    backend="cartopy",
    projection="robinson",
    colormap="turbo",
)
```

Forecast timeline:

```python
scene = viz.Scene(title="SFNO forecast")
scene.add_raster(
    forecast,
    variable="tcwv",
    name="Total column water vapor",
    colormap="magma",
    vmin=0,
    vmax=80,
)
scene.timeline.use_valid_time()
scene.animate("outputs/tcwv.gif", backend="cartopy", fps=8)
```

Sparse observations:

```python
scene = viz.Scene()
scene.add_points(
    stations,
    lat="latitude",
    lon="longitude",
    time="valid_time",
    color="temperature",
    size=4,
)
scene.show(backend="cartopy")
```

Local interactive renderer:

```python
scene = viz.Scene()
scene.add_raster(forecast, variable="msl", colormap="viridis")
scene.camera.set(lon=-40, lat=25, distance=2.0)
viewer = scene.show(backend="ovrtx")
```

Browser-streamed OVRTX globe session:

```python
scene = viz.Scene(title="OVRTX globe")
scene.add_default_texture()
layer = scene.add_raster(forecast, variable="tcwv", colormap="turbo")

session = scene.show(
    backend="ovrtx",
    streaming=True,
    open_browser=True,
    require_renderer=False,
)
layer.append(next_frame, time=next_valid_time)
```

The session payload deliberately mirrors the useful Command Center pieces:
streamed viewport ownership, upper-right layer controls, bottom timeline
coverage, and orbit-camera intent. The browser document exposes the video
surface and state controls; actual RTX pixels should arrive from a backend-owned
`ovrtx` plus `ovstream` render loop rather than a browser-side 3D renderer.

Streaming model loop:

```python
scene = viz.Scene(title="Streaming inference")
layer = scene.add_raster(first_frame, name="t2m", colormap="turbo")

session = scene.show(backend="cartopy", streaming=True, auto_flush=False)
for valid_time, frame in model_frames:
    layer.append(frame, time=valid_time)
    if should_redraw(valid_time):
        session.flush()
session.close()
```

## Why Keep This In Earth2 Studio?

An in-package visualization module offers value that general-purpose BI and map
tools do not provide out of the box:

- It preserves xarray labels, forecast dimensions, `lead_time`, valid-time
  semantics, variable names, units, and model provenance.
- It avoids mandatory export to CSV, GeoJSON, image frames, tiles, dashboards,
  or external services just to inspect a forecast.
- It gives recipes and examples one canonical plotting surface instead of many
  local Matplotlib/CartoPy fragments.
- It can consume pandas and cuDF frames through the same layer API, preserving a
  path for GPU-local sparse data.
- It lets Earth2 Studio define weather-specific defaults: longitude wrapping,
  forecast timelines, colormap normalization across frames, pressure-level
  slicing, and Earth-centric camera conventions.
- It lets regional workflows keep CRS, vertical datum, terrain scale,
  topography, and local cube coordinates attached to the data instead of
  flattening them into anonymous images or dashboard tables.
- It makes external tools downstream choices. Tableau, Power BI, Looker Studio,
  Cesium, or deck.gl can become exporters or publication targets after the user
  has already validated the data in Python.

## Implementation Plan

### Phase 0: Scaffolding and API Contracts

- Add `earth2studio/viz` with no backend work at module import time.
- Define `Scene`, `Layer`, `Timeline`, `Camera`, `LayerStyle`, and backend
  protocol.
- Add xarray and dataframe adapters.
- Add synthetic unit tests for regular lat/lon `xr.DataArray`, curvilinear
  coordinate detection, pandas frames, and cuDF skip/xfail behavior.

### Phase 1: Static Plot MVP

- Implement `viz.plot`, `Scene.add_raster`, `Scene.add_points`, and
  `Scene.show(backend="matplotlib")`.
- Add CartoPy backend with projection selection and clear import/runtime errors.
- Support variable/time/lead_time selections by label or integer index.
- Add examples that replace repeated ad hoc Matplotlib/CartoPy code currently
  found in examples and recipes.

### Phase 2: Timeline and Animation

- Add timeline inference and frame iteration.
- Add `Scene.animate` for GIF/MP4 generation via Matplotlib/CartoPy.
- Support valid-time mode for forecast outputs.
- Add style/range helpers for stable colormap scaling across frames.
- Add `TextureFrame`, `TextureSequence`, and `TextureCachePolicy` tests for
  timestamp selection and prefetch windows.

### Phase 3: Local Renderer and Regional Scene Backends

- Implement OVRTX, OpenUSD, and ANARI backend prototypes behind the one
  consolidated `viz` dependency group.
- Use the same scene/layer/timeline model; do not expose renderer-specific
  objects in the top-level API.
- Add `RegionSpec`, `TerrainLayer`, `DrapedRasterLayer`, and
  `RegionCubeLayer`.
- Add OpenUSD scene export for terrain tiles, textures, cameras, and layer
  metadata.
- Add backend-owned texture managers for OVRTX/OpenUSD with decode, staging,
  upload, current-frame pinning, prefetch, and eviction.
- Add GPU data handoff support when these backends can consume CUDA arrays,
  DLPack, or compatible buffers directly.

### Phase 4: Volumes and Advanced Layers

- Add `VolumeLayer` after confirming target data conventions: pressure-level
  grids, 3D Cartesian volumes, spherical shells, or unstructured vertical
  coordinates.
- Add OpenUSD `UsdVol` / OpenVDB volume export where supported by the NVIDIA
  renderer stack.
- Add vector glyphs, streamlines, and track overlays.
- Add backend capability tests for volume, mesh, camera, and timeline behavior.

### Phase 5: Exporters and Deferred Web Work

- Add exporters only after the local API proves stable: image stacks, GeoTIFF,
  GeoJSON, deck.gl/kepler.gl configs, Cesium manifests, or BI-ready tables.
- Reconsider web server/widget support later. It should be implemented as
  another backend/export layer, not as a constraint on the core API.

## Testing Strategy

- Unit-test selection, coordinate inference, and timeline construction without
  requiring renderer initialization.
- Unit-test backend registry and import-gated error messages.
- Unit-test asset descriptors, texture sequences, cache policy validation, and
  scene asset helpers without renderer mocks.
- Use image comparison tests only for small deterministic Matplotlib outputs.
- Use skip markers for CartoPy, cuDF, CuPy, OVRTX, OpenUSD, ANARI, and volume
  export support.
- Add at least one smoke example that creates an `XarrayBackend` forecast-like
  dataset and visualizes it through `viz.plot`.
- Add a regional smoke example that builds a small synthetic DEM, drapes a
  raster field onto it, and exports an OpenUSD scene.
- Add dependency install smoke tests before pinning the consolidated `viz` group
  into release automation.

## Open Questions

- Should the single `viz` dependency group remain optional, or should Earth2
  Studio eventually make the static subset part of the base install?
- Should OVRTX be pinned to `0.3.0.312915` immediately, or should it use a
  compatible range after the first install smoke?
- How much of the ANARI-SDK native viewer bridge should live in Python versus a
  small compiled viewer application that consumes the `anari` backend payload?
- What CRS/projection metadata should be standardized for non-lat/lon grids
  such as HRRR?
- Which vertical datum and height conventions should be first-class for regional
  terrain and cube data?
- Which OpenVDB writing path should Earth2 Studio standardize on for volume
  exports?
- Which weather-specific layer defaults belong in Earth2 Studio core versus a
  downstream Earth2 application?

## Recommended MVP

The best first milestone is a small but real `earth2studio.viz` package with:

- `viz.plot` for `xr.DataArray` and `xr.Dataset`.
- `Scene.add_raster` and `Scene.add_points`.
- Matplotlib and CartoPy backends in one consolidated `earth2studio[viz]`
  dependency group.
- Timeline inference for `time` and `lead_time`.
- `RegionSpec`, `TerrainLayer`, and `DrapedRasterLayer` for local terrain and
  topography presentation.
- A design-compatible backend registry that can host OVRTX, OpenUSD, and ANARI
  without changing user code.
- No web server/backend implementation in the first pass.

This gives users immediate value while preserving the Command Center-inspired
path to timeline-aware globe and scientific visualization.

The manual plotting migration is tracked separately in
`docs/design/viz_examples_migration.md`.

## Research Sources

- Earth2 Weather Analytics / Command Center public repository:
  <https://github.com/NVIDIA-Omniverse-blueprints/earth2-weather-analytics>
- NVIDIA Earth-2 Weather Analytics blueprint:
  <https://build.nvidia.com/nvidia/earth2-weather-analytics/blueprintcard>
- OVRTX PyPI package:
  <https://pypi.org/project/ovrtx/>
- NVIDIA Omniverse libraries overview:
  <https://docs.nvidia.com/omniverse/index.html>
- NVIDIA package index for OVRTX:
  <https://pypi.nvidia.com/ovrtx/>
- NVIDIA RTX Scientific / IndeX documentation:
  <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx_scientific_index.html>
- NVIDIA Aerial Omniverse Digital Twin scene importer:
  <https://docs.nvidia.com/aerial/aerial-dt/text/scene_importer.html>
- OpenUSD `usd-core` package:
  <https://pypi.org/project/usd-core/>
- OpenUSD `UsdGeomMesh` documentation:
  <https://openusd.org/dev/api/class_usd_geom_mesh.html>
- OpenUSD `UsdVol` volume schema:
  <https://openusd.org/docs/api/usd_vol_page_front.html>
- OpenUSD volume user guide:
  <https://openusd.org/dev/user_guides/schemas/usdVol/Volume.html>
- Khronos ANARI overview:
  <https://www.khronos.org/anari/>
- Khronos ANARI SDK:
  <https://github.com/KhronosGroup/ANARI-SDK>
- ANARI 1.1 specification:
  <https://registry.khronos.org/ANARI/specs/1.1/ANARI-1.1.html>
- ANARI Python binding package:
  <https://pypi.org/project/anari/>
- CartoPy introduction:
  <https://scitools.org.uk/cartopy/docs/v0.17/>
- CartoPy Matplotlib interface:
  <https://scitools.org.uk/cartopy/docs/v0.22/reference/matplotlib.html>
- Matplotlib `pcolormesh` documentation:
  <https://matplotlib.org/stable/gallery/images_contours_and_fields/pcolormesh_levels.html>
- xarray `DataArray.plot` documentation:
  <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.plot.html>
- HoloViz overview:
  <https://holoviz.org/>
- hvPlot overview:
  <https://hvplot.holoviz.org/>
- Plotly maps documentation:
  <https://plotly.com/python/maps/>
- deck.gl introduction and layer catalog:
  <https://deck.gl/docs>
  <https://deck.gl/docs/api-reference/layers>
- kepler.gl documentation:
  <https://docs.kepler.gl/>
- CesiumJS fundamentals:
  <https://cesium.com/learn/cesiumjs-fundamentals/>
- Google Earth Engine overview and image visualization:
  <https://developers.google.com/earth-engine>
  <https://developers.google.com/earth-engine/guides/image_visualization>
- Looker Studio overview and geo chart reference:
  <https://lookerstudio.google.com/docs>
  <https://cloud.google.com/looker/docs/studio/geo-chart-reference>
- Azure Maps Web SDK and Power BI layer docs:
  <https://learn.microsoft.com/en-us/azure/azure-maps/webgl-custom-layer>
  <https://learn.microsoft.com/azure/azure-maps/power-bi-visual-understanding-layers>
- ArcGIS Maps SDK layer docs:
  <https://developers.arcgis.com/javascript/latest/layers/>
- Tableau map documentation:
  <https://www.tableau.com/solutions/maps>
  <https://help.tableau.com/current/pro/desktop/en-gb/maps_build.htm>
- GDAL GeoTIFF documentation:
  <https://gdal.org/en/stable/drivers/raster/gtiff.html>
- GDAL DEM tooling:
  <https://gdal.org/en/stable/programs/gdaldem.html>
- Cloud Optimized GeoTIFF overview:
  <https://cogeo.org/in-depth.html>
- OGC 3D Tiles standard:
  <https://www.ogc.org/standards/3DTiles/>
