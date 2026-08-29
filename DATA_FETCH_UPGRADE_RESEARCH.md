# Data Fetch Upgrade Research

## Status

Research proposal for upgrading `fetch_data` to use xarray `DataArray` objects as
the canonical representation while preserving the existing Torch tensor and
coordinate-dictionary API during migration.

This document complements:

- `CUPY_DESIGN.md`, which defines the broader NumPy, CuPy, and Torch transition.
- `COORDINATE_SIGNATURE_DESIGN.md`, which defines payload-free coordinate
  signatures, dimension order, grid metadata, alignment, and handshakes.

## Goals

1. Keep the current `fetch_data` call pattern and legacy return type working.
2. Make `xr.DataArray` the canonical internal and future public representation.
3. Preserve support for both `DataSource` and `ForecastSource`, including lead time.
4. Use the new coordinate signature to describe target dimensions, labels, and grids.
5. Perform interpolation and temporal reductions on the GPU whenever the requested
   backend is CUDA.
6. Avoid hidden host transfers, unnecessary copies, and repeated interpolation setup.
7. Preserve names, attributes, coordinate attributes, grid metadata, and dimension
   order through fetching and regridding.
8. Provide a gradual migration with numerical parity and clear warnings.

## Non-goals

- Replacing every data source implementation in the first change.
- Making all remote formats decode directly into GPU memory immediately.
- Adding a new public grid or regridding class.
- Folding spatial regridding into coordinate alignment.
- Changing deterministic, diagnostic, or prognostic workflow signatures in this
  stage.

## Current behavior

`fetch_data` currently performs four responsibilities:

1. Detect whether a source accepts lead time.
2. Assemble analysis data into a `(time, lead_time, ...)` result.
3. Optionally interpolate spatial coordinates.
4. Convert the result to Torch or CuPy depending on `legacy` and `device`.

The implementation has several limitations:

- Source type is inferred by inspecting the `__call__` signature.
- A `DataSource` is called separately for every lead time.
- The legacy rectilinear path calls `xarray.interp`, which uses NumPy/SciPy.
- CUDA interpolation therefore runs on CPU before transferring the result to CUDA.
- The non-legacy CUDA path uses `.values`, which can materialize data on the host.
- The non-legacy path does not support `interp_to`.
- Curvilinear interpolation creates a new interpolation mapping for every fetch.
- `map_coords_xr` mixes exact alignment with nearest-neighbor interpolation and
  begins with a deep copy of the input array.
- Reconstructed arrays may lose names, attributes, encoding, and coordinate metadata.

The existing interpolation utilities contain a useful pattern worth retaining:
mapping indices or weights can be planned once on CPU and then applied repeatedly as
GPU gathers. The upgrade should generalize this pattern to NumPy and CuPy payloads
without making Torch the canonical fetch representation.

## Coordinate signature requirements

The coordinate signature is an `xr.DataArray` whose payload uses the Earth2Studio
shape-only backend. `fetch_data` must inspect the signature without allocating its
declared field data.

The following rules are authoritative:

1. `DataArray.dims` defines payload axis order.
2. Fixed dimension coordinates define required labels and label order.
3. Dynamic dimensions such as `time`, `lead_time`, and `batch` may be resolved from
   the request without allocating field data.
4. Auxiliary coordinates declare their own dimension tuples explicitly.
5. Grid attributes identify CRS, spatial dimensions, geometry, or indexed topology.
6. Known-grid coordinates are generated only when comparison or regridding needs
   them.
7. Alignment may select, reorder, add batch, batch dimensions, or transpose.
8. Alignment must never interpolate or change projection.
9. A strict handshake validates dimensions, labels, auxiliary coordinates, grid
   metadata, dtype, and required Earth2Studio attributes.
10. Grid mismatch requires an explicit regridding step.

These rules imply three separate operations:

```text
normalize request -> align labels and dimensions -> regrid spatial geometry
                                      |
                                      +-> strict handshake
```

`map_coords_xr` should therefore not be expanded into the new regridding layer. Its
exact-selection behavior can move into `align_coords`; its interpolation behavior
should be replaced by explicit backend-aware regridding.

## Proposed public API

Retain the current public arguments while extending `interp_to` to accept a coordinate
signature:

```python
@overload
def fetch_data(
    source: DataSource | ForecastSource,
    time: TimeArray,
    variable: VariableArray,
    lead_time: LeadTimeArray = ZERO_LEAD_TIME,
    device: torch.device = "cpu",
    interp_to: CoordSystem | xr.DataArray | None = None,
    interp_method: str = "nearest",
    legacy: Literal[True] = True,
) -> tuple[torch.Tensor, CoordSystem]: ...


@overload
def fetch_data(
    source: DataSource | ForecastSource,
    time: TimeArray,
    variable: VariableArray,
    lead_time: LeadTimeArray = ZERO_LEAD_TIME,
    device: torch.device = "cpu",
    interp_to: CoordSystem | xr.DataArray | None = None,
    interp_method: str = "nearest",
    legacy: Literal[False] = False,
) -> xr.DataArray: ...
```

During the environment-controlled transition described in `CUPY_DESIGN.md`,
`legacy=None` can select the configured mode. Existing explicit values remain
authoritative.

A legacy `CoordSystem` target is converted internally to a coordinate signature. New
code should pass a model signature directly:

```python
target = model.input_coords().reindex(
    time=time,
    lead_time=lead_time,
)

data = fetch_data(
    source,
    time,
    model.input_coords().coords["variable"],
    lead_time,
    device="cuda:0",
    interp_to=target,
    legacy=False,
)
```

No payload is allocated for `target`. Its `dims`, coordinates, and grid attributes
describe the requested result.

## Canonical fetch pipeline

```python
def fetch_data(...):
    request = _normalize_fetch_request(time, lead_time, variable)
    target = _normalize_target_signature(interp_to, request)

    array = _fetch_source_array(source, request)
    array = _normalize_source_array(array, request)
    array = align_coords(array, _request_signature(request))

    if target is not None and not same_grid(array, target):
        array = _place_payload(array, device)
        array = regrid_data_array(array, target, method=interp_method)
    else:
        array = _place_payload(array, device)
        if target is not None:
            array = align_coords(array, target)

    array = _apply_requested_statistics(array, request)
    if target is not None:
        array = align_coords(array, target)
        handshake_coords(array, target)

    mode = resolve_array_api() if legacy is None else legacy
    return array.e2s.to_torch() if mode else array
```

The important ordering change is that CUDA payload placement occurs before any
GPU-supported regridding or statistical reduction. CPU fallback must not happen
silently after this point.

## Fetching analysis and forecast sources

### ForecastSource

A forecast source receives initialization time, lead time, and variable directly.
The returned array is normalized to the canonical dimensions requested by the fetch:

```text
(time, lead_time, variable, ...source spatial dimensions...)
```

Normalization should reorder dimensions without copying when possible and attach an
auxiliary valid-time coordinate:

```python
valid_time = time[:, None] + lead_time[None, :]
array = array.assign_coords(
    valid_time=(("time", "lead_time"), valid_time)
)
```

### DataSource

An analysis source should be called once with all unique valid times rather than once
per lead time:

```python
valid_time = time[:, None] + lead_time[None, :]
unique_time, inverse = np.unique(valid_time.reshape(-1), return_inverse=True)
fetched = source(unique_time, variable)
array = _reshape_valid_times(fetched, inverse, time, lead_time)
```

This removes repeated remote requests and permits a source to combine reads. The
result retains both request coordinates:

- `time`: requested initialization or anchor time.
- `lead_time`: offset from each requested time.
- `valid_time(time, lead_time)`: actual source timestamp.

Duplicate valid times are fetched once and referenced more than once in the assembled
result. The implementation must preserve times outside the `datetime64[ns]` range.

### Source capability detection

Signature inspection remains as a compatibility fallback, but new sources should be
able to declare a simple capability without introducing a public source class:

```python
class ExampleForecastSource:
    supports_lead_time = True
```

Resolution order:

1. Explicit `supports_lead_time` attribute.
2. Known protocol or registered source metadata.
3. Existing call-signature inspection for compatibility.

## Handling the coordinate signature

### Request signature

`_request_signature` creates a payload-free signature containing the exact requested
time, lead-time, and variable labels. It does not require a spatial grid:

```python
request_signature = e2s.coord_array(
    dims=("time", "lead_time", "variable"),
    coords={
        "time": time,
        "lead_time": lead_time,
        "variable": variable,
    },
)
```

It is used to normalize source output labels before spatial handling.

### Target signature

The target signature describes the complete output dimensions and grid. Its spatial
coordinates can be represented at three levels:

1. Generated grid metadata, such as `latlon-0.25deg` or `hrrr-conus-3km`.
2. Indexed topology, such as `healpix-l6-nested`.
3. Inline or externally referenced coordinates for an irregular grid.

`fetch_data` should first compare normalized grid metadata. If source and target grid
hashes match, no physical latitude/longitude arrays are generated and no regridding is
performed. Only label alignment and transpose are needed.

If grids differ, the regridding planner materializes only the geometry required by
the selected algorithm. Registry-backed projected or HEALPix coordinates should be
generated as temporary planning arrays rather than permanently attached as large 2D
coordinates unless the output contract explicitly requires them.

### Result construction

The regridded result must use the target signature as the spatial contract:

- Spatial dimensions follow `target.dims` order.
- Fixed target dimension coordinates and auxiliary coordinates are retained.
- Target grid metadata becomes authoritative.
- Non-spatial source coordinates such as `time`, `lead_time`, and `valid_time` remain.
- Source variable attributes remain unless the operation changes their meaning.
- DataArray name and encoding are preserved where meaningful.

Result construction should use a shallow structural replacement helper rather than
`DataArray.copy()` with its default deep copy.

## Backend placement

Payload and coordinate metadata have different requirements:

- Field payloads should be NumPy on CPU and CuPy on CUDA.
- Small label coordinates may remain NumPy-backed host metadata.
- Large temporary spatial query arrays should be generated on the execution backend.
- Returned physical auxiliary coordinates may remain host metadata unless downstream
  GPU computation explicitly consumes them.

Backend placement must inspect `array.data`, never `array.values`:

```python
def _place_payload(array: xr.DataArray, device) -> xr.DataArray:
    if device.type == "cuda":
        return array.e2s.as_cupy(device.index)
    return array.e2s.as_numpy()
```

Expected behavior by input backend:

| Input payload | CPU request | CUDA request |
| --- | --- | --- |
| NumPy | Preserve | One `cp.asarray` transfer |
| CuPy | Explicit download | Preserve on matching device |
| Torch CPU | Zero-copy conversion where valid | Transfer once |
| Torch CUDA | Explicit download | DLPack zero-copy |
| Dask+NumPy | Materialize NumPy | Map or materialize into CuPy |
| Dask+CuPy | Explicit CPU compute | Preserve GPU chunks |

Initial implementation may continue returning eager arrays. Dask handling should be
isolated behind backend placement so lazy fetching can be added without redesigning
the regridding interface.

## Explicit regridding

The internal regridding entry point operates on DataArrays and target signatures:

```python
def regrid_data_array(
    source: xr.DataArray,
    target: xr.DataArray,
    *,
    method: Literal["nearest", "linear"],
) -> xr.DataArray:
    ...
```

No new public regrid-specification object is required. A private cached plan may use a
dataclass or plain mapping because it is implementation state, not part of the model
or data-source protocol.

### Dispatch matrix

| Source grid | Target grid | Method | Planning | Application |
| --- | --- | --- | --- | --- |
| Rectilinear | Rectilinear | nearest/linear | Axis search | NumPy or CuPy gather/interpolation |
| Rectilinear | 2D lat/lon | nearest/linear | Axis search and weights | NumPy or CuPy gather and weighted sum |
| Curvilinear | Any lat/lon | nearest | KDTree or GPU KNN | Cached flat-index gather |
| Curvilinear | Any lat/lon | linear | CPU triangulation and barycentric weights | Cached gather and weighted sum |
| Regular lat/lon | HEALPix | supported Earth2Grid method | Earth2Grid planner | Zero-copy CuPy/Torch adapter |
| HEALPix | Regular lat/lon | supported Earth2Grid method | Earth2Grid planner | Zero-copy CuPy/Torch adapter |
| Identical grid hash | Same grid | any | None | Shallow alignment only |

### Rectilinear grids

CuPy provides `cupyx.scipy.interpolate.interpn` and
`RegularGridInterpolator` for nearest and linear interpolation on rectilinear grids.
These provide an initial GPU-native implementation with behavior close to SciPy.

For grids used repeatedly, a specialized planner should precompute source indices and
linear weights. Application then becomes four gathers and a weighted sum for 2D
bilinear interpolation. This avoids rebuilding the interpolator and target query array
for every variable, time, or fetch.

### Curvilinear grids

The current `LatLonInterpolation` and `NearestNeighborInterpolator` already separate
CPU planning from GPU application. The new implementation should store backend-neutral
index and weight arrays in the cache, copy them to CuPy once per CUDA device, and apply
them directly to `DataArray.data`.

Nearest-neighbor planning can initially remain CPU-based because grids are usually
static and the plan is reused. GPU KNN through cuVS should be evaluated for large,
dynamic, or irregular grids, but it is not required for the first implementation.

Linear curvilinear interpolation should use CPU triangulation to produce source
indices, barycentric weights, and a validity mask. The field application remains GPU
native. This is preferable to silently downloading every fetched field for SciPy
interpolation.

### HEALPix and known weather grids

Earth2Grid already implements GPU regridding between regular latitude/longitude and
HEALPix using Torch. The initial DataArray path can call it through DLPack:

```text
CuPy payload -> zero-copy Torch view -> Earth2Grid -> zero-copy CuPy view
```

The adapter must preserve dimension order and should isolate Torch from the public
fetch return type. Known-grid registry metadata supplies the Earth2Grid grid
construction parameters.

### Projection changes

PyProj remains the CRS parser and metadata authority, but it should not run a Python
projection transform once per field value. Projection transforms belong in regridding
planning:

1. Resolve source and target CRS once.
2. Transform target sample positions to source coordinates once.
3. Compute reusable source indices and weights.
4. Apply the plan to every field on NumPy or CuPy.

GPU projection kernels, similar to xarray-spatial, can be evaluated later for grids
whose target geometry changes frequently.

## Regridding plan cache

The cache key should include:

```text
source grid hash
target grid hash
interpolation method
longitude convention and wrap policy
bounds and fill policy
source and target spatial dimension order
plan version
```

The reusable plan contains only the minimum required arrays:

- Source indices.
- Weights for linear interpolation.
- Validity or out-of-bounds mask.
- Source and target spatial shapes.
- Expected dimension names and order.

Host planning arrays can be shared between devices. Device copies should be cached by
CUDA device. Cache entries require a size limit because high-resolution irregular-grid
plans can consume significant memory.

## Bounds, longitude, and missing-data behavior

The upgraded path must define behavior currently inherited inconsistently from
xarray, SciPy, and Torch utilities:

- Accept ascending or descending rectilinear axes.
- Normalize longitude only during planning; preserve the target convention in output.
- Handle cyclic global longitude without duplicating a full field.
- Make extrapolation opt-in. Default out-of-bounds values should be `NaN`.
- Preserve source `NaN` values according to the selected interpolation method.
- Reject non-monotonic rectilinear coordinates with an actionable error.
- Require explicit maximum distance for regional nearest-neighbor extrapolation when
  appropriate.
- Validate projected coordinate units and CRS before planning.

These policies must be identical on CPU and GPU.

## Direct GPU data ingestion

The core fetch utility cannot guarantee GPU-native decoding for every HTTP, GRIB,
NetCDF, or cloud source. It can avoid preventing it:

1. Preserve CuPy arrays returned by a source.
2. Never coerce through `.values` or `np.asarray` in the CUDA path.
3. Allow source-specific readers to declare GPU output capability.
4. Transfer NumPy output exactly once when no GPU reader exists.

Zarr 3 supports CuPy-backed GPU buffers through `zarr.config.enable_gpu()`. This
should be prototyped for Zarr-backed Earth2Studio sources in a scoped context rather
than enabled globally. The prototype must confirm that xarray wrapping preserves the
CuPy payload and that codecs do not introduce an unexpected host round trip.

KvikIO and GPUDirect Storage can later accelerate local or suitably configured Zarr
stores. This belongs in source or storage adapters, not in the generic `fetch_data`
control flow.

## Temporal statistics

Temporal statistics should be resolved in valid-time space after fetch normalization.
The variable coordinate remains a simple array of Earth2Studio variable strings;
statistical meaning remains in Earth2Studio attributes defined by the coordinate
signature design.

For a requested statistic:

1. Expand its anchor and offsets into required valid timestamps.
2. Fetch the union of required timestamps once.
3. Assemble the request's `time` and `lead_time` dimensions.
4. Place the payload on the selected backend.
5. Reduce on NumPy or CuPy.
6. Attach the normalized statistic attributes to the result.

This keeps reductions GPU-native and prevents statistics from being encoded as new
variable names.

## CPU fallback policy

GPU-centric behavior requires an explicit policy:

- A CUDA request must not silently download data for interpolation.
- A supported CUDA method runs entirely on CuPy, except one-time grid planning.
- An unsupported CUDA grid or method raises `NotImplementedError` with the supported
  alternatives.
- If a temporary compatibility fallback is required, it must be explicitly enabled
  and emit a warning that identifies the host transfer.

Silent fallback would make performance unpredictable and make tests unable to prove
that the new path is GPU native.

## Migration plan

### Phase 1: Structural separation

- Extract `_fetch_source_array` and `_normalize_source_array`.
- Fetch unique valid times once for analysis sources.
- Add and preserve `valid_time(time, lead_time)`.
- Convert legacy interpolation targets into coordinate signatures.
- Separate `align_coords`, `handshake_coords`, and `regrid_data_array`.
- Remove deep copies and `.values` from the DataArray path.
- Preserve current numerical behavior and legacy defaults.

### Phase 2: DataArray parity

- Support `interp_to` when `legacy=False`.
- Return NumPy-backed CPU and CuPy-backed CUDA DataArrays.
- Preserve metadata and target dimension order.
- Convert to Torch only at the legacy return boundary.
- Emit the environment-controlled migration warning once.

### Phase 3: GPU regridding

- Implement rectilinear nearest and linear CuPy paths.
- Generalize cached curvilinear plans to NumPy and CuPy application.
- Add Earth2Grid HEALPix adapters.
- Add plan caching and memory limits.
- Reject hidden CPU fallback.

### Phase 4: GPU-aware sources

- Prototype Zarr GPU-buffer reads.
- Add optional source capability metadata.
- Preserve CuPy and Dask+CuPy source output.
- Evaluate KvikIO for local Zarr stores.
- Benchmark remote decode, transfer, planning, and application separately.

### Phase 5: Cleanup

- Deprecate interpolation behavior in `map_coords` and `map_coords_xr`.
- Move all exact mapping to `align_coords`.
- Remove legacy-only fetch preparation after the transition window.
- Update data-fetch skills, examples, and API documentation.

## Validation strategy

### Functional matrix

Cover the smallest set of parametrized tests spanning:

- `DataSource` and `ForecastSource`.
- One and multiple initialization times.
- One and multiple lead times, including duplicate valid times.
- NumPy CPU and CuPy CUDA payloads.
- Legacy and DataArray returns.
- Same-grid alignment without interpolation.
- Rectilinear to rectilinear and rectilinear to 2D target grids.
- Curvilinear nearest and linear interpolation.
- Regular latitude/longitude and HEALPix.
- Ascending and descending axes.
- Cyclic longitude and regional bounds.
- Out-of-bounds and `NaN` behavior.
- Target dimension order differing from source order.
- Coordinate and attribute preservation.

### GPU-native assertions

CUDA tests should fail if the execution path calls:

- `cupy.asnumpy`.
- `cupy.ndarray.get`.
- `DataArray.values` on a CuPy-backed field.
- NumPy coercion of the payload after placement.

Tests should also verify:

- The result remains a CuPy array.
- A reused grid pair hits the plan cache.
- Plan arrays live on the requested CUDA device.
- Legacy Torch conversion uses DLPack without a payload copy.
- Same-grid alignment shares memory where transpose or selection permits a view.

### Numerical parity

Use SciPy/xarray results as CPU references during migration. Define tolerances by
method and dtype rather than requiring bitwise equality across CPU and GPU. Include
analytic fields whose exact nearest and bilinear results are known.

### Performance measurements

Measure independently:

1. Remote read and decode.
2. Host-to-device transfer count and bytes.
3. Regridding plan creation.
4. Cached plan application.
5. Peak CPU and GPU memory.
6. End-to-end fetch latency for one and many variables or times.

The primary acceptance criterion is not only lower total latency. A CUDA fetch must
show no full field download after initial placement and no repeated plan construction
for the same grid pair.

## Open questions

1. Should explicit CPU fallback be exposed as a temporary keyword or only an
   environment variable during migration?
2. Which current data sources return lazy Dask arrays that must be supported in the
   first DataArray parity change?
3. Which curvilinear grids are repeated frequently enough to justify persistent disk
   caching of plans in addition to process memory?
4. Should source capability metadata be a simple attribute, a plain mapping, or part
   of the existing source protocol documentation?
5. Which Earth2Grid interpolation methods meet the numerical parity requirements for
   existing HEALPix models?
6. Should large generated target coordinates remain transient, or should selected IO
   backends materialize and persist them with outputs?

## Recommended first implementation slice

The first pull request should remain deliberately narrow:

1. Add `_fetch_source_array`, `_normalize_source_array`, and vectorized valid-time
   assembly.
2. Add `valid_time(time, lead_time)`.
3. Permit a coordinate-signature DataArray as `interp_to`.
4. Add exact `align_coords` and strict target handshake.
5. Support same-grid CPU/CUDA DataArray output without `.values` or deep copies.
6. Retain the current interpolation implementation behind the legacy path.

The second pull request can then introduce explicit `regrid_data_array` with
rectilinear NumPy and CuPy parity. This sequencing prevents source normalization,
coordinate semantics, GPU kernels, and legacy conversion from changing in one large
review.

## References

- [Xarray duck arrays](https://docs.xarray.dev/en/stable/user-guide/duckarrays.html)
- [Xarray interpolation](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.interp.html)
- [CuPy `interpn`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.interpolate.interpn.html)
- [CuPy SciPy comparison](https://docs.cupy.dev/en/stable/reference/comparison.html)
- [Earth2Grid](https://github.com/NVlabs/earth2grid)
- [xarray-spatial](https://github.com/xarray-contrib/xarray-spatial)
- [Zarr GPU buffers](https://zarr.readthedocs.io/en/stable/user-guide/gpu/)
- [KvikIO Zarr integration](https://docs.rapids.ai/api/kvikio/stable/zarr/)
- [cuVS nearest-neighbor APIs](https://docs.rapids.ai/api/cuvs/stable/neighbors/neighbors/)
- [xarray-jax](https://github.com/google-deepmind/xarray_jax)
