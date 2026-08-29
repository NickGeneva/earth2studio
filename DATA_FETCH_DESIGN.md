# Data Fetch Design

## Status

Draft implementation plan for unifying dense and sparse data fetching under
`earth2studio.data.fetch_data` while migrating dense results to the Earth2Studio
DataArray schema.

This document converts `DATA_FETCH_UPGRADE_RESEARCH.md` into a feature-parity design.
It depends on:

- `CUPY_DESIGN.md` for backend conversion and migration configuration.
- `COORDINATE_SIGNATURE_DESIGN.md` for allocation-free DataArray and DataFrame
  signatures, grids, statistics, alignment, and handshakes.

## Decision Summary

1. `fetch_data` is the only high-level public fetch entry point.
2. It accepts array and frame sources and dispatches to `_fetch_data_array` or
   `_fetch_data_frame`.
3. Modern dense mode returns exactly one NumPy- or CuPy-backed `xr.DataArray`.
4. Sparse mode returns exactly one pandas or cuDF DataFrame.
5. Legacy dense mode converts the canonical DataArray once at the return boundary and
   returns `(torch.Tensor, CoordSystem)`.
6. The common inputs remain `source`, `time`, `variable`, `lead_time`, and `device`.
7. One optional `metadata` argument accepts an allocation-free DataArray signature or
   an empty typed DataFrame signature.
8. DataArray metadata carries target dimensions, grids, statistics, and an optional
   compact regridding policy. DataFrame metadata carries columns, dtypes, roles, CRS,
   and optional field requirements.
9. `interp_to` and `interp_method` remain compatibility inputs during migration, but
   are normalized into DataArray metadata internally and then deprecated.
10. The first implementation reaches current feature parity before adding new GPU
    regridders or source-ingestion optimizations.

## Goals

1. Preserve all currently supported `fetch_data` and `fetch_dataframe` calls.
2. Give users one discoverable fetch function for dense and sparse sources.
3. Make DataArray output feature-equivalent to legacy dense output.
4. Preserve pandas and cuDF behavior for frame sources.
5. Support analysis and forecast sources, including multiple initialization and lead
   times.
6. Return NumPy-backed DataArrays on CPU and CuPy-backed DataArrays on CUDA.
7. Preserve dimensions, coordinate values, column order, names, attributes, and grid
   or CRS metadata.
8. Replace bespoke interpolation arguments with metadata-driven grid requirements.
9. Provide explicit internal hooks for statistics, alignment, and regridding.
10. Keep the migration on `main`, controlled by one environment variable for dense
    return behavior.

## Non-goals

- Changing prognostic, diagnostic, or workflow signatures in this stage.
- Returning a Dataset, list, tuple, or collection in modern mode.
- Combining dense arrays and sparse frames in one result.
- Requiring custom sources to inherit from a new base class.
- Introducing a public request, grid, statistic, or regridding class.
- Implementing every proposed regridding backend in the first change.
- Guaranteeing direct GPU decoding for every remote storage format.
- Removing the legacy dense return type during this project.
- Applying spatial regridding to DataFrames in the first implementation.
- Optimizing analysis sources into one vectorized remote request before parity is
  established.

## Current Public Behavior

Earth2Studio currently has two high-level functions:

- `fetch_data` handles `DataSource` and `ForecastSource` and returns either a legacy
  tensor/coordinate pair or a DataArray.
- `fetch_dataframe` handles `DataFrameSource` and `ForecastFrameSource` and returns a
  pandas or cuDF DataFrame.

Together they support:

- One or many initialization times.
- One or many positive, zero, or negative lead times.
- Variable selection and ordering.
- Optional frame-field selection.
- A default zero-hour lead.
- CPU and CUDA devices.
- Nearest and linear interpolation for existing rectilinear array paths.
- Linear interpolation for the existing curvilinear array path.
- One- and two-dimensional target latitude/longitude coordinates.
- Times outside the NumPy nanosecond range.
- Request metadata on frame results.

The unified API must retain this complete behavior before defaults or deprecations
change.

## Feature-Parity Requirement

| Behavior | Array source | Frame source |
| --- | --- | --- |
| Analysis input | `time`, `variable` | `time`, `variable`, optional fields |
| Forecast input | Adds `lead_time` | Adds `lead_time` |
| CPU modern result | NumPy DataArray | pandas DataFrame |
| CUDA modern result | CuPy DataArray | cuDF DataFrame |
| Legacy result | Torch tensor and coordinates | Not applicable |
| Target metadata | DataArray signature | Empty typed DataFrame |
| Spatial transform | Current interpolation parity | No transform |
| Statistics hook | DataArray metadata | Reserved for later design |
| Ordering | `DataArray.dims` | DataFrame columns and rows |

For the same dense source, request, target, method, and device, legacy and modern modes
must represent the same values, shape, dimensions, and coordinates. Legacy conversion
may omit metadata that `CoordSystem` cannot represent, but it must not alter the
payload contract.

## Unified Public API

```python
FetchSource = (
    DataSource
    | ForecastSource
    | DataFrameSource
    | ForecastFrameSource
)
FetchMetadata = xr.DataArray | pd.DataFrame
FetchResult = (
    tuple[torch.Tensor, CoordSystem]
    | xr.DataArray
    | pd.DataFrame
    | cudf.DataFrame
)


def fetch_data(
    source: FetchSource,
    time: TimeArray,
    variable: VariableArray,
    lead_time: LeadTimeArray = ZERO_LEAD_TIME,
    device: torch.device = "cpu",
    interp_to: CoordSystem | None = None,
    interp_method: str = "nearest",
    legacy: bool | None = None,
    *,
    metadata: FetchMetadata | None = None,
    fields: FieldArray | None = None,
) -> FetchResult: ...
```

Typing overloads narrow the return from the source family and explicit legacy mode.
The public dispatcher never returns both an array and a frame.

### Compatibility rules

- Existing dense calls remain valid.
- Existing frame calls move from `fetch_dataframe(...)` to `fetch_data(...)` without
  changing common arguments.
- `fetch_dataframe` remains a compatibility wrapper during the transition.
- `fields` is accepted only for frame sources.
- Frame sources accept `legacy=None` or `legacy=False`; `legacy=True` raises because
  no legacy frame representation exists.
- `metadata` must match the resolved source family.
- Passing `metadata` together with `interp_to` raises to avoid two conflicting target
  descriptions. `interp_method` has no effect without `interp_to`.

## Metadata API Options

### Option A: Free-form mapping

```python
fetch_data(..., metadata={"grid": ..., "statistics": ...})
```

This is short but recreates xarray and pandas schema concepts in an Earth2Studio-only
dictionary. It requires another validator, loses standard coordinate behavior, and is
easy for agents and users to construct incorrectly. This option is not recommended.

### Option B: `target=` signature

```python
fetch_data(..., target=model.input_coords())
```

`target` clearly communicates a desired output contract for arrays. It is less natural
for sparse frames, where the empty DataFrame describes requested fields and metadata
rather than a spatial target.

### Option C: `metadata=` signature

```python
fetch_data(..., metadata=model.input_coords())
```

This accepts the standard objects already used by the model protocols:

- A shape-only `xr.DataArray` for dense dimensions, coordinates, grid, statistics,
  and dtype.
- An empty typed `pd.DataFrame` for sparse columns, dtypes, roles, CRS, and attrs.

This is the recommended option. It introduces no new public class and supports both
source families consistently. The parameter name is broad because it describes
requirements rather than supplying payload data.

## Dense Metadata Example

```python
metadata = e2s.coord_array(
    dims=("time", "lead_time", "variable", "lat", "lon"),
    coords={
        "lead_time": lead_time,
        "variable": ["u10m", "t2m"],
    },
    dynamic=("time",),
    grid="latlon-0.25deg",
    statistics={"u10m": "mean:-24h:0h"},
)

array = fetch_data(
    source,
    time,
    metadata.coords["variable"].data,
    lead_time,
    device="cuda:0",
    metadata=metadata,
    legacy=False,
)
```

The signature allocates no field payload. It can be the same object returned by a
model's `input_coords()` method after resolving dynamic request coordinates.

## Sparse Metadata Example

```python
metadata = station_model.input_coords()  # Empty typed DataFrame

frame = fetch_data(
    station_source,
    time,
    variable=["u10m", "v10m"],
    lead_time=lead_time,
    device="cpu",
    metadata=metadata,
)
```

The DataFrame signature supplies requested columns and validates dtypes, roles, CRS,
and attributes. It contains zero observation rows.

## Runtime Mode Selection

Dense return mode follows the shared migration configuration:

```bash
EARTH2STUDIO_ARRAY_API=legacy
EARTH2STUDIO_ARRAY_API=xarray
```

Resolution precedence is:

1. Explicit `legacy=True` or `legacy=False`.
2. `EARTH2STUDIO_ARRAY_API` when `legacy=None`.
3. The release default when the environment variable is unset.

Before the planned switch, the release default remains `legacy`. An unset value emits
one `Earth2StudioArrayAPIFutureWarning` per process from the first affected dense call.
Explicit mode selection emits no pre-switch warning. Frame calls do not inspect this
environment variable because they have no tensor/coordinate return mode.

Mode resolution occurs at the public boundary, never at import time. Private helpers
do not inspect the environment.

## Public Dispatcher

```python
def fetch_data(...):
    source_kind = _resolve_fetch_source_kind(source)

    if source_kind == "array":
        return _fetch_data_array(
            source,
            time,
            variable,
            lead_time,
            device,
            metadata=_require_array_metadata(metadata),
            legacy=legacy,
            interp_to=interp_to,
            interp_method=interp_method,
        )

    return _fetch_data_frame(
        source,
        time,
        variable,
        lead_time,
        device,
        metadata=_require_frame_metadata(metadata),
        fields=fields,
    )
```

The dispatcher owns source-family validation only. Array and frame implementation
details remain isolated below it.

## Source-Kind Resolution

New source implementations may declare:

```python
output_type: Literal["array", "frame"] = "array"
supports_lead_time: bool = True
```

Resolution order is:

1. Explicit `output_type` and `supports_lead_time` attributes.
2. Registered metadata for built-in sources.
3. Existing `SCHEMA`, `fields`, and call-signature inspection for compatibility.
4. Validation of the returned object against the resolved family.

No custom source is required to change. An explicit declaration avoids inspection and
can be added to built-ins incrementally. A source that declares one family and returns
another raises an actionable error.

## Array Fetch Pipeline

```python
def _fetch_data_array(..., metadata, legacy, interp_to, interp_method):
    mode = resolve_array_api() if legacy is None else (
        "legacy" if legacy else "xarray"
    )
    request = _normalize_array_request(time, lead_time, variable)
    target = _normalize_array_metadata(
        metadata,
        interp_to=interp_to,
        interp_method=interp_method,
        request=request,
    )

    array = _fetch_source_array(source, request)
    array = _normalize_source_array(array, request)
    array = _apply_requested_statistics(array, target)
    array = _regrid_for_fetch(array, target)
    array = _align_array_result(array, request, target)
    array = _place_array_result(array, device)
    array = _validate_array_result(array, request, target)

    return array.e2s.to_torch() if mode == "legacy" else array
```

Every dense private helper receives and returns one DataArray. The only tensor and
coordinate conversion is the final legacy boundary.

## Frame Fetch Pipeline

```python
def _fetch_data_frame(..., metadata, fields):
    request = _normalize_frame_request(time, lead_time, variable, fields)
    schema = _normalize_frame_metadata(metadata, request)

    frame = _fetch_source_frame(source, request)
    frame = _normalize_source_frame(frame, request)
    frame = _select_frame_fields(frame, request, schema)
    frame = _place_frame_result(frame, device)
    frame = _validate_frame_result(frame, request, schema)
    return frame
```

The initial frame path preserves current pandas/cuDF conversion and request attrs.
Metadata-driven field selection supplements the existing `fields` argument; it does
not change frame values or row semantics.

## Time and Lead-Time Semantics

- `time` is an initialization/reference time for forecast sources.
- `lead_time` is the offset from that initialization.
- `valid_time = time + lead_time` is the physical timestamp represented by a value.
- For analysis sources, source `time` already represents valid time.

Array results attach `valid_time` as a coordinate over `(time, lead_time)`. Frame
results retain current request attrs and may expose `valid_time` as a column when the
source contract already supplies one. Adding or standardizing frame time columns is a
separate compatibility decision.

The first parity implementation retains existing source invocation patterns.
Vectorizing analysis reads over unique valid times is a follow-up optimization.

## Statistics Hook

Statistics are read from DataArray metadata:

```python
statistics={"u10m": "mean:-24h:0h"}
```

The interval is interpreted in valid-time space. The concrete request selects the
physical reduction dimension:

- Varying `time` with no or one fixed lead reduces over `time`.
- One fixed initialization with varying leads reduces over `lead_time`.
- Varying `time` and `lead_time` is ambiguous and raises until one is selected.

The private entry point is:

```python
def _apply_requested_statistics(
    array: xr.DataArray,
    metadata: xr.DataArray | None,
) -> xr.DataArray:
    ...
```

The parity implementation supports metadata without statistic modifiers. Until the
statistics implementation lands, non-empty modifiers raise `NotImplementedError`
rather than being ignored. Statistics run before spatial regridding so nonlinear
methods such as maximum retain source-grid semantics.

Frame statistics are not inferred from DataFrame attrs in the first implementation.
A frame signature containing unsupported statistic metadata raises explicitly.

## Replacing Interpolation Arguments

The new API should describe the required output grid, not prescribe an interpolation
implementation. Regridding is triggered automatically when normalized source and
metadata grid identities differ.

### Grid discovery

One shared resolver handles explicit grid metadata and coordinate-only arrays:

1. Use normalized Earth2Studio grid metadata when present.
2. Resolve standard CF grid mapping and CRS metadata.
3. Infer a rectilinear grid from one-dimensional latitude/longitude coordinates.
4. Infer a curvilinear grid from two-dimensional latitude/longitude coordinates.
5. Raise when the geometry is insufficient or contradictory.

The regridding planner compares the resolved source and target grids. It calls
`array.e2s.materialize_grid_coords()` only when the selected engine requires physical
coordinates. Registry-backed grids therefore do not permanently allocate latitude and
longitude arrays merely to prove grid identity.

### Compact regridding policy

Regridding control belongs in the DataArray metadata rather than in `fetch_data`.
A single policy string uses `[engine:]method`:

```python
# Select the best installed engine that supports linear regridding.
metadata = model.input_coords().e2s.set_regrid("linear")

# Require SciPy cubic interpolation.
metadata = model.input_coords().e2s.set_regrid("scipy:cubic")
```

Initial normalized methods are `nearest`, `linear`, `cubic`, and `conservative`.
Initial engine identifiers may include `xarray`, `scipy`, `cupy`, `earth2grid`, and
`xesmf` as their adapters become available. Each adapter declares its supported source
grids, target grids, methods, devices, and optional dependencies.

The accessor expands the compact declaration into validated attrs:

```python
metadata.attrs["earth2studio_regrid"]
# {
#     "policy": "scipy:cubic",
#     "engine": "scipy",
#     "method": "cubic",
# }
```

Users normally provide only the policy string. They do not pass library-specific
keyword arguments through `fetch_data`. Advanced engine configuration belongs in a
registered engine/profile so fetch metadata remains portable and serializable.

### Policy resolution

When grids differ, resolution follows:

1. Use an explicit metadata policy when present.
2. Otherwise use a variable-specific method recommendation when all requested
   variables agree.
3. Otherwise use the current parity default, `nearest`, when supported.
4. Otherwise use the only method supported by the resolved grid pair.
5. Raise with available methods and engines when the choice remains ambiguous.

If the policy names an engine, that engine is mandatory. If it names only a method,
the registry selects an engine deterministically from the grid pair, method, execution
device, and installed dependencies. CUDA-native engines take priority for CUDA
payloads; CPU engines take priority for CPU payloads.

Common cases therefore need only a target grid:

```python
metadata = model.input_coords()
array = fetch_data(..., metadata=metadata)
```

An explicit override remains possible without adding another `fetch_data` argument:

```python
metadata = model.input_coords().e2s.set_regrid("scipy:cubic")
array = fetch_data(..., metadata=metadata)
```

`set_regrid()` stores the validated policy. It does not perform the operation or
introduce a public regridding object.

The output records the resolved engine, method, engine version, and source/target grid
hashes as provenance. Automatic engine selection is therefore reproducible and
inspectable rather than hidden.

### Legacy translation

During migration:

```python
fetch_data(
    ...,
    interp_to=legacy_coords,
    interp_method="linear",
)
```

is normalized internally to the equivalent DataArray metadata and regridding policy.
The initial release can support both paths without a warning. Deprecation begins only
after metadata parity is established and a migration guide is published.

Passing both styles raises because source-to-target behavior must have one authority.

## Grid and Regridding Hook

Grid handling remains three operations:

1. **Describe:** Resolve source and metadata grid identity.
2. **Align:** Select or reorder labels on the same grid.
3. **Regrid:** Change spatial geometry or projection.

```python
def _regrid_for_fetch(
    array: xr.DataArray,
    metadata: xr.DataArray | None,
) -> xr.DataArray:
    if metadata is None or array.e2s.same_grid(metadata):
        return array
    policy = _resolve_regrid_policy(array, metadata)
    return _regrid_data_array(array, metadata, policy=policy)
```

The first implementation places current interpolation behavior behind
`_regrid_data_array` and preserves its numerical behavior. This is the parity adapter,
not the final performance implementation.

The hook later permits cached NumPy/CuPy plans, projected grids, HEALPix, and new
methods without changing `fetch_data`. Unsupported grid/method combinations raise and
never silently select a different algorithm.

## Alignment and Result Construction

Dense alignment may select and reorder variables, restore request time/lead order,
transpose dimensions, and attach target spatial coordinates. It never regrids.

During the data stage, it does not add a model `batch` dimension. Workflow/model
alignment remains responsible for batching until the prognostic migration.

Dense results preserve:

- `DataArray.name` and attrs.
- Coordinate attrs and auxiliary coordinates.
- Variable, time, lead-time, and valid-time labels.
- Source grid metadata when unchanged.
- Metadata grid identity after regridding.
- Spatial dimension names and order.

Frame results preserve:

- Column order and dtypes.
- Frame attrs and supported column metadata.
- Source rows and indexes unless field selection requires projection.
- CRS and Earth2Studio role metadata.

## Backend Placement

Dense placement inspects `array.data`, never `array.values`:

```python
def _place_array_result(array, device):
    device = torch.device(device)
    if device.type == "cuda":
        return array.e2s.as_cupy(device.index)
    return array.e2s.as_numpy()
```

Frame placement retains pandas on CPU and uses `cudf.from_pandas` on CUDA until sources
can return compatible cuDF frames directly.

The first implementation guarantees one final placement, preserves already compatible
payloads, and never downloads a CuPy payload after placement. Direct GPU decoding and
GPU-native regridding remain follow-up optimizations.

## `fetch_dataframe` Migration

`fetch_dataframe` becomes a compatibility wrapper:

```python
def fetch_dataframe(...):
    return fetch_data(...)
```

The wrapper initially emits no warning. After unified-fetch documentation and examples
are available, it emits a targeted `FutureWarning` naming `fetch_data`. Removal requires
a separate schedule and is not part of this design.

## Validation

The dispatcher validates source family and metadata type before remote work begins.

Dense validation checks requested labels and order, dimensions, valid time, backend,
target grid, attrs, and no unrequested payload dimensions. A complete DataArray
signature uses the shared fetch-relevant handshake.

Frame validation checks requested variables and fields, column order, dtypes, required
roles, CRS attrs, backend, and source-family return type.

## Implementation Plan

### Pull Request 1: Unified dispatcher

- Add parity tests for the existing array and frame functions.
- Add `_resolve_fetch_source_kind` with compatibility fallbacks.
- Move existing implementations to `_fetch_data_array` and `_fetch_data_frame`.
- Make public `fetch_data` dispatch to both paths.
- Convert `fetch_dataframe` into a no-warning compatibility wrapper.
- Preserve all current arguments, results, and defaults.

### Pull Request 2: Canonical dense pipeline

- Add centralized `EARTH2STUDIO_ARRAY_API` resolution and warning tests.
- Change dense `legacy` behavior to `bool | None` while retaining the legacy default.
- Extract dense request, source normalization, placement, and validation helpers.
- Make every dense helper return one DataArray.
- Add valid-time metadata without changing payload shape.
- Convert to Torch only at the legacy boundary.

### Pull Request 3: Metadata signatures

- Add `metadata` support for DataArray and empty DataFrame signatures.
- Add metadata type and conflict validation in the dispatcher.
- Use DataFrame metadata for field, dtype, role, and CRS validation.
- Normalize legacy `interp_to` and `interp_method` into DataArray metadata.
- Add grid comparison, regridding dispatch, and statistic hooks.
- Reject unsupported metadata instead of ignoring it.

### Pull Request 4: Dense feature parity

- Support current interpolation behavior through `_regrid_data_array` in modern mode.
- Preserve names, coordinate attrs, grid metadata, and target dimension order.
- Remove `.values` and unnecessary deep copies from the modern path.
- Return NumPy on CPU and CuPy on CUDA.
- Run complete array and frame parity tests on built-in random sources.

### Pull Request 5: Documentation and deprecation readiness

- Update data-source guides and API examples around unified `fetch_data`.
- Add dense and sparse metadata examples.
- Document environment-variable and warning behavior.
- Publish an interpolation-argument migration guide.
- Schedule, but do not immediately enable, `fetch_dataframe` and interpolation-argument
  warnings.
- Define the release that will switch the unset dense default to `xarray`.

### Follow-up work

- Implement temporal statistic fetch expansion and NumPy/CuPy reductions.
- Vectorize analysis-source reads over unique valid times.
- Add cached GPU-native regridding plans.
- Add projected-grid and HEALPix adapters.
- Evaluate source-native CuPy/cuDF and Zarr GPU buffers.
- Design sparse temporal statistics and frame spatial operations if required.

## Test Plan

Use a compact parametrized matrix spanning:

- Array, forecast-array, frame, and forecast-frame sources.
- Explicit and inferred source-family capabilities.
- One and multiple initialization and lead times, including negative leads.
- Duplicate valid times and out-of-nanosecond-range dates.
- Variable and frame-field selection and ordering.
- CPU NumPy/pandas and CUDA CuPy/cuDF outputs.
- Explicit legacy, explicit modern, and environment-selected dense modes.
- DataArray and DataFrame metadata signatures.
- Automatic and engine-qualified regridding policies.
- Metadata/source mismatches and conflicting legacy inputs.
- One- and two-dimensional interpolation targets.
- Current rectilinear and curvilinear interpolation methods.
- Same-grid no-op handling.
- Metadata, name, attrs, CRS, coordinate, dtype, and order preservation.
- Legacy tensor/coordinate parity.
- Unsupported statistics and grids raising clearly.

Tests compare the canonical normalized result with both public modes instead of
maintaining independent expected implementations.

CUDA array tests assert that the result remains CuPy and does not call
`DataArray.values`, `cupy.asnumpy`, or `cupy.ndarray.get` after placement. Frame tests
assert pandas/cuDF parity for values, columns, dtypes, and attrs supported by both
backends.

## Completion Criteria

The data-fetch stage is complete when:

1. `fetch_data` dispatches all four source families correctly.
2. Existing `fetch_data` and `fetch_dataframe` calls remain operational.
3. Modern dense output reaches complete legacy feature parity.
4. Frame output preserves current pandas/cuDF behavior.
5. DataArray and DataFrame metadata signatures are supported and validated.
6. Legacy interpolation inputs translate to metadata-driven regridding.
7. Modern CPU output is NumPy-backed and modern CUDA output is CuPy-backed.
8. Existing custom sources require no changes.
9. `EARTH2STUDIO_ARRAY_API` controls implicit dense return behavior.
10. Explicit dense legacy values override configuration.
11. Unsupported grid and statistic requirements never pass silently.
12. Metadata and ordering survive both internal pipelines.
13. Both paths have accurate test coverage and pass in CI.

The dense default remains legacy until these criteria pass and the later model, IO,
and documentation stages are ready for the shared default switch.

## Open Questions

1. Is `metadata` the preferred public name, or is `target` clearer despite being less
   natural for DataFrames?
2. Should the regridding engine registry be public in the first release or remain an
   internal extension point until its adapter contract stabilizes?
3. Which variable metadata is sufficient to choose nearest versus linear regridding
   automatically?
4. Should frame metadata fully replace `fields` after migration, or should `fields`
   remain as a convenience shorthand?
5. Which metadata fields can survive legacy `CoordSystem` conversion?
6. Should source capability metadata remain two booleans/strings or become one private
   mapping when additional source capabilities are needed?
