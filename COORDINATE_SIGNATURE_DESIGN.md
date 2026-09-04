# Ordered Xarray Model Signatures

**Status:** Proposal
**Scope:** Prognostic, diagnostic, statistical, and workflow coordinate contracts
**Related design:** `CUPY_DESIGN.md`

## Decision summary

Earth2Studio should use an `xarray.DataArray` as the coordinate signature advertised
by a model. Authors construct it with `earth2studio.coord_array()`, providing
dimensions and coordinates but no data array. Spatial signatures then use the `.e2s`
grid resolver to attach a PyProj-supported CRS and geometry or topology metadata.
Known Earth2Studio keys provide a concise path for common grids; explicit PyProj input
remains available for arbitrary grids. Internally the signature uses a private
shape-only duck array with no field buffer. The returned public object is a normal
`DataArray`.

The signature separates the required information using native xarray fields:

- `DataArray.dims` is the authoritative, ordered tensor dimension contract.
- Dimension coordinates specify required labels and their order.
- `E2S_DYNAMIC_DIMS` preserves the current wildcard semantics; those dimensions have
  length zero in the signature.
- Auxiliary coordinates describe multidimensional latitude, longitude, bounds,
  masks, and other geometry without becoming tensor dimensions.
- Earth2Studio grid metadata identifies a PyProj-supported CRS without adding an
  implicit coordinate.
- Flat geometry or topology attributes distinguish rectilinear, projected, and
  indexed grids such as HEALPix.
- The signature dtype communicates the expected model input or output dtype.

The existing methods can retain their names during migration:

```python
input_signature = model.input_coords()
output_signature = model.output_coords(input_signature)
```

Both calls operate without a forward pass or a populated data array. Pipelines can
therefore derive the exact order, variables, grid, and one-step output structure
before fetching data or loading model weights.

## Hard requirements

The replacement must preserve the current `CoordSystem` capabilities and add
xarray-native grid descriptions.

1. A model must advertise its ordered input tensor dimensions without real data.
2. A model must derive its ordered one-step output dimensions and coordinates from
   an input signature without running inference.
3. A diagnostic must be able to add, remove, reorder, or resize dimensions.
4. A prognostic must be able to advance lead time while preserving compatible
   input dimensions.
5. Fixed coordinate labels, including variable order, must be inspectable and
   validated.
6. Dynamic dimensions such as `batch` and `time` must retain their current wildcard
   behavior.
7. Rectilinear, projected, curvilinear, and unstructured grids must be describable.
8. Every resolved spatial signature must include a CRS normalized through
   `CRS.from_user_input()` or `CRS.from_cf()`; a known Earth2Studio key is valid only
   when its registry entry supplies that CRS.
9. One-dimensional indexed grids must describe topology, including HEALPix `nside`
   and pixel ordering.
10. Model signatures must compose through prognostic models, diagnostics,
   statistics, and IO planning.
11. Alignment may reorder an existing grid, but it must not silently regrid.
12. No Earth2Studio-specific outer signature class or per-grid metadata object should
    be required.

## Current Earth2Studio contract

Earth2Studio currently passes a Torch tensor and an ordered `CoordSystem` together.
The ordered dictionary does four important jobs:

- Dictionary key order defines tensor axis order.
- Coordinate arrays define the labels at each axis position.
- `np.empty(0)` declares a dynamic dimension.
- `output_coords(input_coords)` simulates one forward pass without computation.

The last behavior is required by more than model validation. Workflows use it to
build rollout coordinates, IO backends use it to initialize stores, wrappers use it
to compose models, and utilities use it to calculate forecast steps.

The current representation becomes ambiguous when metadata is not itself a tensor
axis. For example, a projected grid has tensor dimensions such as `hrrr_y` and
`hrrr_x`, two-dimensional latitude and longitude coordinates over those dimensions,
and projection metadata. Adding all of those fields to the ordered dictionary makes
it unclear which keys are tensor axes. Current code has encountered the same problem
with per-sample time metadata.

## Research findings

### Xarray

Standalone `xr.Coordinates` is a dictionary-like collection of coordinate variables
and indexes. It does not provide a sufficiently strong single-tensor axis contract.
In contrast, [`DataArray.dims`](https://docs.xarray.dev/en/latest/api/dataarray.html)
is explicitly a tuple of dimension names associated with the array, and
`DataArray.sizes` is an ordered mapping in that same order.

This makes `DataArray` the relevant xarray object for Earth2Studio. Earth2Studio
passes one packed tensor with a `variable` dimension, so one ordered `DataArray`
describes the model boundary directly. An `xr.Dataset` is less suitable as the core
signature because each data variable may have a different dimension tuple and the
dataset does not define one global tensor order.

### GraphCast and GenCast

GraphCast's xarray predictor API accepts a `targets_template` dataset. The template
communicates the variables, levels, lead times, dimensions, and shapes that the
predictor must produce without using the target values. See the
[`Predictor` interface](https://github.com/google-deepmind/graphcast/blob/main/graphcast/predictor_base.py).

This validates the template approach. Earth2Studio should use a `DataArray` rather
than a `Dataset` because Earth2Studio's model boundary is one packed tensor and must
have one unambiguous axis order.

The [`xarray-jax` project](https://github.com/google-deepmind/xarray_jax) similarly
preserves `DataArray` dimension names and coordinates while exposing the underlying
arrays to JAX. It treats normal xarray coordinates as static metadata, which matches
the proposed separation between model signature metadata and GPU payload data.

### WeatherBench

WeatherBench explicitly declares a dimension tuple for every generated xarray
variable rather than inferring computational order from coordinate iteration. Its
[`schema.py`](https://github.com/google-research/weatherbench2/blob/main/weatherbench2/schema.py)
constructs variables using tuples such as `("time", "level", "longitude",
"latitude")`.

The relevant lesson is that coordinate names alone are insufficient. Computational
dimension order must be attached to the array variable. `DataArray.dims` provides
that attachment directly.

### Grid-aware xarray packages

[MetPy](https://unidata.github.io/MetPy/latest/api/generated/metpy.xarray.html)
uses CF coordinate metadata to identify x, y, vertical, and time axes. It stores
projection information in a scalar grid-mapping coordinate and can derive 2D
latitude and longitude from projected x/y coordinates.

[rioxarray](https://corteva.github.io/rioxarray/stable/getting_started/crs_management.html)
also stores the CRS in a scalar coordinate using CF or GDAL-compatible metadata and
records which dimensions are spatial.

[xESMF](https://xesmf.readthedocs.io/en/stable/user_api.html) accepts xarray grid
descriptions with either 1D longitude/latitude coordinates for rectilinear grids or
2D longitude/latitude coordinates for general curvilinear grids. Regridding is an
explicit operation separate from ordinary coordinate selection.

Earth2Studio should use xarray's ordered dimensions and auxiliary coordinates, but
its internal model signature does not need a CF grid-mapping coordinate. The grid
registry and namespaced attributes are the source of truth. A format adapter may
create CF-compatible grid-mapping metadata when exporting data.

### HEALPix

[HEALPix](https://healpix.sourceforge.io/doc/html/intro_Geometric_Algebraic_Propert.htm)
identifies a pixel using its index, `nside`, and either RING or NESTED ordering. A
full-resolution map contains `12 * nside**2` pixels. PROJ provides a [HEALPix map
projection](https://proj.org/en/stable/operations/projections/healpix.html), but a
projection does not encode the discrete resolution or pixel numbering. Earth2Studio
must therefore store a PyProj-supported geographic CRS plus explicit HEALPix topology
metadata when the model tensor uses a one-dimensional pixel index.

## Proposed signature representation

### Allocation-free signature

A model signature is an ordinary DataArray whose private backend stores shape and
dtype metadata but never allocates field values. Model authors do not initialize an
empty NumPy array or instantiate the backend directly.

```python
import earth2studio as e2s

signature = e2s.coord_array(
    dims=("batch", "lead_time", "variable", "lat", "lon"),
    coords={
        "lead_time": np.array([np.timedelta64(0, "h")]),
        "variable": VARIABLES,
    },
    dynamic=("batch",),
    grid="latlon-0.25deg",
)

assert signature.dims == (
    "batch",
    "lead_time",
    "variable",
    "lat",
    "lon",
)
assert signature.size == 0
assert signature.data.nbytes == 0
assert isinstance(signature, xr.DataArray)
assert signature.e2s.dynamic_dims == ("batch",)
```

The repository's current xarray version was tested with a shape-only duck array and
nominal shape of `(0, 1, 73, 721, 1440)`. Construction, transpose, coordinate
assignment, and basic slicing preserve the backend without allocating field data.
List-based indexing, label reordering, and reindexing currently raise because the
prototype supports only basic indexers. Those operations are required for alignment
and must be implemented before the signature contract is complete. Coordinate arrays
that are explicitly attached still consume their normal metadata memory.

### Coordinate signatures and coordinate memory

"Zero allocation" means zero field-data allocation, not literally a zero-byte Python
object. The DataArray header, dimension names, variable names, and metadata must exist
somewhere so that the signature can communicate its contract. These objects are small
and unavoidable. Large spatial coordinate arrays do not need to be materialized in
the signature.

Xarray permits dimensions without coordinate labels and falls back to a compact
`RangeIndex` when positional indexing is requested. A coordinate signature can
therefore retain spatial sizes while omitting explicit x, y, latitude, and longitude
arrays.

The public constructor should have an xarray-like signature:

```python
def coord_array(
    dims: Sequence[Hashable],
    coords: Mapping[Hashable, Any] | None = None,
    *,
    dynamic: Sequence[Hashable] = (),
    sizes: Mapping[Hashable, int] | None = None,
    grid: str | pyproj.CRS | Mapping[str, Any] | None = None,
    statistics: Mapping[str, str] | None = None,
    dtype: DTypeLike = np.float32,
    name: Hashable | None = None,
    attrs: Mapping[Hashable, Any] | None = None,
) -> xr.DataArray:
    """Create an allocation-free Earth2Studio coordinate DataArray."""
```

[Xarray's accessor
registration](https://docs.xarray.dev/en/stable/internals/extending-xarray.html)
attaches `.e2s` to `DataArray` instances; it does not create a module namespace such
as `xr.e2s`. [Xarray
backends](https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html) are
file IO engines, not constructor plugins. Adding `xr.e2s` would therefore require
monkeypatching xarray. A top-level `earth2studio.coord_array()` is the only bespoke
construction step, and the returned standard DataArray immediately exposes
`array.e2s` for all later operations.

Dimension sizes are inferred from attached dimension coordinates, a known `grid` key,
and then explicit `sizes`. A CRS-only `grid` input does not define shape, so its
spatial dimensions still require coordinates or sizes. Dimensions listed in `dynamic`
have length zero and need no empty coordinate value. Missing dimensions raise rather
than being silently treated as dynamic. The constructor adds the standard
Earth2Studio kind, schema-version, and dynamic-dimension attributes before returning
the DataArray. The optional `statistics` mapping is resolved against the plain string
values in the `variable` coordinate and packed into normalized DataArray attributes.

The constructor internally uses a deliberately small xarray-compatible duck array.
It is private because model authors should never need to interact with it. The
following shows its required interface; indexing details are omitted:

```python
class _CoordinateArray:
    """Shape-only storage for an xarray coordinate signature."""

    def __init__(
        self,
        sizes: Mapping[str, int | None],
        *,
        dtype: DTypeLike = np.float32,
    ) -> None:
        self.dims = tuple(sizes)
        self.shape = tuple(0 if size is None else size for size in sizes.values())
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)

    @property
    def nbytes(self) -> int:
        return 0

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> Self:
        ...

    def transpose(self, axes: tuple[int, ...]) -> Self:
        ...

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any):
        ...

    def __array_ufunc__(self, ufunc: Any, method: str, *args: Any, **kwargs: Any):
        return NotImplemented

    def __array__(self, *args: Any, **kwargs: Any) -> Never:
        raise TypeError("CoordinateArray has no materialized values")
```

The implementation only needs shape-preserving indexing and transpose operations
used by xarray's structural APIs. Numerical operations, device conversion, and field
serialization must raise. This prevents a coordinate signature from being mistaken
for runtime data.

The internal array supplies ordered dimensions, defaults to `float32`, maps dynamic
dimensions to zero-length axes, and rejects materialization. It allocates a small
Python object for shape and dtype metadata but zero bytes for field values.

```python
def input_coords(self) -> xr.DataArray:
    return e2s.coord_array(
        dims=(
            "batch",
            "time",
            "lead_time",
            "variable",
            "hrrr_y",
            "hrrr_x",
        ),
        coords={
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": self.variables,
        },
        dynamic=("batch", "time"),
        grid="hrrr-conus-3km",
    )
```

### Reserved Earth2Studio metadata

Earth2Studio-specific metadata should use a small set of exported constants whose
values all begin with `earth2studio_`. Model authors should import the constants
rather than repeat string literals.

| Constant | Stored key | Purpose |
| --- | --- | --- |
| `E2S_KIND` | `earth2studio_kind` | Distinguish coordinate arrays and frames |
| `E2S_SCHEMA_VERSION` | `earth2studio_schema_version` | Version the metadata contract |
| `E2S_DYNAMIC_DIMS` | `earth2studio_dynamic_dims` | Identify wildcard dimensions |
| `E2S_GRID_ID` | `earth2studio_grid_id` | Store the canonical resolved grid key |
| `E2S_CRS` | `earth2studio_crs` | Store a normalized CRS for custom grids |
| `E2S_GRID_HASH` | `earth2studio_grid_hash` | Verify exact grid identity |
| `E2S_SPATIAL_DIMS` | `earth2studio_spatial_dims` | Identify ordered spatial dimensions |
| `E2S_GRID_TOPOLOGY` | `earth2studio_grid_topology` | Name indexed grid topology |
| `E2S_HEALPIX_NSIDE` | `earth2studio_healpix_nside` | Set HEALPix resolution |
| `E2S_HEALPIX_ORDERING` | `earth2studio_healpix_ordering` | Set RING or NESTED indexing |
| `E2S_ROLE` | `earth2studio_role` | Label a coordinate or frame column role |

These attributes keep the internal signature compact and independent of optional CF
serialization conventions. Known grids resolve their CRS through `E2S_GRID_ID`;
custom grids store a normalized CRS in `E2S_CRS`. No implicit `spatial_ref`
coordinate or `grid_mapping` attribute is added. Values should remain flat and
serializable; the metadata exporter can produce a compact JSON description for
agents without writing a dummy field array. `coord_array()` autofills `E2S_KIND`,
`E2S_SCHEMA_VERSION`, and `E2S_DYNAMIC_DIMS`; a model only adds metadata specific to
its variables or grid.

### Grid accessor

The reserved keys should be managed through the existing `.e2s` xarray accessor.
Setters return a new DataArray, while getters return plain serializable mappings or
standard library objects rather than Earth2Studio metadata classes.

```python
signature = signature.e2s.set_grid("hrrr-conus-3km")
grid = signature.e2s.get_grid()
crs = signature.e2s.crs
```

`set_grid()` accepts either a known Earth2Studio grid key or PyProj-normalizable CRS
input. A known key supplies complete geometry. A CRS-only input uses the array's
coordinates and explicit transform or topology arguments. `get_grid()` returns a
plain mapping containing the normalized CRS, spatial dimensions, geometry or
topology, hash, and optional canonical key.

### Grid resolver and registry

The same resolver backs `coord_array(grid=...)` and `array.e2s.set_grid(...)`:

1. Normalize aliases and check exact Earth2Studio custom grids.
2. Check parameterized grid families such as regular lat/lon and HEALPix.
3. Attempt `pyproj.CRS.from_user_input()` or `CRS.from_cf()`.
4. Raise with close known-key suggestions when neither path resolves.

```python
known = array.e2s.set_grid("hrrr-conus-3km")
standard = array.e2s.set_grid(
    "EPSG:4326",
    spatial_dims=("lat", "lon"),
)
```

Initial keys are:

| Canonical key | Alias | Definition |
| --- | --- | --- |
| `hrrr-conus-3km` | `hrrr` | Lambert CRS, 1059 by 1799 shape, HRRR transform |
| `latlon-0.25deg` | `latlon025` | EPSG:4326, descending latitude, 0-360 longitude |
| `fcn-global-0.25deg` | `fcn` | EPSG:4326, 720 by 1440, south pole excluded |
| `healpix-l6-nested` | `hpx6` | `nside=64`, NESTED, 49,152 pixels |
| `healpix-l10-nested` | `hpx10` | `nside=1024`, NESTED, 12,582,912 pixels |

The registry should implement `latlon-{resolution}deg` and
`healpix-l{level}-{ring|nested}` as validated resolver families. This avoids a large
enumerated table while retaining canonical keys. Short aliases resolve to one fixed,
documented canonical key; `hpx6` therefore always means `healpix-l6-nested`.
The regular lat/lon family includes both poles, orders latitude north-to-south, uses
longitudes in `[0, 360)`, and requires a resolution that evenly divides both 180 and
360 degrees. Other conventions use explicit coordinates and an EPSG CRS.

Exact custom grids such as HRRR are private registry entries containing CRS, ordered
spatial dimensions, shape, transform, units, and a lazy coordinate generator.
Registry entries may be plain mappings or callables; no public `GridSpec` class is
required. Names that PyProj already recognizes are reserved and cannot be shadowed by
custom entries.

Resolution stores the canonical grid key and required geometry or topology on the
DataArray. Known grids resolve their normalized CRS from the registry; custom grids
store it in `E2S_CRS`.

For `coord_array()`, a known key fills missing fixed spatial sizes and generated
coordinates. For an existing runtime DataArray, `set_grid()` only validates and
annotates matching dimensions and shape; it never reshapes or regrids data.

There is no `compact_grid_metadata` object in this design. CRS, geometry, spatial
dimension names, and indexed topology use flat serializable attributes managed by
`.e2s`.

Every spatial grid carries a CRS. A regular latitude-longitude grid normally uses an
EPSG geographic CRS. A projected grid such as HRRR uses a PyProj-supported Lambert
conformal CRS. A HEALPix grid uses a PyProj-supported geographic CRS plus explicit
topology metadata because its tensor coordinate is a pixel index rather than x/y.

This signature allocates no field buffer regardless of its declared sizes. It also
avoids the 1D projected coordinates and 2D latitude/longitude arrays. Those
coordinates can be generated only when an operation needs them:

```python
with_coords = signature.e2s.materialize_grid_coords()
```

Grid coordinate storage follows three levels:

1. **Generated grid:** Store shape, affine coordinate parameters, and CRS metadata.
   Generate x/y and optional 2D latitude/longitude on demand.
2. **Indexed topology:** Store shape, CRS, and topology parameters such as HEALPix
   `nside` and ordering. Generate pixel centers on demand.
3. **External grid:** Store a URI plus a content hash for arbitrary irregular grids.
   Load coordinates lazily only when exact comparison or regridding requires them.

An optional registry ID may be added to any level to cache a known coordinate
generator, but the serialized CRS and geometry or topology remain authoritative.

Inline coordinate arrays remain supported for small or genuinely irregular grids,
but they are not the default model-signature representation. An arbitrary irregular
grid cannot be represented exactly with literally zero metadata or without storing
or referencing its coordinate values somewhere.

The experimental xarray `CoordinateTransformIndex` can create lazy coordinates from
a transform, but Earth2Studio should not make its initial protocol depend on an
experimental API. A small `.e2s.materialize_grid_coords()` utility provides the same
required behavior while keeping the serialized metadata stable.

Coordinate signatures are runtime metadata declarations and should not be written as
dummy NetCDF or Zarr arrays. IO backends should reject the private shape-only backend
as field data. If a signature must be persisted, Earth2Studio should serialize only
its dimensions, sizes, dtype, fixed labels, grid metadata, and external references to
JSON or another compact metadata representation.

All native model signatures should include a dynamic leading `batch` dimension.
This is already the Earth2Studio batching convention. A legacy model that does not
natively batch can be wrapped with a singleton batch internally while still
advertising the common signature.

### Why the dimension order is unambiguous

Coordinate mapping order is not used. Only the DataArray variable's dimension tuple
defines tensor order:

```python
model.input_coords().dims
model.output_coords(model.input_coords()).dims
```

Moving, sorting, or reconstructing coordinates does not change the declared tensor
order unless the DataArray itself is transposed. A model that requires `(y, x)` and
a model that requires `(x, y)` advertise different signatures even if they reference
the same physical grid.

### Dynamic dimensions

The constructor's `dynamic` argument retains the current wildcard meaning and maps
those names to zero-length signature dimensions:

```python
signature = e2s.coord_array(
    dims=("batch", "time", "variable", "lat", "lon"),
    coords={"variable": self.variables},
    dynamic=("batch", "time"),
    sizes={"lat": self.nlat, "lon": self.nlon},
)
```

The dimension name and dynamic role remain inspectable through `DataArray.dims` and
`E2S_DYNAMIC_DIMS`. A dynamic dimension is required but accepts any nonzero runtime
size and compatible values. Fixed dimensions contain their required coordinate
labels. A model that needs a specific dynamic coordinate dtype records that
requirement on the coordinate or with a dedicated Earth2Studio role attribute.

A pipeline can resolve a dynamic coordinate without allocating payload data because
the batch dimension remains empty:

```python
generic = model.input_coords()
requested = generic.reindex(time=[np.datetime64("2026-08-28T00")])

assert requested.size == 0
```

This allows output coordinates to be derived for a specific initialization time
without downloading or allocating the input fields.

## Spatial grid examples

### Regular latitude-longitude

A regular grid uses latitude and longitude as both tensor dimensions and physical
coordinates. The known key resolves its EPSG CRS, shape, coordinate direction, and
spacing:

```python
latlon = e2s.coord_array(
    dims=("batch", "time", "variable", "lat", "lon"),
    coords={"variable": variables},
    dynamic=("batch", "time"),
    grid="latlon-0.25deg",
)
```

`latlon.dims` fixes the model tensor order. The grid key resolves the CRS, shape,
coordinate direction, and spacing without adding a scalar coordinate. CRS axis order
does not transpose the model tensor.

### Lambert conformal

HRRR uses a custom Lambert conformal CRS. The common path uses its registry key:

```python
hrrr = e2s.coord_array(
    dims=("batch", "time", "variable", "hrrr_y", "hrrr_x"),
    coords={"variable": variables},
    dynamic=("batch", "time"),
    grid="hrrr-conus-3km",
)
```

The registry resolves the 1059 by 1799 shape, transform, and the custom Lambert CRS.
The same grid can be declared explicitly when no key exists:

```python
hrrr_crs = pyproj.CRS.from_user_input(
    "+proj=lcc +lon_0=262.5 +lat_0=38.5 +lat_1=38.5 "
    "+lat_2=38.5 +R=6371229 +units=m +type=crs"
)

hrrr = e2s.coord_array(
    dims=("batch", "time", "variable", "hrrr_y", "hrrr_x"),
    coords={"variable": variables},
    dynamic=("batch", "time"),
    sizes={"hrrr_y": 1059, "hrrr_x": 1799},
).e2s.set_grid(
    hrrr_crs,
    spatial_dims=("hrrr_y", "hrrr_x"),
    transform=hrrr_transform,
)
```

When coordinates are materialized for inspection, regridding, or IO, the runtime
DataArray follows normal xarray and CF conventions:

```python
hrrr = hrrr.assign_coords(
    {
        "hrrr_y": xr.Variable(
            "hrrr_y",
            projected_y,
            attrs={
                "axis": "Y",
                "standard_name": "projection_y_coordinate",
                "units": "m",
            },
        ),
        "hrrr_x": xr.Variable(
            "hrrr_x",
            projected_x,
            attrs={
                "axis": "X",
                "standard_name": "projection_x_coordinate",
                "units": "m",
            },
        ),
        "lat": (("hrrr_y", "hrrr_x"), latitude),
        "lon": (("hrrr_y", "hrrr_x"), longitude),
    }
)
```

### One-dimensional HEALPix

HEALPix uses one tensor dimension for the pixel index. Parameterized keys resolve the
geographic CRS, `nside`, ordering, shape, and coordinate generator:

```python
def healpix_signature(level: int) -> xr.DataArray:
    return e2s.coord_array(
        dims=("batch", "time", "variable", "hpx"),
        coords={"variable": variables},
        dynamic=("batch", "time"),
        grid=f"healpix-l{level}-nested",
    )


hpx_level_6 = healpix_signature(6)  # shape: (..., 49_152)
hpx_level_10 = healpix_signature(10)  # shape: (..., 12_582_912)
```

The coordinate signature does not allocate `np.arange(npix)`. The index is implied by
the dimension size and topology. A runtime array can expose pixel centers as
one-dimensional auxiliary coordinates without changing tensor order:

```python
npix = 12 * 64**2
hpx = xr.DataArray(
    data,
    dims=("time", "variable", "hpx"),
    coords={
        "hpx": np.arange(npix),
        "lat": ("hpx", pixel_latitude),
        "lon": ("hpx", pixel_longitude),
    },
).e2s.set_grid(
    "healpix-l6-nested",
)
```

PROJ's `+proj=healpix` operation is useful for projecting the sphere to a 2D map, but
it does not encode `nside`, the RING/NESTED scheme, or the one-dimensional pixel
index. Those remain required topology attributes.

### Grid identity

The minimum grid identity is:

1. PyProj-normalized WKT2 or PROJJSON
2. Ordered spatial tensor dimensions and shape
3. Coordinate units
4. Exact coordinates, an affine transform, or indexed topology parameters
5. Coordinate digests or external references where geometry is not generatable

The fingerprint hashes these fields. A human-readable grid ID is optional and
excluded from the hash. Serialized arrays remain self-describing without requiring a
registry lookup.

## Temporal Statistic Modifiers

A statistic is an optional modifier on a variable, not a new variable identity.
`u10m` remains `u10m` whether it represents an instantaneous value, a daily mean, or
a daily maximum. The `variable` coordinate remains a plain array of strings. The
modifier is independent of the spatial grid and does not affect tensor dimension
order.

### Representation

`coord_array()` accepts one optional mapping from variable strings to compact
modifiers:

```python
def input_coords(self) -> xr.DataArray:
    return e2s.coord_array(
        dims=("batch", "time", "variable", "lat", "lon"),
        coords={"variable": np.array(["u10m", "t2m"])},
        dynamic=("batch", "time"),
        grid="latlon-0.25deg",
        statistics={"u10m": "mean:24h"},
    )
```

This mirrors `grid=`: the model supplies a short declaration, and the resolver packs a
fully descriptive, JSON-serializable mapping into the attributes:

```python
signature = model.input_coords()

signature.coords["variable"].values
# np.array(["u10m", "t2m"])

signature.attrs["earth2studio_statistics"]
# {
#     "u10m": {
#         "modifier": "mean:24h",
#         "method": "mean",
#         "window": "PT24H",
#         "start_offset": "-PT24H",
#         "end_offset": "PT0S",
#         "closed": "left",
#     }
# }
```

Variables absent from the mapping are instantaneous. Modifiers use the compact form
`method:window`:

- `mean:24h` means a trailing 24-hour mean
- `max:24h` means a trailing 24-hour maximum
- `min:24h` means a trailing 24-hour minimum
- `sum:6h` means a trailing six-hour sum or accumulation

Custom intervals use `method:start_offset:end_offset`:

```python
centered = e2s.coord_array(
    dims=("batch", "time", "variable", "lat", "lon"),
    coords={"variable": ["u10m"]},
    dynamic=("batch", "time"),
    grid="latlon-0.25deg",
    statistics={"u10m": "mean:-12h:+12h"},
)
```

The modifier resolves to:

```python
centered.attrs["earth2studio_statistics"]["u10m"]
# {
#     "modifier": "mean:-12h:+12h",
#     "method": "mean",
#     "window": "PT24H",
#     "start_offset": "-PT12H",
#     "end_offset": "PT12H",
#     "closed": "left",
# }
```

At target valid time `2026-08-28T12:00`, this covers
`2026-08-28T00:00 <= t < 2026-08-29T00:00`. The two-part `mean:24h` form is
shorthand for `mean:-24h:0h`.

The string only defines the method and offsets from `T`, where `T` is always
`valid_time`. The dimensions determine how valid time is represented:

```python
# Data or analysis: valid_time = time
analysis = e2s.coord_array(
    dims=("time", "variable", "lat", "lon"),
    coords={"variable": ["u10m"]},
    dynamic=("time",),
    grid="latlon-0.25deg",
    statistics={"u10m": "mean:-12h:+12h"},
)

# Forecast: valid_time = time + lead_time
forecast = e2s.coord_array(
    dims=("time", "lead_time", "variable", "lat", "lon"),
    coords={"lead_time": [np.timedelta64(6, "h")], "variable": ["u10m"]},
    dynamic=("time",),
    grid="latlon-0.25deg",
    statistics={"u10m": "mean:-12h:+12h"},
)
```

This lets data and forecast sources use the same statistic strings without exposing
their storage-specific temporal dimension in the modifier.

If every variable is instantaneous, the attribute is omitted.
`get_statistic()` returns `None` for a variable without a modifier.

The accessor owns parsing and validation. Model authors do not provide a metadata
class, source variable, reduced dimension, offsets, closure, or sample interval. The
variable is the mapping key, and the remaining fields follow the versioned modifier
convention.

### Accessor API

The public API contains only the information that changes:

```python
array = array.e2s.set_statistic("u10m", method="mean", window="24h")
modifier = array.e2s.get_statistic("u10m")

assert modifier == "mean:24h"
assert list(array.coords["variable"].values) == ["u10m", "t2m"]
```

The window follows one convention: it is trailing and half-open. At target valid time
`T`, `mean:24h` means `mean(u10m[t] for T - 24h <= t < T)`. `T` is the physical
timestamp for which the statistic is requested, not another array dimension.

For a forecast, `time` is its initialization or reference time, while `lead_time` is
the offset from that initialization. `valid_time` is the physical timestamp described
by the forecast value:

```python
time = np.datetime64("2026-08-28T00:00")
lead_time = np.timedelta64(6, "h")
valid_time = time + lead_time
# np.datetime64("2026-08-28T06:00")
```

An observation or analysis normally has no `lead_time`; its `time` coordinate is
already its valid time. For multiple initialization and lead coordinates,
`valid_time` is a derived two-dimensional coordinate over `(time, lead_time)`.

The statistic is selected in valid-time space. The concrete request determines which
storage dimension changes across that interval:

1. Varying `time` with no `lead_time`, or one fixed lead, reduces over `time`.
2. One fixed `time` with varying `lead_time` reduces over `lead_time`.
3. Varying `time` and `lead_time` is ambiguous. Earth2Studio raises and requires the
   caller to select one initialization or one lead before applying the statistic.

The static attribute stores no physical reduction dimension because a model signature
may have dynamic temporal coordinates. For a forecast request with one initialization
and multiple leads, the resolved description is:

```python
resolved = request.e2s.resolve_statistic(
    "u10m",
    valid_time=np.datetime64("2026-08-28T06:00"),
)

resolved
# {
#     "modifier": "mean:24h",
#     "method": "mean",
#     "window": np.timedelta64(24, "h"),
#     "valid_time": np.datetime64("2026-08-28T06:00"),
#     "source_dimension": "lead_time",
#     "start": np.datetime64("2026-08-27T06:00"),
#     "end": np.datetime64("2026-08-28T06:00"),
#     "closed": "left",
# }
```

The stored representation is portable; `resolve_statistic()` adds the target valid
time and concrete bounds only at runtime. Backend serialization can translate the
descriptor to CF `cell_methods` and time bounds without changing the variable label.

### Downstream use

Fetching and reduction use the unchanged variable label:

```python
modifier = request.e2s.get_statistic("u10m")
source_times = request.e2s.statistic_times("u10m", valid_time)
source = fetch_data(data_source, source_times, "u10m")
result = source.e2s.apply_statistic("u10m", modifier)
```

Reading a modifier never computes a reduction or moves data between devices. When
unpacking for IO, Earth2Studio converts it to CF `cell_methods` and concrete time
bounds.

Generic xarray operations copy attributes but do not align mapping keys. The accessor
therefore ignores keys not present in the current string-valued `variable` coordinate,
and Earth2Studio validation and IO prune those stale entries before handshakes or
serialization.

The initial protocol permits one modifier per variable label in a packed DataArray.
If a workflow needs both instantaneous and aggregated `u10m` simultaneously, it uses
separate arrays. A more complex representation should be introduced only if real
model interfaces require it.

## Input and output derivation

### Prognostic model

A prognostic exposes a generic input signature and transforms any compatible input
signature into the one-step output signature.

```python
def output_coords(self, input_coords: xr.DataArray) -> xr.DataArray:
    input_coords = handshake_coords(input_coords, self.input_coords())

    return input_coords.assign_coords(
        lead_time=input_coords.lead_time + np.timedelta64(1, "h")
    )
```

For a generic input, dynamic dimensions remain empty. For a resolved signature,
concrete times and batch labels can be preserved. No payload values are inspected.

### Diagnostic model

A diagnostic explicitly declares additions, removals, and ordering changes in its
returned DataArray dimensions.

```python
def output_coords(self, input_coords: xr.DataArray) -> xr.DataArray:
    input_coords = handshake_coords(input_coords, self.input_coords())

    coords = dict(input_coords.coords)
    coords.update(
        {
            "sample": np.arange(self.number_of_samples),
            "variable": self.output_variables,
            "hrrr_y": self.output_y,
            "hrrr_x": self.output_x,
        }
    )
    return e2s.coord_array(
        dims=(
            "batch",
            "sample",
            "time",
            "lead_time",
            "variable",
            "hrrr_y",
            "hrrr_x",
        ),
        coords=coords,
        dynamic=input_coords.e2s.dynamic_dims,
        sizes={
            "batch": input_coords.sizes["batch"],
            "time": input_coords.sizes["time"],
        },
        dtype=input_coords.dtype,
        attrs=input_coords.attrs,
    )
```

The output order is explicit even when it differs from the input. A scalar or reduced
diagnostic simply omits the removed dimensions from its returned `dims` tuple.

### Pipeline composition

Signatures can be composed without model execution:

```python
signature = prognostic.input_coords()
signature = prognostic.output_coords(signature)
signature = diagnostic.output_coords(signature)
signature = statistic.output_coords(signature)

print(signature.dims)
print(signature.coords)
```

This supports compatibility checks, wrapper construction, agent inspection, graph
planning, and IO initialization before any data or checkpoints are loaded.

## DataFrame signatures

Frame-based models need the same planning capability, but their structure is an
ordered set of fields rather than an N-dimensional tensor. An empty, typed
`pandas.DataFrame` is the direct equivalent of an empty DataArray signature. Pandas
already supports zero-row typed columns, so Earth2Studio should not introduce a
`CoordinateFrame` or hide pandas behind a factory. The model constructs a normal
DataFrame directly.

```python
def input_coords(self) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "time": pd.Series(dtype="datetime64[ns]"),
            "station": pd.Series(dtype="string"),
            "lat": pd.Series(dtype="float64"),
            "lon": pd.Series(dtype="float64"),
            "variable": pd.Series(
                pd.Categorical([], categories=self.variables, ordered=True)
            ),
            "observation": pd.Series(dtype="float32"),
        }
    )
    frame.attrs.update(
        {
            E2S_KIND: "coordinate_frame",
            E2S_SCHEMA_VERSION: 1,
            "earth2studio_geometry": "point",
            "earth2studio_crs": "EPSG:4326",
        }
    )
    return frame
```

The DataFrame contains zero rows, so it has no observation payload. Its metadata
communicates:

- `DataFrame.columns` for required field order
- `DataFrame.dtypes` for field types
- `CategoricalDtype` for fixed or allowed labels
- `DataFrame.attrs` for roles, CRS, units, and other schema metadata

`output_coords(input_signature)` returns another empty typed DataFrame, allowing a
frame diagnostic or assimilation model to add, remove, or reorder fields without
processing any observations.

```python
output_signature = model.output_coords(model.input_coords())

assert output_signature.empty
print(tuple(output_signature.columns))
print(output_signature.dtypes)
```

As with DataArray signatures, the DataFrame object and its column metadata consume a
small amount of Python memory. No row data is allocated or written to disk. Multiple
frame or tensor inputs continue to use the existing tuple convention, with one empty
signature per input.

## Alignment and handshakes

Earth2Studio should distinguish normalization from strict validation.

```python
array = align_coords(array, model.input_coords())
handshake_coords(array, model.input_coords())
```

### Alignment

`align_coords` may perform only structural, label-preserving operations:

1. Pack arbitrary leading dimensions into the model's dynamic batch dimension.
2. Add an implicit singleton batch when only batch is missing.
3. Select and reorder fixed variable labels.
4. Select and reorder fixed coordinate labels.
5. Transpose the DataArray into the model signature's `dims` order.

For example, input with dimensions `(batch, variable, hrrr_x, hrrr_y)` can be
transposed to `(batch, variable, hrrr_y, hrrr_x)`. Xarray transposes dependent
coordinates such as `lat(hrrr_y, hrrr_x)` and `lon(hrrr_y, hrrr_x)` with the data.
For NumPy and CuPy arrays, transpose is normally a view; the later Torch conversion
may make the payload contiguous only when the model requires it.

### Strict handshake

After alignment, `handshake_coords` validates:

- The runtime `array.dims` exactly equals the signature `required.dims`.
- Every required dimension is present.
- Fixed coordinate labels and their order match.
- Dynamic coordinates have an accepted dtype and nonzero runtime size.
- Required auxiliary coordinates are present.
- Auxiliary coordinate dimension tuples match exactly.
- Auxiliary coordinate shapes agree with their referenced dimensions.
- Projected x/y values, units, and grid-mapping metadata match.
- Required DataArray dtype and model-specific metadata match.

Signature-to-signature handshakes use the same checks but permit a dynamic dimension
to remain length zero.

### Regridding boundary

Alignment must never perform nearest-neighbor interpolation or projection changes.
If x/y values, latitude/longitude geometry, or the CRS differ, the handshake raises a
grid mismatch and identifies the required grid. The caller must explicitly invoke a
regridding utility before model execution.

This removes the current ambiguity in `map_coords`, where label selection,
reordering, nearest matching, and partial interpolation share one function.

## IO and rollout planning

The one-step output signature contains the information currently consumed by IO:

```python
one_step = prognostic.output_coords(prognostic.input_coords())
```

The workflow can replace the dynamic dimensions with requested values while retaining
zero payload size:

```python
store_signature = one_step.reindex(
    time=requested_times,
    lead_time=requested_lead_times,
)

io.add_array(store_signature)
```

This avoids reconstructing ordered dictionaries and manually moving `time` and
`lead_time` keys to the front.

## Transition from `CoordSystem`

No flag day is required.

1. Add converters between an ordered `CoordSystem` and an empty DataArray signature.
2. Allow protocol metadata methods to return either representation during migration.
3. Wrap legacy `output_coords` by converting the input signature to an ordered
   dictionary and converting its result back to a signature.
4. Let migrated models return signatures directly.
5. Keep runtime model adapters responsible for converting DataArrays to Torch until
   each model becomes natively DataArray-aware.

```python
def legacy_output_signature(model, input_signature):
    input_coords = signature_to_coord_system(input_signature)
    output_coords = model.output_coords(input_coords)
    return coord_system_to_signature(output_coords, dtype=input_signature.dtype)
```

The ordered legacy dictionary remains authoritative inside the adapter, so current
models preserve their exact behavior. New model declarations use xarray dimensions,
auxiliary coordinates, and CF grid metadata without introducing a new schema class.

## Alternatives rejected

### Standalone `xr.Coordinates`

Rejected because it does not own the ordered dimensions of one tensor. Inferring axis
order from coordinate iteration would recreate the ambiguity this migration is meant
to remove.

### Coordinate attributes containing an axis number

Rejected because it duplicates structural information, is verbose to declare, and
can become inconsistent after dimension operations. `DataArray.dims` already carries
the ordered structure.

### A new outer signature or per-grid class

Rejected for the initial migration because it increases the number of concepts model
authors and users must learn. The required structure already exists in DataArray,
xarray coordinates, and CF metadata. The private shape-only backend is not a public
schema and does not carry grid-specific behavior.

### An `xr.Dataset` signature

Rejected as the default because a Dataset permits each variable to have independent
dimensions and does not expose one global tensor order. It remains appropriate for a
future protocol whose model boundary genuinely consists of several independently
shaped arrays.

## Open questions

1. Which coordinate attributes are mandatory for exact grid comparison.
2. Whether 2D latitude and longitude are required in every projected-grid signature
   or derived lazily from x/y and the CRS.
3. Whether signature dtype represents model input dtype, public physical-data dtype,
   or both through separate metadata.

## Recommendation

Prototype this representation on three models before changing the protocol:

1. FCN for a fixed regular latitude-longitude prognostic
2. StormCast for a projected HRRR prognostic
3. CorrDiff for a diagnostic that adds `sample`, changes variables, and changes grid

The prototype should prove generic and resolved signature derivation, dimension
transposition, grid mismatch reporting, wrapper composition, and IO initialization.
If these cases work without model-specific exceptions, the representation satisfies
the current coordinate contract while enabling the DataArray migration.
