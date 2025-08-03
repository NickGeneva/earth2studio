from earth2studio.utils.xarray import Earth2StudioDatasetAccessor
import numpy as np
import xarray as xr

# Create random data array
data = np.random.random((20, 20, 20, 20))
data2 = np.random.random((20, 20, 20))

# Create coordinate arrays
times = np.arange(20)
lats = np.linspace(-90, 90, 20)
lons = np.linspace(-180, 180, 20)

# Create dataset with coords
ds = xr.Dataset(
    data_vars={
        "data": (["time", "lead_time", "lat", "lon"], data),
        "data2": (["time", "lat", "lon"], data2),
    },
    coords={
        "time": times,
        "lat": lats,
        "lon": lons
    }
)

print(ds)

ds = ds.e2s.as_cupy()


tensors = ds.e2s.as_torch()
print(tensors)


test = xr.DataArray(
    coords={
        "batch": np.empty(0),
        "lat": lats,
        "lon": lons
    },
    dims=["batch", "lat", "lon"],
)

coords = xr.Dataset(
    data_vars={"data": test},
)

print(coords)

ds = ds.e2s.batch(coords)

print(ds)

# print(ds.e2s.unbatch())
