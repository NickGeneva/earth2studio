
import pytest
import numpy as np
import cupy as cp
import pandas as pd
import torch
import xarray as xr
from earth2studio.utils.xarray import Earth2StudioDataArrayAccessor


class TestEarth2StudioDataArrayAccessor:
    """Test suite for Earth2StudioDataArrayAccessor."""

    @pytest.fixture(scope="class")
    def sample_data_array(self):
        """Create a sample DataArray for testing."""
        data = np.random.rand(10, 20, 30)
        coords = {
            'time': np.arange(10),
            'lat': np.linspace(-90, 90, 20),
            'lon': np.linspace(-180, 180, 30)
        }
        return xr.DataArray(data, dims=['time', 'lat', 'lon'], coords=coords)

    @pytest.fixture(scope="class")
    def sample_cupy_data_array(self):
        """Create a sample DataArray with cupy data for testing."""
        data = cp.random.rand(10, 20, 30)
        coords = {
            'time': np.arange(10),
            'lat': np.linspace(-90, 90, 20),
            'lon': np.linspace(-180, 180, 30)
        }
        return xr.DataArray(data, dims=['time', 'lat', 'lon'], coords=coords)

    @pytest.mark.parametrize("dtype", [
        np.float32, np.float64, np.int32, np.int64
    ])
    def test_xarray_device_cpu(self, sample_data_array, dtype):
        """Test device management functionality with different data types."""
        # Convert data to specified dtype
        data = sample_data_array.astype(dtype)
        da = data.e2s

        assert not da.is_cupy
        assert da.device == torch.device("cpu")

        cpu_result = da.to("cpu")
        assert isinstance(cpu_result.data, np.ndarray)
        assert cpu_result.data.dtype == dtype
        assert np.array_equal(cpu_result.data, data.data)

        numpy_result = da.numpy()
        assert isinstance(numpy_result.data, np.ndarray)
        assert numpy_result.data.dtype == dtype

        cpu_result2 = da.cpu()
        assert isinstance(cpu_result2.data, np.ndarray)
        assert cpu_result2.data.dtype == dtype

        torch_result = da.as_torch()
        assert isinstance(torch_result, torch.Tensor)
        assert torch_result.dtype == torch.float32 if dtype == np.float32 else torch.float64
        assert torch_result.shape == data.shape

        # Test to() method - CUDA (if available)
        if torch.cuda.is_available():
            cuda_result = da.to("cuda")
            assert hasattr(cuda_result.data, "__cuda_array_interface__")
            assert cuda_result.data.dtype == dtype

            # Test device property for CUDA
            cuda_da = cuda_result.e2s
            assert cuda_da.device.type == "cuda"
            assert cuda_da.is_cupy

            # Test conversion back to CPU
            back_to_cpu = cuda_da.to("cpu")
            assert isinstance(back_to_cpu.data, np.ndarray)
            assert np.allclose(back_to_cpu.data, data.data)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dtype", [
        np.float32, np.float64, np.int32, np.int64
    ])
    def test_xarray_device_gpu(self, sample_data_array, dtype):
        """Test device management functionality with different data types."""
        # Convert data to specified dtype
        data = sample_data_array.astype(dtype)
        da = data.e2s

        assert not da.is_cupy
        assert da.device == torch.device("cpu")

        da = da.to("cuda:0")
        assert hasattr(da.data, "__cuda_array_interface__")
        assert da.data.dtype == dtype
        assert da.e2s.device.type == "cuda"
        assert da.e2s.is_cupy

        torch_result = da.e2s.as_torch()
        assert isinstance(torch_result, torch.Tensor)
        assert torch_result.dtype == torch.float32 if dtype == np.float32 else torch.float64
        assert torch_result.shape == data.shape
        assert torch_result.device.type == "cuda"
        assert torch_result.get_device() == 0

        # Now check it was indeed a zero copy, namely modifying a element of
        # torch_result should also impact the cupy data array in object
        torch_result[0, 0] = 42.0
        assert cp.allclose(da.data[0, 0], 42.0)

        cpy_result = da.e2s.from_torch(torch_result)
        cpy_result.data[0, 0] = 43.0
        assert cp.allclose(cpy_result.data, da.data)

        back_to_cpu = da.e2s.to("cpu")
        assert isinstance(back_to_cpu.data, np.ndarray)
        # Should be different on CPU now because we went from CPU -> GPU -> CPU
        assert not np.allclose(back_to_cpu.data, data.data)
        data.data[0,0] = 43.0
        assert np.allclose(back_to_cpu.data, data.data)
        
    def test_xarray_batch_simple(self):
        data = np.random.rand(5, 10, 15)
        time = np.array(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'], dtype='datetime64[D]')
        da = xr.DataArray(
            data,
            dims=['time', 'lat', 'lon'],
            coords={
                'time': np.array(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'], dtype='datetime64[D]'),
                'lat': np.linspace(-90, 90, 10),
                'lon': np.linspace(-180, 180, 15)
            }
        )
        target_coords = xr.DataArray(
            dims=['batch', 'lat', 'lon'],
            coords={
                'batch': np.empty(0),
                'lat': np.linspace(-90, 90, 10),
                'lon': np.linspace(-180, 180, 15)
            }
        )
        batched = da.e2s.batch(target_coords)
        assert 'batch' in batched.dims
        assert batched.shape == (5, 10, 15)  # batch, lat, lon
        assert batched.dims == ('batch', 'lat', 'lon')
        assert len(batched.coords['batch']) == 5

        unbatched = batched.e2s.unbatch()
        assert 'time' in unbatched.dims
        assert unbatched.shape == (5, 10, 15)
        assert unbatched.dims == ('time', 'lat', 'lon')
        assert np.all(batched.coords['time'] == time)

    def test_xarray_batch_multiple_dims(self):
        da = xr.DataArray(
            data=np.random.rand(4, 3, 10, 15),
            dims=['ensemble', 'time', 'lat', 'lon'],
            coords={
                'ensemble': np.arange(4),
                'time': np.arange(3),
                'lat': np.linspace(-90, 90, 10),
                'lon': np.linspace(-180, 180, 15)
            }
        )
        target_coords = xr.DataArray(
            dims=['batch', 'lat', 'lon'],
            coords={
                'batch': np.arange(12),
                'lat': np.linspace(-90, 90, 10),
                'lon': np.linspace(-180, 180, 15)
            }
        )
        batched = da.e2s.batch(target_coords)
        assert 'batch' in batched.dims
        assert batched.shape == (12, 10, 15)  # 4*3=12, lat, lon
        assert batched.dims == ('batch', 'lat', 'lon')

        unbatched = batched.e2s.unbatch()
        assert 'ensemble' in unbatched.dims
        assert 'time' in unbatched.dims
        assert unbatched.shape == (4, 3, 10, 15)
        assert unbatched.dims == ('ensemble', 'time', 'lat', 'lon')

    def test_xarray_batch_no_batch_coords(self):
        data = np.random.rand(10, 15)
        da = xr.DataArray(
            data,
            dims=['lat', 'lon'],
            coords={
                'lat': np.linspace(-90, 90, 10),
                'lon': np.linspace(-180, 180, 15)
            }
        )
        target_coords = xr.DataArray(
            dims=['batch', 'lat', 'lon'],
            coords={
                'batch': np.empty(0),
                'lat': np.linspace(-90, 90, 10),
                'lon': np.linspace(-180, 180, 15)
            }
        )

        batched = da.e2s.batch(target_coords)
        assert 'batch' in batched.dims
        assert batched.shape == (1, 10, 15)
        assert batched.dims == ('batch', 'lat', 'lon')

        unbatched = batched.e2s.unbatch()
        assert unbatched.shape == (10, 15)
        assert unbatched.dims == ('lat', 'lon')

    def test_xarray_batch_different_coord_systems(self):
        data = np.random.rand(2, 3, 5, 7)
        da = xr.DataArray(
            data,
            dims=['level', 'time', 'y', 'x'],
            coords={
                'level': [1000, 500],
                'time': pd.date_range('2023-01-01', periods=3),
                'y': np.linspace(-60, 60, 5),
                'x': np.linspace(-120, 120, 7)
            }
        )

        # Create target coordinates with different naming
        target_coords = xr.DataArray(
            dims=['batch', 'y', 'x'],
            coords={
                'batch': np.empty(0),
                'y': np.linspace(-60, 60, 5),
                'x': np.linspace(-120, 120, 7)
            }
        )

        # Test batching
        batched = da.e2s.batch(target_coords)
        assert 'batch' in batched.dims
        assert batched.shape == (6, 5, 7)

        batched_0 = batched.sum(dim="x")

        # Test unbatching
        unbatched = batched_0.e2s.unbatch()
        assert 'level' in unbatched.dims
        assert 'time' in unbatched.dims
        assert unbatched.shape == (2, 3, 5)
        assert unbatched.dims == ('level', 'time', 'y')
        
        batched_0 = batched.transpose('batch', 'x', 'y')

        unbatched = batched_0.e2s.unbatch()
        assert 'level' in unbatched.dims
        assert 'time' in unbatched.dims
        assert unbatched.shape == (2, 3, 7, 5)
        assert unbatched.dims == ('level', 'time', 'x', 'y')
