:mod:`earth2studio.viz`: Visualization
--------------------------------------

Visualization helpers for xarray data arrays, datasets, pandas or cuDF-like
data frames, regional terrain scenes, and backend-neutral scene descriptions.

.. automodule:: earth2studio.viz
    :no-members:
    :no-inherited-members:

.. currentmodule:: earth2studio

.. autosummary::
    :nosignatures:
    :toctree: generated/viz/

    viz.plot
    viz.raster_dataarray
    viz.raster_panel
    viz.plot_raster_grid
    viz.save_raster_grid
    viz.series_panel
    viz.plot_series
    viz.save_series
    viz.point_panel
    viz.plot_points
    viz.plot_point_sets
    viz.save_points
    viz.save_point_sets
    viz.track_panel
    viz.plot_tracks
    viz.save_tracks
    viz.Scene
    viz.Camera
    viz.Timeline
    viz.RegionSpec
    viz.RasterPanel
    viz.SeriesPanel
    viz.PointPanel
    viz.TrackPanel
    viz.common_cache_root
    viz.viz_cache_root
    viz.readable_cache_filename
    viz.default_texture_domain
    viz.infer_grid_spec_from_xarray
    viz.LayerProtocol
    viz.SceneProtocol
    viz.BackendProtocol
    viz.AssetSourceProtocol
    viz.TextureManagerProtocol
    viz.Layer
    viz.RasterLayer
    viz.PointLayer
    viz.VectorLayer
    viz.TerrainLayer
    viz.DrapedRasterLayer
    viz.ImageLayer
    viz.GeoTiffLayer
    viz.MeshLayer
    viz.RegionCubeLayer
    viz.VolumeLayer
    viz.AssetSource
    viz.TextureSource
    viz.MeshSource
    viz.TextureFrame
    viz.TextureSequence
    viz.TextureCachePolicy
    viz.TextureDomain
    viz.TextureDomainAsset
    viz.GridSpec
    viz.infer_asset_kind
    viz.LayerStyle
    viz.ProjectionSpec
    viz.BackendCapabilities
    viz.RenderResult
    viz.VizDependencyError
    viz.register_backend
    viz.get_backend
    viz.available_backends
