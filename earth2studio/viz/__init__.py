# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Agent-friendly summary: public entrypoint for Earth2 Studio visualization.

Key APIs: `Scene`, `plot`, layer dataclasses, `Timeline`, `Camera`,
`RegionSpec`, asset and texture descriptors, native-grid heatmap helpers,
one-off quick plot helpers, and backend registry helpers. Importing this module
does not import heavy renderer dependencies; backend factories are resolved
lazily.
"""

from earth2studio.viz.api import plot
from earth2studio.viz.assets import (
    AssetSource,
    MeshSource,
    TextureSource,
    infer_asset_kind,
)
from earth2studio.viz.backends.base import (
    BackendCapabilities,
    RenderResult,
    VizDependencyError,
    available_backends,
    get_backend,
    register_backend,
)
from earth2studio.viz.base import (
    AssetSourceProtocol,
    BackendProtocol,
    LayerProtocol,
    SceneEventProtocol,
    SceneProtocol,
    SceneSessionProtocol,
    TextureManagerProtocol,
)
from earth2studio.viz.cache import (
    DEFAULT_VIZ_CACHE_VERSION,
    common_cache_root,
    readable_cache_filename,
    viz_cache_root,
)
from earth2studio.viz.camera import Camera
from earth2studio.viz.domains import (
    DEFAULT_TEXTURE_DOMAIN_NAME,
    TextureDomain,
    TextureDomainAsset,
    default_texture_domain,
)
from earth2studio.viz.grids import GridSpec, infer_grid_spec_from_xarray
from earth2studio.viz.layers import (
    DrapedRasterLayer,
    GeoTiffLayer,
    ImageLayer,
    Layer,
    MeshLayer,
    PointLayer,
    RasterLayer,
    RegionCubeLayer,
    TerrainLayer,
    TrackLayer,
    VectorLayer,
    VolumeLayer,
)
from earth2studio.viz.native import (
    can_native_heatmap,
    native_grid_heatmap,
    supports_native_heatmap,
)
from earth2studio.viz.quick import (
    PointPanel,
    RasterPanel,
    SeriesPanel,
    TrackPanel,
    plot_point_sets,
    plot_points,
    plot_raster_grid,
    plot_series,
    plot_tracks,
    point_panel,
    raster_dataarray,
    raster_panel,
    save_point_sets,
    save_points,
    save_raster_grid,
    save_series,
    save_tracks,
    series_panel,
    track_panel,
)
from earth2studio.viz.regional import RegionSpec
from earth2studio.viz.scene import Scene, SceneEvent
from earth2studio.viz.styles import LayerStyle, ProjectionSpec
from earth2studio.viz.textures import (
    TextureCachePolicy,
    TextureFrame,
    TextureSequence,
)
from earth2studio.viz.timeline import Timeline

__all__ = [
    "AssetSource",
    "AssetSourceProtocol",
    "BackendCapabilities",
    "BackendProtocol",
    "Camera",
    "DEFAULT_TEXTURE_DOMAIN_NAME",
    "DEFAULT_VIZ_CACHE_VERSION",
    "DrapedRasterLayer",
    "GeoTiffLayer",
    "GridSpec",
    "ImageLayer",
    "Layer",
    "LayerProtocol",
    "LayerStyle",
    "MeshLayer",
    "MeshSource",
    "PointLayer",
    "PointPanel",
    "ProjectionSpec",
    "RasterPanel",
    "RasterLayer",
    "RegionCubeLayer",
    "RegionSpec",
    "RenderResult",
    "Scene",
    "SceneEvent",
    "SceneEventProtocol",
    "SceneProtocol",
    "SceneSessionProtocol",
    "SeriesPanel",
    "TerrainLayer",
    "TextureCachePolicy",
    "TextureDomain",
    "TextureDomainAsset",
    "TextureFrame",
    "TextureManagerProtocol",
    "TextureSequence",
    "TextureSource",
    "Timeline",
    "TrackPanel",
    "TrackLayer",
    "VectorLayer",
    "VizDependencyError",
    "VolumeLayer",
    "available_backends",
    "can_native_heatmap",
    "common_cache_root",
    "default_texture_domain",
    "get_backend",
    "infer_asset_kind",
    "infer_grid_spec_from_xarray",
    "native_grid_heatmap",
    "point_panel",
    "plot",
    "plot_points",
    "plot_point_sets",
    "plot_raster_grid",
    "plot_series",
    "plot_tracks",
    "raster_dataarray",
    "raster_panel",
    "readable_cache_filename",
    "register_backend",
    "save_points",
    "save_point_sets",
    "save_raster_grid",
    "save_series",
    "save_tracks",
    "series_panel",
    "supports_native_heatmap",
    "track_panel",
    "viz_cache_root",
]
