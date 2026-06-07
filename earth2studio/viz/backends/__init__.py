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
"""Agent-friendly summary: visualization backend registry exports.

Key APIs: `register_backend`, `get_backend`, `available_backends`,
`BackendCapabilities`, `RenderResult`, and `SummaryBackend`.
"""

from earth2studio.viz.backends.base import (
    BackendCapabilities,
    RenderResult,
    SummaryBackend,
    VizBackend,
    VizDependencyError,
    available_backends,
    get_backend,
    register_backend,
)

__all__ = [
    "BackendCapabilities",
    "RenderResult",
    "SummaryBackend",
    "VizBackend",
    "VizDependencyError",
    "available_backends",
    "get_backend",
    "register_backend",
]
