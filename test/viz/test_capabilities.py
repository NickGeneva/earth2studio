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
"""Agent-friendly summary: tests for internal viz parity inventory.

Key APIs under test: `default_capability_inventory` and
`summarize_capability_inventory` record Command Center feature-parity status
without expanding the public scene API.
"""

from earth2studio.viz.capabilities import (
    default_capability_inventory,
    summarize_capability_inventory,
)


def test_default_capability_inventory_tracks_current_gaps() -> None:
    inventory = default_capability_inventory()
    by_id = {capability.id: capability for capability in inventory}

    assert by_id["default-global-textures"].status == "implemented"
    assert by_id["application-session"].status == "partial"
    assert by_id["dynamic-texture-streaming"].status == "partial"
    assert by_id["data-to-visual-payload"].public_api_required is False
    assert (
        "Concrete OVRTX texture manager" in by_id["dynamic-texture-streaming"].missing
    )
    assert "globe_view" in by_id["application-session"].command_center_concept


def test_capability_inventory_summary_counts_statuses() -> None:
    summary = summarize_capability_inventory()
    counts = summary["counts"]
    gaps = summary["missing_or_partial"]

    assert counts["implemented"] >= 2
    assert counts["partial"] >= 1
    assert counts["missing"] >= 1
    assert all(gap["status"] in {"partial", "missing"} for gap in gaps)
    assert any(gap["id"] == "regional-terrain" for gap in gaps)
