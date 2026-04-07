# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Rust backend."""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip entire module if Rust backend is not compiled
try:
    from earth2bufrio._lib import read_bufr_rust

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = [
    pytest.mark.rust,
    pytest.mark.skipif(not HAS_RUST, reason="Rust backend not compiled"),
]


DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture()
def table_b_json() -> str:
    """Load table_b.json as string."""
    import importlib.resources

    ref = importlib.resources.files("earth2bufrio.tables").joinpath("table_b.json")
    return ref.read_text(encoding="utf-8")


@pytest.fixture()
def table_d_json() -> str:
    """Load table_d.json as string."""
    import importlib.resources

    ref = importlib.resources.files("earth2bufrio.tables").joinpath("table_d.json")
    return ref.read_text(encoding="utf-8")


class TestReadBufrRustDirect:
    """Test the raw Rust function directly."""

    def test_empty_file(self, tmp_path, table_b_json, table_d_json):
        """Empty file returns empty batch with fixed columns."""
        empty_file = tmp_path / "empty.bufr"
        empty_file.write_bytes(b"")
        batch = read_bufr_rust(str(empty_file), table_b_json, table_d_json, None, None)
        assert batch.num_rows == 0
        assert "message_type" in batch.schema.names
        assert "message_index" in batch.schema.names

    def test_nonexistent_file_raises(self, table_b_json, table_d_json):
        """Non-existent file raises an error."""
        with pytest.raises((OSError, RuntimeError)):
            read_bufr_rust(
                "/nonexistent/path.bufr", table_b_json, table_d_json, None, None
            )


class TestReadBufrRustIntegration:
    """Integration tests using read_bufr() with backend='rust'."""

    def test_api_dispatch(self, tmp_path):
        """read_bufr(backend='rust') dispatches to the Rust backend."""
        import earth2bufrio

        empty_file = tmp_path / "empty.bufr"
        empty_file.write_bytes(b"")
        table = earth2bufrio.read_bufr(empty_file, backend="rust")
        assert table.num_rows == 0
        assert "message_type" in table.column_names


class TestReadBufrRustCrossval:
    """Cross-validate Rust vs Python backends on real BUFR files."""

    @pytest.mark.parametrize(
        "filename",
        [
            "profiler_european.bufr",
            "207003.bufr",
            "uegabe.bufr",
        ],
    )
    def test_crossval_row_counts(self, filename, table_b_json, table_d_json):
        """Rust and Python backends produce same row count."""
        import earth2bufrio

        bufr_path = DATA_DIR / filename
        if not bufr_path.exists():
            pytest.skip(f"Test fixture {filename} not found")

        py_table = earth2bufrio.read_bufr(bufr_path, backend="python")
        rust_table = earth2bufrio.read_bufr(bufr_path, backend="rust")

        assert (
            py_table.num_rows == rust_table.num_rows
        ), f"{filename}: Python={py_table.num_rows} rows, Rust={rust_table.num_rows} rows"

    @pytest.mark.parametrize(
        "filename",
        [
            "profiler_european.bufr",
            "207003.bufr",
            "uegabe.bufr",
        ],
    )
    def test_crossval_fixed_columns(self, filename, table_b_json, table_d_json):
        """Rust and Python backends produce same fixed column values."""
        import earth2bufrio

        bufr_path = DATA_DIR / filename
        if not bufr_path.exists():
            pytest.skip(f"Test fixture {filename} not found")

        py_table = earth2bufrio.read_bufr(bufr_path, backend="python")
        rust_table = earth2bufrio.read_bufr(bufr_path, backend="rust")

        for col in ["message_index", "subset_index", "YEAR", "MNTH", "DAYS"]:
            if col in py_table.column_names and col in rust_table.column_names:
                py_vals = py_table.column(col).to_pylist()
                rust_vals = rust_table.column(col).to_pylist()
                assert py_vals == rust_vals, f"{filename}: {col} mismatch"
