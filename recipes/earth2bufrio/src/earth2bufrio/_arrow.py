# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert decoded BUFR data into wide-format PyArrow Tables."""

from __future__ import annotations

from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Fixed columns — always present in every table
# ---------------------------------------------------------------------------
_FIXED_COLUMNS = ("message_type", "message_index", "subset_index")
_TIME_COLUMNS = ("YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO")
_ALL_FIXED = _FIXED_COLUMNS + _TIME_COLUMNS


def build_table(
    rows: list[dict[str, Any]],
    mnemonics: list[str] | None = None,
) -> pa.Table:
    """Convert rows of mnemonic-keyed data into a wide-format PyArrow Table.

    Each dict in *rows* represents one BUFR subset.  Keys include the
    fixed columns (``message_type``, ``message_index``, ``subset_index``,
    ``YEAR``, ``MNTH``, ``DAYS``, ``HOUR``, ``MINU``, ``SECO``) plus
    one key per extracted mnemonic.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        One dict per subset.  Values are scalars (``float``, ``int``,
        ``str``) or lists (replicated data).
    mnemonics : list[str] | None, optional
        If given, only these mnemonic columns are included (fixed columns
        are always present).

    Returns
    -------
    pa.Table
        Wide-format table with one row per subset.
    """
    if not rows:
        schema = pa.schema(
            [
                pa.field("message_type", pa.string()),
                pa.field("message_index", pa.int32()),
                pa.field("subset_index", pa.int32()),
                pa.field("YEAR", pa.int32()),
                pa.field("MNTH", pa.int32()),
                pa.field("DAYS", pa.int32()),
                pa.field("HOUR", pa.int32()),
                pa.field("MINU", pa.int32()),
                pa.field("SECO", pa.int32()),
            ]
        )
        return pa.table(
            {
                name: pa.array([], type=f.type)
                for name, f in zip(schema.names, schema, strict=True)
            },
            schema=schema,
        )

    # Discover all mnemonic keys across rows (preserving insertion order)
    mnemonic_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in _ALL_FIXED and key not in seen:
                seen.add(key)
                mnemonic_keys.append(key)

    # Apply mnemonics filter
    if mnemonics is not None:
        allowed = set(mnemonics)
        mnemonic_keys = [k for k in mnemonic_keys if k in allowed]

    # Determine type for each mnemonic column by inspecting first non-None value
    col_types: dict[str, pa.DataType] = {}
    for key in mnemonic_keys:
        for row in rows:
            val = row.get(key)
            if val is not None:
                if isinstance(val, list):
                    # Check inner type
                    if val and isinstance(val[0], str):
                        col_types[key] = pa.list_(pa.string())
                    else:
                        col_types[key] = pa.list_(pa.float64())
                elif isinstance(val, str):
                    col_types[key] = pa.string()
                else:
                    col_types[key] = pa.float64()
                break
        else:
            # All None — default to float64
            col_types[key] = pa.float64()

    # Build column arrays
    col_data: dict[str, list[Any]] = {name: [] for name in _ALL_FIXED}
    for key in mnemonic_keys:
        col_data[key] = []

    for row in rows:
        col_data["message_type"].append(row.get("message_type", ""))
        col_data["message_index"].append(row.get("message_index", 0))
        col_data["subset_index"].append(row.get("subset_index", 0))
        for tc in _TIME_COLUMNS:
            col_data[tc].append(row.get(tc, 0))
        for key in mnemonic_keys:
            col_data[key].append(row.get(key))

    # Build schema
    fields: list[pa.Field] = [
        pa.field("message_type", pa.string()),
        pa.field("message_index", pa.int32()),
        pa.field("subset_index", pa.int32()),
    ]
    for tc in _TIME_COLUMNS:
        fields.append(pa.field(tc, pa.int32()))
    for key in mnemonic_keys:
        fields.append(pa.field(key, col_types[key]))

    schema = pa.schema(fields)

    # Build arrays
    arrays: dict[str, pa.Array] = {}
    arrays["message_type"] = pa.array(col_data["message_type"], type=pa.string())
    arrays["message_index"] = pa.array(col_data["message_index"], type=pa.int32())
    arrays["subset_index"] = pa.array(col_data["subset_index"], type=pa.int32())
    for tc in _TIME_COLUMNS:
        arrays[tc] = pa.array(col_data[tc], type=pa.int32())
    for key in mnemonic_keys:
        arrays[key] = pa.array(col_data[key], type=col_types[key])

    return pa.table(arrays, schema=schema)
