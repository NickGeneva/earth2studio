# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate eccodes reference JSON files for cross-validation.

Usage::

    .venv/bin/python3 recipes/earth2bufrio/tests/generate_eccodes_references.py

This script decodes each ``.bufr`` fixture in ``tests/data/`` using
eccodes and writes a companion ``.eccodes.ref.json`` file with the
structure::

    {
      "messages": [
        {
          "message_index": 0,
          "num_subsets": 1,
          "compressed": false,
          "expanded_codes": [1001, 1002, ...],
          "expanded_abbreviations": ["blockNumber", "stationNumber", ...],
          "subsets": [
            {
              "subset_index": 0,
              "values": {
                "blockNumber": 8.0,
                "windDirection": [51.0, 103.0, ...],
                ...
              }
            }
          ]
        }
      ]
    }

For uncompressed messages (1 subset), values are scalar.  For compressed
messages with N subsets, eccodes may return either a scalar (all subsets
identical) or an array of length N.  We split these into per-subset dicts
so the cross-validation test can compare one subset at a time.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import eccodes

DATA_DIR = Path(__file__).parent / "data"

FIXTURES = [
    "profiler_european.bufr",
    "207003.bufr",
    "uegabe.bufr",
    "g2nd_208.bufr",
    "b005_89.bufr",
]


def _read_key_value(
    msgid: int,
    key: str,
) -> float | str | list[float] | list[str] | None:
    """Read a single key from an unpacked eccodes BUFR message.

    Tries numeric (double array) first, then string, then scalar.
    Returns ``None`` for keys that cannot be read (replication markers,
    operator descriptors, etc.).
    """
    try:
        sz = eccodes.codes_get_size(msgid, key)
    except eccodes.KeyValueNotFoundError:
        return None

    if sz == 0:
        return None

    # Check native type to handle string descriptors (CCITT IA5) correctly.
    # Without this, codes_get_double_array may succeed on string keys and
    # return a spurious numeric value.
    try:
        native_type = eccodes.codes_get_native_type(msgid, key)
    except Exception:
        native_type = None

    if native_type is str:
        # String descriptor — read as string
        try:
            val = eccodes.codes_get_string(msgid, key)
            if val is not None:
                return val.rstrip(" \x00")
        except Exception:
            pass
        return None

    # Try double array for numeric descriptors
    try:
        arr = eccodes.codes_get_double_array(msgid, key)
        # Replace NaN / BUFR missing with None
        cleaned: list[float | None] = []
        for v in arr:
            if math.isnan(v) or v > 1.0e30:
                cleaned.append(None)
            else:
                cleaned.append(float(v))
        if len(cleaned) == 1:
            return cleaned[0]
        return cleaned  # type: ignore[return-value]
    except Exception:
        pass

    # Try string as fallback
    try:
        val = eccodes.codes_get_string(msgid, key)
        if val is not None:
            return val
    except Exception:
        pass

    # Try generic get
    try:
        val = eccodes.codes_get(msgid, key)
        if isinstance(val, int | float):
            return float(val)
        if isinstance(val, str):
            return val
        return None
    except Exception:
        return None


def _sanitize_for_json(val: object) -> object:
    """Replace non-JSON-serialisable float values with None."""
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(val, list):
        return [_sanitize_for_json(v) for v in val]
    return val


def generate_reference(bufr_path: Path) -> dict:
    """Decode a BUFR file with eccodes and return reference data.

    Parameters
    ----------
    bufr_path : Path
        Path to the .bufr file.

    Returns
    -------
    dict
        Reference dict with ``"messages"`` key.
    """
    messages = []

    with bufr_path.open("rb") as f:
        msg_index = 0
        while True:
            msgid = eccodes.codes_bufr_new_from_file(f)
            if msgid is None:
                break

            try:
                eccodes.codes_set(msgid, "unpack", 1)

                num_subsets = eccodes.codes_get(msgid, "numberOfSubsets")
                compressed = eccodes.codes_get(msgid, "compressedData")

                # Get expanded descriptor info
                raw_codes = eccodes.codes_get_long_array(msgid, "expandedDescriptors")
                expanded_codes = (
                    raw_codes.tolist()
                    if hasattr(raw_codes, "tolist")
                    else list(raw_codes)
                )
                raw_abbrevs = eccodes.codes_get_string_array(
                    msgid, "expandedAbbreviations"
                )
                expanded_abbrevs = list(raw_abbrevs)

                # Build ranked key names (eccodes uses #N for duplicates)
                key_counts: dict[str, int] = {}
                ranked_keys: list[str] = []
                for abbrev in expanded_abbrevs:
                    if abbrev in key_counts:
                        key_counts[abbrev] += 1
                        ranked_keys.append(f"{abbrev}#{key_counts[abbrev]}")
                    else:
                        key_counts[abbrev] = 1
                        ranked_keys.append(abbrev)

                # Read all values from the full (unpacked) message
                full_values: dict[str, object] = {}
                for ranked_key in ranked_keys:
                    val = _read_key_value(msgid, ranked_key)
                    full_values[ranked_key] = _sanitize_for_json(val)

                # Split into per-subset dicts
                subsets: list[dict] = []
                for s in range(num_subsets):
                    subset_vals: dict[str, object] = {}
                    for ranked_key in ranked_keys:
                        raw = full_values[ranked_key]
                        if isinstance(raw, list) and len(raw) == num_subsets:
                            # Per-subset array: pick the s-th element
                            subset_vals[ranked_key] = raw[s]
                        else:
                            # Scalar or replicated array — same for all subsets
                            subset_vals[ranked_key] = raw
                    subsets.append(
                        {
                            "subset_index": s,
                            "values": subset_vals,
                        }
                    )

                messages.append(
                    {
                        "message_index": msg_index,
                        "num_subsets": num_subsets,
                        "compressed": bool(compressed),
                        "expanded_codes": expanded_codes,
                        "expanded_abbreviations": expanded_abbrevs,
                        "subsets": subsets,
                    }
                )
            finally:
                eccodes.codes_release(msgid)

            msg_index += 1

    return {"messages": messages}


def main() -> None:
    """Generate .eccodes.ref.json files for all fixtures."""
    for fname in FIXTURES:
        bufr_path = DATA_DIR / fname
        if not bufr_path.exists():
            print(f"SKIP {fname}: file not found", file=sys.stderr)
            continue

        ref_path = DATA_DIR / f"{Path(fname).stem}.eccodes.ref.json"
        try:
            ref_data = generate_reference(bufr_path)
            n_msgs = len(ref_data["messages"])
            total_subsets = sum(len(m["subsets"]) for m in ref_data["messages"])
            total_keys = sum(
                len(s["values"]) for m in ref_data["messages"] for s in m["subsets"]
            )
            ref_path.write_text(
                json.dumps(ref_data, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            print(
                f"OK   {fname}: {n_msgs} msgs, {total_subsets} subsets, "
                f"{total_keys} total key-values -> {ref_path.name}"
            )
        except Exception as exc:
            print(f"FAIL {fname}: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
