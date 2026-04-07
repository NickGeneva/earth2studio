# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate reference JSON files from pybufrkit for cross-validation.

Usage::

    python tests/generate_references.py

This script decodes each ``.bufr`` fixture in ``tests/data/`` using
pybufrkit and writes a companion ``.ref.json`` file with the structure::

    [
      {
        "message": 0,
        "subset": 0,
        "values": [[fxy_int, value], ...]
      },
      ...
    ]

Numeric values are the decoded physical values, strings are UTF-8, and
missing values are ``null``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pybufrkit.decoder import Decoder

DATA_DIR = Path(__file__).parent / "data"

FIXTURES = [
    "profiler_european.bufr",
    "207003.bufr",
    "uegabe.bufr",
    "g2nd_208.bufr",
    "b005_89.bufr",
]


def _fxy_str_to_int(fxy_str: str) -> int:
    """Convert a pybufrkit descriptor string like '001001' to int 1001.

    Handles special prefixes used by pybufrkit:
    - 'A01001': associated field -> negative sentinel
    - 'F08090': first-order statistics substitution -> negative sentinel
    - 'R01031': replaced/retained -> negative sentinel
    """
    s = str(fxy_str)
    if s[0].isalpha():
        # Special-prefix descriptor (A=associated, F=first-order, R=replaced)
        prefix = s[0]
        numeric = int(s[1:])
        # Encode prefix in the sentinel: A=-1, F=-2, R=-3
        prefix_map = {"A": -1, "F": -2, "R": -3}
        multiplier = prefix_map.get(prefix, -9)
        return multiplier * 1000000 - numeric
    return int(s)


def _coerce_value(val: object) -> float | str | None:
    """Coerce a pybufrkit value to JSON-serialisable form."""
    if val is None:
        return None
    if isinstance(val, bytes):
        # CCITT IA5 string
        return val.decode("ascii", errors="replace").rstrip(" \x00")
    if isinstance(val, float):
        return val
    if isinstance(val, int):
        return float(val)
    # Fallback
    return float(val)


def generate_reference(bufr_path: Path) -> list[dict]:
    """Decode a BUFR file with pybufrkit and return reference data."""
    decoder = Decoder()
    raw = bufr_path.read_bytes()
    msg = decoder.process(raw)

    td = msg.template_data.value
    descriptors_all = td.decoded_descriptors_all_subsets
    values_all = td.decoded_values_all_subsets

    records: list[dict] = []
    for subset_idx in range(len(values_all)):
        descs = descriptors_all[subset_idx]
        vals = values_all[subset_idx]
        pairs: list[list] = []
        for d, v in zip(descs, vals, strict=True):
            fxy = _fxy_str_to_int(d)
            pairs.append([fxy, _coerce_value(v)])
        records.append(
            {
                "message": 0,
                "subset": subset_idx,
                "values": pairs,
            }
        )
    return records


def main() -> None:
    """Generate .ref.json files for all fixtures."""
    decoder_check = Decoder()  # noqa: F841 — verify import works

    for fname in FIXTURES:
        bufr_path = DATA_DIR / fname
        if not bufr_path.exists():
            print(f"SKIP {fname}: file not found", file=sys.stderr)
            continue

        ref_path = bufr_path.with_suffix(".ref.json")
        try:
            records = generate_reference(bufr_path)
            ref_path.write_text(
                json.dumps(records, indent=2, allow_nan=False), encoding="utf-8"
            )
            total_values = sum(len(r["values"]) for r in records)
            print(
                f"OK   {fname}: {len(records)} subsets, {total_values} total values -> {ref_path.name}"
            )
        except Exception as exc:
            print(f"FAIL {fname}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
