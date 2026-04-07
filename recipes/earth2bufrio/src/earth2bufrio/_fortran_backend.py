# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fortran backend for reading NCEP BUFR/PrepBUFR files via ctypes."""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from earth2bufrio._arrow import build_table

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default mnemonic sets per message type
# ---------------------------------------------------------------------------
_DEFAULT_MNEMONICS: dict[str, list[str]] = {
    # Satellite radiance — ATMS
    "NC021203": [
        "SAID",
        "CLATH",
        "CLONH",
        "SAZA",
        "SOZA",
        "IANG",
        "TMBR",
        "CHNM",
        "YEAR",
        "MNTH",
        "DAYS",
        "HOUR",
        "MINU",
        "SECO",
    ],
    # Satellite radiance — AMSU-A
    "NC021023": [
        "SAID",
        "CLAT",
        "CLON",
        "SAZA",
        "SOZA",
        "IANG",
        "TMBR",
        "CHNM",
        "YEAR",
        "MNTH",
        "DAYS",
        "HOUR",
        "MINU",
        "SECO",
    ],
    # Satellite radiance — MHS
    "NC021027": [
        "SAID",
        "CLAT",
        "CLON",
        "SAZA",
        "SOZA",
        "IANG",
        "TMBR",
        "CHNM",
        "YEAR",
        "MNTH",
        "DAYS",
        "HOUR",
        "MINU",
        "SECO",
    ],
    # PrepBUFR conventional observations
    "_PREPBUFR": [
        "YOB",
        "XOB",
        "DHR",
        "ELV",
        "TYP",
        "POB",
        "QOB",
        "TOB",
        "ZOB",
        "UOB",
        "VOB",
        "PWO",
        "TDO",
        "PMO",
        "XDR",
        "YDR",
        "HRDR",
    ],
}

# Replicated mnemonics — use ufbrep instead of ufbint
_REPLICATED_MNEMONICS: frozenset[str] = frozenset(
    {
        "TMBR",
        "CHNM",
        "POB",
        "QOB",
        "TOB",
        "ZOB",
        "UOB",
        "VOB",
        "PWO",
        "TDO",
        "XDR",
        "YDR",
        "HRDR",
    }
)

# Time mnemonics — always extracted for the fixed time columns
_TIME_MNEMONICS: tuple[str, ...] = ("YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO")

# Maximum number of levels per ufbint/ufbrep call
_MAX_LEVELS: int = 255


def _load_lib() -> ctypes.CDLL:
    """Find and load ``libearth2bufrio_fort.so``.

    Raises
    ------
    RuntimeError
        If the shared library cannot be found.
    """
    lib_name = "libearth2bufrio_fort.so"
    pkg_dir = Path(__file__).parent
    candidates = [pkg_dir / lib_name, lib_name]
    for candidate in candidates:
        try:
            return ctypes.CDLL(str(candidate))
        except OSError:
            continue
    msg = f"Could not load {lib_name}. Build with 'make fortran' first."
    raise RuntimeError(msg)


def read_ncep(
    path: str | Path,
    *,
    mnemonics: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    workers: int = 1,
) -> pa.Table:
    """Read an NCEP BUFR/PrepBUFR file using the Fortran backend.

    Parameters
    ----------
    path : str | Path
        Path to the BUFR file.
    mnemonics : list[str] | None, optional
        Mnemonic strings to extract.  ``None`` discovers defaults
        per message type.
    filters : dict[str, Any] | None, optional
        Message-level filters.  Supports ``"message_type"`` (str)
        and ``"data_category"`` (int, not used by Fortran backend).
    workers : int, optional
        Unused (reserved for future multi-file parallelism).

    Returns
    -------
    pa.Table
        Wide-format table with one row per subset.
    """
    lib = _load_lib()

    # Set up ctypes signatures
    lib.e2b_open.restype = ctypes.c_int
    lib.e2b_next_message.restype = ctypes.c_int
    lib.e2b_next_subset.restype = ctypes.c_int
    lib.e2b_read_values.restype = ctypes.c_int
    lib.e2b_read_replicated.restype = ctypes.c_int
    lib.e2b_get_bmiss.restype = ctypes.c_double

    file_path = Path(path)
    filepath_bytes = str(file_path).encode("utf-8")

    # Open the file
    lun = lib.e2b_open(
        filepath_bytes,
        ctypes.c_int(len(filepath_bytes)),
    )
    if lun < 0:
        msg = f"Fortran backend failed to open: {file_path}"
        raise RuntimeError(msg)

    bmiss = lib.e2b_get_bmiss()

    msg_type_filter = None
    if filters is not None:
        msg_type_filter = filters.get("message_type")

    rows: list[dict[str, Any]] = []
    msg_index = 0

    try:
        while True:
            # Read next message
            msg_type_buf = ctypes.create_string_buffer(9)
            msg_type_len = ctypes.c_int(0)
            idate = ctypes.c_int(0)

            ret = lib.e2b_next_message(
                ctypes.c_int(lun),
                msg_type_buf,
                ctypes.byref(msg_type_len),
                ctypes.byref(idate),
            )
            if ret != 0:
                break

            msg_type = msg_type_buf.raw[: msg_type_len.value].decode("ascii").strip()

            # Apply message_type filter
            if msg_type_filter is not None and msg_type != msg_type_filter:
                msg_index += 1
                continue

            # Determine which mnemonics to read for this message
            if mnemonics is not None:
                read_mnems = mnemonics
            elif msg_type in _DEFAULT_MNEMONICS:
                read_mnems = _DEFAULT_MNEMONICS[msg_type]
            else:
                read_mnems = _DEFAULT_MNEMONICS["_PREPBUFR"]

            # Read subsets
            subset_index = 0
            while True:
                ret = lib.e2b_next_subset(ctypes.c_int(lun))
                if ret != 0:
                    break

                row: dict[str, Any] = {
                    "message_type": msg_type,
                    "message_index": msg_index,
                    "subset_index": subset_index,
                }

                # Read each mnemonic
                for mnem in read_mnems:
                    values_buf = (ctypes.c_double * _MAX_LEVELS)()
                    nvalues = ctypes.c_int(0)
                    mnem_bytes = mnem.encode("ascii")

                    if mnem in _REPLICATED_MNEMONICS:
                        lib.e2b_read_replicated(
                            ctypes.c_int(lun),
                            mnem_bytes,
                            ctypes.c_int(len(mnem_bytes)),
                            values_buf,
                            ctypes.c_int(_MAX_LEVELS),
                            ctypes.byref(nvalues),
                        )
                    else:
                        lib.e2b_read_values(
                            ctypes.c_int(lun),
                            mnem_bytes,
                            ctypes.c_int(len(mnem_bytes)),
                            values_buf,
                            ctypes.c_int(_MAX_LEVELS),
                            ctypes.byref(nvalues),
                        )

                    n = nvalues.value
                    if n == 0:
                        continue

                    # Extract values, replacing bmiss with None
                    raw_vals = [
                        None if values_buf[i] >= bmiss else values_buf[i]
                        for i in range(n)
                    ]

                    if n == 1:
                        val = raw_vals[0]
                        if mnem in _TIME_MNEMONICS and val is not None:
                            row[mnem] = int(val)
                        else:
                            row[mnem] = val
                    else:
                        # Multi-level: store as list
                        row[mnem] = raw_vals

                # Ensure time columns exist
                for tc in _TIME_MNEMONICS:
                    if tc not in row:
                        row[tc] = 0

                rows.append(row)
                subset_index += 1

            msg_index += 1
    finally:
        lib.e2b_close(ctypes.c_int(lun))

    return build_table(rows, mnemonics=mnemonics)
