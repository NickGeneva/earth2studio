# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API for reading BUFR files into PyArrow Tables."""

from __future__ import annotations

import functools
import importlib.resources
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from earth2bufrio._arrow import build_table
from earth2bufrio._decoder import decode
from earth2bufrio._descriptors import expand_descriptors
from earth2bufrio._reader import read_messages
from earth2bufrio._section import parse_message
from earth2bufrio._tables import TableSet

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore[import-untyped]

    from earth2bufrio._types import DecodedSubset, ParsedMessage

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=2)
def _load_table_json(filename: str) -> str:
    """Load a bundled table JSON file and return its content as a string.

    Parameters
    ----------
    filename : str
        Filename inside the ``earth2bufrio.tables`` package (e.g.
        ``"table_b.json"``).

    Returns
    -------
    str
        The raw JSON string.
    """
    ref = importlib.resources.files("earth2bufrio.tables").joinpath(filename)
    return ref.read_text(encoding="utf-8")


def read_bufr(
    path: str | Path,
    *,
    mnemonics: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    workers: int = 1,
    backend: str = "python",
) -> pa.Table:
    """Read a BUFR file and return its contents as a PyArrow Table.

    Parameters
    ----------
    path : str | Path
        Path to the BUFR file on disk.
    mnemonics : list[str] | None, optional
        Mnemonic strings to extract.  ``None`` returns all available
        fields.
    filters : dict[str, Any] | None, optional
        Key-value pairs to filter messages.  Supports
        ``"data_category"`` (int) and ``"message_type"`` (str).
    workers : int, optional
        Number of parallel workers for decoding.  ``1`` (default) uses
        the current process; values ``>1`` use a
        :class:`~concurrent.futures.ProcessPoolExecutor`.
    backend : str, optional
        Decoding backend.  ``"python"`` (default) uses the pure-Python
        decoder.  ``"fortran"`` uses the NCEPLIBS-bufr Fortran backend
        (requires ``make fortran`` first).  ``"rust"`` uses the Rust
        backend with Rayon parallelism (requires ``make rust`` first).

    Returns
    -------
    pa.Table
        Wide-format table with one row per subset.  Columns are named
        after BUFR mnemonics.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    BufrDecodeError
        If the file contains malformed BUFR data.
    ValueError
        If *backend* is not ``"python"``, ``"fortran"``, or ``"rust"``.

    Examples
    --------
    >>> import earth2bufrio
    >>> table = earth2bufrio.read_bufr("observations.bufr")
    >>> table.column_names[:3]
    ['message_type', 'message_index', 'subset_index']
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = f"BUFR file not found: {file_path}"
        raise FileNotFoundError(msg)

    if backend == "fortran":
        from earth2bufrio._fortran_backend import read_ncep

        return read_ncep(
            file_path,
            mnemonics=mnemonics,
            filters=filters,
            workers=workers,
        )

    if backend == "rust":
        import pyarrow as pa  # type: ignore[import-untyped]

        from earth2bufrio._lib import read_bufr_rust

        table_b_str = _load_table_json("table_b.json")
        table_d_str = _load_table_json("table_d.json")
        cat_filter = filters.get("data_category") if filters else None
        batch = read_bufr_rust(
            str(file_path), table_b_str, table_d_str, mnemonics, cat_filter
        )
        return pa.Table.from_batches([batch])

    if backend != "python":
        msg = f"Unknown backend: {backend!r}. Use 'python', 'fortran', or 'rust'."
        raise ValueError(msg)

    # --- Python backend ---
    raw_data = file_path.read_bytes()
    if len(raw_data) == 0:
        return build_table([], mnemonics=mnemonics)

    messages = list(read_messages(raw_data))
    if not messages:
        return build_table([], mnemonics=mnemonics)

    parsed_messages: list[tuple[int, ParsedMessage]] = []
    for msg in messages:
        parsed = parse_message(msg)
        ident = parsed.identification

        if ident.data_category == 11:
            logger.debug("Skipping DX table message %d", msg.index)
            continue

        parsed_messages.append((int(msg.index), parsed))

    if filters is not None:
        data_cat_filter = filters.get("data_category")
        if data_cat_filter is not None:
            parsed_messages = [
                (idx, pm)
                for idx, pm in parsed_messages
                if pm.identification.data_category == data_cat_filter
            ]

    if not parsed_messages:
        return build_table([], mnemonics=mnemonics)

    if workers > 1 and len(parsed_messages) > 1:
        decoded_msgs = _decode_parallel(parsed_messages, workers)
    else:
        decoded_msgs = _decode_sequential(parsed_messages)

    rows = _python_subsets_to_rows(decoded_msgs)
    return build_table(rows, mnemonics=mnemonics)


def _python_subsets_to_rows(
    decoded_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Python backend decoded messages to wide-format row dicts.

    Parameters
    ----------
    decoded_messages : list[dict[str, Any]]
        Output from ``_decode_single()``.

    Returns
    -------
    list[dict[str, Any]]
        One dict per subset, keyed by mnemonic name.
    """
    rows: list[dict[str, Any]] = []
    for msg in decoded_messages:
        subsets: list[DecodedSubset] = msg["subsets"]
        for subset_idx, subset in enumerate(subsets):
            row: dict[str, Any] = {
                "message_type": str(msg.get("data_category", "")),
                "message_index": msg["message_index"],
                "subset_index": subset_idx,
                "YEAR": msg["year"],
                "MNTH": msg["month"],
                "DAYS": msg["day"],
                "HOUR": msg["hour"],
                "MINU": msg["minute"],
                "SECO": msg["second"],
            }
            for desc, val in subset.values:
                name = desc.entry.name
                if name in row and isinstance(row[name], list):
                    row[name].append(val)
                elif name in row and name not in _FIXED_ROW_KEYS:
                    row[name] = [row[name], val]
                else:
                    row[name] = val
            rows.append(row)
    return rows


_FIXED_ROW_KEYS = frozenset(
    {
        "message_type",
        "message_index",
        "subset_index",
        "YEAR",
        "MNTH",
        "DAYS",
        "HOUR",
        "MINU",
        "SECO",
    }
)


def _decode_single(
    msg_index: int,
    parsed: ParsedMessage,
) -> dict[str, Any]:
    """Decode a single parsed message into the dict format for build_table.

    A per-message :class:`TableSet` is created using the originating
    centre and local table version from the BUFR identification section,
    ensuring that local table overrides are applied correctly.

    Parameters
    ----------
    msg_index : int
        The original message index in the file.
    parsed : ParsedMessage
        The parsed (but not decoded) BUFR message.

    Returns
    -------
    dict[str, Any]
        Dict with keys expected by :func:`_python_subsets_to_rows`.
    """
    ident = parsed.identification
    desc_section = parsed.data_description

    tables = TableSet(
        centre=ident.originating_center,
        local_table_version=ident.local_table_version,
    )
    expanded = expand_descriptors(desc_section.descriptors, tables)

    subsets = decode(
        expanded,
        parsed.data_bytes,
        ident.num_subsets,
        ident.compressed,
    )

    return {
        "message_index": msg_index,
        "data_category": ident.data_category,
        "year": ident.year,
        "month": ident.month,
        "day": ident.day,
        "hour": ident.hour,
        "minute": ident.minute,
        "second": ident.second,
        "subsets": subsets,
    }


def _decode_sequential(
    parsed_messages: list[tuple[int, ParsedMessage]],
) -> list[dict[str, Any]]:
    """Decode messages sequentially in the current process.

    Parameters
    ----------
    parsed_messages : list[tuple[int, ParsedMessage]]
        List of (message_index, ParsedMessage) tuples.

    Returns
    -------
    list[dict[str, Any]]
        Decoded message dicts ready for :func:`_python_subsets_to_rows`.
    """
    results: list[dict[str, Any]] = []
    for msg_index, parsed in parsed_messages:
        try:
            result = _decode_single(msg_index, parsed)
            results.append(result)
        except Exception:
            logger.warning("Failed to decode message %d, skipping", msg_index)
    return results


def _decode_parallel(
    parsed_messages: list[tuple[int, ParsedMessage]],
    workers: int,
) -> list[dict[str, Any]]:
    """Decode messages using a ProcessPoolExecutor.

    Parameters
    ----------
    parsed_messages : list[tuple[int, ParsedMessage]]
        List of (message_index, ParsedMessage) tuples.
    workers : int
        Number of worker processes.

    Returns
    -------
    list[dict[str, Any]]
        Decoded message dicts ready for :func:`_python_subsets_to_rows`.
    """
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_decode_single, msg_index, parsed): msg_index
            for msg_index, parsed in parsed_messages
        }
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception:
                msg_index = futures[future]
                logger.warning("Failed to decode message %d, skipping", msg_index)
    return results
