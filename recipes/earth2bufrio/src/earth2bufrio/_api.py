# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API for reading BUFR files into PyArrow Tables."""

from __future__ import annotations

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

    from earth2bufrio._types import ParsedMessage

logger = logging.getLogger(__name__)


def read_bufr(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    workers: int = 1,
) -> pa.Table:
    """Read a BUFR file and return its contents as a PyArrow Table.

    Parameters
    ----------
    path : str | Path
        Path to the BUFR file on disk.
    columns : list[str] | None, optional
        If given, only these columns are included in the returned table.
    filters : dict[str, Any] | None, optional
        Key-value pairs to filter messages.  Currently supports
        ``"data_category"`` (int) to select only messages with a
        matching data category.
    workers : int, optional
        Number of parallel workers for decoding.  ``1`` (default) uses
        the current process; values ``>1`` use a
        :class:`~concurrent.futures.ProcessPoolExecutor`.

    Returns
    -------
    pa.Table
        Long-format table with the 14-column BUFR observation schema.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    BufrDecodeError
        If the file contains malformed BUFR data.

    Examples
    --------
    >>> import earth2bufrio
    >>> table = earth2bufrio.read_bufr("observations.bufr")
    >>> table.schema.names[:3]
    ['message_index', 'subset_index', 'data_category']
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = f"BUFR file not found: {file_path}"
        raise FileNotFoundError(msg)

    raw_data = file_path.read_bytes()
    if len(raw_data) == 0:
        return build_table([], columns=columns)

    # Step 1: Split into messages
    messages = list(read_messages(raw_data))
    if not messages:
        return build_table([], columns=columns)

    # Step 2: Load tables
    tables = TableSet()

    # Step 3: Parse all messages and extract DX tables (data_category=11)
    parsed_messages: list[tuple[int, ParsedMessage]] = []
    for msg in messages:
        parsed = parse_message(msg)
        ident = parsed.identification

        # DX table messages (PrepBUFR): data_category=11
        if ident.data_category == 11:
            logger.debug("Skipping DX table message %d", msg.index)
            continue

        parsed_messages.append((int(msg.index), parsed))

    # Step 4: Apply filters
    if filters is not None:
        data_cat_filter = filters.get("data_category")
        if data_cat_filter is not None:
            parsed_messages = [
                (idx, pm)
                for idx, pm in parsed_messages
                if pm.identification.data_category == data_cat_filter
            ]

    if not parsed_messages:
        return build_table([], columns=columns)

    # Step 5: Expand descriptors + decode
    if workers > 1 and len(parsed_messages) > 1:
        decoded_msgs = _decode_parallel(parsed_messages, tables, workers)
    else:
        decoded_msgs = _decode_sequential(parsed_messages, tables)

    # Step 6: Build Arrow table
    return build_table(decoded_msgs, columns=columns)


def _decode_single(
    msg_index: int,
    parsed: ParsedMessage,
    tables: TableSet,
) -> dict[str, Any]:
    """Decode a single parsed message into the dict format for build_table.

    Parameters
    ----------
    msg_index : int
        The original message index in the file.
    parsed : ParsedMessage
        The parsed (but not decoded) BUFR message.
    tables : TableSet
        The BUFR table set for descriptor look-ups.

    Returns
    -------
    dict[str, Any]
        Dict with keys expected by :func:`~earth2bufrio._arrow.build_table`.
    """
    ident = parsed.identification
    desc_section = parsed.data_description

    # Expand descriptors
    expanded = expand_descriptors(desc_section.descriptors, tables)

    # Decode data section
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
    tables: TableSet,
) -> list[dict[str, Any]]:
    """Decode messages sequentially in the current process.

    Parameters
    ----------
    parsed_messages : list[tuple[int, ParsedMessage]]
        List of (message_index, ParsedMessage) tuples.
    tables : TableSet
        BUFR table set.

    Returns
    -------
    list[dict[str, Any]]
        Decoded message dicts ready for :func:`build_table`.
    """
    results: list[dict[str, Any]] = []
    for msg_index, parsed in parsed_messages:
        try:
            result = _decode_single(msg_index, parsed, tables)
            results.append(result)
        except Exception:
            logger.warning("Failed to decode message %d, skipping", msg_index)
    return results


def _decode_parallel(
    parsed_messages: list[tuple[int, ParsedMessage]],
    tables: TableSet,
    workers: int,
) -> list[dict[str, Any]]:
    """Decode messages using a ProcessPoolExecutor.

    Parameters
    ----------
    parsed_messages : list[tuple[int, ParsedMessage]]
        List of (message_index, ParsedMessage) tuples.
    tables : TableSet
        BUFR table set.
    workers : int
        Number of worker processes.

    Returns
    -------
    list[dict[str, Any]]
        Decoded message dicts ready for :func:`build_table`.
    """
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_decode_single, msg_index, parsed, tables): msg_index
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
