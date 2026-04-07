# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor expansion for BUFR decoding.

Expands raw FXY descriptor sequences into a flat list of
:class:`~earth2bufrio._types.ExpandedDescriptor` objects, handling
Table D sequences, replication operators (F=1), and data-description
operators (F=2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from earth2bufrio._types import BufrDecodeError, ExpandedDescriptor, TableBEntry

if TYPE_CHECKING:
    from earth2bufrio._tables import TableSet

logger = logging.getLogger(__name__)

_MAX_DEPTH = 50


@dataclass
class _OperatorState:
    """Mutable state tracking active F=2 operator modifications."""

    width_delta: int = 0
    scale_delta: int = 0


def expand_descriptors(
    raw_ids: tuple[int, ...] | list[int],
    tables: TableSet,
) -> list[ExpandedDescriptor]:
    """Expand a raw FXY descriptor sequence into resolved Table B elements.

    Parameters
    ----------
    raw_ids : tuple[int, ...] | list[int]
        The unexpanded descriptor sequence (integer FXY values).
    tables : TableSet
        Table B / D lookup container.

    Returns
    -------
    list[ExpandedDescriptor]
        Flat list of expanded descriptors with their Table B entries.

    Raises
    ------
    BufrDecodeError
        If recursion exceeds the maximum depth (50) or an unknown
        descriptor is encountered.
    """
    state = _OperatorState()
    return _expand(list(raw_ids), tables, state, depth=0)


def _expand(
    ids: list[int],
    tables: TableSet,
    state: _OperatorState,
    depth: int,
) -> list[ExpandedDescriptor]:
    """Recursively expand descriptors with depth tracking.

    Parameters
    ----------
    ids : list[int]
        Descriptor integers to expand.
    tables : TableSet
        Table B / D lookup container.
    state : _OperatorState
        Current operator modification state.
    depth : int
        Current recursion depth.

    Returns
    -------
    list[ExpandedDescriptor]
        Expanded descriptors.

    Raises
    ------
    BufrDecodeError
        If recursion depth exceeds ``_MAX_DEPTH``.
    """
    if depth > _MAX_DEPTH:
        raise BufrDecodeError(
            f"Descriptor expansion exceeded maximum recursion depth ({_MAX_DEPTH})"
        )

    result: list[ExpandedDescriptor] = []
    idx = 0
    while idx < len(ids):
        fxy = ids[idx]
        f = fxy // 100000
        x = (fxy % 100000) // 1000
        y = fxy % 1000

        if f == 0:
            # Table B element descriptor
            try:
                entry = tables.lookup_b(fxy)
            except KeyError as err:
                raise BufrDecodeError(f"Unknown Table B descriptor: {fxy:06d}") from err

            # Apply active operator modifications
            if state.width_delta != 0 or state.scale_delta != 0:
                entry = TableBEntry(
                    name=entry.name,
                    units=entry.units,
                    scale=entry.scale + state.scale_delta,
                    reference_value=entry.reference_value,
                    bit_width=entry.bit_width + state.width_delta,
                )

            result.append(ExpandedDescriptor(fxy=fxy, entry=entry))
            idx += 1

        elif f == 1:
            # Replication operator: X = number of descriptors, Y = count
            num_descriptors = x
            replication_count = y

            if replication_count == 0:
                # Delayed replication: next descriptor is the replication
                # factor (e.g. 31001), followed by the descriptors to replicate.
                idx += 1
                if idx >= len(ids):
                    raise BufrDecodeError(
                        "Delayed replication missing factor descriptor"
                    )
                factor_fxy = ids[idx]
                # Expand the factor descriptor (it's a Table B element)
                factor_expanded = _expand([factor_fxy], tables, state, depth + 1)
                result.extend(factor_expanded)

                # Collect the next num_descriptors descriptors
                idx += 1
                replicated_ids = ids[idx : idx + num_descriptors]
                if len(replicated_ids) < num_descriptors:
                    raise BufrDecodeError("Delayed replication: not enough descriptors")

                # Expand them once (decoder handles actual count at runtime)
                expanded_group = _expand(replicated_ids, tables, state, depth + 1)
                result.extend(expanded_group)
                idx += num_descriptors
            else:
                # Regular replication: repeat the next X descriptors Y times
                idx += 1
                replicated_ids = ids[idx : idx + num_descriptors]
                if len(replicated_ids) < num_descriptors:
                    raise BufrDecodeError("Replication: not enough descriptors")

                for _ in range(replication_count):
                    expanded_group = _expand(replicated_ids, tables, state, depth + 1)
                    result.extend(expanded_group)
                idx += num_descriptors

        elif f == 2:
            # Operator descriptors
            operator = x
            if operator == 1:
                # 201YYY: change data width
                if y == 0:
                    state.width_delta = 0
                else:
                    state.width_delta = y - 128
            elif operator == 2:
                # 202YYY: change scale
                if y == 0:
                    state.scale_delta = 0
                else:
                    state.scale_delta = y - 128
            else:
                logger.warning(
                    "Unsupported F=2 operator %03d%03d — ignoring", operator, y
                )
            idx += 1

        elif f == 3:
            # Table D sequence descriptor
            try:
                d_entry = tables.lookup_d(fxy)
            except KeyError as err:
                raise BufrDecodeError(f"Unknown Table D descriptor: {fxy:06d}") from err

            expanded = _expand(list(d_entry.descriptors), tables, state, depth + 1)
            result.extend(expanded)
            idx += 1

        else:
            raise BufrDecodeError(f"Unknown descriptor class F={f} in {fxy:06d}")

    return result
