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

from earth2bufrio._types import (
    BufrDecodeError,
    DelayedReplicationMarker,
    ExpandedDescriptor,
    TableBEntry,
)

if TYPE_CHECKING:
    from earth2bufrio._tables import TableSet

logger = logging.getLogger(__name__)

_MAX_DEPTH = 50

# Union type for expanded items (element descriptors or delayed-replication groups)
ExpandedItem = ExpandedDescriptor | DelayedReplicationMarker


@dataclass
class _OperatorState:
    """Mutable state tracking active F=2 operator modifications."""

    width_delta: int = 0
    scale_delta: int = 0
    assoc_field_width: int = 0
    bitmap_context: bool = False


# Synthetic zero-bit-width Table B entries for QC/substitution operator markers.
# These produce no data in the stream — they are bookkeeping markers that
# pybufrkit also emits as (fxy, 0.0) pairs.
_OPERATOR_MARKER_ENTRY = TableBEntry(
    name="OPERATOR MARKER",
    units="NUMERIC",
    scale=0,
    reference_value=0,
    bit_width=0,
)


def expand_descriptors(
    raw_ids: tuple[int, ...] | list[int],
    tables: TableSet,
) -> list[ExpandedItem]:
    """Expand a raw FXY descriptor sequence into resolved Table B elements.

    Parameters
    ----------
    raw_ids : tuple[int, ...] | list[int]
        The unexpanded descriptor sequence (integer FXY values).
    tables : TableSet
        Table B / D lookup container.

    Returns
    -------
    list[ExpandedItem]
        List of expanded descriptors and delayed replication markers.

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
) -> list[ExpandedItem]:
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
    list[ExpandedItem]
        Expanded descriptors and markers.

    Raises
    ------
    BufrDecodeError
        If recursion depth exceeds ``_MAX_DEPTH``.
    """
    if depth > _MAX_DEPTH:
        raise BufrDecodeError(
            f"Descriptor expansion exceeded maximum recursion depth ({_MAX_DEPTH})"
        )

    result: list[ExpandedItem] = []
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

            # Apply active operator modifications (201/202 only, not 204)
            if state.width_delta != 0 or state.scale_delta != 0:
                entry = TableBEntry(
                    name=entry.name,
                    units=entry.units,
                    scale=entry.scale + state.scale_delta,
                    reference_value=entry.reference_value,
                    bit_width=entry.bit_width + state.width_delta,
                )

            # If associated field width is active and this is not an
            # associated-field significance descriptor (031021) or other
            # class-31 control descriptor, emit a synthetic associated-field
            # descriptor before the real one.
            if state.assoc_field_width > 0 and x != 31:
                assoc_entry = TableBEntry(
                    name="ASSOCIATED FIELD",
                    units="CODE TABLE",
                    scale=0,
                    reference_value=0,
                    bit_width=state.assoc_field_width,
                )
                result.append(ExpandedDescriptor(fxy=999999, entry=assoc_entry))

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
                factor_list = _expand([factor_fxy], tables, state, depth + 1)
                if not factor_list or not isinstance(
                    factor_list[0], ExpandedDescriptor
                ):
                    raise BufrDecodeError(
                        "Delayed replication factor did not expand to a descriptor"
                    )
                factor_desc = factor_list[0]

                # Collect the next num_descriptors descriptors
                idx += 1
                replicated_ids = ids[idx : idx + num_descriptors]
                if len(replicated_ids) < num_descriptors:
                    raise BufrDecodeError("Delayed replication: not enough descriptors")

                # Expand the group once
                expanded_group = _expand(replicated_ids, tables, state, depth + 1)

                # Emit a delayed-replication marker
                result.append(
                    DelayedReplicationMarker(
                        factor_desc=factor_desc,
                        group=tuple(expanded_group),
                    )
                )
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
            elif operator == 4:
                # 204YYY: associated field
                if y == 0:
                    state.assoc_field_width = 0
                else:
                    state.assoc_field_width = y
            elif operator == 7:
                # 207YYY: increase scale, reference value, and data width
                if y == 0:
                    state.width_delta = 0
                    state.scale_delta = 0
                else:
                    state.scale_delta = y
                    state.width_delta = ((10 * y) + 2) // 3
            elif operator in (22, 23, 24, 25, 32, 35, 36, 37):
                # Operators 222-225, 232, 235-237: quality information,
                # substituted/replaced values, and bitmap operators.
                #
                # These are pass-through markers — the expander emits a
                # zero-width synthetic descriptor so the decoder sees them
                # in sequence.  The subsequent descriptors in the
                # unexpanded list (bitmap 031031 indicators, QC/replacement
                # Table B elements) are already regular descriptors and
                # will be expanded normally.
                #
                # 222000 — quality information follows
                # 223000 — substituted operator values follow
                # 224000 — first-order statistical values follow
                # 225000 — difference statistical values follow
                # 232000 — replaced/retained values follow
                # 235000 — cancel backward data reference
                # 236000 — define data present bitmap
                # 237000 — use defined data present bitmap
                # 2XX255 — cancel the corresponding XX context
                result.append(ExpandedDescriptor(fxy=fxy, entry=_OPERATOR_MARKER_ENTRY))
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
