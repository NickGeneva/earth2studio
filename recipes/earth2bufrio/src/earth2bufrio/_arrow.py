# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert decoded BUFR subsets into a PyArrow Table."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from earth2bufrio._types import DecodedSubset

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------
TABLE_SCHEMA = pa.schema(
    [
        pa.field("message_index", pa.int32()),
        pa.field("subset_index", pa.int32()),
        pa.field("data_category", pa.int32()),
        pa.field("latitude", pa.float64()),
        pa.field("longitude", pa.float64()),
        pa.field("time", pa.timestamp("us")),
        pa.field("station_id", pa.string()),
        pa.field("pressure", pa.float64()),
        pa.field("elevation", pa.float64()),
        pa.field("descriptor_id", pa.string()),
        pa.field("descriptor_name", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("units", pa.string()),
        pa.field("quality_mark", pa.int32()),
    ]
)

# ---------------------------------------------------------------------------
# Well-known descriptor sets  (FXY int -> promoted column name)
# ---------------------------------------------------------------------------
_LATITUDE_FXYS: frozenset[int] = frozenset({5001, 5002})
_LONGITUDE_FXYS: frozenset[int] = frozenset({6001, 6002})
_STATION_ID_FXYS: frozenset[int] = frozenset({1015, 1018, 1019})
_PRESSURE_FXYS: frozenset[int] = frozenset({7004, 10004})
_ELEVATION_FXYS: frozenset[int] = frozenset({7001, 7002, 10199})
_TIME_FXYS: frozenset[int] = frozenset({4001, 4002, 4003, 4004, 4005, 4006})
_QUALITY_FXYS: frozenset[int] = frozenset({33007})

_ALL_PROMOTED: frozenset[int] = (
    _LATITUDE_FXYS
    | _LONGITUDE_FXYS
    | _STATION_ID_FXYS
    | _PRESSURE_FXYS
    | _ELEVATION_FXYS
    | _TIME_FXYS
    | _QUALITY_FXYS
)


def _extract_promoted(
    subset: DecodedSubset,
) -> tuple[dict[str, Any], dict[int, int]]:
    """Extract well-known descriptor values from a decoded subset.

    Parameters
    ----------
    subset : DecodedSubset
        A single decoded BUFR subset.

    Returns
    -------
    tuple[dict[str, Any], dict[int, int]]
        A 2-tuple of (promoted column values, time-part FXY-to-int mapping).
    """
    promoted: dict[str, Any] = {
        "latitude": None,
        "longitude": None,
        "station_id": None,
        "pressure": None,
        "elevation": None,
        "quality_mark": None,
    }
    time_parts: dict[int, int] = {}

    for desc, val in subset.values:
        fxy = desc.fxy
        if fxy in _LATITUDE_FXYS and val is not None:
            promoted["latitude"] = float(val) if not isinstance(val, str) else None
        elif fxy in _LONGITUDE_FXYS and val is not None:
            promoted["longitude"] = float(val) if not isinstance(val, str) else None
        elif fxy in _STATION_ID_FXYS and val is not None:
            promoted["station_id"] = str(val)
        elif fxy in _PRESSURE_FXYS and val is not None:
            promoted["pressure"] = float(val) if not isinstance(val, str) else None
        elif fxy in _ELEVATION_FXYS and val is not None:
            promoted["elevation"] = float(val) if not isinstance(val, str) else None
        elif fxy in _TIME_FXYS and val is not None:
            time_parts[fxy] = int(val)
        elif fxy in _QUALITY_FXYS and val is not None:
            promoted["quality_mark"] = int(val)

    return promoted, time_parts


def _build_timestamp(
    time_parts: dict[int, int],
    msg_year: int,
    msg_month: int,
    msg_day: int,
    msg_hour: int,
    msg_minute: int,
    msg_second: int,
) -> datetime.datetime | None:
    """Construct a datetime from descriptor time parts or message-level fields.

    Parameters
    ----------
    time_parts : dict[int, int]
        Mapping of FXY -> integer value for time descriptors found in the subset.
    msg_year : int
        Message-level year.
    msg_month : int
        Message-level month.
    msg_day : int
        Message-level day.
    msg_hour : int
        Message-level hour.
    msg_minute : int
        Message-level minute.
    msg_second : int
        Message-level second.

    Returns
    -------
    datetime.datetime | None
        The constructed timestamp, or ``None`` if the year is 0.
    """
    year = time_parts.get(4001, msg_year)
    month = time_parts.get(4002, msg_month)
    day = time_parts.get(4003, msg_day)
    hour = time_parts.get(4004, msg_hour)
    minute = time_parts.get(4005, msg_minute)
    second = time_parts.get(4006, msg_second)

    if year == 0:
        return None

    try:
        return datetime.datetime(year, month, day, hour, minute, second)
    except (ValueError, OverflowError):
        return None


def build_table(
    decoded_messages: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> pa.Table:
    """Convert decoded BUFR messages into a PyArrow Table.

    Each decoded subset produces one row per non-promoted descriptor. Promoted
    (well-known) descriptor values are replicated across all rows from that
    subset.

    Parameters
    ----------
    decoded_messages : list[dict]
        Each dict has keys: ``message_index``, ``data_category``, ``year``,
        ``month``, ``day``, ``hour``, ``minute``, ``second``, ``subsets``
        (a list of :class:`DecodedSubset`).
    columns : list[str] | None, optional
        If given, only these columns are included in the returned table.

    Returns
    -------
    pa.Table
        Long-format table with the 14-column BUFR observation schema.
    """
    # Column accumulators
    col_message_index: list[int] = []
    col_subset_index: list[int] = []
    col_data_category: list[int] = []
    col_latitude: list[float | None] = []
    col_longitude: list[float | None] = []
    col_time: list[datetime.datetime | None] = []
    col_station_id: list[str | None] = []
    col_pressure: list[float | None] = []
    col_elevation: list[float | None] = []
    col_descriptor_id: list[str] = []
    col_descriptor_name: list[str] = []
    col_value: list[float | None] = []
    col_units: list[str] = []
    col_quality_mark: list[int | None] = []

    for msg in decoded_messages:
        msg_idx: int = msg["message_index"]
        data_cat: int = msg["data_category"]
        subsets: list[DecodedSubset] = msg["subsets"]

        for subset_idx, subset in enumerate(subsets):
            # First pass: extract promoted values
            promoted, time_parts = _extract_promoted(subset)

            timestamp = _build_timestamp(
                time_parts,
                msg["year"],
                msg["month"],
                msg["day"],
                msg["hour"],
                msg["minute"],
                msg["second"],
            )

            # Second pass: create rows for non-promoted descriptors
            for desc, val in subset.values:
                if desc.fxy in _ALL_PROMOTED:
                    continue

                col_message_index.append(msg_idx)
                col_subset_index.append(subset_idx)
                col_data_category.append(data_cat)
                col_latitude.append(promoted["latitude"])
                col_longitude.append(promoted["longitude"])
                col_time.append(timestamp)
                col_station_id.append(promoted["station_id"])
                col_pressure.append(promoted["pressure"])
                col_elevation.append(promoted["elevation"])
                col_descriptor_id.append(f"{desc.fxy:06d}")
                col_descriptor_name.append(desc.entry.name)
                col_units.append(desc.entry.units)
                col_quality_mark.append(promoted["quality_mark"])

                # Value: only floats/ints go in the value column
                if val is None or isinstance(val, str):
                    col_value.append(None)
                else:
                    col_value.append(float(val))

    table = pa.table(
        {
            "message_index": pa.array(col_message_index, type=pa.int32()),
            "subset_index": pa.array(col_subset_index, type=pa.int32()),
            "data_category": pa.array(col_data_category, type=pa.int32()),
            "latitude": pa.array(col_latitude, type=pa.float64()),
            "longitude": pa.array(col_longitude, type=pa.float64()),
            "time": pa.array(col_time, type=pa.timestamp("us")),
            "station_id": pa.array(col_station_id, type=pa.string()),
            "pressure": pa.array(col_pressure, type=pa.float64()),
            "elevation": pa.array(col_elevation, type=pa.float64()),
            "descriptor_id": pa.array(col_descriptor_id, type=pa.string()),
            "descriptor_name": pa.array(col_descriptor_name, type=pa.string()),
            "value": pa.array(col_value, type=pa.float64()),
            "units": pa.array(col_units, type=pa.string()),
            "quality_mark": pa.array(col_quality_mark, type=pa.int32()),
        },
        schema=TABLE_SCHEMA,
    )

    if columns is not None:
        table = table.select(columns)

    return table
