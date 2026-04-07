# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WMO BUFR Table B / D management.

Loads the bundled WMO table JSON files and provides a :class:`TableSet`
container for descriptor look-ups during BUFR decoding.  Supports
optional local table overlays keyed by originating centre and local
table version.
"""

from __future__ import annotations

import contextlib
import importlib.resources
import json
from typing import TYPE_CHECKING, Any

from earth2bufrio._types import TableBEntry, TableDEntry

if TYPE_CHECKING:
    from collections.abc import Generator


# Mapping of (centre, local_table_version) to local override file.
_LOCAL_TABLE_B_FILES: dict[tuple[int, int], str] = {
    (98, 101): "table_b_local_98_101.json",
    (98, 1): "table_b_local_98_1.json",
}


def load_table_b() -> dict[int, TableBEntry]:
    """Load the bundled WMO Table B from JSON.

    Reads ``tables/table_b.json`` shipped with the package and returns a
    dictionary mapping integer FXY descriptors to :class:`TableBEntry`
    instances.

    Returns
    -------
    dict[int, TableBEntry]
        Mapping of integer FXY descriptor to its Table B metadata.
    """
    ref = importlib.resources.files("earth2bufrio.tables").joinpath("table_b.json")
    raw: dict[str, Any] = json.loads(ref.read_text(encoding="utf-8"))
    table: dict[int, TableBEntry] = {}
    for key, val in raw.items():
        fxy = int(key)
        table[fxy] = TableBEntry(
            name=val["name"],
            units=val["units"],
            scale=int(val["scale"]),
            reference_value=int(val["reference_value"]),
            bit_width=int(val["bit_width"]),
        )
    return table


def load_table_d() -> dict[int, TableDEntry]:
    """Load the bundled WMO Table D from JSON.

    Reads ``tables/table_d.json`` shipped with the package and returns a
    dictionary mapping integer FXY descriptors to :class:`TableDEntry`
    instances.

    Returns
    -------
    dict[int, TableDEntry]
        Mapping of integer FXY descriptor to its Table D sequence entry.
    """
    ref = importlib.resources.files("earth2bufrio.tables").joinpath("table_d.json")
    raw: dict[str, list[int]] = json.loads(ref.read_text(encoding="utf-8"))
    table: dict[int, TableDEntry] = {}
    for key, members in raw.items():
        fxy = int(key)
        table[fxy] = TableDEntry(descriptors=tuple(int(m) for m in members))
    return table


def load_local_table_b(centre: int, local_version: int) -> dict[int, TableBEntry]:
    """Load a bundled local Table B overlay for the given centre and version.

    Parameters
    ----------
    centre : int
        Originating centre code (e.g. 98 for ECMWF).
    local_version : int
        Local table version number from BUFR Section 1.

    Returns
    -------
    dict[int, TableBEntry]
        Mapping of integer FXY descriptor to the local Table B entry.
        Returns an empty dict if no local table is bundled for the
        given ``(centre, local_version)`` pair.
    """
    filename = _LOCAL_TABLE_B_FILES.get((centre, local_version))
    if filename is None:
        return {}
    ref = importlib.resources.files("earth2bufrio.tables").joinpath(filename)
    raw: dict[str, Any] = json.loads(ref.read_text(encoding="utf-8"))
    table: dict[int, TableBEntry] = {}
    for key, val in raw.items():
        fxy = int(key)
        table[fxy] = TableBEntry(
            name=val["name"],
            units=val["units"],
            scale=int(val["scale"]),
            reference_value=int(val["reference_value"]),
            bit_width=int(val["bit_width"]),
        )
    return table


class TableSet:
    """Container for BUFR Table B and Table D look-ups.

    On construction the bundled WMO tables are loaded.  Entries can be
    overridden via :meth:`add_b` / :meth:`add_d`, and the :meth:`scope`
    context manager allows temporary modifications that are automatically
    rolled back on exit (used for DX-table scoping in PrepBUFR decoding).

    Parameters
    ----------
    centre : int | None, optional
        Originating centre code.  When provided together with
        *local_table_version*, the corresponding local table overlay is
        automatically applied on top of the WMO base table.
    local_table_version : int | None, optional
        Local table version from the BUFR Section 1 header.
    """

    def __init__(
        self,
        centre: int | None = None,
        local_table_version: int | None = None,
    ) -> None:
        self._table_b: dict[int, TableBEntry] = load_table_b()
        self._table_d: dict[int, TableDEntry] = load_table_d()
        if centre is not None and local_table_version is not None:
            local_entries = load_local_table_b(centre, local_table_version)
            self._table_b.update(local_entries)

    def lookup_b(self, fxy: int) -> TableBEntry:
        """Look up a Table B element descriptor.

        Parameters
        ----------
        fxy : int
            Integer FXY descriptor (e.g. ``12001`` for temperature).

        Returns
        -------
        TableBEntry
            The corresponding Table B entry.

        Raises
        ------
        KeyError
            If *fxy* is not present in Table B.
        """
        return self._table_b[fxy]

    def lookup_d(self, fxy: int) -> TableDEntry:
        """Look up a Table D sequence descriptor.

        Parameters
        ----------
        fxy : int
            Integer FXY descriptor (e.g. ``301011``).

        Returns
        -------
        TableDEntry
            The corresponding Table D entry.

        Raises
        ------
        KeyError
            If *fxy* is not present in Table D.
        """
        return self._table_d[fxy]

    def add_b(self, fxy: int, entry: TableBEntry) -> None:
        """Add or override a Table B entry.

        Parameters
        ----------
        fxy : int
            Integer FXY descriptor.
        entry : TableBEntry
            The entry to insert.
        """
        self._table_b[fxy] = entry

    def add_d(self, fxy: int, entry: TableDEntry) -> None:
        """Add or override a Table D entry.

        Parameters
        ----------
        fxy : int
            Integer FXY descriptor.
        entry : TableDEntry
            The entry to insert.
        """
        self._table_d[fxy] = entry

    @contextlib.contextmanager
    def scope(self) -> Generator[None, None, None]:
        """Temporarily scope table modifications.

        Any entries added or overridden inside the ``with`` block are
        reverted when the block exits.  This is used for PrepBUFR DX-table
        overlays that should not persist beyond one message.

        Yields
        ------
        None
        """
        saved_b = dict(self._table_b)
        saved_d = dict(self._table_d)
        try:
            yield
        finally:
            self._table_b = saved_b
            self._table_d = saved_d


def parse_dx_table(data: bytes) -> dict[int, TableBEntry]:
    """Parse a PrepBUFR DX (dictionary) table from raw bytes.

    Parameters
    ----------
    data : bytes
        Raw bytes of the DX table section.

    Returns
    -------
    dict[int, TableBEntry]
        Parsed Table B entries from the DX table.

    Notes
    -----
    This is a stub — full implementation will be added in a later task.
    """
    return {}
