# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOAA-NASA Joint Archive (NNJA) of Observations for Earth System Reanalysis.
#
# Reference: https://psl.noaa.gov/data/nnja_obs/
# Public S3 bucket: s3://noaa-reanalyses-pds/observations/reanalysis/
#
# BUFR decoding is handled by the ``bufr-hound`` Rust-based parser which
# natively extracts DX tables from PrepBUFR files and returns PyArrow
# RecordBatches.  No multiprocessing is needed on the Python side as
# bufr-hound uses Rayon internally for parallel message decoding.

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import s3fs
from loguru import logger

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    managed_session,
    prep_data_inputs,
)
from earth2studio.lexicon import NNJAObsConvLexicon
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import bufr_hound  # noqa: F401
except ImportError:
    OptionalDependencyFailure("data")
    bufr_hound = None  # type: ignore[assignment]
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

NNJA_BUCKET = "noaa-reanalyses-pds"
NNJA_PREFIX = "observations/reanalysis"

# PrepBUFR section-1 dataCategory -> NCEP message-type class string
_PREPBUFR_OBS_TYPES: dict[int, str] = {
    102: "ADPUPA",  # Upper air: radiosondes, pilot balloons, dropsondes
    104: "AIRCFT",  # Aircraft
    105: "SATWND",  # Satellite-derived winds
    107: "VADWND",  # VAD (NEXRAD) winds
    109: "ADPSFC",  # Surface land
    110: "SFCSHP",  # Surface marine
    112: "GPSIPW",  # GPS precipitable water
    113: "SYNDAT",  # Synthetic bogus data
    119: "RASSDA",  # RASS virtual temperature
    121: "ASCATW",  # ASCAT scatterometer winds
}

# ── Schemas ─────────────────────────────────────────────────────────

_NNJA_CONV_SCHEMA = pa.schema(
    [
        E2STUDIO_SCHEMA.field("time"),
        E2STUDIO_SCHEMA.field("pres"),
        E2STUDIO_SCHEMA.field("elev"),
        # NNJA stores PrepBUFR report-type code as uint16 (numeric)
        pa.field("type", pa.uint16(), nullable=True),
        E2STUDIO_SCHEMA.field("class"),
        E2STUDIO_SCHEMA.field("lat"),
        E2STUDIO_SCHEMA.field("lon"),
        E2STUDIO_SCHEMA.field("station"),
        E2STUDIO_SCHEMA.field("station_elev"),
        E2STUDIO_SCHEMA.field("observation"),
        E2STUDIO_SCHEMA.field("variable"),
    ]
)

# ── Async-task dataclasses ──────────────────────────────────────────


@dataclass
class _NNJAConvTask:
    """Async task for a single PrepBUFR cycle file (route ``prepbufr``)."""

    s3_uri: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    var_plan: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = field(
        default_factory=dict
    )


@dataclass
class _NNJAGpsRoTask:
    """Async task for a single gps/gpsro cycle BUFR file (route ``gpsro``)."""

    s3_uri: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    # Map var_name -> (bufr_descriptor_id, modifier)
    var_plan: dict[str, tuple[int, Callable[[pd.DataFrame], pd.DataFrame]]] = field(
        default_factory=dict
    )


class _NNJAObsBase:
    """Shared infrastructure for NNJA DataFrame data sources.

    Subclasses must define ``SOURCE_ID``, ``SCHEMA``, ``MIN_DATE``, and
    implement ``_create_tasks(time_list, variable)`` and
    ``_decode_file(local_path, task)``.
    """

    SOURCE_ID: str
    SCHEMA: pa.Schema
    MIN_DATE: datetime = datetime(1979, 1, 1)

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(0, "m"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 24,
        retries: int = 3,
    ) -> None:
        self._verbose = verbose
        self._cache = cache
        self._async_workers = async_workers
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None
        self.fs: s3fs.S3FileSystem | None = None

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    async def _async_init(self) -> None:
        """Async initialization of S3 filesystem."""
        self.fs = s3fs.S3FileSystem(
            anon=True, client_kwargs={}, asynchronous=True, skip_instance_cache=True
        )

    # ------------------------------------------------------------------
    # Synchronous entry point
    # ------------------------------------------------------------------
    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch observations for a set of timestamps.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Cycle timestamps (UTC). Must align to a 6-hour cycle (00, 06,
            12, 18z); the time tolerance is used to bracket the cycle when
            selecting observations.
        variable : str | list[str] | VariableArray
            Variable ids defined in
            :py:class:`earth2studio.lexicon.NNJAObsConvLexicon`.
        fields : str | list[str] | pa.Schema | None, optional
            Output column subset. ``None`` (default) returns all schema
            fields.

        Returns
        -------
        pd.DataFrame
            Observation DataFrame with columns matching the resolved schema.
        """
        try:
            df = _sync_async(
                self.fetch, time, variable, fields, timeout=self.async_timeout
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

        return df

    # ------------------------------------------------------------------
    # Async fetch (downloads + decode)
    # ------------------------------------------------------------------
    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get data."""
        if self.fs is None:
            await self._async_init()

        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        async_tasks = self._create_tasks(time_list, variable_list)
        file_uri_set = list({task.s3_uri for task in async_tasks})

        async with managed_session(self.fs):
            coros = [
                async_retry(
                    self._fetch_remote_file,
                    uri,
                    retries=self._retries,
                    backoff=1.0,
                    task_timeout=120.0,
                    exceptions=(OSError, IOError, TimeoutError, ConnectionError),
                )
                for uri in file_uri_set
            ]
            await gather_with_concurrency(
                coros,
                max_workers=self._async_workers,
                desc="Fetching NNJA files",
                verbose=(not self._verbose),
            )

        df = self._compile_dataframe(async_tasks, variable_list, schema)
        return df

    # ------------------------------------------------------------------
    # File fetch
    # ------------------------------------------------------------------
    async def _fetch_remote_file(self, path: str) -> None:
        """Download a single remote file into the cache directory."""
        if self.fs is None:
            raise ValueError("File system is not initialized")

        cache_path = self._cache_path(path)
        if pathlib.Path(cache_path).is_file():
            return
        try:
            data = await self.fs._cat_file(path)
            with open(cache_path, "wb") as fh:
                fh.write(data)
        except FileNotFoundError:
            self._handle_missing_file(path)

    def _handle_missing_file(self, path: str) -> None:
        """Handle missing file during fetch. Override in subclasses if a
        warn-only behaviour is preferred."""
        logger.error(f"File {path} not found")
        raise FileNotFoundError(f"File {path} not found")

    # ------------------------------------------------------------------
    # Compile DataFrame
    # ------------------------------------------------------------------
    def _compile_dataframe(
        self,
        async_tasks: list,
        variables: list[str],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Decode each fetched file and concatenate into a single DataFrame."""
        tables: list[pa.Table] = []
        n_tasks = len(async_tasks)
        compile_t0 = time.perf_counter()
        for idx, task in enumerate(async_tasks, start=1):
            local_path = self._cache_path(task.s3_uri)
            if not pathlib.Path(local_path).is_file():
                logger.warning(f"Cached file missing for {task.s3_uri}, skipping")
                continue
            short_uri = task.s3_uri.rsplit("/", 1)[-1]
            logger.info(f"[{self.SOURCE_ID}] decode {idx}/{n_tasks} start: {short_uri}")
            t0 = time.perf_counter()
            try:
                table = self._decode_file(local_path, task)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(f"Failed to decode {local_path}: {exc}")
                continue
            elapsed = time.perf_counter() - t0
            if table is None or table.num_rows == 0:
                logger.info(
                    f"[{self.SOURCE_ID}] decode {idx}/{n_tasks} done : "
                    f"{short_uri} (empty) in {elapsed:.1f}s"
                )
                continue
            logger.info(
                f"[{self.SOURCE_ID}] decode {idx}/{n_tasks} done : "
                f"{short_uri} ({table.num_rows:,} rows) in {elapsed:.1f}s"
            )
            tables.append(table)

        logger.info(
            f"[{self.SOURCE_ID}] compile finished: {len(tables)} non-empty "
            f"frames, total {time.perf_counter() - compile_t0:.1f}s"
        )

        if not tables:
            return pd.DataFrame(
                {name: pd.Series(dtype=object) for name in self.SCHEMA.names}
            )[[name for name in schema.names if name in self.SCHEMA.names]]

        result_table = pa.concat_tables(tables, promote_options="default")
        # Select only the requested schema columns
        select_cols = [
            name for name in schema.names if name in result_table.schema.names
        ]
        result_table = result_table.select(select_cols)
        # Convert to pandas only at the very end
        df = result_table.to_pandas()
        df.attrs["source"] = self.SOURCE_ID
        return df

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _create_tasks(self, time_list: list[datetime], variable: list[str]) -> list:
        raise NotImplementedError("Subclasses must implement _create_tasks.")

    def _decode_file(self, local_path: str, task: Any) -> pa.Table | None:
        raise NotImplementedError("Subclasses must implement _decode_file.")

    # ------------------------------------------------------------------
    # Cycle iteration shared by all NNJA subclasses
    # ------------------------------------------------------------------
    def _cycle_windows(
        self, time_list: list[datetime]
    ) -> dict[datetime, tuple[datetime, datetime]]:
        """Map each unique 6-hour cycle to the union of requested time windows.

        For each ``t`` in ``time_list`` we cover all 6-hour cycles
        whose synoptic time falls within ``[t + tol_lower, t + tol_upper]``.
        Multiple input times that map to the same cycle are merged by
        taking the union of their windows so the cycle file is fetched
        once but filtering keeps observations valid for any of them.
        """
        windows: dict[datetime, tuple[datetime, datetime]] = {}
        for t in time_list:
            tmin = t + self._tolerance_lower
            tmax = t + self._tolerance_upper
            day = tmin.replace(minute=0, second=0, microsecond=0)
            day = day.replace(hour=(day.hour // 6) * 6)
            while day <= tmax:
                existing = windows.get(day)
                windows[day] = (
                    (min(existing[0], tmin), max(existing[1], tmax))
                    if existing is not None
                    else (tmin, tmax)
                )
                day += timedelta(hours=6)
        return windows

    # ------------------------------------------------------------------
    # Time validation / cache / fields
    # ------------------------------------------------------------------
    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate that times align to a 6-hour cycle and are in range."""
        for t in times:
            if t.minute != 0 or t.second != 0 or t.microsecond != 0:
                raise ValueError(
                    f"Requested datetime {t} must be on a whole hour "
                    f"(NNJA cycles are 6-hourly)."
                )
            if t.hour % 6 != 0:
                raise ValueError(
                    f"Requested datetime {t} must align to a 6-hour cycle "
                    f"(00, 06, 12, 18z)."
                )
            if t < cls.MIN_DATE:
                raise ValueError(
                    f"Requested datetime {t} is earlier than {cls.__name__}.MIN_DATE "
                    f"({cls.MIN_DATE.isoformat()})."
                )

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if given date time is available.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check

        Returns
        -------
        bool
            If date time is available
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True

    def _cache_path(self, s3_uri: str) -> str:
        """Deterministic cache path for an S3 URI."""
        sha = hashlib.sha256(s3_uri.encode()).hexdigest()
        return os.path.join(self.cache, sha)

    @property
    def cache(self) -> str:
        """Local cache directory for this data source."""
        cache_location = os.path.join(datasource_cache_root(), "nnja")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_nnja_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Resolve ``fields`` into a validated PyArrow schema subset."""
        if fields is None:
            return cls.SCHEMA
        if isinstance(fields, str):
            fields = [fields]
        if isinstance(fields, pa.Schema):
            for f in fields:
                if f.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{f.name}' not in {cls.__name__} SCHEMA. "
                        f"Available: {cls.SCHEMA.names}"
                    )
                expected = cls.SCHEMA.field(f.name).type
                if f.type != expected:
                    raise TypeError(
                        f"Field '{f.name}' has type {f.type}, expected "
                        f"{expected} from class SCHEMA"
                    )
            return fields
        selected = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not in {cls.__name__} SCHEMA. "
                    f"Available: {cls.SCHEMA.names}"
                )
            selected.append(cls.SCHEMA.field(name))
        return pa.schema(selected)


# ── PrepBUFR mnemonic → variable mapping ─────────────────────────────
# Mnemonics we request from bufr-hound for the PrepBUFR observation levels
_PREPBUFR_HEADER_MNEMONICS = ["SID", "XOB", "YOB", "DHR", "ELV", "TYP"]
_PREPBUFR_OBS_MNEMONICS = ["POB", "TOB", "QOB", "UOB", "VOB"]
_PREPBUFR_ALL_MNEMONICS = _PREPBUFR_HEADER_MNEMONICS + _PREPBUFR_OBS_MNEMONICS

# Lexicon key → bufr-hound mnemonic column name
_LEXICON_KEY_TO_MNEMONIC: dict[str, str] = {
    "TOB": "TOB",
    "QOB": "QOB",
    "POB": "POB",
    "wind::u": "UOB",
    "wind::v": "VOB",
}


def _arrow_table_to_nnja_schema(
    table: pa.Table,
    cycle_time: datetime,
    dt_min: datetime,
    dt_max: datetime,
    var_plan: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]],
) -> pa.Table | None:
    """Transform a flattened bufr-hound PyArrow table to the NNJA long-format schema.

    The input ``table`` has columns from bufr-hound after flatten:
    scalar header columns (SID, XOB, YOB, DHR, ELV, TYP) and per-level
    observation columns (POB, TOB, QOB, UOB, VOB), plus ``_data_category``.

    The output is a long-format table with one row per (observation, variable)
    matching the ``_NNJA_CONV_SCHEMA``.

    Parameters
    ----------
    table : pa.Table
        Flattened bufr-hound output (wide format, one row per obs level).
    cycle_time : datetime
        The 6-hour cycle time for this file.
    dt_min : datetime
        Minimum observation time (inclusive).
    dt_max : datetime
        Maximum observation time (inclusive).
    var_plan : dict
        Maps variable name -> (lexicon_key, modifier_fn).

    Returns
    -------
    pa.Table or None
        Long-format table matching _NNJA_CONV_SCHEMA, or None if empty.
    """
    if table.num_rows == 0:
        return None

    # Compute observation time: cycle_time + DHR (hours offset)
    # DHR is in hours as a float
    cycle_ts = pa.scalar(np.datetime64(cycle_time, "us"), type=pa.timestamp("us"))

    if "DHR" in table.schema.names:
        dhr_col = table.column("DHR")
        # Convert DHR (hours) to microseconds integer, then to duration
        dhr_us = pc.multiply(pc.cast(dhr_col, pa.float64()), 3_600_000_000.0)
        dhr_us_int = pc.cast(pc.round(dhr_us), pa.int64())
        dhr_duration = pc.cast(dhr_us_int, pa.duration("us"))
        obs_time = pc.add(cycle_ts, dhr_duration)
    else:
        # No DHR column — all observations at cycle time
        obs_time = pa.array(
            [cycle_time] * table.num_rows, type=pa.timestamp("us")
        )

    # Filter by time window [dt_min, dt_max]
    ts_min = pa.scalar(np.datetime64(dt_min, "us"), type=pa.timestamp("us"))
    ts_max = pa.scalar(np.datetime64(dt_max, "us"), type=pa.timestamp("us"))
    time_mask = pc.and_(
        pc.greater_equal(obs_time, ts_min),
        pc.less_equal(obs_time, ts_max),
    )
    table = table.filter(time_mask)
    obs_time = pc.filter(obs_time, time_mask)

    if table.num_rows == 0:
        return None

    # Extract header columns
    lat = (
        pc.cast(table.column("YOB"), pa.float32())
        if "YOB" in table.schema.names
        else pa.array([None] * table.num_rows, type=pa.float32())
    )
    lon_raw = (
        pc.cast(table.column("XOB"), pa.float64())
        if "XOB" in table.schema.names
        else pa.array([None] * table.num_rows, type=pa.float64())
    )
    # Normalize longitude to [0, 360)
    lon = pc.cast(
        pc.if_else(
            pc.is_null(lon_raw),
            lon_raw,
            pc.subtract(
                lon_raw,
                pc.multiply(pc.floor(pc.divide(lon_raw, pa.scalar(360.0))), pa.scalar(360.0)),
            ),
        ),
        pa.float32(),
    )

    # Filter invalid lat/lon
    valid_lat = pc.and_(
        pc.greater_equal(lat, pa.scalar(-90.0, pa.float32())),
        pc.less_equal(lat, pa.scalar(90.0, pa.float32())),
    )
    valid_geo = pc.and_(valid_lat, pc.is_valid(lat))
    table = table.filter(valid_geo)
    obs_time = pc.filter(obs_time, valid_geo)
    lat = pc.filter(lat, valid_geo)
    lon = pc.filter(lon, valid_geo)

    if table.num_rows == 0:
        return None

    # Station ID
    if "SID" in table.schema.names:
        sid_col = table.column("SID")
        if pa.types.is_binary(sid_col.type) or pa.types.is_large_binary(sid_col.type):
            station = pc.binary_join_element_wise(
                pc.cast(sid_col, pa.utf8()), pa.scalar("")
            )
        elif pa.types.is_string(sid_col.type) or pa.types.is_large_string(
            sid_col.type
        ):
            station = sid_col
        else:
            station = pc.cast(sid_col, pa.utf8())
    else:
        station = pa.array([None] * table.num_rows, type=pa.utf8())

    # Station elevation
    station_elev = (
        pc.cast(table.column("ELV"), pa.float32())
        if "ELV" in table.schema.names
        else pa.array([None] * table.num_rows, type=pa.float32())
    )

    # Report type
    if "TYP" in table.schema.names:
        typ_col = table.column("TYP")
        report_type = pc.cast(typ_col, pa.uint16())
    else:
        report_type = pa.array([None] * table.num_rows, type=pa.uint16())

    # Obs class from _data_category
    if "_data_category" in table.schema.names:
        data_cat_col = table.column("_data_category")
        # Map data category codes to class strings
        obs_class_list = [
            _PREPBUFR_OBS_TYPES.get(
                cat.as_py() if cat.is_valid else 0, ""
            )
            for cat in data_cat_col
        ]
        obs_class = pa.array(obs_class_list, type=pa.utf8())
    else:
        obs_class = pa.array([""] * table.num_rows, type=pa.utf8())

    # Pressure (POB) in MB — will be converted to Pa later
    pres_mb = (
        pc.cast(table.column("POB"), pa.float32())
        if "POB" in table.schema.names
        else pa.array([None] * table.num_rows, type=pa.float32())
    )

    # Now melt from wide to long: for each requested variable, extract
    # observation values from the corresponding mnemonic column.
    var_tables: list[pa.Table] = []
    for var_name, (lexicon_key, _modifier) in var_plan.items():
        mnemonic = _LEXICON_KEY_TO_MNEMONIC.get(lexicon_key)
        if mnemonic is None:
            continue
        if mnemonic not in table.schema.names:
            continue

        obs_col = pc.cast(table.column(mnemonic), pa.float32())
        # Filter to rows where observation is not null
        valid_mask = pc.is_valid(obs_col)

        if pc.sum(pc.cast(valid_mask, pa.int64())).as_py() == 0:
            continue

        var_table = pa.table(
            {
                "time": pc.filter(obs_time, valid_mask),
                "pres": pc.filter(pres_mb, valid_mask),
                "elev": pa.array(
                    [None] * pc.sum(pc.cast(valid_mask, pa.int64())).as_py(),
                    type=pa.float32(),
                ),
                "type": pc.filter(report_type, valid_mask),
                "class": pc.filter(obs_class, valid_mask),
                "lat": pc.filter(lat, valid_mask),
                "lon": pc.filter(lon, valid_mask),
                "station": pc.filter(station, valid_mask),
                "station_elev": pc.filter(station_elev, valid_mask),
                "observation": pc.filter(obs_col, valid_mask),
                "variable": pa.array(
                    [var_name]
                    * pc.sum(pc.cast(valid_mask, pa.int64())).as_py(),
                    type=pa.utf8(),
                ),
            },
            schema=_NNJA_CONV_SCHEMA,
        )
        var_tables.append(var_table)

    if not var_tables:
        return None

    result = pa.concat_tables(var_tables)
    return result


def _apply_modifiers_arrow(
    table: pa.Table,
    var_plan: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]],
    *,
    convert_pres_mb_to_pa: bool,
) -> pa.Table:
    """Apply unit conversions to the observation column using PyArrow compute.

    Applies the known conversions directly in Arrow without going through
    pandas.  The var_plan modifiers are defined in the lexicon; we replicate
    the same numeric transforms here in PyArrow compute.

    Parameters
    ----------
    table : pa.Table
        Long-format table with 'variable' and 'observation' columns.
    var_plan : dict
        Variable plan from the task.
    convert_pres_mb_to_pa : bool
        Whether to convert the 'pres' column from MB to Pa.

    Returns
    -------
    pa.Table
        Table with converted units.
    """
    if table.num_rows == 0:
        return table

    # Apply per-variable observation modifiers
    obs_col = table.column("observation")
    var_col = table.column("variable")

    # Determine which variables need transformation
    transforms: dict[str, str] = {}  # var_name -> transform_type
    for var_name, (lexicon_key, _mod) in var_plan.items():
        if lexicon_key == "TOB":
            transforms[var_name] = "c_to_k"
        elif lexicon_key == "QOB":
            transforms[var_name] = "mgkg_to_kgkg"
        elif lexicon_key == "POB":
            transforms[var_name] = "mb_to_pa"
        # wind (UOB/VOB) needs no conversion

    if transforms:
        # Process using PyArrow for each variable type needing conversion
        for var_name, transform_type in transforms.items():
            mask = pc.equal(var_col, pa.scalar(var_name, pa.utf8()))
            if transform_type == "c_to_k":
                # °C → K: add 273.15
                new_obs = pc.if_else(
                    mask,
                    pc.cast(
                        pc.add(pc.cast(obs_col, pa.float64()), pa.scalar(273.15)),
                        pa.float32(),
                    ),
                    obs_col,
                )
            elif transform_type == "mgkg_to_kgkg":
                # mg/kg → kg/kg: multiply by 1e-6
                new_obs = pc.if_else(
                    mask,
                    pc.cast(
                        pc.multiply(pc.cast(obs_col, pa.float64()), pa.scalar(1e-6)),
                        pa.float32(),
                    ),
                    obs_col,
                )
            elif transform_type == "mb_to_pa":
                # MB → Pa: multiply by 100
                new_obs = pc.if_else(
                    mask,
                    pc.cast(
                        pc.multiply(pc.cast(obs_col, pa.float64()), pa.scalar(100.0)),
                        pa.float32(),
                    ),
                    obs_col,
                )
            else:
                continue
            obs_col = new_obs

        table = table.set_column(
            table.schema.get_field_index("observation"), "observation", obs_col
        )

    # Convert pres column from MB to Pa
    if convert_pres_mb_to_pa and "pres" in table.schema.names:
        pres_col = table.column("pres")
        pres_pa = pc.cast(
            pc.multiply(pc.cast(pres_col, pa.float64()), pa.scalar(100.0)),
            pa.float32(),
        )
        table = table.set_column(
            table.schema.get_field_index("pres"), "pres", pres_pa
        )

    return table


@check_optional_dependencies()
class NNJAObsConv(_NNJAObsBase):
    """NNJA conventional (in-situ + GPS RO) observational data source. NOAA-NASA Joint
    Archive (NNJA) of Observations for Earth System Reanalysis is an archive ideal for
    developing observation-driven weather forecasting tools, as it includes a wide
    cross-section of data from a plethora of sensing platforms (satellites, surface
    stations, weather balloons, and more) and features data from 1979 to the present.

    Parameters
    ----------
    source : {"prepbufr", "convbufr", "prepbufr.acft_profiles"}, optional
        Which encoding family of the NNJA conventional archive to read,
        by default ``"prepbufr"``.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single
        value (symmetric ± window) or a tuple ``(lower, upper)`` for
        asymmetric windows, by default ``np.timedelta64(0, 'm')``.
    cache : bool, optional
        Cache downloaded files in the local filesystem cache, by default True.
    verbose : bool, optional
        Show progress bars, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the async fetch, by default 600.
    async_workers : int, optional
        Maximum number of concurrent async fetch tasks, by default 24.
    retries : int, optional
        Number of retry attempts per failed fetch task with exponential
        backoff, by default 3.

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.brightband.com/data/nnja-ai/
    - https://psl.noaa.gov/data/nnja_obs/
    - https://registry.opendata.aws/noaa-reanalyses-obs/
    - https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/document.htm

    Badges
    ------
    region:global dataclass:observation product:wind product:temp product:atmos product:insitu
    """

    SOURCE_ID = "earth2studio.data.NNJAObsConv"
    SCHEMA = _NNJA_CONV_SCHEMA
    MIN_DATE = datetime(1979, 1, 1)

    VALID_SOURCES = frozenset(["prepbufr", "convbufr", "prepbufr.acft_profiles"])

    def __init__(
        self,
        source: str = "prepbufr",
        time_tolerance: TimeTolerance = np.timedelta64(0, "m"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 24,
        retries: int = 3,
    ) -> None:
        if source not in self.VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Valid sources: {sorted(self.VALID_SOURCES)}"
            )
        self._source = source
        super().__init__(
            time_tolerance=time_tolerance,
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
            async_workers=async_workers,
            retries=retries,
        )

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------
    def _create_tasks(self, time_list: list[datetime], variable: list[str]) -> list:
        # Partition variables by lexicon route prefix:
        #   "prepbufr::..." -> conv/prepbufr/ tasks (PrepBUFR decoder)
        #   "gpsro::..."    -> gps/gpsro/ tasks (GPS RO BUFR decoder)
        prepbufr_plan: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = (
            {}
        )
        gpsro_plan: dict[str, tuple[int, Callable[[pd.DataFrame], pd.DataFrame]]] = {}

        for v in variable:
            try:
                source_key, modifier = NNJAObsConvLexicon[v]  # type: ignore[misc]
            except KeyError:
                logger.error(f"Variable id '{v}' not found in NNJAObsConvLexicon")
                raise
            route, _, rest = source_key.partition("::")
            if route == "prepbufr":
                prepbufr_plan[v] = (rest, modifier)
            elif route == "gpsro":  # pragma: no cover - GPS RO not yet in lexicon
                try:
                    desc_id = int(rest)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid gpsro lexicon entry '{source_key}' for {v}: "
                        f"expected an integer BUFR descriptor id"
                    ) from exc
                gpsro_plan[v] = (desc_id, modifier)
            else:
                raise ValueError(
                    f"Unknown route '{route}' in NNJAObsConvLexicon entry "
                    f"'{source_key}' for variable '{v}' (expected 'prepbufr' or 'gpsro')"
                )

        # Build one task per unique cycle file; when multiple requested
        # times map to the same cycle the task's window is the union of
        # those time windows (see ``_NNJAObsBase._cycle_windows``).
        windows = self._cycle_windows(time_list) if prepbufr_plan or gpsro_plan else {}
        tasks: list = []
        for cycle_dt, (tmin, tmax) in windows.items():
            if prepbufr_plan:
                tasks.append(
                    _NNJAConvTask(
                        s3_uri=self._build_prepbufr_uri(cycle_dt),
                        datetime_file=cycle_dt,
                        datetime_min=tmin,
                        datetime_max=tmax,
                        var_plan=prepbufr_plan,
                    )
                )
            if gpsro_plan:  # pragma: no cover - GPS RO not yet in lexicon
                tasks.append(
                    _NNJAGpsRoTask(
                        s3_uri=self._build_gpsro_uri(cycle_dt),
                        datetime_file=cycle_dt,
                        datetime_min=tmin,
                        datetime_max=tmax,
                        var_plan=gpsro_plan,
                    )
                )
        return tasks

    def _build_prepbufr_uri(self, cycle: datetime) -> str:
        """Build the NNJA S3 URI for a single PrepBUFR cycle."""
        year_key = cycle.strftime("%Y")
        month_key = cycle.strftime("%m")
        date_key = cycle.strftime("%Y%m%d")
        hour_key = f"{cycle.hour:02d}"
        return (
            f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/conv/{self._source}/"
            f"{year_key}/{month_key}/{self._source}/"
            f"gdas.{date_key}.t{hour_key}z.{self._source}.nr"
        )

    def _build_gpsro_uri(self, cycle: datetime) -> str:
        """Build the NNJA S3 URI for a single gps/gpsro cycle file."""
        year_key = cycle.strftime("%Y")
        month_key = cycle.strftime("%m")
        date_key = cycle.strftime("%Y%m%d")
        hour_key = f"{cycle.hour:02d}"
        return (
            f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/gps/gpsro/"
            f"{year_key}/{month_key}/bufr/"
            f"gdas.{date_key}.t{hour_key}z.gpsro.tm00.bufr_d"
        )

    # Back-compat alias used by tests that targeted the v1 method name.
    def _build_uri(self, cycle: datetime) -> str:
        return self._build_prepbufr_uri(cycle)

    def _handle_missing_file(self, path: str) -> None:
        """Warn instead of raising on missing NNJA cycle files.

        NNJA does not guarantee every cycle/sub-archive combination
        exists (e.g. the ``gps/gpsro/`` archive only goes back to the
        early 2000s, and individual cycles can be absent). Returning a
        partial DataFrame is more useful than aborting a multi-cycle
        request because of one missing file.
        """
        logger.warning(f"NNJA conventional file {path} not found, skipping")

    # ------------------------------------------------------------------
    # File decode (dispatch by task type)
    # ------------------------------------------------------------------
    def _decode_file(
        self, local_path: str, task: _NNJAConvTask | _NNJAGpsRoTask
    ) -> pa.Table | None:
        if isinstance(task, _NNJAGpsRoTask):
            return self._decode_gpsro_file(local_path, task)
        return self._decode_prepbufr_file(local_path, task)

    def _decode_prepbufr_file(
        self, local_path: str, task: _NNJAConvTask
    ) -> pa.Table | None:
        """Decode a PrepBUFR cycle file into a PyArrow Table using bufr-hound.

        Uses the Rust-based bufr-hound parser which handles DX-table
        extraction and parallel message decoding internally.
        """
        import bufr_hound

        # Determine which mnemonics we need based on the var_plan
        needed_mnemonics: set[str] = set(_PREPBUFR_HEADER_MNEMONICS)
        needed_mnemonics.add("POB")  # Always need pressure for the schema
        for lexicon_key, _mod in task.var_plan.values():
            mnemonic = _LEXICON_KEY_TO_MNEMONIC.get(lexicon_key)
            if mnemonic:
                needed_mnemonics.add(mnemonic)

        logger.info(
            f"[NNJAObsConv prepbufr] cycle={task.datetime_file:%Y-%m-%d %H:%MZ} "
            f"mnemonics={sorted(needed_mnemonics)} "
            f"categories={sorted(_PREPBUFR_OBS_TYPES.keys())}"
        )
        decode_t0 = time.perf_counter()

        # Call bufr-hound: returns list of RecordBatches, one per data category
        batches = bufr_hound.read_prepbufr(
            local_path,
            mnemonics=sorted(needed_mnemonics),
            data_category_filter=list(_PREPBUFR_OBS_TYPES.keys()),
            flatten=True,
        )

        if not batches:
            logger.info(
                f"[NNJAObsConv prepbufr] cycle={task.datetime_file:%Y-%m-%d %H:%MZ} "
                f"no data batches returned"
            )
            return None

        # Convert RecordBatches to a single Table
        # Each batch is already flattened (one row per obs level)
        tables = [
            pa.Table.from_batches([batch])
            for batch in batches
            if batch.num_rows > 0
        ]

        if not tables:
            return None

        wide_table = pa.concat_tables(tables, promote_options="default")
        decode_elapsed = time.perf_counter() - decode_t0
        logger.info(
            f"[NNJAObsConv prepbufr] cycle={task.datetime_file:%Y-%m-%d %H:%MZ} "
            f"bufr-hound decoded {wide_table.num_rows:,} raw rows in "
            f"{decode_elapsed:.1f}s"
        )

        # Transform wide format to NNJA long format
        result = _arrow_table_to_nnja_schema(
            wide_table,
            cycle_time=task.datetime_file,
            dt_min=task.datetime_min,
            dt_max=task.datetime_max,
            var_plan=task.var_plan,
        )

        if result is None:
            return None

        # Apply unit conversions
        result = _apply_modifiers_arrow(
            result, task.var_plan, convert_pres_mb_to_pa=True
        )

        logger.info(
            f"[NNJAObsConv prepbufr] cycle={task.datetime_file:%Y-%m-%d %H:%MZ} "
            f"final schema rows: {result.num_rows:,}"
        )
        return result

    def _decode_gpsro_file(  # pragma: no cover - GPS RO not yet in lexicon
        self, local_path: str, task: _NNJAGpsRoTask
    ) -> pa.Table | None:
        """Decode a single NNJA gps/gpsro cycle BUFR file into a PyArrow Table.

        GPS RO support is not yet active in the lexicon. This is a
        placeholder that will be implemented when GPS RO variables are
        added to `NNJAObsConvLexicon`.
        """
        import bufr_hound

        logger.info(
            f"[NNJAObsConv gpsro]    cycle={task.datetime_file:%Y-%m-%d %H:%MZ} "
            f"decoding gpsro file"
        )
        decode_t0 = time.perf_counter()

        # GPS RO files use standard BUFR (not PrepBUFR) encoding
        batches = bufr_hound.read_bufr(
            local_path,
            flatten=True,
        )

        if not batches:
            return None

        # TODO: Implement GPS RO schema transformation when lexicon entries
        # are added. For now return None.
        logger.warning(
            f"[NNJAObsConv gpsro] GPS RO decoding not yet implemented with "
            f"bufr-hound (decoded {sum(b.num_rows for b in batches)} rows in "
            f"{time.perf_counter() - decode_t0:.1f}s)"
        )
        return None
