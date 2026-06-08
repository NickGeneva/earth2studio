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

from __future__ import annotations

import json
import os
import random
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from earth2studio.io import IOBackend, ZarrBackend
from earth2studio.models.px import PrognosticModel
from earth2studio.utils.type import CoordSystem

_COMPLETE_NAME = "_complete"
_META_KEY = "earth2studio_restart"
_META_VERSION = 1
_SELECTOR_DIMS = ("time", "ensemble", "batch")


@runtime_checkable
class ReadableIOBackend(IOBackend, Protocol):
    """IO backend protocol for restart stores that can read tensors back."""

    coords: CoordSystem

    def read(
        self,
        coords: CoordSystem,
        array_name: str,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Read an array from the backend."""
        pass


@runtime_checkable
class ModelRNGState(Protocol):
    """Optional protocol for models with private RNG state."""

    def get_rng_state(self) -> Any:
        """Return model-specific RNG state."""
        pass

    def set_rng_state(self, state: Any) -> None:
        """Restore model-specific RNG state."""
        pass


@dataclass(frozen=True)
class RNGState:
    """Captured global and optional model RNG state."""

    python: object | None
    numpy: object | None
    torch_cpu: torch.Tensor | None
    torch_cuda: list[torch.Tensor]
    model: object | None = None


@dataclass(frozen=True)
class RestartState:
    """Tensor state loaded from a restart store."""

    x: torch.Tensor
    coords: CoordSystem
    step: int
    rng: RNGState | None = None


def default_restart_path(name: str, root: str | Path | None = None) -> Path:
    """Return the default local path for a named restart zarr store.

    Parameters
    ----------
    name : str
        Logical restart name. If it has no suffix, ``.zarr`` is appended.
    root : str | Path, optional
        Root directory. Defaults to ``EARTH2STUDIO_CACHE`` or
        ``~/.cache/earth2studio``.

    Returns
    -------
    Path
        Restart store path under ``<root>/restart``.
    """
    cache_root = Path(
        root
        if root is not None
        else os.environ.get(
            "EARTH2STUDIO_CACHE",
            Path.home() / ".cache" / "earth2studio",
        )
    )
    path = cache_root / "restart" / name
    if path.suffix != ".zarr":
        path = path.with_suffix(".zarr")
    return path


def open_restart_io(
    name: str,
    root: str | Path | None = None,
    *,
    overwrite: bool = False,
    chunks: dict[str, int] | None = None,
) -> ZarrBackend:
    """Open a default Zarr restart store.

    Parameters
    ----------
    name : str
        Logical restart store name.
    root : str | Path, optional
        Optional restart root directory.
    overwrite : bool, optional
        Remove an existing restart store before opening it, by default False.
    chunks : dict[str, int], optional
        Optional Zarr chunk overrides.

    Returns
    -------
    ZarrBackend
        Zarr-backed restart IO object.
    """
    path = default_restart_path(name, root)
    if overwrite and path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return ZarrBackend(
        file_name=str(path),
        chunks=chunks if chunks is not None else {"time": 1, "ensemble": 1},
        backend_kwargs={"overwrite": False},
    )


def init_restart(
    restart_io: IOBackend,
    state_coords: CoordSystem,
    steps: np.ndarray,
    *,
    state_name: str = "state",
) -> None:
    """Initialize restart state and progress arrays if missing.

    Parameters
    ----------
    restart_io : IOBackend
        Restart IO backend.
    state_coords : CoordSystem
        Coordinate domain that can hold restart state writes.
    steps : np.ndarray
        Step coordinate values that may be marked complete.
    state_name : str, optional
        Restart state array name, by default ``"state"``.
    """
    if state_name not in restart_io:
        restart_io.add_array(state_coords, state_name)

    if _COMPLETE_NAME not in restart_io:
        complete_coords = _complete_coords(state_coords, steps)
        data = torch.zeros(
            tuple(len(values) for values in complete_coords.values()),
            dtype=torch.int8,
        )
        restart_io.add_array(complete_coords, _COMPLETE_NAME, data=data)

    _write_metadata(restart_io, {"version": _META_VERSION, "state_name": state_name})


def save_restart(
    restart_io: IOBackend,
    x: torch.Tensor,
    coords: CoordSystem,
    *,
    step: int,
    rng: RNGState | None = None,
    state_name: str = "state",
) -> None:
    """Write a restart tensor and mark the step complete.

    The completion marker is written after the state and RNG metadata so interrupted
    writes are not considered valid by :func:`get_last_step`.
    """
    if state_name not in restart_io or _COMPLETE_NAME not in restart_io:
        raise RuntimeError(
            "Restart store is not initialized. Call init_restart() before "
            "save_restart()."
        )
    _validate_tensor_coords(x, coords)
    restart_io.write(x, coords, state_name)

    if rng is not None:
        _write_rng_metadata(restart_io, coords, step, rng)

    marker_coords = _marker_coords(restart_io, coords, step)
    marker = torch.ones(
        tuple(len(values) for values in marker_coords.values()),
        dtype=torch.int8,
    )
    restart_io.write(marker, marker_coords, _COMPLETE_NAME)


def get_last_step(
    restart_io: ReadableIOBackend,
    selector: CoordSystem | None = None,
) -> int | None:
    """Return the latest committed step for a selector.

    Parameters
    ----------
    restart_io : ReadableIOBackend
        Restart IO backend.
    selector : CoordSystem, optional
        Optional coordinate selector, typically containing ``time`` and/or
        ``ensemble``. When multiple selector values are supplied, a step is valid only
        if every selected cell has a completion marker.
    """
    if _COMPLETE_NAME not in restart_io:
        return None

    complete_coords = _read_complete_coords(restart_io)
    read_coords = _select_complete_coords(complete_coords, selector)
    complete, _ = restart_io.read(read_coords, _COMPLETE_NAME, device="cpu")
    complete = complete.to(dtype=torch.bool)

    step_axis = list(read_coords).index("step")
    complete = torch.movedim(complete, step_axis, -1)
    complete = complete.reshape(-1, complete.shape[-1])
    valid = complete.all(dim=0)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return None

    step_values = np.asarray(read_coords["step"])
    return int(step_values[int(valid_indices[-1])])


def load_restart(
    restart_io: ReadableIOBackend,
    coords: CoordSystem,
    *,
    step: int,
    device: torch.device | str = "cpu",
    state_name: str = "state",
) -> RestartState:
    """Load a restart tensor and any stored RNG state."""
    x, read_coords = restart_io.read(coords, state_name, device=device)
    rng = _read_rng_metadata(restart_io, coords, step)
    return RestartState(x=x, coords=read_coords, step=step, rng=rng)


def validate_restart_compatibility(
    state: RestartState,
    prognostic: PrognosticModel,
) -> None:
    """Validate that a restart state can seed a prognostic iterator."""
    _validate_tensor_coords(state.x, state.coords)
    try:
        prognostic.output_coords(state.coords.copy())
    except Exception as exc:
        raise ValueError(
            "Restart state coordinates are not compatible with the prognostic "
            f"model {prognostic.__class__.__name__}. The saved state must be "
            "accepted by prognostic.create_iterator()."
        ) from exc


def capture_rng_state(model: object | None = None) -> RNGState:
    """Capture Python, NumPy, Torch, and optional model RNG state."""
    model_state = None
    if isinstance(model, ModelRNGState):
        model_state = model.get_rng_state()

    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = [state.cpu() for state in torch.cuda.get_rng_state_all()]

    return RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=cuda_states,
        model=model_state,
    )


def restore_rng_state(state: RNGState, model: object | None = None) -> None:
    """Restore Python, NumPy, Torch, and optional model RNG state."""
    if state.python is not None:
        random.setstate(_to_tuple(state.python))
    if state.numpy is not None:
        if isinstance(state.numpy, dict):
            np.random.set_state(_deserialize_numpy_state(state.numpy))
        else:
            np.random.set_state(state.numpy)
    if state.torch_cpu is not None:
        torch.random.set_rng_state(state.torch_cpu.cpu())
    if state.torch_cuda and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all([s.cpu() for s in state.torch_cuda])
    if model is not None and state.model is not None:
        if not isinstance(model, ModelRNGState):
            raise TypeError("Model does not implement set_rng_state")
        model.set_rng_state(state.model)


def _complete_coords(state_coords: CoordSystem, steps: np.ndarray) -> CoordSystem:
    coords: CoordSystem = OrderedDict()
    for dim in _SELECTOR_DIMS:
        if dim in state_coords:
            coords[dim] = np.asarray(state_coords[dim])
    coords["step"] = np.asarray(steps, dtype=np.int64)
    return coords


def _read_complete_coords(restart_io: ReadableIOBackend) -> CoordSystem:
    return OrderedDict(
        (dim, values)
        for dim, values in restart_io.coords.items()
        if dim in _SELECTOR_DIMS or dim == "step"
    )


def _select_complete_coords(
    complete_coords: CoordSystem,
    selector: CoordSystem | None,
) -> CoordSystem:
    read_coords = complete_coords.copy()
    if selector is None:
        return read_coords

    for dim, values in selector.items():
        if dim not in complete_coords:
            raise KeyError(f"Restart selector dimension {dim} is not in progress store")
        values = np.asarray(values)
        if values.size == 0:
            raise ValueError(f"Restart selector dimension {dim} is empty")
        if not np.all(np.isin(values, complete_coords[dim])):
            raise ValueError(
                f"Restart selector values for dimension {dim} are not in progress store"
            )
        read_coords[dim] = values
    return read_coords


def _marker_coords(restart_io: IOBackend, coords: CoordSystem, step: int) -> CoordSystem:
    complete_coords = OrderedDict(
        (dim, values)
        for dim, values in restart_io.coords.items()
        if dim in _SELECTOR_DIMS or dim == "step"
    )
    if "step" not in complete_coords:
        raise KeyError("Restart progress store has no step coordinate")

    marker_coords: CoordSystem = OrderedDict()
    for dim in complete_coords:
        if dim == "step":
            marker_coords[dim] = np.asarray([step], dtype=complete_coords[dim].dtype)
        else:
            if dim not in coords:
                raise KeyError(f"Restart state coordinates missing selector {dim}")
            marker_coords[dim] = np.asarray(coords[dim])
    return marker_coords


def _validate_tensor_coords(x: torch.Tensor, coords: CoordSystem) -> None:
    if x.ndim != len(coords):
        raise ValueError(
            f"Tensor has {x.ndim} dimensions but coords have {len(coords)} dimensions"
        )
    for axis, (dim, values) in enumerate(coords.items()):
        if x.shape[axis] != len(values):
            raise ValueError(
                f"Tensor shape for dim {dim} is {x.shape[axis]} but coords length "
                f"is {len(values)}"
            )


def _metadata(restart_io: IOBackend) -> dict[str, Any]:
    root = getattr(restart_io, "root", None)
    attrs = getattr(root, "attrs", None)
    if attrs is None:
        return {}
    value = attrs.get(_META_KEY, {})
    return dict(value) if isinstance(value, dict) else {}


def _write_metadata(restart_io: IOBackend, update: dict[str, Any]) -> None:
    root = getattr(restart_io, "root", None)
    attrs = getattr(root, "attrs", None)
    if attrs is None:
        return
    meta = _metadata(restart_io)
    meta.update(update)
    attrs[_META_KEY] = meta


def _write_rng_metadata(
    restart_io: IOBackend,
    coords: CoordSystem,
    step: int,
    rng: RNGState,
) -> None:
    meta = _metadata(restart_io)
    rng_meta = dict(meta.get("rng", {}))
    rng_meta[_metadata_key(restart_io, coords, step)] = _serialize_rng_state(rng)
    meta["rng"] = rng_meta
    _write_metadata(restart_io, meta)


def _read_rng_metadata(
    restart_io: IOBackend,
    coords: CoordSystem,
    step: int,
) -> RNGState | None:
    meta = _metadata(restart_io)
    rng_meta = meta.get("rng", {})
    if not isinstance(rng_meta, dict):
        return None
    value = rng_meta.get(_metadata_key(restart_io, coords, step))
    if value is None:
        return None
    return _deserialize_rng_state(value)


def _metadata_key(restart_io: IOBackend, coords: CoordSystem, step: int) -> str:
    selector: dict[str, Any] = {"step": int(step)}
    complete_coords = OrderedDict(
        (dim, values)
        for dim, values in restart_io.coords.items()
        if dim in _SELECTOR_DIMS
    )
    for dim in complete_coords:
        selector[dim] = _serialize_array(np.asarray(coords[dim]))
    return json.dumps(selector, sort_keys=True, separators=(",", ":"))


def _serialize_array(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        dtype = "datetime64[ns]"
        payload = values.astype(dtype).astype(np.int64).tolist()
    elif np.issubdtype(values.dtype, np.timedelta64):
        dtype = "timedelta64[ns]"
        payload = values.astype(dtype).astype(np.int64).tolist()
    else:
        dtype = str(values.dtype)
        payload = values.tolist()
    return {"dtype": dtype, "values": payload}


def _serialize_numpy_state(state: object) -> dict[str, Any]:
    name, keys, pos, has_gauss, cached_gaussian = state  # type: ignore[misc]
    keys_array = np.asarray(keys)
    return {
        "name": name,
        "keys": keys_array.tolist(),
        "keys_dtype": str(keys_array.dtype),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _deserialize_numpy_state(state: dict[str, Any]) -> tuple[Any, ...]:
    return (
        state["name"],
        np.asarray(state["keys"], dtype=np.dtype(state["keys_dtype"])),
        int(state["pos"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    )


def _serialize_rng_state(state: RNGState) -> dict[str, Any]:
    model = state.model
    if model is not None:
        json.dumps(model)

    return {
        "python": _to_jsonable(state.python) if state.python is not None else None,
        "numpy": _serialize_numpy_state(state.numpy) if state.numpy is not None else None,
        "torch_cpu": _serialize_torch_state(state.torch_cpu),
        "torch_cuda": [_serialize_torch_state(s) for s in state.torch_cuda],
        "model": model,
    }


def _deserialize_rng_state(state: dict[str, Any]) -> RNGState:
    torch_cpu = _deserialize_torch_state(state.get("torch_cpu"))
    return RNGState(
        python=_to_tuple(state.get("python")),
        numpy=_deserialize_numpy_state(state["numpy"])
        if state.get("numpy") is not None
        else None,
        torch_cpu=torch_cpu,
        torch_cuda=[
            t for t in (_deserialize_torch_state(s) for s in state.get("torch_cuda", [])) if t is not None
        ],
        model=state.get("model"),
    )


def _serialize_torch_state(state: torch.Tensor | None) -> dict[str, Any] | None:
    if state is None:
        return None
    data = state.detach().cpu().to(dtype=torch.uint8).numpy()
    return {"values": data.tolist()}


def _deserialize_torch_state(state: dict[str, Any] | None) -> torch.Tensor | None:
    if state is None:
        return None
    return torch.as_tensor(state["values"], dtype=torch.uint8)


def _to_jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _to_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_to_tuple(v) for v in value)
    return value
