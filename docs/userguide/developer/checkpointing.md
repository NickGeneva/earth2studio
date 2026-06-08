(checkpointing_design)=

# Checkpointing Design

This page is a design note for first-class restart support in Earth2Studio inference
workflows. It focuses on a small, IO-backed utility layer that custom workflows can use
directly, with optional integration into the built-in workflows in {mod}`earth2studio.run`.

## Existing Shape

The package already has most of the right primitives:

- Forecast state moves through the package as explicit `(x, coords)` pairs.
- `IOBackend` is a protocol, not a base class, and built-in IO backends expose simple
  `add_array` and `write` methods.
- `ZarrBackend` supports partial writes and `read`, and is already recommended for
  datetime-compatible output.
- The built-in workflows in `earth2studio/run.py` are thin loops over
  `prognostic.create_iterator(x, coords)`.
- The eval recipe already has item-level resume markers for `(time, ensemble)` work
  items, but that only skips completed forecasts. It does not restart from a partial
  rollout.

The main subtlety is that the tensor yielded by a prognostic iterator is not always the
same as the iterator's private next-input state. For simple re-entrant models these are
equivalent enough to resume. For models with sliding windows, multi-frame inputs, or
private sampler state, the yielded output may not be sufficient. The restart utility
should make that distinction explicit instead of silently writing an unusable restart.

## Goals

- Keep checkpointing off by default and cheap when disabled.
- Let users enable it with one extra IO object and an interval.
- Store restart data through the same IO/Zarr tooling users already inspect.
- Treat custom workflows as the primary surface; built-in workflows should use the same
  helper functions a user can call by hand.
- Preserve enough progress metadata to detect the last committed state after a crash.
- Preserve global RNG state, and allow model-specific RNG hooks without requiring every
  model to inherit from a new base class.
- Validate compatibility before resume and fail loudly when a restart cannot be used.

## Non-Goals For The First Pass

- No optimizer/training checkpointing.
- No restart inside one model forward call or inside one diffusion sampling call.
- No checkpoint rotation or stale checkpoint cleanup.
- No distributed checkpoint coordinator. The store layout should support disjoint writes
  from ranks, but rank synchronization belongs to recipes or launch infrastructure.

## Proposed Module

Add `earth2studio/utils/checkpoint.py` with a small function-first API:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from earth2studio.io import IOBackend, ZarrBackend
from earth2studio.models.px import PrognosticModel
from earth2studio.utils.type import CoordSystem


class ReadableIOBackend(IOBackend, Protocol):
    coords: CoordSystem

    def read(
        self,
        coords: CoordSystem,
        array_name: str,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[torch.Tensor, CoordSystem]: ...


class ModelRNGState(Protocol):
    def get_rng_state(self) -> object: ...
    def set_rng_state(self, state: object) -> None: ...


@dataclass(frozen=True)
class RNGState:
    python: object | None
    numpy: object | None
    torch_cpu: torch.Tensor | None
    torch_cuda: list[torch.Tensor]
    model: object | None = None


@dataclass(frozen=True)
class RestartState:
    x: torch.Tensor
    coords: CoordSystem
    step: int
    rng: RNGState | None = None


def default_restart_path(name: str, root: str | Path | None = None) -> Path: ...


def open_restart_io(
    name: str,
    root: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> ZarrBackend: ...


def init_restart(
    restart_io: IOBackend,
    state_coords: CoordSystem,
    steps: np.ndarray,
    *,
    state_name: str = "state",
) -> None: ...


def save_restart(
    restart_io: IOBackend,
    x: torch.Tensor,
    coords: CoordSystem,
    *,
    step: int,
    rng: RNGState | None = None,
    state_name: str = "state",
) -> None: ...


def get_last_step(
    restart_io: ReadableIOBackend,
    selector: CoordSystem,
) -> int | None: ...


def load_restart(
    restart_io: ReadableIOBackend,
    coords: CoordSystem,
    *,
    step: int,
    device: torch.device | str = "cpu",
    state_name: str = "state",
) -> RestartState: ...


def validate_restart_compatibility(
    state: RestartState,
    prognostic: PrognosticModel,
) -> None: ...


def capture_rng_state(model: object | None = None) -> RNGState: ...


def restore_rng_state(state: RNGState) -> None: ...
```

The only public data containers are `RNGState` and `RestartState`. They are explicit and
typed, avoiding free-form dictionaries in user code. `ReadableIOBackend` is a protocol so
custom IO backends can opt in structurally.

## Default Store Location

`default_restart_path(name)` should use the package-wide cache convention:

```text
${EARTH2STUDIO_CACHE:-~/.cache/earth2studio}/restart/<name>.zarr
```

This deliberately does not use `EARTH2STUDIO_MODEL_CACHE`, because restart state is run
state rather than model artifact state. Production users should usually pass an explicit
shared filesystem path instead of relying on the local cache.

## Restart Store Layout

Use one restart store per logical run. The default implementation should be a
`ZarrBackend`, but the state writes go through the IO protocol.

The store contains:

- `state`: the full restart tensor. It keeps the `variable` dimension inside the tensor
  instead of splitting variables into array names, because resume needs the state as one
  tensor.
- `_complete`: a small integer marker array indexed by `step` and any selector
  dimensions such as `time`, `ensemble`, or `batch`. A value of `1` means the state and
  metadata for that selector and step were committed.
- Small metadata for RNG and coordinate schema. In the Zarr implementation this can live
  under a versioned `root.attrs["earth2studio_restart"]` entry. If a future backend does
  not support attrs, the same metadata can be stored in tiny arrays next to `_complete`.

Write order matters:

1. Write the `state` tensor.
2. Write RNG metadata.
3. Write the `_complete` marker last.

`get_last_step` must only look at `_complete`. A crash during the large state write then
leaves no committed marker, so resume falls back to the previous valid checkpoint.

## State Contract

The restart state must be the tensor and coordinate system that can be passed back to
`prognostic.create_iterator` to continue. It is not necessarily the same tensor that was
written to the forecast output.

For custom workflows this is straightforward: save the workflow's own next-input state.
For built-in workflows, the first implementation should only resume when the iterator's
yielded state passes `validate_restart_compatibility`. The validation should call
`prognostic.output_coords(state.coords)` and use the model's own coordinate checks. If a
model requires hidden iterator state that is not present in `(x, coords)`, checkpointing
should raise with an actionable message.

This prevents an attractive but wrong design: treating forecast output as restart state
for every model.

## RNG Handling

`capture_rng_state` should capture:

- Python `random` state.
- NumPy random state.
- Torch CPU RNG state.
- Torch CUDA RNG states when CUDA is initialized.
- Optional model RNG state when the model implements `get_rng_state` and
  `set_rng_state`.

On resume, `restore_rng_state` runs before creating the new prognostic iterator. Models
that already use deterministic per-member seeds still work without model-specific hooks;
stochastic models with private generators need to implement the optional protocol for
bit-identical resume.

If strict reproducibility is requested later, the helper can grow a `strict_rng=True`
flag that raises when a known-stochastic model lacks restorable RNG state. The first pass
does not need that extra public option.

## Built-In Workflow Integration

Add two optional parameters to each built-in workflow:

```python
restart_io: ReadableIOBackend | None = None
restart_interval: int = 1
```

When `restart_io is None`, the workflow remains unchanged.

For `deterministic`:

1. Build the normal forecast output store exactly as today.
2. Initialize or validate the restart store.
3. If `_complete` contains a valid step for the requested initial-condition time, load
   that state, restore RNG, skip data fetch, and resume from `step + 1`.
4. Otherwise fetch data normally and start from step 0.
5. After writing forecast output for each checkpoint interval, save restart state and
   mark it complete.

For `diagnostic`:

- Save the prognostic state before applying diagnostics. On resume, recompute diagnostic
  output for subsequent steps. This keeps the restart store independent of diagnostic
  model changes.

For `ensemble`:

- Use `ensemble` coordinates as part of the restart selector.
- The built-in batched ensemble loop can resume a partial mini-batch when the same
  ensemble coordinate subset is used. For maximum resilience, users can set
  `batch_size=1`, which makes every ensemble member independently restartable.
- The eval recipe already runs one `WorkItem` per `(time, ensemble)` and is the better
  target for robust large distributed ensembles.

The built-in workflows should not gain a large restart mode matrix. Passing `restart_io`
turns restart on; passing `None` turns it off. A fresh or overwritten `restart_io` starts
from scratch.

## Custom Workflow Example

The low-level user pattern should stay explicit:

```python
from earth2studio.utils.checkpoint import (
    capture_rng_state,
    get_last_step,
    load_restart,
    open_restart_io,
    restore_rng_state,
    save_restart,
    validate_restart_compatibility,
)

restart_io = open_restart_io("my-forecast")
selector = OrderedDict({"time": np.asarray([time[0]])})
last_step = get_last_step(restart_io, selector)

if last_step is None:
    x, coords = fetch_data(...)
    start_step = 0
else:
    state = load_restart(restart_io, restart_coords_for_step(last_step), step=last_step)
    validate_restart_compatibility(state, prognostic)
    if state.rng is not None:
        restore_rng_state(state.rng)
    x, coords = state.x, state.coords
    start_step = state.step + 1

iterator = prognostic.create_iterator(x, coords)
if last_step is not None:
    next(iterator)  # discard the checkpoint state itself

for step, (x_step, coords_step) in enumerate(iterator, start=start_step):
    output_io.write(*split_coords(x_step, coords_step))
    if step % 6 == 0:
        save_restart(
            restart_io,
            x_step,
            coords_step,
            step=step,
            rng=capture_rng_state(prognostic),
        )
```

For models where `x_step, coords_step` is not a valid next-input state, custom workflows
should save their own internal next-input state instead. That is the main reason the
utility should be function-first rather than tied to the built-in workflow iterator.

## Test Plan

- Unit-test `get_last_step` over empty, partial, and complete marker arrays.
- Unit-test `validate_restart_compatibility` with matching variables/spatial coords and
  with intentional coordinate mismatches.
- Unit-test RNG capture and restore for Python, NumPy, Torch CPU, and CUDA when
  available.
- Integration-test deterministic resume with `Persistence` and `ZarrBackend`: full run
  equals interrupted+resumed run bit-for-bit.
- Integration-test ensemble resume with fixed seeds and `batch_size=1`: full run equals
  interrupted+resumed run bit-for-bit.
- Add an eval recipe test that replaces a marker-only skip with a mid-rollout restart
  for one `WorkItem`.

## Rejected Alternatives

- `torch.save` files: simple, but duplicates the IO stack, is less inspectable with
  xarray/zarr tools, and creates a separate artifact format.
- A large `CheckpointManager` class: too much surface area for the first pass and less
  natural for custom loops.
- Hook-only checkpointing: tempting because `PrognosticMixin` exposes hooks, but it can
  conflict with user hooks and does not cover every model iterator.
- Treating forecast output as restart state: works for some re-entrant models, but is
  wrong for models with sliding windows or private rollout state.

## Recommended Implementation Order

1. Add `earth2studio/utils/checkpoint.py` with the low-level helpers and Zarr-backed
   metadata implementation.
2. Add focused unit tests for utilities and Zarr marker behavior.
3. Add deterministic and ensemble integration tests with lightweight models.
4. Add `restart_io` and `restart_interval` to `earth2studio.run` using the helpers.
5. Add a gallery example in `examples/07_misc` or `examples/01_getting_started`.
6. Integrate the same helper into the eval recipe once the package utility is stable.

The important constraint is to keep the utility usable without the built-in workflows.
If a user can add three lines to a custom loop and understand exactly what state is being
saved, the design is heading in the right direction.
