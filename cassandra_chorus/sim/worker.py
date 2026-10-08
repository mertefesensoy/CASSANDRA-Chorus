"""One simulated worker's round: build its model, train H local steps, return what the coordinator expects.

Modes (docs/implementations/2026-10-08-simulation-harness.md):

``sliced``
    A slice model holding only the worker's experts, so routing is masked to
    them by construction (S0-F-07). Returns the shared part and its experts.
``partial`` (S0-F-26)
    The full model with every expert routable; experts not assigned to the
    worker are frozen. Returns the shared part and its assigned experts.
``full`` (S0-F-14)
    The full model, everything trained and returned.

The optimizer is created fresh for every round (owner decision D19).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Literal

import torch

from cassandra_chorus.coordinator import Slice
from cassandra_chorus.model import ModelSection, MoETransformer, expert_param_owner
from cassandra_chorus.train import OptimSection, TrainStats, make_optimizer, train_steps

Mode = Literal["sliced", "partial", "full"]
Batch = tuple[torch.Tensor, torch.Tensor]


def _keeps(name: str, slice_: Slice, mode: Mode) -> bool:
    owner = expert_param_owner(name)
    return owner is None or mode == "full" or slice_.holds(*owner)


def build_worker_model(
    cfg: ModelSection, global_state: Mapping[str, torch.Tensor], slice_: Slice, mode: Mode, device: torch.device | str
) -> MoETransformer:
    """The worker's model for this round, loaded from ``global_state``.

    Construction does not disturb PyTorch's global random state (its random
    initialization is immediately overwritten by the loaded parameters).
    """
    with torch.random.fork_rng(devices=[]):
        model = MoETransformer(cfg, slice_.experts if mode == "sliced" else None)
    state = {k: v for k, v in global_state.items() if mode != "sliced" or _keeps(k, slice_, mode)}
    model.load_state_dict(state, strict=True)
    model.to(device)
    if mode == "partial":
        for name, param in model.named_parameters():
            if not _keeps(name, slice_, mode):
                param.requires_grad_(False)
    return model


def returned_state(model: MoETransformer, slice_: Slice, mode: Mode) -> dict[str, torch.Tensor]:
    """The parameters the worker sends back, detached copies on the model's device."""
    return {k: v.detach().clone() for k, v in model.state_dict().items() if _keeps(k, slice_, mode)}


@dataclasses.dataclass
class WorkerRound:
    """One worker's round: its model after training, what it returns, and training statistics."""

    model: MoETransformer
    state: dict[str, torch.Tensor]
    stats: TrainStats


def train_worker(
    cfg: ModelSection,
    optim: OptimSection,
    global_state: Mapping[str, torch.Tensor],
    slice_: Slice,
    mode: Mode,
    next_batch: Callable[[], Batch],
    round_index: int,
    local_steps: int,
    total_local_steps: int,
    device: torch.device | str,
) -> WorkerRound:
    """Train one worker for ``local_steps`` steps of its trajectory.

    The learning-rate schedule runs over the worker's whole trajectory: this
    round's steps are ``round_index * local_steps`` onward, out of
    ``total_local_steps`` (owner-confirmed 2026-10-08). The optimizer is fresh.
    """
    model = build_worker_model(cfg, global_state, slice_, mode, device)
    optimizer = make_optimizer(model, optim)
    stats = train_steps(
        model, optimizer, next_batch, local_steps, optim, device,
        start_step=round_index * local_steps, total_steps=total_local_steps,
    )
    return WorkerRound(model=model, state=returned_state(model, slice_, mode), stats=stats)
