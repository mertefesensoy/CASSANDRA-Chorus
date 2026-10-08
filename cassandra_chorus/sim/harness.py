"""Single-machine simulation settings and round machinery (PLAN step 6; SRS S0-F-04, S0-F-11, S0-F-12, S0-N-04).

Task-independent: the entry point supplies data and evaluation. Every random
stream is derived from the run seed, the worker and the round, so a round's
outcome depends only on the global state it starts from; that is what makes
resuming bitwise identical to an uninterrupted run.
"""

from __future__ import annotations

import dataclasses
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch

from cassandra_chorus.coordinator import NesterovOuter, PlainAverage, Slice, WorkerSpec, assign_slices
from cassandra_chorus.repro import derive_seed


@dataclasses.dataclass(frozen=True)
class SimSection:
    """Simulation settings (the ``[sim]`` section). Defaults are the Gate A setting (D18, D20)."""

    n_workers: int = 4
    capacity: int = 4
    capacities: tuple[int, ...] = ()  # optional per-worker capacities (S0-F-05); empty: all use `capacity`
    local_steps: int = 125
    rounds: int = 10
    mode: Literal["sliced", "partial", "full"] = "sliced"
    policy: Literal["random", "coverage", "rolling"] = "coverage"
    cross_layer: Literal["same", "independent"] = "same"
    rolling_shift: int = 1
    router_rule: Literal["holders", "all"] = "holders"
    outer: Literal["plain", "nesterov"] = "plain"
    outer_lr: float = 1.0
    outer_momentum: float = 0.9
    drop_prob: float = 0.0
    skew: float = 0.0
    equal_compute_steps: int = 5000


def worker_specs(sim: SimSection) -> list[WorkerSpec]:
    caps = list(sim.capacities) if sim.capacities else [sim.capacity] * sim.n_workers
    return [WorkerSpec(f"w{i}", c) for i, c in enumerate(caps)]


def validate_sim(sim: SimSection, n_experts: int, top_k: int) -> None:
    """Raise ``ValueError`` listing every inconsistency (equal compute D17, D21, capacities)."""
    problems = []
    if sim.n_workers < 1 or sim.local_steps < 1 or sim.rounds < 1:
        problems.append("n_workers, local_steps and rounds must be positive")
    if sim.capacities and len(sim.capacities) != sim.n_workers:
        problems.append(f"capacities has {len(sim.capacities)} entries for {sim.n_workers} workers")
    total = sim.n_workers * sim.local_steps * sim.rounds
    if total != sim.equal_compute_steps:
        problems.append(f"n_workers x local_steps x rounds = {total}, not the equal-compute {sim.equal_compute_steps} (D17)")
    if sim.mode in ("partial", "full") and sim.router_rule != "all":
        problems.append(f"mode {sim.mode!r} trains every router row, so router_rule must be 'all' (D21)")
    if sim.mode != "full":
        bad = [c for c in (sim.capacities or (sim.capacity,)) if not top_k <= c <= n_experts]
        if bad:
            problems.append(f"capacities {bad} outside [top_k={top_k}, n_experts={n_experts}]")
    if not 0.0 <= sim.drop_prob <= 1.0 or not 0.0 <= sim.skew <= 1.0:
        problems.append("drop_prob and skew must be in [0, 1]")
    if problems:
        raise ValueError("invalid simulation configuration: " + "; ".join(problems))


def round_slices(sim: SimSection, n_layers: int, n_experts: int, top_k: int, round_index: int, seed: int) -> dict[str, Slice]:
    """Slices for a round. In ``full`` mode every worker holds every expert."""
    specs = worker_specs(sim)
    if sim.mode == "full":
        everything = tuple(range(n_experts))
        return {w.name: Slice(w.name, {layer: everything for layer in range(n_layers)}) for w in specs}
    return assign_slices(specs, n_layers, n_experts, top_k, round_index, sim.policy, sim.cross_layer, seed, sim.rolling_shift)


def dropped_workers(sim: SimSection, round_index: int, seed: int) -> list[str]:
    """Workers whose result is lost this round, each independently with probability ``drop_prob`` (S0-F-11)."""
    rng = random.Random(derive_seed(seed, f"sim/faults/r{round_index}"))
    return [w.name for w in worker_specs(sim) if rng.random() < sim.drop_prob]


def skew_weights(n_maps: int, n_workers: int, worker_index: int, skew: float) -> list[float]:
    """Map sampling weights for one worker (S0-F-12): uniform mixed with the worker's preferred maps."""
    preferred = [m for m in range(n_maps) if m % n_workers == worker_index % n_workers]
    if not preferred:
        return [1.0 / n_maps] * n_maps
    return [(1.0 - skew) / n_maps + (skew / len(preferred) if m in preferred else 0.0) for m in range(n_maps)]


def data_generator(seed: int, worker_index: int, round_index: int) -> torch.Generator:
    return torch.Generator().manual_seed(derive_seed(seed, f"sim/data/w{worker_index}/r{round_index}"))


def make_outer(sim: SimSection) -> PlainAverage | NesterovOuter:
    return PlainAverage() if sim.outer == "plain" else NesterovOuter(sim.outer_lr, sim.outer_momentum)


# --------------------------------------------------------------------------
# Checkpoints (S0-N-04)

CHECKPOINT_NAME = "checkpoint.pt"


def save_checkpoint(run_dir: Path, completed_round: int, global_state: Mapping[str, torch.Tensor],
                    outer: PlainAverage | NesterovOuter, config_hash: str) -> Path:
    """Atomically write the state after ``completed_round`` (temporary file, then replace)."""
    path = Path(run_dir) / CHECKPOINT_NAME
    tmp = path.with_suffix(".pt.tmp")
    payload: dict[str, Any] = {
        "completed_round": completed_round,
        "config_hash": config_hash,
        "global_state": {k: v.detach().cpu() for k, v in global_state.items()},
        "outer_velocity": {k: v.cpu() for k, v in outer.velocity.items()} if isinstance(outer, NesterovOuter) else None,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(run_dir: Path, config_hash: str, outer: PlainAverage | NesterovOuter,
                    device: torch.device | str) -> tuple[int, dict[str, torch.Tensor]]:
    """Return (next round, global state) and restore the outer optimizer's velocity."""
    payload = torch.load(Path(run_dir) / CHECKPOINT_NAME, map_location="cpu")
    if payload["config_hash"] != config_hash:
        raise ValueError("checkpoint was written for a different configuration")
    if isinstance(outer, NesterovOuter) and payload["outer_velocity"] is not None:
        outer.velocity = {k: v.to(device) for k, v in payload["outer_velocity"].items()}
    state = {k: v.to(device) for k, v in payload["global_state"].items()}
    return payload["completed_round"] + 1, state
