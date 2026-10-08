"""Slice assignment: which experts each worker holds in a round (SRS S0-F-05, S0-F-06, S0-F-25).

Policies: ``random``, ``coverage`` (every expert held by at least one worker)
and ``rolling`` (windows that rotate by ``shift`` experts per round, owner
decision D12). Cross-layer: ``same`` indices in every layer (default, D6) or
an ``independent`` choice per layer. Deterministic given the arguments.
Algorithms: docs/implementations/2026-10-08-coordinator.md.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Mapping, Sequence
from typing import Literal

from cassandra_chorus.repro import derive_seed

Policy = Literal["random", "coverage", "rolling"]
CrossLayer = Literal["same", "independent"]


@dataclasses.dataclass(frozen=True)
class WorkerSpec:
    """A worker and how many experts per mixture layer it can hold this round."""

    name: str
    capacity: int


@dataclasses.dataclass(frozen=True)
class Slice:
    """The experts a worker holds: every layer mapped to sorted global expert indices.

    ``experts`` has the same form as the ``held`` argument of
    :class:`cassandra_chorus.model.MoETransformer`.
    """

    worker: str
    experts: Mapping[int, tuple[int, ...]]

    def holds(self, layer: int, expert: int) -> bool:
        return expert in self.experts[layer]


def _check(workers: Sequence[WorkerSpec], n_layers: int, n_experts: int, top_k: int, policy: str) -> None:
    names = [w.name for w in workers]
    if not workers:
        raise ValueError("at least one worker is needed")
    if len(set(names)) != len(names):
        raise ValueError(f"worker names must be unique, got {names}")
    if n_layers < 1 or n_experts < 1:
        raise ValueError("n_layers and n_experts must be at least 1")
    bad = [w.name for w in workers if not top_k <= w.capacity <= n_experts]
    if bad:
        raise ValueError(f"capacity must be between top_k={top_k} and n_experts={n_experts}; workers {bad} are not")
    if policy in ("coverage", "rolling") and sum(w.capacity for w in workers) < n_experts:
        raise ValueError(
            f"policy {policy!r} needs total capacity of at least {n_experts}, got {sum(w.capacity for w in workers)}"
        )


def _random(workers: Sequence[WorkerSpec], n_experts: int, rng: random.Random) -> dict[str, tuple[int, ...]]:
    return {w.name: tuple(sorted(rng.sample(range(n_experts), w.capacity))) for w in workers}


def _coverage(workers: Sequence[WorkerSpec], n_experts: int, rng: random.Random) -> dict[str, tuple[int, ...]]:
    held: dict[str, set[int]] = {w.name: set() for w in workers}
    experts = list(range(n_experts))
    rng.shuffle(experts)
    order = [w for w in workers]
    turn = 0
    for e in experts:  # deal every expert once, in turn, skipping full workers
        while len(held[order[turn % len(order)].name]) >= order[turn % len(order)].capacity:
            turn += 1
        held[order[turn % len(order)].name].add(e)
        turn += 1
    for w in workers:  # fill remaining capacity with experts this worker does not hold yet
        free = [e for e in range(n_experts) if e not in held[w.name]]
        held[w.name].update(rng.sample(free, w.capacity - len(held[w.name])))
    return {name: tuple(sorted(s)) for name, s in held.items()}


def _rolling(workers: Sequence[WorkerSpec], n_experts: int, round_index: int, shift: int) -> dict[str, tuple[int, ...]]:
    result = {}
    offset = 0
    for w in workers:
        start = (offset + round_index * shift) % n_experts
        result[w.name] = tuple(sorted((start + j) % n_experts for j in range(w.capacity)))
        offset = (offset + w.capacity) % n_experts
    return result


def assign_slices(
    workers: Sequence[WorkerSpec],
    n_layers: int,
    n_experts: int,
    top_k: int,
    round_index: int,
    policy: Policy = "coverage",
    cross_layer: CrossLayer = "same",
    seed: int = 0,
    shift: int = 1,
) -> dict[str, Slice]:
    """One :class:`Slice` per worker for round ``round_index``. No side effects.

    Raises ``ValueError`` for an impossible request: a capacity outside
    [top_k, n_experts], duplicate worker names, or (for coverage and rolling)
    total capacity below the number of experts.
    """
    if policy not in ("random", "coverage", "rolling"):
        raise ValueError(f"unknown assignment policy {policy!r}")
    if cross_layer not in ("same", "independent"):
        raise ValueError(f"unknown cross-layer option {cross_layer!r}")
    _check(workers, n_layers, n_experts, top_k, policy)
    rng = random.Random(derive_seed(seed, f"assignment/round{round_index}"))

    def draw() -> dict[str, tuple[int, ...]]:
        if policy == "random":
            return _random(workers, n_experts, rng)
        if policy == "coverage":
            return _coverage(workers, n_experts, rng)
        return _rolling(workers, n_experts, round_index, shift)

    per_layer: list[dict[str, tuple[int, ...]]]
    if cross_layer == "same":
        one = draw()
        per_layer = [one] * n_layers
    elif policy == "rolling":
        # Same rotation in every layer, seen through a fixed per-layer relabelling of the experts.
        base = draw()
        layer_rng = random.Random(derive_seed(seed, "assignment/rolling-permutations"))
        per_layer = []
        for _ in range(n_layers):
            perm = list(range(n_experts))
            layer_rng.shuffle(perm)
            per_layer.append({name: tuple(sorted(perm[e] for e in experts)) for name, experts in base.items()})
    else:
        per_layer = [draw() for _ in range(n_layers)]

    return {w.name: Slice(w.name, {layer: per_layer[layer][w.name] for layer in range(n_layers)}) for w in workers}


def holders(slices: Mapping[str, Slice], layer: int, expert: int) -> list[str]:
    """Workers whose slice holds ``expert`` in ``layer``, in the order of ``slices``."""
    return [name for name, s in slices.items() if s.holds(layer, expert)]
