"""Merging worker results into the global model (SRS S0-F-08 to S0-F-10, decision D11).

Works on plain parameter dictionaries (name to tensor), so the same code
merges simulated workers in Stage 0 and real workers in Stage 1 (S0-N-05).
Parameters are classified by name with the model's naming helpers:

* expert parameters: averaged over the returning workers that held the expert;
* router weights: by default each row (one per expert) over that expert's
  holders (``router_rule="holders"``); ``"all"`` averages the whole router
  over every returning worker, as in SRS Draft 0.4;
* everything else (the shared part): over every returning worker.

Averages are weighted by examples processed (S0-F-09), accumulated in float64
in the order the results are given, and anything without a contributor is left
unchanged. Formulas: docs/implementations/2026-10-08-coordinator.md.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from cassandra_chorus.coordinator.assignment import Slice
from cassandra_chorus.model import expert_param_owner, router_param_layer

State = Mapping[str, torch.Tensor]
RouterRule = Literal["holders", "all"]


@dataclasses.dataclass(frozen=True)
class WorkerResult:
    """What one worker sends back: its slice, its parameters after local training, examples processed."""

    slice: Slice
    state: State
    examples: int


@dataclasses.dataclass
class MergeReport:
    """What a merge did. Holder counts are per layer, indexed by global expert."""

    workers: list[str]
    total_examples: int
    holder_counts: dict[int, list[int]]
    unchanged_experts: dict[int, list[int]]
    unchanged_router_rows: dict[int, list[int]]
    router_rule: str
    outer: dict[str, Any]


class PlainAverage:
    """Default outer optimizer: the merged model is the weighted average (S0-F-10)."""

    def describe(self) -> dict[str, Any]:
        return {"name": "plain_average"}

    def step(self, name: str, current: torch.Tensor, averaged: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return averaged.clone()
        return torch.where(mask, averaged, current)


class NesterovOuter:
    """DiLoCo-style outer step with Nesterov momentum on Δ = average - current.

    Only entries that had contributors this round move; elsewhere both the
    parameter and its velocity are left as they are. lr=1, momentum=0 gives
    :class:`PlainAverage`.
    """

    def __init__(self, lr: float = 1.0, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.velocity: dict[str, torch.Tensor] = {}

    def describe(self) -> dict[str, Any]:
        return {"name": "nesterov", "lr": self.lr, "momentum": self.momentum}

    def step(self, name: str, current: torch.Tensor, averaged: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        delta = (averaged.double() - current.double())
        v = self.velocity.get(name)
        if v is None:
            v = torch.zeros_like(delta)
        new_v = self.momentum * v + delta
        update = self.lr * (delta + self.momentum * new_v)
        if mask is not None:
            maskd = mask.to(torch.bool)
            new_v = torch.where(maskd, new_v, v)
            update = torch.where(maskd, update, torch.zeros_like(update))
        self.velocity[name] = new_v
        return (current.double() + update).to(current.dtype)


def _expected_names(global_state: State, slice_: Slice) -> set[str]:
    names = set()
    for name in global_state:
        owner = expert_param_owner(name)
        if owner is None or slice_.holds(*owner):
            names.add(name)
    return names


def _validate(global_state: State, results: Sequence[WorkerResult]) -> None:
    seen = set()
    for r in results:
        if r.slice.worker in seen:
            raise ValueError(f"worker {r.slice.worker!r} returned more than once")
        seen.add(r.slice.worker)
        if r.examples <= 0:
            raise ValueError(f"worker {r.slice.worker!r} reports {r.examples} examples; must be positive")
        expected = _expected_names(global_state, r.slice)
        got = set(r.state)
        if got != expected:
            missing, extra = sorted(expected - got)[:5], sorted(got - expected)[:5]
            raise ValueError(f"worker {r.slice.worker!r} returned the wrong parameters; missing {missing}, unexpected {extra}")
        for name in got:
            if tuple(r.state[name].shape) != tuple(global_state[name].shape):
                raise ValueError(f"worker {r.slice.worker!r}: {name} has shape {tuple(r.state[name].shape)}, "
                                 f"expected {tuple(global_state[name].shape)}")


def _weighted_mean(tensors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    total = sum(weights)
    acc = torch.zeros_like(tensors[0], dtype=torch.float64)
    for t, w in zip(tensors, weights):
        acc += t.to(torch.float64) * w
    return acc / total


def merge(
    global_state: State,
    results: Sequence[WorkerResult],
    router_rule: RouterRule = "holders",
    outer: PlainAverage | NesterovOuter | None = None,
) -> tuple[dict[str, torch.Tensor], MergeReport]:
    """Merge ``results`` into ``global_state``; return the new state and a report.

    Does not modify its inputs. Raises ``ValueError`` if a result carries the
    wrong parameter names or shapes, a non-positive example count, or a worker
    twice. An empty ``results`` (every worker dropped) returns an unchanged copy.
    """
    if router_rule not in ("holders", "all"):
        raise ValueError(f"unknown router rule {router_rule!r}")
    outer = outer or PlainAverage()
    _validate(global_state, results)

    layers = sorted({o[0] for n in global_state if (o := expert_param_owner(n)) is not None}
                    | {lay for n in global_state if (lay := router_param_layer(n)) is not None})
    router_rows = {router_param_layer(n): global_state[n].shape[0] for n in global_state if router_param_layer(n) is not None}
    n_experts = max(router_rows.values()) if router_rows else 0
    holder_counts = {lay: [sum(1 for r in results if r.slice.holds(lay, e)) for e in range(n_experts)] for lay in layers}

    new_state: dict[str, torch.Tensor] = {}
    for name, current in global_state.items():
        owner = expert_param_owner(name)
        router_layer = router_param_layer(name)
        if owner is not None:
            contributors = [r for r in results if r.slice.holds(*owner)]
        else:
            contributors = list(results)
        if not contributors:
            new_state[name] = current.clone()
            continue
        if router_layer is not None and router_rule == "holders":
            # Row e is averaged over the holders of expert e only; rows without holders stay unchanged.
            averaged = current.to(torch.float64).clone()
            mask = torch.zeros(current.shape[0], dtype=torch.bool)
            for e in range(current.shape[0]):
                row_holders = [r for r in results if r.slice.holds(router_layer, e)]
                if row_holders:
                    averaged[e] = _weighted_mean([r.state[name][e] for r in row_holders],
                                                 [float(r.examples) for r in row_holders])
                    mask[e] = True
            row_mask = mask.view(-1, *([1] * (current.dim() - 1))).expand_as(current).to(current.device)
            new_state[name] = outer.step(name, current, averaged.to(current.dtype), row_mask)
        else:
            averaged = _weighted_mean([r.state[name] for r in contributors], [float(r.examples) for r in contributors])
            new_state[name] = outer.step(name, current, averaged.to(current.dtype), None)

    unheld = {lay: [e for e, c in enumerate(counts) if c == 0] for lay, counts in holder_counts.items()}
    if router_rule == "holders":
        unchanged_rows = unheld  # a row moves only if its expert had a holder
    elif results:
        unchanged_rows = {lay: [] for lay in layers}  # the whole router is averaged over every worker
    else:
        unchanged_rows = {lay: list(range(n_experts)) for lay in layers}
    report = MergeReport(
        workers=[r.slice.worker for r in results],
        total_examples=sum(r.examples for r in results),
        holder_counts=holder_counts,
        unchanged_experts=unheld,
        unchanged_router_rows=unchanged_rows,
        router_rule=router_rule,
        outer=outer.describe(),
    )
    return new_state, report
