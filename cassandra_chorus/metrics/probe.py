"""Redundancy probe (PLAN step 9a, SRS D24): what each expert of a saved model learned.

Two evaluations per mixture layer, each on a fixed evaluation set, with every
other layer routing as trained:

- **bypass**: the layer's mixture output is replaced by zeros through a forward
  hook, so the residual stream passes through that layer's mixture unchanged;
- **forced pairs**: the layer's routing is restricted to each pair of experts
  with the model's existing expert mask (with k = 2 every token goes to both,
  weighted by the router's softmax over the pair).

The model code is not changed. Derived quantities and the reading registered
before any probe run: docs/implementations/2026-10-08-redundancy-probe.md.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch

from cassandra_chorus.data.task_a import TaskABatch
from cassandra_chorus.metrics.task_a import evaluate_task_a

COMPETENT = 0.99      # a forced cell is competent if it keeps 99% of the trained per-map accuracy
NEEDED_POINTS = 5.0   # a layer is needed if bypassing it costs at least 5 points on its worst map


@contextlib.contextmanager
def bypass_layer(model, layer: int) -> Iterator[None]:
    """Within the block, layer ``layer``'s mixture output is zeros; the hook is always removed."""

    def hook(module, inputs, result):
        return dataclasses.replace(result, output=torch.zeros_like(result.output))

    handle = model.blocks[layer].moe.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def probe_model(model, batch: TaskABatch, device: torch.device | str, n_maps: int) -> dict[str, Any]:
    """Per-map accuracies: trained routing, each layer bypassed, each layer forced to each expert pair.

    Returns ``{"trained_per_map": [M], "layers": [{"bypass_per_map": [M],
    "pairs": [{"experts": [a, b], "per_map": [M]}, ...]}, ...]}`` with pairs in
    lexicographic order. No side effects on the model beyond evaluation.
    """
    n_layers, n_experts = model.cfg.n_layers, model.cfg.n_experts
    trained = evaluate_task_a(model, batch, device, n_maps)["per_map_accuracy"]
    layers = []
    for layer in range(n_layers):
        with bypass_layer(model, layer):
            bypassed = evaluate_task_a(model, batch, device, n_maps)["per_map_accuracy"]
        pairs = [
            {"experts": list(pair),
             "per_map": evaluate_task_a(model, batch, device, n_maps, expert_mask={layer: pair})["per_map_accuracy"]}
            for pair in itertools.combinations(range(n_experts), 2)
        ]
        layers.append({"bypass_per_map": bypassed, "pairs": pairs})
    return {"trained_per_map": trained, "layers": layers}


def retention(forced: float, trained: float) -> float:
    """Forced accuracy as a fraction of trained accuracy on the same map; 1 when there was nothing to lose."""
    return forced / trained if trained > 0 else 1.0


def competence(trained: Sequence[float], pairs: Sequence[Mapping[str, Any]], n_experts: int) -> list[list[float]]:
    """c[e][m]: mean retention on map m over the pairs that contain expert e (1 = as good as trained routing)."""
    table = []
    for e in range(n_experts):
        rows = [p["per_map"] for p in pairs if e in p["experts"]]
        table.append([sum(retention(r[m], t) for r in rows) / len(rows) for m, t in enumerate(trained)])
    return table


def summarize_layer(trained: Sequence[float], layer: Mapping[str, Any], n_experts: int) -> dict[str, Any]:
    """Bypass cost (points, worst map), competent share, competence table and specialization of one layer."""
    bypass_cost = 100 * max(t - b for t, b in zip(trained, layer["bypass_per_map"]))
    cells = [f >= COMPETENT * t for p in layer["pairs"] for f, t in zip(p["per_map"], trained)]
    table = competence(trained, layer["pairs"], n_experts)
    spread = [max(row) - min(row) for row in table]
    return {
        "bypass_cost_points": bypass_cost,
        "needed": bypass_cost >= NEEDED_POINTS,
        "competent_share": sum(cells) / len(cells),
        "specialization": sum(spread) / len(spread),
        "competence": table,
    }


def summarize(probe: Mapping[str, Any], n_experts: int) -> list[dict[str, Any]]:
    return [summarize_layer(probe["trained_per_map"], layer, n_experts) for layer in probe["layers"]]


def pooled_share(summary: Sequence[Mapping[str, Any]], layers: Sequence[int]) -> float | None:
    """Competent share pooled over the given layers (each layer has the same number of cells)."""
    if not layers:
        return None
    return sum(summary[layer]["competent_share"] for layer in layers) / len(layers)


def evaluate_reading(summaries: Mapping[tuple[str, str, int], Sequence[Mapping[str, Any]]],
                     seeds: Sequence[int] = (7, 11, 19)) -> dict[str, Any]:
    """The reading registered before any probe run, per seed and variant, then overall.

    ``summaries`` maps (arm, variant, seed) to a model's per-layer summary. For
    each seed and variant: (1) the main arm has at least one needed layer;
    (2) pooled over the main arm's needed layers, its competent share is higher
    than the centralized, partial and full runs' over the same layer indices.
    Overall: "supported" if (1) and (2) hold everywhere, "bypassed" if no main
    layer is needed anywhere, "mixed" otherwise; "incomplete" if a model is missing.
    """
    rows, missing = {}, False
    for variant in ("marked", "unmarked"):
        for seed in seeds:
            main = summaries.get(("main", variant, seed))
            others = {arm: summaries.get((arm, variant, seed)) for arm in ("centralized", "partial", "full")}
            if main is None or any(v is None for v in others.values()):
                rows[(variant, seed)] = {"status": "missing"}
                missing = True
                continue
            needed = [layer for layer, s in enumerate(main) if s["needed"]]
            main_share = pooled_share(main, needed)
            other_share = {arm: pooled_share(s, needed) for arm, s in others.items()}
            higher = bool(needed) and all(main_share > v for v in other_share.values())
            rows[(variant, seed)] = {"needed_layers": needed, "main_share": main_share, "other_share": other_share,
                                     "condition_1": bool(needed), "condition_2": higher}
    if missing:
        overall = "incomplete"
    elif all(r["condition_1"] and r["condition_2"] for r in rows.values()):
        overall = "supported"
    elif not any(r["condition_1"] for r in rows.values()):
        overall = "bypassed"
    else:
        overall = "mixed"
    return {"rows": rows, "overall": overall}
