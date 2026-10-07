"""Task A evaluation (SRS S0-F-20, raw material for S0-F-21 and S0-F-22).

Accuracy is measured on scored positions only (first occurrences of keys), for
the full model with all experts available unless a mask is given. The
map-by-expert table counts, per layer, which experts the token at each scored
position was routed to, split by the map that produced the sequence; step 8
builds router-consistency and usage metrics from it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from cassandra_chorus.data.task_a import TaskABatch


@torch.no_grad()
def evaluate_task_a(
    model,
    batch: TaskABatch,
    device: torch.device | str,
    n_maps: int,
    chunk: int = 512,
    expert_mask=None,
    select: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
) -> dict:
    """Evaluate ``model`` on ``batch``; no side effects except evaluation mode.

    Returns a dictionary with ``accuracy`` (pooled), ``per_map_accuracy``
    (list of M), ``min_map_accuracy``, ``scored_loss`` (nats), ``scored``
    (count), ``expert_counts`` ([layers][E], all positions) and
    ``map_expert_counts`` ([layers][M][E], scored positions).
    """
    was_training = model.training
    model.eval()
    n_layers, n_experts = model.cfg.n_layers, model.cfg.n_experts
    correct = torch.zeros(n_maps, dtype=torch.long)
    total = torch.zeros(n_maps, dtype=torch.long)
    loss_sum = 0.0
    expert_counts = torch.zeros(n_layers, n_experts, dtype=torch.long)
    map_expert = torch.zeros(n_layers, n_maps, n_experts, dtype=torch.long)
    for start in range(0, batch.inputs.shape[0], chunk):
        inputs = batch.inputs[start : start + chunk].to(device)
        targets = batch.targets[start : start + chunk].to(device)
        map_ids = batch.map_ids[start : start + chunk].to(device)
        out = model(inputs, expert_mask=expert_mask, return_routing=True, select=select)
        scored = targets != -100
        logits = out.logits[scored]
        tgt = targets[scored]
        loss_sum += F.cross_entropy(logits, tgt, reduction="sum").item()
        # Counting happens on the CPU: exact, and free of GPU operations that
        # PyTorch flags as non-deterministic (such as bincount).
        hits = (logits.argmax(dim=-1) == tgt).cpu()
        seq_map = map_ids.unsqueeze(1).expand_as(targets)[scored].cpu()
        correct += torch.bincount(seq_map[hits], minlength=n_maps)
        total += torch.bincount(seq_map, minlength=n_maps)
        expert_counts += out.expert_counts.cpu()
        for layer, routing in enumerate(out.routing):
            chosen = routing[scored].cpu()  # [S, k]
            pair = seq_map.unsqueeze(1) * n_experts + chosen  # map and expert as one index
            map_expert[layer] += torch.bincount(pair.reshape(-1), minlength=n_maps * n_experts).view(n_maps, n_experts)
    model.train(was_training)
    per_map = (correct.double() / total.clamp(min=1).double()).tolist()
    return {
        "accuracy": float(correct.sum()) / max(int(total.sum()), 1),
        "per_map_accuracy": per_map,
        "min_map_accuracy": min(per_map),
        "scored_loss": loss_sum / max(int(total.sum()), 1),
        "scored": int(total.sum()),
        "expert_counts": expert_counts.tolist(),
        "map_expert_counts": map_expert.tolist(),
    }


def random_selector(n_experts: int, top_k: int, seed: int, device: torch.device | str):
    """A ``select`` hook choosing k distinct available experts uniformly at random per token.

    Draws one uniform score per expert from its own generator, sets unavailable
    experts to minus infinity, and keeps the k highest: a uniform random
    k-subset of the available experts.
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    def select(layer: int, logits: torch.Tensor) -> torch.Tensor:
        scores = torch.rand(logits.shape, generator=generator, device=logits.device)
        scores = scores.masked_fill(torch.isinf(logits), float("-inf"))
        return scores.topk(top_k, dim=-1).indices

    return select


def routing_necessity(model, batch: TaskABatch, device: torch.device | str, n_maps: int, seed: int) -> dict:
    """Accuracy with the trained router, with random routing, and with each expert removed.

    Removing expert e means masking it in every layer. Returns pooled and
    minimum per-map accuracy for each condition.
    """
    cfg = model.cfg

    def summary(result: dict) -> dict:
        return {"accuracy": result["accuracy"], "min_map_accuracy": result["min_map_accuracy"]}

    trained = evaluate_task_a(model, batch, device, n_maps)
    random_routing = evaluate_task_a(
        model, batch, device, n_maps, select=random_selector(cfg.n_experts, cfg.top_k, seed, device)
    )
    leave_one_out = []
    for e in range(cfg.n_experts):
        mask = {layer: [x for x in range(cfg.n_experts) if x != e] for layer in range(cfg.n_layers)}
        leave_one_out.append({"expert": e, **summary(evaluate_task_a(model, batch, device, n_maps, expert_mask=mask))})
    return {"trained": summary(trained), "random_routing": summary(random_routing), "leave_one_out": leave_one_out}
