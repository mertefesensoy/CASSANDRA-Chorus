"""Task-independent training loop.

Used by the centralized baseline (step 4), by each simulated worker's local
training (step 6) and, in Stage 1, by real worker clients. It knows nothing
about tasks: batches come from a caller-supplied function.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import torch
from torch import nn

Batch = tuple[torch.Tensor, torch.Tensor]  # (inputs, targets); targets of -100 are ignored


@dataclasses.dataclass(frozen=True)
class OptimSection:
    """Optimizer and schedule (the ``[optim]`` section).

    Weight decay defaults to 0: on a sliced worker, the router rows of experts
    it does not hold get no gradient, and decoupled decay would still shrink
    them, confounding the comparison with centralized training.
    """

    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    warmup_steps: int = 100
    grad_clip: float = 1.0
    batch_size: int = 64


@dataclasses.dataclass
class TrainStats:
    """What a call to :func:`train_steps` processed. Losses are means over its steps."""

    steps: int = 0
    sequences: int = 0
    scored_targets: int = 0
    mean_loss: float = float("nan")
    mean_ce_loss: float = float("nan")
    mean_balance_loss: float = float("nan")


def make_optimizer(model: nn.Module, cfg: OptimSection) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.weight_decay
    )


def lr_at(step: int, cfg: OptimSection) -> float:
    """Linear warm-up to ``lr`` over ``warmup_steps`` (step counted from 0), then constant."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


def train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    next_batch: Callable[[], Batch],
    n_steps: int,
    cfg: OptimSection,
    device: torch.device | str,
    start_step: int = 0,
    expert_mask=None,
) -> TrainStats:
    """Run ``n_steps`` optimizer steps; return what was processed.

    Each step: set the scheduled learning rate for global step
    ``start_step + i``, compute the model's ``loss`` (cross-entropy plus the
    weighted balance loss) on the next batch, backpropagate, clip the global
    gradient norm to ``grad_clip`` (if positive) and step the optimizer.
    Side effects: updates the model's parameters and the optimizer state.
    Raises ``FloatingPointError`` on a non-finite loss rather than continuing.
    """
    model.train()
    stats = TrainStats()
    loss_sum = 0.0
    ce_sum = balance_sum = scored = torch.zeros((), device=device)  # accumulated on the device
    for i in range(n_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(start_step + i, cfg)
        inputs, targets = next_batch()
        inputs, targets = inputs.to(device), targets.to(device)
        out = model(inputs, targets=targets, expert_mask=expert_mask)
        loss_value = out.loss.item()  # the one host synchronization per step
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"non-finite loss {loss_value} at step {start_step + i}")
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        stats.steps += 1
        stats.sequences += inputs.shape[0]
        loss_sum += loss_value
        ce_sum = ce_sum + out.ce_loss.detach()
        balance_sum = balance_sum + out.balance_loss.detach()
        scored = scored + (targets != -100).sum()
    if stats.steps:
        stats.mean_loss = loss_sum / stats.steps
        stats.mean_ce_loss = ce_sum.item() / stats.steps
        stats.mean_balance_loss = balance_sum.item() / stats.steps
        stats.scored_targets = int(scored.item())
    return stats
