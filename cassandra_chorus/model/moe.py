"""Mixture-of-experts layer with a maskable router (SRS S0-F-01, S0-F-02).

Routing rule (owner decision 2026-10-07): unavailable experts get logit minus
infinity, the top-k available experts are selected, and their gate weights are
a softmax over just those k logits, so the weights sum to 1 whatever subset of
experts is available. Formulas: ``docs/implementations/2026-10-07-moe-transformer.md``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from cassandra_chorus.model.layers import SwiGLUExpert


@dataclasses.dataclass
class MoEResult:
    """Output of one mixture-of-experts layer for a batch.

    ``output`` has the input's shape. ``top_idx`` and ``gates`` are ``[B, T, k]``
    (global expert indices and their weights). ``counts`` is ``[E]``: how many
    tokens were routed to each expert. ``balance_loss`` is a scalar.
    """

    output: torch.Tensor
    top_idx: torch.Tensor
    gates: torch.Tensor
    counts: torch.Tensor
    balance_loss: torch.Tensor


def switch_balance_loss(top_idx: torch.Tensor, probs: torch.Tensor, n_available: int) -> torch.Tensor:
    """Switch-Transformer load-balancing loss restricted to the available experts.

    ``top_idx``: ``[N, k]`` selected experts; ``probs``: ``[N, E]`` router
    probabilities (softmax over available experts, zero elsewhere). Returns
    ``n_available * sum_i f_i * P_i`` where f_i is the fraction of the N*k
    assignments that went to expert i and P_i the mean probability of expert i.
    Equals 1 under uniform routing; only P carries gradient.
    """
    n_tokens, k = top_idx.shape
    n_experts = probs.shape[-1]
    counts = F.one_hot(top_idx.reshape(-1), n_experts).sum(dim=0)
    f = counts.to(probs.dtype) / (n_tokens * k)
    p = probs.mean(dim=0)
    return n_available * (f * p).sum()


class MoELayer(nn.Module):
    """A router over ``n_experts`` global experts, of which this layer holds ``held``.

    Experts live in a ``ModuleDict`` keyed by their global index, so parameter
    names identify the expert (``experts.5.w1.weight``) whether this is the full
    model or a slice. The router always has all ``n_experts`` rows.
    """

    def __init__(self, d_model: int, hidden: int, n_experts: int, top_k: int, held: Iterable[int]) -> None:
        super().__init__()
        held = tuple(sorted(set(held)))
        bad = [e for e in held if not 0 <= e < n_experts]
        if bad:
            raise ValueError(f"expert indices {bad} are outside 0..{n_experts - 1}")
        if len(held) < top_k:
            raise ValueError(f"a layer holding {len(held)} experts cannot route each token to top_k={top_k}")
        self.n_experts = n_experts
        self.top_k = top_k
        self.held = held
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleDict({str(e): SwiGLUExpert(d_model, hidden) for e in held})

    def available(self, allowed: Iterable[int] | None) -> tuple[int, ...]:
        """Held experts that ``allowed`` (None: no restriction) also permits."""
        if allowed is None:
            return self.held
        allowed = set(allowed)
        return tuple(e for e in self.held if e in allowed)

    def forward(self, x: torch.Tensor, allowed: Iterable[int] | None = None) -> MoEResult:
        batch, tokens, channels = x.shape
        flat = x.reshape(-1, channels)
        available = self.available(allowed)
        if len(available) < self.top_k:
            raise ValueError(
                f"only {len(available)} experts are available ({list(available)}) but top_k={self.top_k}"
            )

        logits = self.router(flat)
        is_available = torch.zeros(self.n_experts, dtype=torch.bool, device=x.device)
        is_available[list(available)] = True
        logits = logits.masked_fill(~is_available, float("-inf"))
        top_logits, top_idx = logits.topk(self.top_k, dim=-1)
        gates = F.softmax(top_logits, dim=-1)

        # Sparse dispatch: each expert processes only the tokens that selected it,
        # in ascending global index, and its weighted output is added back.
        out = torch.zeros_like(flat)
        for e in available:
            rows, slots = (top_idx == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            expert_out = self.experts[str(e)](flat[rows])
            out.index_add_(0, rows, expert_out * gates[rows, slots].unsqueeze(-1))

        probs = F.softmax(logits, dim=-1)
        counts = F.one_hot(top_idx.reshape(-1), self.n_experts).sum(dim=0)
        return MoEResult(
            output=out.view(batch, tokens, channels),
            top_idx=top_idx.view(batch, tokens, self.top_k),
            gates=gates.view(batch, tokens, self.top_k),
            counts=counts,
            balance_loss=switch_balance_loss(top_idx, probs, len(available)),
        )
