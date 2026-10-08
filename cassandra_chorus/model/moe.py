"""Mixture-of-experts layer with a maskable router (SRS S0-F-01, S0-F-02).

Routing rule (owner decision 2026-10-07): unavailable experts get logit minus
infinity, the top-k available experts are selected, and their gate weights are
a softmax over just those k logits, so the weights sum to 1 whatever subset of
experts is available. Formulas: ``docs/implementations/2026-10-07-moe-transformer.md``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable

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

    def __init__(
        self, d_model: int, hidden: int, n_experts: int, top_k: int, held: Iterable[int], dispatch: str = "sparse"
    ) -> None:
        super().__init__()
        held = tuple(sorted(set(held)))
        bad = [e for e in held if not 0 <= e < n_experts]
        if bad:
            raise ValueError(f"expert indices {bad} are outside 0..{n_experts - 1}")
        if len(held) < top_k:
            raise ValueError(f"a layer holding {len(held)} experts cannot route each token to top_k={top_k}")
        if dispatch not in ("sparse", "dense"):
            raise ValueError(f"unknown dispatch {dispatch!r}")
        self.n_experts = n_experts
        self.top_k = top_k
        self.held = held
        self.dispatch = dispatch
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleDict({str(e): SwiGLUExpert(d_model, hidden) for e in held})

    def available(self, allowed: Iterable[int] | None) -> tuple[int, ...]:
        """Held experts that ``allowed`` (None: no restriction) also permits."""
        if allowed is None:
            return self.held
        allowed = set(allowed)
        return tuple(e for e in self.held if e in allowed)

    def forward(
        self,
        x: torch.Tensor,
        allowed: Iterable[int] | None = None,
        select: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> MoEResult:
        """Route ``x`` (``[B, T, d]``) through the available experts.

        ``select``, used only by diagnostics, replaces top-k selection: it gets
        the masked logits ``[N, E]`` and returns ``[N, k]`` distinct available
        experts. Gate weights are always the softmax over the router's logits
        of the selected experts.
        """
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
        if select is None:
            top_logits, top_idx = logits.topk(self.top_k, dim=-1)
        else:
            top_idx = select(logits)
            self._check_selection(top_idx, is_available, flat.shape[0])
            top_logits = logits.gather(1, top_idx)
        gates = F.softmax(top_logits, dim=-1)

        if self.dispatch == "sparse":
            out = self._sparse(flat, available, top_idx, gates)
        else:
            out = self._dense(flat, available, top_idx, gates)

        probs = F.softmax(logits, dim=-1)
        counts = F.one_hot(top_idx.reshape(-1), self.n_experts).sum(dim=0)
        return MoEResult(
            output=out.view(batch, tokens, channels),
            top_idx=top_idx.view(batch, tokens, self.top_k),
            gates=gates.view(batch, tokens, self.top_k),
            counts=counts,
            balance_loss=switch_balance_loss(top_idx, probs, len(available)),
        )

    def _sparse(
        self, flat: torch.Tensor, available: tuple[int, ...], top_idx: torch.Tensor, gates: torch.Tensor
    ) -> torch.Tensor:
        """Each expert processes only the tokens that selected it, in ascending global
        index, and its gate-weighted output is added back."""
        out = torch.zeros_like(flat)
        for e in available:
            rows, slots = (top_idx == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            expert_out = self.experts[str(e)](flat[rows])
            out.index_add_(0, rows, expert_out * gates[rows, slots].unsqueeze(-1))
        return out

    def _dense(
        self, flat: torch.Tensor, available: tuple[int, ...], top_idx: torch.Tensor, gates: torch.Tensor
    ) -> torch.Tensor:
        """Every available expert processes every token in a few batched products;
        a gate matrix that is zero outside each token's selected k combines them.

        Same result as :meth:`_sparse` up to rounding. An available expert that no
        token selects gets an all-zero gradient here instead of none.
        """
        experts = [self.experts[str(e)] for e in available]
        w1 = torch.stack([m.w1.weight for m in experts])  # [A, h, d]
        w3 = torch.stack([m.w3.weight for m in experts])  # [A, h, d]
        w2 = torch.stack([m.w2.weight for m in experts])  # [A, d, h]
        hidden = F.silu(torch.einsum("nd,ahd->anh", flat, w1)) * torch.einsum("nd,ahd->anh", flat, w3)
        expert_out = torch.einsum("anh,adh->and", hidden, w2)  # [A, N, d]
        # Gate matrix [N, A] from one-hot products (element-wise ops and a sum only).
        positions = torch.tensor(available, device=flat.device)
        chosen = (top_idx.unsqueeze(-1) == positions).to(gates.dtype)  # [N, k, A]
        gate_matrix = (chosen * gates.unsqueeze(-1)).sum(dim=1)
        return torch.einsum("and,na->nd", expert_out, gate_matrix)

    def _check_selection(self, top_idx: torch.Tensor, is_available: torch.Tensor, n_tokens: int) -> None:
        if top_idx.shape != (n_tokens, self.top_k):
            raise ValueError(f"select must return shape {(n_tokens, self.top_k)}, got {tuple(top_idx.shape)}")
        if not bool(is_available[top_idx].all()):
            raise ValueError("select chose an unavailable expert")
        ordered = top_idx.sort(dim=1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            raise ValueError("select must choose k distinct experts per token")
