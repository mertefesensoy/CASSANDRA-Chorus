"""Decoder-only mixture-of-experts transformer (SRS S0-F-01 to S0-F-03, S0-F-25).

``MoETransformer(cfg)`` is the full model; ``MoETransformer(cfg, held)`` is a
slice holding, for every layer, only the experts listed in ``held``. Parameter
names carry global expert indices, so a slice's state dict is the full model's
state dict filtered with :func:`expert_param_owner`.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from cassandra_chorus.model.layers import CausalSelfAttention
from cassandra_chorus.model.moe import MoELayer, MoEResult

HeldExperts = Mapping[int, Iterable[int]]


@dataclasses.dataclass(frozen=True)
class ModelSection:
    """Model configuration (the ``[model]`` section of a run configuration)."""

    vocab_size: int
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    expert_hidden: int
    n_experts: int = 8
    top_k: int = 2
    balance_coef: float = 0.0
    dropout: float = 0.0
    rope_base: float = 10000.0
    attention: Literal["explicit", "sdpa"] = "explicit"
    init_std: float = 0.02


def validate_model_config(cfg: ModelSection) -> None:
    """Raise ``ValueError`` listing every inconsistency in ``cfg``."""
    problems = []
    for name in ("vocab_size", "context_length", "d_model", "n_layers", "n_heads", "expert_hidden", "n_experts"):
        if getattr(cfg, name) < 1:
            problems.append(f"{name} must be at least 1")
    if cfg.n_heads >= 1 and cfg.d_model % cfg.n_heads != 0:
        problems.append(f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})")
    elif cfg.n_heads >= 1 and (cfg.d_model // cfg.n_heads) % 2 != 0:
        problems.append("the head size d_model / n_heads must be even for rotary embeddings")
    if cfg.top_k < 2:
        problems.append(
            "top_k must be at least 2: gate weights are a softmax over the selected k, "
            "so with k = 1 the weight is always 1 and the router gets no gradient"
        )
    if cfg.top_k > cfg.n_experts:
        problems.append(f"top_k ({cfg.top_k}) cannot exceed n_experts ({cfg.n_experts})")
    if not 0.0 <= cfg.dropout < 1.0:
        problems.append("dropout must be in [0, 1)")
    if cfg.balance_coef < 0:
        problems.append("balance_coef must be non-negative")
    if problems:
        raise ValueError("invalid model configuration: " + "; ".join(problems))


def normalize_held(cfg: ModelSection, held: HeldExperts | None) -> dict[int, tuple[int, ...]]:
    """Every layer's held experts as sorted tuples. ``None`` means all experts everywhere.

    A mapping must name every layer exactly once; indices must be in range and
    each layer must hold at least ``top_k`` experts.
    """
    if held is None:
        return {layer: tuple(range(cfg.n_experts)) for layer in range(cfg.n_layers)}
    layers = set(held)
    expected = set(range(cfg.n_layers))
    if layers != expected:
        raise ValueError(f"held must name every layer {sorted(expected)}, got {sorted(layers)}")
    result = {}
    for layer in range(cfg.n_layers):
        experts = tuple(sorted(set(held[layer])))
        if not experts or experts[0] < 0 or experts[-1] >= cfg.n_experts:
            raise ValueError(f"layer {layer}: expert indices must be in 0..{cfg.n_experts - 1}, got {experts}")
        if len(experts) < cfg.top_k:
            raise ValueError(f"layer {layer} holds {len(experts)} experts, fewer than top_k={cfg.top_k}")
        result[layer] = experts
    return result


@dataclasses.dataclass
class ModelOutput:
    """``logits`` ``[B, T, V]``; losses are None without targets.

    ``balance_loss`` is the mean over layers of the per-layer load-balancing
    loss. ``expert_counts`` ``[n_layers, E]`` counts tokens routed to each
    global expert. ``routing`` (only if requested) holds each layer's top-k
    expert indices ``[B, T, k]``.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None
    ce_loss: torch.Tensor | None
    balance_loss: torch.Tensor
    expert_counts: torch.Tensor
    routing: list[torch.Tensor] | None


class Block(nn.Module):
    def __init__(self, cfg: ModelSection, held: tuple[int, ...]) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.context_length, cfg.rope_base, cfg.dropout, cfg.attention
        )
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.moe = MoELayer(cfg.d_model, cfg.expert_hidden, cfg.n_experts, cfg.top_k, held)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, allowed: Iterable[int] | None) -> tuple[torch.Tensor, MoEResult]:
        x = x + self.attn(self.norm1(x))
        routed = self.moe(self.norm2(x), allowed)
        return x + self.drop(routed.output), routed


class MoETransformer(nn.Module):
    """Pre-norm decoder-only transformer whose feed-forward layers are mixtures of experts."""

    def __init__(self, cfg: ModelSection, held: HeldExperts | None = None) -> None:
        super().__init__()
        validate_model_config(cfg)
        held_map = normalize_held(cfg, held)
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, held_map[layer]) for layer in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def held_experts(self) -> dict[int, tuple[int, ...]]:
        return {layer: block.moe.held for layer, block in enumerate(self.blocks)}

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        expert_mask: HeldExperts | None = None,
        return_routing: bool = False,
    ) -> ModelOutput:
        """Run the model on token indices ``idx`` of shape ``[B, T]``.

        ``expert_mask`` optionally restricts routing per layer to the given
        global experts (layers not named are unrestricted). Targets equal to
        -100 are ignored by the cross-entropy.
        """
        if idx.dim() != 2:
            raise ValueError(f"idx must have shape [B, T], got {tuple(idx.shape)}")
        if idx.shape[1] > self.cfg.context_length:
            raise ValueError(f"sequence length {idx.shape[1]} exceeds context_length {self.cfg.context_length}")
        if expert_mask is not None:
            unknown = set(expert_mask) - set(range(self.cfg.n_layers))
            if unknown:
                raise ValueError(f"expert_mask names layers that do not exist: {sorted(unknown)}")

        x = self.embed(idx)
        routed: list[MoEResult] = []
        for layer, block in enumerate(self.blocks):
            allowed = None if expert_mask is None else expert_mask.get(layer)
            x, result = block(x, allowed)
            routed.append(result)
        logits = self.head(self.norm(x))

        balance_loss = torch.stack([r.balance_loss for r in routed]).mean()
        ce_loss = loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
            loss = ce_loss + self.cfg.balance_coef * balance_loss
        return ModelOutput(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            balance_loss=balance_loss,
            expert_counts=torch.stack([r.counts for r in routed]),
            routing=[r.top_idx for r in routed] if return_routing else None,
        )


_EXPERT_PARAM = re.compile(r"^blocks\.(\d+)\.moe\.experts\.(\d+)\.")


def expert_param_owner(name: str) -> tuple[int, int] | None:
    """``(layer, global expert)`` for an expert parameter name, ``None`` for the shared part."""
    match = _EXPERT_PARAM.match(name)
    return (int(match.group(1)), int(match.group(2))) if match else None


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Parameter counts split into the shared part and the experts."""
    shared = experts = 0
    for name, param in model.named_parameters():
        if expert_param_owner(name) is None:
            shared += param.numel()
        else:
            experts += param.numel()
    return {"total": shared + experts, "shared": shared, "experts": experts}


def expected_parameter_count(cfg: ModelSection, held: HeldExperts | None = None) -> dict[str, int]:
    """Analytic parameter count (see the step 2 implementation doc)."""
    d, h = cfg.d_model, cfg.expert_hidden
    per_layer_shared = 4 * d * d + 2 * d + cfg.n_experts * d
    shared = 2 * cfg.vocab_size * d + d + cfg.n_layers * per_layer_shared
    held_map = normalize_held(cfg, held)
    experts = sum(len(e) for e in held_map.values()) * 3 * d * h
    return {"total": shared + experts, "shared": shared, "experts": experts}
