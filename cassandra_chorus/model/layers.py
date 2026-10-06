"""Transformer building blocks: rotary position embedding, causal self-attention, SwiGLU expert.

Formulas are written out in ``docs/implementations/2026-10-07-moe-transformer.md``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary position embedding, half-split convention.

    For a vector split into halves (a, b) at position p:
    (a cos(p theta) - b sin(p theta), a sin(p theta) + b cos(p theta)),
    with theta_j = base^(-2j / head_size). Holds no parameters; the tables are
    non-persistent buffers, so they never appear in a state dict.
    """

    def __init__(self, head_size: int, context_length: int, base: float) -> None:
        super().__init__()
        if head_size % 2 != 0:
            raise ValueError(f"rotary embedding needs an even head size, got {head_size}")
        inv_freq = base ** (-torch.arange(0, head_size, 2, dtype=torch.float64) / head_size)
        angles = torch.outer(torch.arange(context_length, dtype=torch.float64), inv_freq)
        self.register_buffer("cos", angles.cos().to(torch.float32), persistent=False)
        self.register_buffer("sin", angles.sin().to(torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` of shape ``[..., T, head_size]`` by position 0 to T-1."""
        tokens = x.shape[-2]
        half = x.shape[-1] // 2
        cos = self.cos[:tokens].to(x.dtype)
        sin = self.sin[:tokens].to(x.dtype)
        a, b = x[..., :half], x[..., half:]
        return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary positions and no biases.

    ``impl="explicit"`` computes scores, mask, softmax and product directly;
    ``impl="sdpa"`` uses PyTorch's fused kernel, which in fp32 on the reference
    GPU has a non-deterministic backward pass unless determinism is ``strict``.
    """

    def __init__(
        self, d_model: int, n_heads: int, context_length: int, rope_base: float, dropout: float, impl: str
    ) -> None:
        super().__init__()
        if impl not in ("explicit", "sdpa"):
            raise ValueError(f"unknown attention implementation {impl!r}")
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.impl = impl
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_size, context_length, rope_base)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        future = torch.ones(context_length, context_length, dtype=torch.bool).triu(1)
        self.register_buffer("future", future, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        shape = (batch, tokens, self.n_heads, self.head_size)
        q = self.rope(q.view(shape).transpose(1, 2))
        k = self.rope(k.view(shape).transpose(1, 2))
        v = v.view(shape).transpose(1, 2)
        if self.impl == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            scores = scores.masked_fill(self.future[:tokens, :tokens], float("-inf"))
            y = self.attn_drop(F.softmax(scores, dim=-1)) @ v
        y = y.transpose(1, 2).contiguous().view(batch, tokens, channels)
        return self.resid_drop(self.out(y))


class SwiGLUExpert(nn.Module):
    """One expert: ``w2(silu(w1 x) * w3 x)``, no biases."""

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
