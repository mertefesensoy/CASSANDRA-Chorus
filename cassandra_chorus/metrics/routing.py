"""Routing metrics for Task A (SRS S0-F-21, S0-F-22, S0-F-27; definitions fixed by the owner as D22).

All functions are pure: they take count tables already in the run logs and
return numbers. Formulas: docs/implementations/2026-10-08-metrics-and-gate-a-report.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

Table = Sequence[Sequence[float]]  # rows: maps; columns: experts


def _log2(x: float) -> float:
    return math.log(x, 2)


def router_consistency(table: Table) -> float:
    """Normalized mutual information between map (rows) and chosen expert (columns).

    I(M; E) / H(M) in [0, 1]: 0 when routing ignores the map. With E = M and
    k = 2 choices per token the attainable maximum is 2/3. Returns 0.0 for an
    empty table or a single map.
    """
    total = float(sum(sum(row) for row in table))
    if total <= 0:
        return 0.0
    pm = [sum(row) / total for row in table]
    pe = [sum(table[m][e] for m in range(len(table))) / total for e in range(len(table[0]))]
    h_m = -sum(p * _log2(p) for p in pm if p > 0)
    if h_m <= 0:
        return 0.0
    mi = 0.0
    for m, row in enumerate(table):
        for e, count in enumerate(row):
            if count > 0:
                p = count / total
                mi += p * _log2(p / (pm[m] * pe[e]))
    return max(0.0, mi / h_m)


def top2_share(table: Table) -> float:
    """Mean over maps of the share of a map's choices going to its two most-chosen experts."""
    shares = []
    for row in table:
        s = sum(row)
        if s > 0:
            shares.append(sum(sorted(row, reverse=True)[:2]) / s)
    return sum(shares) / len(shares) if shares else 0.0


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence, base 2, of two distributions (normalized here); in [0, 1]."""
    sp, sq = float(sum(p)), float(sum(q))
    if sp <= 0 or sq <= 0:
        raise ValueError("both distributions need positive mass")
    p = [x / sp for x in p]
    q = [x / sq for x in q]
    a = [(x + y) / 2 for x, y in zip(p, q)]

    def kl(u, v):
        return sum(x * _log2(x / y) for x, y in zip(u, v) if x > 0)

    return min(1.0, max(0.0, 0.5 * kl(p, a) + 0.5 * kl(q, a)))


def expert_drift(holder_tables: Sequence[Table], held: Sequence[Sequence[int]], n_experts: int) -> dict:
    """Drift of one layer in one round from each holder's map-by-expert table.

    ``holder_tables[h]`` is worker h's table for this layer and ``held[h]`` the
    experts it held. For each expert with at least two holders that routed
    tokens to it, drift is the mean Jensen-Shannon divergence between the map
    distributions of its holder pairs. Returns ``{"mean", "max", "experts"}``,
    with mean and max None when no expert qualifies.
    """
    per_expert: dict[int, float] = {}
    for e in range(n_experts):
        columns = []
        for table, experts in zip(holder_tables, held):
            if e in experts:
                column = [row[e] for row in table]
                if sum(column) > 0:
                    columns.append(column)
        if len(columns) >= 2:
            pairs = [js_divergence(a, b) for a, b in combinations(columns, 2)]
            per_expert[e] = sum(pairs) / len(pairs)
    values = list(per_expert.values())
    return {
        "mean": sum(values) / len(values) if values else None,
        "max": max(values) if values else None,
        "experts": per_expert,
    }


def negligible_experts(expert_counts: Sequence[float], top_k: int, threshold_fraction: float = 0.1) -> list[int]:
    """Experts whose share of tokens is below ``threshold_fraction * k / E`` (S0-A-03).

    ``expert_counts`` counts routing choices for one layer over all positions;
    a token makes k choices, so the token count is ``sum / k``.
    """
    n_experts = len(expert_counts)
    tokens = sum(expert_counts) / top_k
    if tokens <= 0:
        return list(range(n_experts))
    limit = threshold_fraction * top_k / n_experts
    return [e for e, c in enumerate(expert_counts) if c / tokens < limit]


def routing_necessity_gap(trained_accuracy: float, random_routing_accuracy: float) -> float:
    """How much accuracy depends on routing: trained minus random-routing accuracy."""
    return trained_accuracy - random_routing_accuracy
