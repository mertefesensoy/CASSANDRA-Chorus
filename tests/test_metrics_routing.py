from __future__ import annotations

import math

import pytest

from cassandra_chorus.metrics import (
    expert_drift,
    js_divergence,
    negligible_experts,
    router_consistency,
    routing_necessity_gap,
    top2_share,
)


def test_consistency_is_zero_when_routing_ignores_the_map():
    table = [[10] * 8 for _ in range(8)]
    assert router_consistency(table) == pytest.approx(0.0, abs=1e-12)


def test_consistency_is_one_for_one_expert_per_map():
    table = [[10 if e == m else 0 for e in range(8)] for m in range(8)]
    assert router_consistency(table) == pytest.approx(1.0)


def test_consistency_ceiling_with_two_choices_per_token():
    # Each map split evenly over a dedicated pair; pairs overlap cyclically (8 maps, 8 experts).
    table = [[10 if e in (m, (m + 1) % 8) else 0 for e in range(8)] for m in range(8)]
    assert router_consistency(table) == pytest.approx(2 / 3)


def test_consistency_edge_cases():
    assert router_consistency([[0, 0], [0, 0]]) == 0.0
    assert router_consistency([[3, 5, 2]]) == 0.0  # one map: no map entropy


def test_top2_share():
    assert top2_share([[10] * 8]) == pytest.approx(0.25)
    assert top2_share([[5, 5, 0, 0], [1, 1, 1, 1]]) == pytest.approx((1.0 + 0.5) / 2)


def test_js_divergence():
    assert js_divergence([1, 2, 3], [2, 4, 6]) == pytest.approx(0.0, abs=1e-12)  # same after normalizing
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1.0)  # disjoint supports
    value = js_divergence([1, 1], [3, 1])
    p, q = [0.5, 0.5], [0.75, 0.25]
    a = [(x + y) / 2 for x, y in zip(p, q)]
    kl = lambda u, v: sum(x * math.log2(x / y) for x, y in zip(u, v) if x > 0)  # noqa: E731
    assert value == pytest.approx(0.5 * kl(p, a) + 0.5 * kl(q, a))
    with pytest.raises(ValueError):
        js_divergence([0, 0], [1, 1])


def test_expert_drift():
    # Two holders of expert 0 sending it disjoint maps (drift 1); expert 1 identical (drift 0);
    # expert 2 held by one worker only (undefined); expert 3 held by both but unused by one (undefined).
    a = [[5, 2, 1, 0], [0, 2, 0, 0]]
    b = [[0, 4, 0, 3], [7, 4, 0, 0]]
    drift = expert_drift([a, b], [[0, 1, 2, 3], [0, 1, 3]], 4)
    assert drift["experts"] == pytest.approx({0: 1.0, 1: 0.0})
    assert drift["mean"] == pytest.approx(0.5) and drift["max"] == pytest.approx(1.0)
    assert expert_drift([a], [[0, 1]], 4) == {"mean": None, "max": None, "experts": {}}


def test_negligible_experts():
    # k = 2, E = 8: threshold share 0.1 * 2 / 8 = 0.025 of tokens. 1000 tokens make 2000 choices.
    counts = [400, 400, 400, 400, 200, 176, 24, 0]
    assert negligible_experts(counts, top_k=2) == [6, 7]
    assert negligible_experts([0] * 4, top_k=2) == [0, 1, 2, 3]


def test_routing_necessity_gap():
    assert routing_necessity_gap(1.0, 0.49) == pytest.approx(0.51)
