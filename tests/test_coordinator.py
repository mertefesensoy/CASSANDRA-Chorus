from __future__ import annotations

import pytest
import torch

from cassandra_chorus.coordinator import (
    NesterovOuter,
    PlainAverage,
    Slice,
    WorkerResult,
    WorkerSpec,
    assign_slices,
    holders,
    merge,
)
from cassandra_chorus.model import ModelSection, MoETransformer, expert_param_owner, router_param_layer

CFG = ModelSection(vocab_size=11, context_length=8, d_model=16, n_layers=2, n_heads=2, expert_hidden=8, n_experts=4)
E, L = CFG.n_experts, CFG.n_layers


def global_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {k: v.clone() for k, v in MoETransformer(CFG).state_dict().items()}


def worker_state(state, slice_: Slice, offset: float) -> dict[str, torch.Tensor]:
    """A worker's result: its slice of ``state``, every parameter shifted by ``offset``.

    For router rows of experts the worker does not hold, the row is returned
    unchanged, as a real masked worker returns it (zero gradient, no weight decay).
    """
    out = {}
    for name, t in state.items():
        owner = expert_param_owner(name)
        if owner is not None and not slice_.holds(*owner):
            continue
        new = t.clone() + offset
        layer = router_param_layer(name)
        if layer is not None:
            for e in range(t.shape[0]):
                if not slice_.holds(layer, e):
                    new[e] = t[e]
        out[name] = new
    return out


def make_slice(name, *layers) -> Slice:
    return Slice(name, {i: tuple(sorted(experts)) for i, experts in enumerate(layers)})


# ---------------------------------------------------------------- assignment


def specs(*caps):
    return [WorkerSpec(f"w{i}", c) for i, c in enumerate(caps)]


@pytest.mark.parametrize("cross_layer", ["same", "independent"])
def test_coverage_covers_every_expert_in_every_layer(cross_layer):
    for r in range(10):
        slices = assign_slices(specs(2, 2, 3), L, 6, 2, r, "coverage", cross_layer, seed=7)
        for layer in range(L):
            assert set().union(*(set(s.experts[layer]) for s in slices.values())) == set(range(6))
        assert [len(slices[f"w{i}"].experts[0]) for i in range(3)] == [2, 2, 3]  # unequal capacities kept


def test_coverage_needs_enough_capacity():
    with pytest.raises(ValueError, match="total capacity"):
        assign_slices(specs(2, 2), L, 6, 2, 0, "coverage")


def test_random_respects_capacity_and_is_deterministic():
    a = assign_slices(specs(2, 3), L, 8, 2, 4, "random", seed=1)
    b = assign_slices(specs(2, 3), L, 8, 2, 4, "random", seed=1)
    c = assign_slices(specs(2, 3), L, 8, 2, 5, "random", seed=1)
    assert a == b and a != c
    for s in a.values():
        for experts in s.experts.values():
            assert len(set(experts)) == len(experts) and all(0 <= e < 8 for e in experts)


def test_rolling_windows_tile_and_rotate():
    r0 = assign_slices(specs(3, 2, 3), 1, 8, 2, 0, "rolling")
    assert [r0[f"w{i}"].experts[0] for i in range(3)] == [(0, 1, 2), (3, 4), (5, 6, 7)]
    r1 = assign_slices(specs(3, 2, 3), 1, 8, 2, 1, "rolling")
    assert [r1[f"w{i}"].experts[0] for i in range(3)] == [(1, 2, 3), (4, 5), (0, 6, 7)]
    counts = {name: [0] * 8 for name in ("w0", "w1", "w2")}
    for r in range(8):  # over E rounds with shift 1, each worker holds each expert capacity times
        for name, s in assign_slices(specs(3, 2, 3), 1, 8, 2, r, "rolling").items():
            for e in s.experts[0]:
                counts[name][e] += 1
    assert counts == {"w0": [3] * 8, "w1": [2] * 8, "w2": [3] * 8}


def test_rolling_independent_layers_are_relabelled_but_still_cover():
    slices = assign_slices(specs(2, 2), 3, 4, 2, 0, "rolling", "independent", seed=3)
    for layer in range(3):
        assert set().union(*(set(s.experts[layer]) for s in slices.values())) == set(range(4))
    assert len({tuple(slices["w0"].experts[layer]) for layer in range(3)}) > 1  # some relabelling differs


def test_same_cross_layer_uses_identical_indices():
    slices = assign_slices(specs(2, 2), 3, 4, 2, 0, "random", "same", seed=5)
    for s in slices.values():
        assert s.experts[0] == s.experts[1] == s.experts[2]


@pytest.mark.parametrize(
    ("workers", "message"),
    [(specs(1, 3), "capacity"), (specs(5, 3), "capacity"), ([WorkerSpec("a", 2), WorkerSpec("a", 2)], "unique")],
)
def test_invalid_assignment(workers, message):
    with pytest.raises(ValueError, match=message):
        assign_slices(workers, L, E, 2, 0, "random")


def test_slices_build_slice_models():
    for s in assign_slices(specs(2, 3), L, E, 2, 0, "coverage", seed=1).values():
        model = MoETransformer(CFG, s.experts)
        assert model.held_experts() == dict(s.experts)


# ---------------------------------------------------------------- merge


def test_shared_part_is_the_example_weighted_average():
    g = global_state()
    a, b = make_slice("a", [0, 1], [0, 1]), make_slice("b", [2, 3], [2, 3])
    new, _ = merge(g, [WorkerResult(a, worker_state(g, a, 1.0), 30), WorkerResult(b, worker_state(g, b, 4.0), 10)])
    # (30 * 1 + 10 * 4) / 40 = 1.75
    assert torch.allclose(new["embed.weight"], g["embed.weight"] + 1.75)
    assert torch.allclose(new["blocks.0.attn.qkv.weight"], g["blocks.0.attn.qkv.weight"] + 1.75)


def test_expert_held_by_nobody_is_unchanged():
    g = global_state()
    a, b = make_slice("a", [0, 1], [0, 1]), make_slice("b", [0, 2], [1, 2])  # expert 3 held by nobody
    new, report = merge(g, [WorkerResult(a, worker_state(g, a, 1.0), 5), WorkerResult(b, worker_state(g, b, 3.0), 5)])
    for name in g:
        owner = expert_param_owner(name)
        if owner is not None and owner[1] == 3:
            assert torch.equal(new[name], g[name]), name
    # Layer 0: a holds {0, 1}, b holds {0, 2}. Layer 1: a holds {0, 1}, b holds {1, 2}.
    assert report.unchanged_experts == {0: [3], 1: [3]}
    assert report.holder_counts == {0: [2, 1, 1, 0], 1: [1, 2, 1, 0]}


def test_experts_are_averaged_over_their_holders_with_unequal_slices():
    g = global_state()
    a = make_slice("a", [0, 1, 2], [0, 1, 2])  # larger slice
    b = make_slice("b", [2, 3], [2, 3])
    new, _ = merge(g, [WorkerResult(a, worker_state(g, a, 1.0), 10), WorkerResult(b, worker_state(g, b, 5.0), 30)])
    w1 = "blocks.0.moe.experts.{}.w1.weight"
    assert torch.allclose(new[w1.format(0)], g[w1.format(0)] + 1.0)  # only a holds 0
    assert torch.allclose(new[w1.format(3)], g[w1.format(3)] + 5.0)  # only b holds 3
    assert torch.allclose(new[w1.format(2)], g[w1.format(2)] + (10 * 1 + 30 * 5) / 40)  # both hold 2


def test_dropped_worker_is_simply_absent():
    g = global_state()
    a, b, c = (make_slice(n, [0, 1], [0, 1]) for n in "abc")
    full, _ = merge(g, [WorkerResult(s, worker_state(g, s, o), 10) for s, o in ((a, 1.0), (b, 2.0), (c, 9.0))])
    dropped, report = merge(g, [WorkerResult(s, worker_state(g, s, o), 10) for s, o in ((a, 1.0), (b, 2.0))])
    assert torch.allclose(dropped["embed.weight"], g["embed.weight"] + 1.5)
    assert not torch.allclose(full["embed.weight"], dropped["embed.weight"])
    assert report.workers == ["a", "b"] and report.total_examples == 20


def test_router_rows_follow_their_holders_by_default():
    g = global_state()
    a, b = make_slice("a", [0, 1], [0, 1]), make_slice("b", [1, 2], [1, 2])
    results = [WorkerResult(a, worker_state(g, a, 2.0), 10), WorkerResult(b, worker_state(g, b, 6.0), 10)]
    new, report = merge(g, results)
    router = "blocks.0.moe.router.weight"
    assert torch.allclose(new[router][0], g[router][0] + 2.0)  # held by a only
    assert torch.allclose(new[router][1], g[router][1] + 4.0)  # held by both: (2 + 6) / 2
    assert torch.allclose(new[router][2], g[router][2] + 6.0)  # held by b only
    assert torch.equal(new[router][3], g[router][3])  # held by nobody: unchanged
    assert report.unchanged_router_rows[0] == [3]


def test_router_rule_all_dilutes_rows_by_holder_share():
    g = global_state()
    a, b = make_slice("a", [0, 1], [0, 1]), make_slice("b", [1, 2], [1, 2])
    results = [WorkerResult(a, worker_state(g, a, 2.0), 10), WorkerResult(b, worker_state(g, b, 6.0), 30)]
    new, _ = merge(g, results, router_rule="all")
    router = "blocks.0.moe.router.weight"
    # Row 0 is held only by a (n=10 of 40): old + 10 * 2 / 40 = old + 0.5, not old + 2.
    assert torch.allclose(new[router][0], g[router][0] + 0.5)
    # Row 2 only by b (30 of 40): old + 30 * 6 / 40 = old + 4.5.
    assert torch.allclose(new[router][2], g[router][2] + 4.5)


def test_full_model_workers_reduce_to_federated_averaging():
    g = global_state()
    everything = tuple(range(E))
    results = [WorkerResult(make_slice(n, everything, everything), worker_state(g, make_slice(n, everything, everything), o), w)
               for n, o, w in (("a", 1.0, 1), ("b", 2.0, 2), ("c", 4.0, 1))]
    new, _ = merge(g, results)
    expected_shift = (1 * 1 + 2 * 2 + 1 * 4) / 4
    for name in g:
        assert torch.allclose(new[name], g[name] + expected_shift, atol=1e-6), name


def test_merging_unchanged_copies_is_the_identity():
    g = global_state()
    a, b = make_slice("a", [0, 1], [2, 3]), make_slice("b", [1, 2], [0, 1])
    new, _ = merge(g, [WorkerResult(a, worker_state(g, a, 0.0), 3), WorkerResult(b, worker_state(g, b, 0.0), 7)])
    for name in g:
        assert torch.equal(new[name], g[name]), name


def test_no_results_and_inputs_untouched():
    g = global_state()
    snapshot = {k: v.clone() for k, v in g.items()}
    new, report = merge(g, [])
    assert all(torch.equal(new[k], g[k]) for k in g) and report.total_examples == 0
    a = make_slice("a", [0, 1], [0, 1])
    ws = worker_state(g, a, 1.0)
    ws_snapshot = {k: v.clone() for k, v in ws.items()}
    merge(g, [WorkerResult(a, ws, 5)])
    assert all(torch.equal(g[k], snapshot[k]) for k in g) and all(torch.equal(ws[k], ws_snapshot[k]) for k in ws)


def test_merged_state_loads_into_the_full_model():
    g = global_state()
    slices = assign_slices(specs(2, 2, 3), L, E, 2, 0, "coverage", seed=2)
    new, _ = merge(g, [WorkerResult(s, worker_state(g, s, 0.1 * i), 10 + i) for i, s in enumerate(slices.values())])
    MoETransformer(CFG).load_state_dict(new, strict=True)


@pytest.mark.parametrize("problem", ["extra", "missing", "shape", "duplicate", "examples"])
def test_invalid_results_raise(problem):
    g = global_state()
    a = make_slice("a", [0, 1], [0, 1])
    state = worker_state(g, a, 1.0)
    results = [WorkerResult(a, state, 5)]
    if problem == "extra":
        state["blocks.0.moe.experts.3.w1.weight"] = g["blocks.0.moe.experts.3.w1.weight"].clone()
    elif problem == "missing":
        del state["embed.weight"]
    elif problem == "shape":
        state["embed.weight"] = torch.zeros(3, 3)
    elif problem == "duplicate":
        results = results * 2
    else:
        results = [WorkerResult(a, state, 0)]
    with pytest.raises(ValueError):
        merge(g, results)


def test_holders_helper():
    slices = {"a": make_slice("a", [0, 1]), "b": make_slice("b", [1, 2])}
    assert holders(slices, 0, 1) == ["a", "b"] and holders(slices, 0, 3) == []


# ---------------------------------------------------------------- outer optimizer


def test_nesterov_with_lr1_momentum0_equals_plain_average():
    g = global_state()
    a, b = make_slice("a", [0, 1], [0, 1]), make_slice("b", [1, 2], [1, 2])
    results = [WorkerResult(a, worker_state(g, a, 1.0), 10), WorkerResult(b, worker_state(g, b, 3.0), 10)]
    plain, _ = merge(g, results, outer=PlainAverage())
    nest, _ = merge(g, results, outer=NesterovOuter(lr=1.0, momentum=0.0))
    for name in g:
        assert torch.allclose(plain[name], nest[name], atol=1e-6), name


def test_nesterov_momentum_and_untouched_entries():
    g = global_state()
    outer = NesterovOuter(lr=1.0, momentum=0.5)
    a = make_slice("a", [0, 1], [0, 1])  # experts 2, 3 and their router rows untouched
    s1, _ = merge(g, [WorkerResult(a, worker_state(g, a, 1.0), 4)], outer=outer)
    # Round 1: delta 1, v = 1, update = 1 * (1 + 0.5 * 1) = 1.5.
    assert torch.allclose(s1["embed.weight"], g["embed.weight"] + 1.5)
    router = "blocks.0.moe.router.weight"
    assert torch.equal(s1[router][2], g[router][2])
    assert torch.count_nonzero(outer.velocity[router][2]) == 0
    s2, _ = merge(s1, [WorkerResult(a, worker_state(s1, a, 1.0), 4)], outer=outer)
    # Round 2: delta 1, v = 0.5 * 1 + 1 = 1.5, update = 1 + 0.5 * 1.5 = 1.75.
    assert torch.allclose(s2["embed.weight"], s1["embed.weight"] + 1.75)
    expert3 = "blocks.0.moe.experts.3.w1.weight"
    assert torch.equal(s2[expert3], g[expert3]) and expert3 not in outer.velocity


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("router_rule", ["holders", "all"])
def test_merge_on_cuda_matches_cpu(router_rule):
    g = global_state()
    slices = assign_slices(specs(2, 3, 2), L, E, 2, 1, "coverage", seed=4)
    results = [WorkerResult(s, worker_state(g, s, 0.3 * (i + 1)), 7 + i) for i, s in enumerate(slices.values())]
    cpu, _ = merge(g, results, router_rule=router_rule, outer=NesterovOuter(0.7, 0.9))
    g_cuda = {k: v.cuda() for k, v in g.items()}
    results_cuda = [WorkerResult(r.slice, {k: v.cuda() for k, v in r.state.items()}, r.examples) for r in results]
    gpu, _ = merge(g_cuda, results_cuda, router_rule=router_rule, outer=NesterovOuter(0.7, 0.9))
    for name in g:
        assert gpu[name].device.type == "cuda"
        assert torch.allclose(gpu[name].cpu(), cpu[name], atol=1e-6), name
