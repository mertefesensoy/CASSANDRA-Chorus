from __future__ import annotations

import torch

from cassandra_chorus.data.task_a import TaskA, TaskASection
from cassandra_chorus.metrics.probe import (
    bypass_layer,
    bypass_layers,
    competence,
    evaluate_reading,
    probe_model,
    summarize,
    summarize_layer,
)
from cassandra_chorus.metrics.task_a import evaluate_task_a
from cassandra_chorus.model import ModelSection, MoETransformer

TASK = TaskA(TaskASection())
BATCH = TASK.evaluation_set(4, 1001)
CFG = ModelSection(vocab_size=TASK.vocab_size, context_length=64, d_model=32, n_layers=2, n_heads=4,
                   expert_hidden=32, n_experts=4)


def model(seed=0):
    torch.manual_seed(seed)
    return MoETransformer(CFG)


def test_identical_experts_make_every_pair_competent():
    m = model()
    with torch.no_grad():
        for block in m.blocks:
            for e in range(1, CFG.n_experts):
                block.moe.experts[str(e)].load_state_dict(block.moe.experts["0"].state_dict())
    probe = probe_model(m, BATCH, "cpu", TASK.cfg.n_maps)
    for layer in summarize(probe, CFG.n_experts):
        assert layer["competent_share"] == 1.0
        assert layer["specialization"] < 1e-9


def test_silent_layer_costs_nothing_to_bypass():
    m = model()
    with torch.no_grad():
        for e in range(CFG.n_experts):
            m.blocks[1].moe.experts[str(e)].w2.weight.zero_()  # layer 1's mixture outputs exactly zero
    probe = probe_model(m, BATCH, "cpu", TASK.cfg.n_maps)
    assert probe["layers"][1]["bypass_per_map"] == probe["trained_per_map"]
    assert summarize(probe, CFG.n_experts)[1]["bypass_cost_points"] == 0.0


def test_bypass_hook_is_removed():
    m = model()
    before = evaluate_task_a(m, BATCH, "cpu", TASK.cfg.n_maps)["per_map_accuracy"]
    with bypass_layer(m, 0):
        pass
    assert evaluate_task_a(m, BATCH, "cpu", TASK.cfg.n_maps)["per_map_accuracy"] == before
    assert not m.blocks[0].moe._forward_hooks


def test_probe_shape():
    probe = probe_model(model(), BATCH, "cpu", TASK.cfg.n_maps)
    assert len(probe["layers"]) == CFG.n_layers
    assert [p["experts"] for p in probe["layers"][0]["pairs"]] == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]


def test_summary_arithmetic():
    trained = [1.0, 0.5]
    layer = {
        "bypass_per_map": [0.90, 0.48],  # worst map loses 10 points
        "pairs": [
            {"experts": [0, 1], "per_map": [1.0, 0.5]},    # both cells competent
            {"experts": [0, 2], "per_map": [0.98, 0.5]},   # map 0 below 0.99 of trained
            {"experts": [1, 2], "per_map": [0.5, 0.496]},  # map 1 exactly 0.992 of trained: competent
        ],
    }
    s = summarize_layer(trained, layer, 3)
    assert abs(s["bypass_cost_points"] - 10.0) < 1e-9 and s["needed"]
    assert s["competent_share"] == 4 / 6
    # retention per pair and map: [0,1] -> (1.0, 1.0); [0,2] -> (0.98, 1.0); [1,2] -> (0.5, 0.992)
    table = competence(trained, layer["pairs"], 3)
    assert table[0] == [0.99, 1.0] and table[1] == [0.75, 0.996] and table[2] == [0.74, 0.996]
    assert abs(s["specialization"] - ((1.0 - 0.99) + (0.996 - 0.75) + (0.996 - 0.74)) / 3) < 1e-9


def summary(needed, share):
    return [{"needed": n, "competent_share": s} for n, s in zip(needed, share)]


def matrix(main, others):
    out = {}
    for variant in ("marked", "unmarked"):
        for seed in (7, 11, 19):
            out[("main", variant, seed)] = main
            for arm in ("centralized", "partial", "full"):
                out[(arm, variant, seed)] = others
    return out


def test_reading_supported():
    r = evaluate_reading(matrix(summary([False, True], [1.0, 0.9]), summary([True, True], [0.95, 0.3])))
    assert r["overall"] == "supported"
    assert r["rows"][("marked", 7)]["needed_layers"] == [1]  # compared on the main arm's needed layers only


def test_reading_bypassed_and_mixed_and_incomplete():
    assert evaluate_reading(matrix(summary([False, False], [1.0, 1.0]), summary([True, True], [0.2, 0.2])))["overall"] == "bypassed"
    assert evaluate_reading(matrix(summary([True, True], [0.2, 0.2]), summary([True, True], [0.5, 0.5])))["overall"] == "mixed"
    partial = matrix(summary([True], [0.9]), summary([True], [0.1]))
    del partial[("full", "unmarked", 19)]
    assert evaluate_reading(partial)["overall"] == "incomplete"


def test_bypassing_every_layer_equals_a_model_without_mixtures():
    m = model()
    with torch.no_grad():
        for block in m.blocks:
            for e in range(CFG.n_experts):
                block.moe.experts[str(e)].w2.weight.zero_()  # every mixture outputs zero already
    silent = evaluate_task_a(m, BATCH, "cpu", TASK.cfg.n_maps)["per_map_accuracy"]
    m2 = model()
    with bypass_layers(m2, range(CFG.n_layers)):
        bypassed = evaluate_task_a(m2, BATCH, "cpu", TASK.cfg.n_maps)["per_map_accuracy"]
    assert bypassed == silent  # same weights outside the experts (same seed), so the same outputs
    assert not any(block.moe._forward_hooks for block in m2.blocks)
