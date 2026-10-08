from __future__ import annotations

import json

import pytest

from cassandra_chorus.metrics.gate_a import AmbiguousRunsError, collect, evaluate_gate_a, run_row
from scripts.analyze_task_a import main as analyze_main

L, E, M = 2, 8, 8
UNIFORM_TABLE = [[10] * E for _ in range(M)]
UNIFORM_COUNTS = [[250] * E for _ in range(L)]  # 1000 tokens x k = 2 choices per layer


def write_run(root, name, seed, marked, *, gate=1.0, min_map=1.0, curve=None, random_acc=0.5, counts=None, sim=True,
              status="completed", rounds=None, run_id=None):
    run_id = run_id or f"2026T{seed}{name}_s{seed}"
    d = root / run_id
    d.mkdir(parents=True)
    config = {
        "run": {"name": name, "seed": seed},
        "model": {"n_layers": L, "n_experts": E, "top_k": 2},
        "task_a": {"marked": marked},
    }
    if sim:
        config["sim"] = {"capacity": 4, "capacities": [], "mode": "sliced"}
    records = [
        {"type": "header", "run_id": run_id, "config_hash": "h" + run_id, "config": config, "git": {"commit": "abc", "dirty": False}},
        *[{"type": "round", **r} for r in (rounds or [])],
        {"type": "final", "gate": {"accuracy": gate, "min_map_accuracy": min_map, "map_expert_counts": [UNIFORM_TABLE] * L,
                                   "expert_counts": counts or UNIFORM_COUNTS},
         "bayes_optimal_gate": 1.0 if marked else 0.951, "diagnostic": {"trained": {"accuracy": gate if curve is None else curve},
                        "random_routing": {"accuracy": random_acc},
                        "leave_one_out": [{"expert": e, "accuracy": (gate if curve is None else curve) - 0.01 * e}
                                          for e in range(E)]}},
        {"type": "end", "status": status},
    ]
    (d / "log.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return d


def full_matrix(root, **main_overrides):
    for seed in (7, 11, 19):
        write_run(root, "taskA-central-marked", seed, True, sim=False, random_acc=0.49)
        write_run(root, "taskA-central-unmarked", seed, False, gate=0.951, min_map=0.945, sim=False, random_acc=0.25)
        write_run(root, "taskA-sliced-coverage-marked", seed, True, **main_overrides)
        write_run(root, "taskA-sliced-coverage-unmarked", seed, False, gate=0.950, min_map=0.944)


def test_full_passing_matrix(tmp_path):
    full_matrix(tmp_path)
    result = evaluate_gate_a(collect(tmp_path.iterdir()))
    assert result["overall"] == "pass"
    assert set(result["all_seeds"].values()) == {"pass"}
    assert result["redundancy"]["marked"]["flag"] is False  # necessity 0.5 vs 0.51


def test_a01_fails_on_one_map_below_99_percent(tmp_path):
    full_matrix(tmp_path)
    rows = collect(tmp_path.iterdir())
    rows[("main", "marked", 11)].min_map_accuracy = 0.985
    result = evaluate_gate_a(rows)
    assert result["verdicts"]["S0-A-01"] == {7: "pass", 11: "fail", 19: "pass"}
    assert result["all_seeds"]["S0-A-01"] == "fail" and result["overall"] == "fail"


def test_a02_compares_with_the_same_seed(tmp_path):
    full_matrix(tmp_path)
    rows = collect(tmp_path.iterdir())
    rows[("main", "unmarked", 19)].gate_accuracy = 0.925  # 2.6 points below centralized 0.951
    verdicts = evaluate_gate_a(rows)["verdicts"]["S0-A-02"]
    assert verdicts == {7: "pass", 11: "pass", 19: "fail"}


def test_a03_negligible_expert(tmp_path):
    starved = [[400, 400, 400, 400, 200, 176, 24, 0], [250] * E]
    full_matrix(tmp_path, counts=starved)
    result = evaluate_gate_a(collect(tmp_path.iterdir()))
    assert result["all_seeds"]["S0-A-03"] == "fail"
    assert "negligible" in result["details"]["S0-A-03"][7]


def test_missing_runs_make_the_gate_incomplete(tmp_path):
    write_run(tmp_path, "taskA-central-marked", 7, True, sim=False)
    write_run(tmp_path, "taskA-sliced-coverage-marked", 7, True)
    result = evaluate_gate_a(collect(tmp_path.iterdir()))
    assert result["verdicts"]["S0-A-01"][7] == "pass" and result["verdicts"]["S0-A-01"][11] == "missing"
    assert result["overall"] == "incomplete"


def test_redundancy_flag_reported_but_not_a_failure(tmp_path):
    full_matrix(tmp_path, random_acc=0.9)  # necessity 0.1 vs centralized 0.51
    result = evaluate_gate_a(collect(tmp_path.iterdir()))
    assert result["overall"] == "pass"  # D23: pass with a recorded finding
    assert result["redundancy"]["marked"]["flag"] is True
    assert result["redundancy"]["marked"]["necessity_ratio_by_seed"][0] == pytest.approx(0.1 / 0.51)


def test_necessity_uses_the_curve_set_on_both_sides(tmp_path):
    d = write_run(tmp_path, "taskA-sliced-coverage-unmarked", 7, False, gate=0.951, curve=0.90, random_acc=0.40)
    assert run_row(d).necessity == pytest.approx(0.50)  # 0.90 - 0.40, not 0.951 - 0.40
    assert run_row(d).removal_cost == pytest.approx(0.07)  # worst single-expert removal, also on the curve set


def test_run_selection(tmp_path):
    write_run(tmp_path, "taskA-sliced-coverage-marked", 7, True, status="stopped", run_id="a_stopped")
    write_run(tmp_path, "taskA-central-pilot-bal0", 7, True, sim=False, run_id="b_pilot")
    write_run(tmp_path, "simcheck", 7, True, run_id="c_check")
    assert collect(tmp_path.iterdir()) == {}
    write_run(tmp_path, "taskA-sliced-coverage-marked", 7, True, run_id="d_done")
    assert list(collect(tmp_path.iterdir())) == [("main", "marked", 7)]
    write_run(tmp_path, "taskA-sliced-coverage-marked", 7, True, run_id="e_again")
    with pytest.raises(AmbiguousRunsError, match="d_done"):
        collect(tmp_path.iterdir())


def test_name_must_match_variant(tmp_path):
    d = write_run(tmp_path, "taskA-sliced-coverage-marked", 7, False)
    with pytest.raises(ValueError, match="run name says marked"):
        run_row(d)


def test_drift_from_round_records(tmp_path):
    a = [[5 if (e == 0 and m < 4) else 0 for e in range(E)] for m in range(M)]  # expert 0 gets maps 0-3
    b = [[5 if (e == 0 and m >= 4) else 0 for e in range(E)] for m in range(M)]  # expert 0 gets maps 4-7
    workers = [
        {"worker": "w0", "dropped": False, "held": {"0": [0, 1, 2, 3], "1": [0, 1, 2, 3]}, "map_expert_counts": [a, a]},
        {"worker": "w1", "dropped": False, "held": {"0": [0, 4, 5, 6], "1": [0, 4, 5, 6]}, "map_expert_counts": [b, a]},
        {"worker": "w2", "dropped": True, "held": {"0": [0], "1": [0]}},
    ]
    d = write_run(tmp_path, "taskA-sliced-coverage-marked", 7, True, rounds=[{"round": 0, "workers": workers}])
    row = run_row(d)
    assert row.drift_final[0]["mean"] == pytest.approx(1.0)  # disjoint maps in layer 0
    assert row.drift_final[1]["mean"] == pytest.approx(0.0)  # identical in layer 1
    assert row.drift_mean_over_rounds == pytest.approx([1.0, 0.0])


def test_report_for_a_full_matrix(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "report.md"
    full_matrix(runs, random_acc=0.9)
    write_run(runs, "taskA-fullavg-marked", 7, True)  # a comparison arm gets its own table
    assert analyze_main(["--runs-dir", str(runs), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## Criteria verdict: PASS" in text
    assert "Recorded finding (D23).** In 3 of 6 seed and variant pairs" in text
    assert "### Sliced, coverage (main arm)" in text and "### Full-model local averaging (S0-F-14)" in text
    assert "### Partial update" not in text  # arms with no runs are left out
    assert "| Full-model local averaging (S0-F-14) | marked | 1 |" in text  # arms side by side


def test_report_with_missing_runs(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "report.md"
    write_run(runs, "taskA-central-marked", 7, True, sim=False)
    analyze_main(["--runs-dir", str(runs), "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert "## Criteria verdict: INCOMPLETE" in text and "missing: no sliced marked run" in text
    assert "Recorded finding" not in text
