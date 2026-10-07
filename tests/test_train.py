from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys

import pytest
import torch

from cassandra_chorus import repro
from cassandra_chorus.data import TaskA, TaskASection
from cassandra_chorus.model import ModelSection, MoETransformer
from cassandra_chorus.paths import REPO_ROOT, RUNS_DIR_ENV
from cassandra_chorus.train import OptimSection, lr_at, make_optimizer, train_steps

TINY = ModelSection(vocab_size=34, context_length=64, d_model=32, n_layers=2, n_heads=4, expert_hidden=32)


def test_learning_rate_schedule():
    cfg = OptimSection(lr=1e-3, warmup_steps=4)
    assert [lr_at(s, cfg) for s in range(6)] == pytest.approx([2.5e-4, 5e-4, 7.5e-4, 1e-3, 1e-3, 1e-3])
    assert lr_at(0, dataclasses.replace(cfg, warmup_steps=0)) == 1e-3


def test_optimizer_settings():
    cfg = OptimSection()
    optimizer = make_optimizer(MoETransformer(TINY), cfg)
    group = optimizer.param_groups[0]
    assert group["weight_decay"] == 0.0 and group["betas"] == (0.9, 0.95)


def test_train_steps_learns_and_reports():
    torch.manual_seed(0)
    task = TaskA(TaskASection())
    model = MoETransformer(TINY)
    cfg = OptimSection(batch_size=16, warmup_steps=5, lr=3e-3)
    optimizer = make_optimizer(model, cfg)
    generator = torch.Generator().manual_seed(0)

    def next_batch():
        batch = task.sample(16, generator)
        return batch.inputs, batch.targets

    first = train_steps(model, optimizer, next_batch, 5, cfg, "cpu")
    later = train_steps(model, optimizer, next_batch, 60, cfg, "cpu", start_step=5)
    assert first.steps == 5 and first.sequences == 80 and first.scored_targets > 0
    assert later.mean_ce_loss < first.mean_ce_loss  # it learns something
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-3)  # warm-up finished


def test_non_finite_loss_stops_training():
    model = MoETransformer(TINY)
    with torch.no_grad():
        model.head.weight.fill_(float("nan"))
    inputs = torch.zeros(2, 64, dtype=torch.long)
    targets = torch.zeros(2, 64, dtype=torch.long)
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train_steps(model, make_optimizer(model, OptimSection()), lambda: (inputs, targets), 1, OptimSection(), "cpu")


def test_derive_seed():
    assert repro.derive_seed(7, "task_a/train") == repro.derive_seed(7, "task_a/train")
    assert repro.derive_seed(7, "task_a/train") != repro.derive_seed(11, "task_a/train")
    assert repro.derive_seed(7, "task_a/train") != repro.derive_seed(7, "other")
    assert 0 <= repro.derive_seed(7, "x") < 2**63


def test_centralized_entry_point_end_to_end_on_cpu(tmp_path):
    overrides = [
        'run.device="cpu"', "run.name=central-test", "model.d_model=32", "model.n_heads=4", "model.n_layers=2",
        "model.expert_hidden=32", "train.steps=6", "train.eval_every=3", "train.curve_n_per_map=2",
        "train.gate_n_per_map=3", "optim.batch_size=8",
    ]
    args = [sys.executable, "-m", "scripts.train_task_a_centralized", "--config", "configs/task_a/centralized_pilot.toml"]
    for o in overrides:
        args += ["--set", o]
    env = {**os.environ, RUNS_DIR_ENV: str(tmp_path)}
    result = subprocess.run(args, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    (run_dir,) = list(tmp_path.iterdir())
    records = [json.loads(line) for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    types = [r["type"] for r in records]
    assert types[:2] == ["header", "setup"] and types[-1] == "end" and records[-1]["status"] == "completed"
    evals = [r for r in records if r["type"] == "eval"]
    assert [r["step"] for r in evals] == [0, 3, 6]
    (final,) = [r for r in records if r["type"] == "final"]
    assert final["gate"]["scored"] > 0 and len(final["gate"]["per_map_accuracy"]) == 8
    assert set(final["diagnostic"]) == {"trained", "random_routing", "leave_one_out"}
    assert final["bayes_optimal_gate"] == 1.0 and final["precondition_met"] in (True, False)
    assert (run_dir / "final_model.pt").exists()


def test_entry_point_rejects_inconsistent_config(tmp_path):
    args = [sys.executable, "-m", "scripts.train_task_a_centralized", "--config", "configs/task_a/centralized_pilot.toml",
            "--set", "model.vocab_size=30", "--set", 'run.device="cpu"']
    env = {**os.environ, RUNS_DIR_ENV: str(tmp_path)}
    result = subprocess.run(args, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode != 0 and "Task A needs 34" in result.stderr
    assert not list(tmp_path.iterdir())  # failed before creating a run folder
