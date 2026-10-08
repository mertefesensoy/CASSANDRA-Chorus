from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from cassandra_chorus.config import config_hash
from cassandra_chorus.coordinator import NesterovOuter, Slice
from cassandra_chorus.data import TaskA, TaskASection
from cassandra_chorus.model import ModelSection, MoETransformer, expert_param_owner, router_param_layer
from cassandra_chorus.paths import REPO_ROOT, RUNS_DIR_ENV
from cassandra_chorus.sim import (
    SimSection,
    data_generator,
    dropped_workers,
    load_checkpoint,
    round_slices,
    save_checkpoint,
    skew_weights,
    train_worker,
    validate_sim,
)
from cassandra_chorus.train import OptimSection, make_optimizer, train_steps

CFG = ModelSection(vocab_size=34, context_length=64, d_model=32, n_layers=2, n_heads=4, expert_hidden=32, n_experts=8)
OPT = OptimSection(batch_size=8, warmup_steps=1)
TASK = TaskA(TaskASection())


def global_state():
    torch.manual_seed(0)
    return {k: v.clone() for k, v in MoETransformer(CFG).state_dict().items()}


def batches(seed=1):
    gen = torch.Generator().manual_seed(seed)

    def next_batch():
        b = TASK.sample(8, gen)
        return b.inputs, b.targets
    return next_batch


SLICE = Slice("w0", {0: (0, 2, 4, 6), 1: (1, 3, 5, 7)})


# ---------------------------------------------------------------- worker modes


def test_sliced_worker_returns_its_slice_and_leaves_unheld_router_rows():
    g = global_state()
    wr = train_worker(CFG, OPT, g, SLICE, "sliced", batches(), 0, 3, 3, "cpu")
    expected = {k for k in g if (o := expert_param_owner(k)) is None or SLICE.holds(*o)}
    assert set(wr.state) == expected
    for name, value in wr.state.items():
        layer = router_param_layer(name)
        if layer is not None:
            for e in range(8):
                if SLICE.holds(layer, e):
                    assert not torch.equal(value[e], g[name][e])  # trained
                else:
                    assert torch.equal(value[e], g[name][e])  # zero gradient, no weight decay: unchanged


def test_partial_worker_freezes_unassigned_experts_and_trains_every_router_row():
    g = global_state()
    wr = train_worker(CFG, OPT, g, SLICE, "partial", batches(), 0, 3, 3, "cpu")
    assert set(wr.state) == {k for k in g if (o := expert_param_owner(k)) is None or SLICE.holds(*o)}
    full = wr.model.state_dict()
    for name in g:
        owner = expert_param_owner(name)
        if owner is not None and not SLICE.holds(*owner):
            assert torch.equal(full[name], g[name]), name  # frozen
    router = wr.state["blocks.0.moe.router.weight"]
    assert all(not torch.equal(router[e], g["blocks.0.moe.router.weight"][e]) for e in range(8))


def test_full_worker_returns_everything():
    g = global_state()
    everything = Slice("w0", {0: tuple(range(8)), 1: tuple(range(8))})
    wr = train_worker(CFG, OPT, g, everything, "full", batches(), 0, 2, 2, "cpu")
    assert set(wr.state) == set(g)


def test_one_full_worker_reproduces_centralized_training_bitwise():
    g = global_state()
    everything = Slice("w0", {0: tuple(range(8)), 1: tuple(range(8))})
    wr = train_worker(CFG, OPT, g, everything, "full", batches(), 0, 4, 4, "cpu")
    model = MoETransformer(CFG)
    model.load_state_dict(g)
    train_steps(model, make_optimizer(model, OPT), batches(), 4, OPT, "cpu", start_step=0, total_steps=4)
    for name, value in model.state_dict().items():
        assert torch.equal(wr.state[name], value), name


# ---------------------------------------------------------------- harness pieces


def test_skew_weights():
    assert skew_weights(8, 4, 1, 0.0) == [0.125] * 8
    only_own = skew_weights(8, 4, 1, 1.0)
    assert only_own == [0, 0.5, 0, 0, 0, 0.5, 0, 0]
    half = skew_weights(8, 4, 2, 0.5)
    assert abs(sum(half) - 1.0) < 1e-12 and half[2] == half[6] > half[0]


def test_dropped_workers_are_deterministic():
    sim = SimSection(drop_prob=0.5)
    assert dropped_workers(sim, 3, 7) == dropped_workers(sim, 3, 7)
    assert dropped_workers(SimSection(drop_prob=0.0), 3, 7) == []
    assert dropped_workers(SimSection(drop_prob=1.0), 3, 7) == ["w0", "w1", "w2", "w3"]


def test_data_streams_depend_only_on_seed_worker_and_round():
    a = torch.randint(0, 100, (5,), generator=data_generator(7, 1, 2))
    b = torch.randint(0, 100, (5,), generator=data_generator(7, 1, 2))
    c = torch.randint(0, 100, (5,), generator=data_generator(7, 2, 2))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_full_mode_slices_hold_everything():
    slices = round_slices(SimSection(mode="full", router_rule="all"), 2, 8, 2, 0, 7)
    assert all(s.experts == {0: tuple(range(8)), 1: tuple(range(8))} for s in slices.values())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rounds": 9}, "equal-compute"),
        ({"mode": "partial"}, "router_rule must be 'all'"),
        ({"capacity": 1}, "capacities"),
        ({"capacities": (4, 4)}, "entries"),
        ({"drop_prob": 2.0}, "drop_prob"),
    ],
)
def test_invalid_sim(changes, message):
    import dataclasses

    with pytest.raises(ValueError, match=message):
        validate_sim(dataclasses.replace(SimSection(), **changes), 8, 2)


def test_checkpoint_round_trip(tmp_path):
    g = global_state()
    outer = NesterovOuter(0.7, 0.9)
    outer.velocity = {"embed.weight": torch.ones_like(g["embed.weight"])}
    save_checkpoint(tmp_path, 4, g, outer, "hash-a")
    restored = NesterovOuter(0.7, 0.9)
    next_round, state = load_checkpoint(tmp_path, "hash-a", restored, "cpu")
    assert next_round == 5 and all(torch.equal(state[k], g[k]) for k in g)
    assert torch.equal(restored.velocity["embed.weight"], outer.velocity["embed.weight"])
    with pytest.raises(ValueError, match="different configuration"):
        load_checkpoint(tmp_path, "hash-b", NesterovOuter(), "cpu")


# ---------------------------------------------------------------- entry point, resume


TINY = [
    "model.d_model=32", "model.n_heads=4", "model.n_layers=2", "model.expert_hidden=32",
    "sim.n_workers=2", "sim.local_steps=3", "sim.rounds=3", "sim.equal_compute_steps=18",
    "optim.batch_size=8", "optim.warmup_steps=1", "eval.curve_n_per_map=2", "eval.gate_n_per_map=2",
]


def run_entry(runs_dir, *args):
    env = {**os.environ, RUNS_DIR_ENV: str(runs_dir)}
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    result = subprocess.run([sys.executable, "-m", "scripts.train_task_a_sliced", *args],
                            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    return result


def config_args(device, *extra):
    args = ["--config", "configs/task_a/sliced.toml"]
    for o in [f'run.device="{device}"', *TINY, *extra]:
        args += ["--set", o]
    return args


def final_bytes(run_dir):
    state = torch.load(run_dir / "final_model.pt")
    return b"".join(state[k].cpu().numpy().tobytes() for k in sorted(state))


def records(run_dir):
    return [json.loads(line) for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")])]
)
def test_resumed_run_ends_bitwise_identical(tmp_path, device):
    whole = run_entry(tmp_path / "whole", *config_args(device, "run.name=whole"))
    assert whole.returncode == 0, whole.stderr
    (whole_dir,) = list((tmp_path / "whole").iterdir())

    first = run_entry(tmp_path / "split", *config_args(device, "run.name=split"), "--stop-after-round", "0")
    assert first.returncode == 3, first.stderr
    (split_dir,) = list((tmp_path / "split").iterdir())
    assert not (split_dir / "final_model.pt").exists() and (split_dir / "checkpoint.pt").exists()
    resumed = run_entry(tmp_path / "split", "--resume", str(split_dir))
    assert resumed.returncode == 0, resumed.stderr

    assert final_bytes(whole_dir) == final_bytes(split_dir)
    a, b = records(whole_dir), records(split_dir)
    final_a = [r for r in a if r["type"] == "final"][0]
    final_b = [r for r in b if r["type"] == "final"][0]
    assert final_a["gate"] == final_b["gate"] and final_a["diagnostic"] == final_b["diagnostic"]
    types_b = [r["type"] for r in b]
    assert types_b.count("round") == 3 and "resume" in types_b
    assert [r["status"] for r in b if r["type"] == "end"] == ["stopped", "completed"]


@pytest.mark.parametrize("extra", [("sim.mode=partial", "sim.router_rule=all"), ("sim.mode=full", "sim.router_rule=all"),
                                   ("sim.policy=rolling",), ("sim.drop_prob=0.5",), ("sim.outer=nesterov",)])
def test_other_arms_run_end_to_end(tmp_path, extra):
    result = run_entry(tmp_path, *config_args("cpu", "run.name=arm", "eval.save_model=false", *extra))
    assert result.returncode == 0, result.stderr
    (run_dir,) = list(tmp_path.iterdir())
    recs = records(run_dir)
    rounds = [r for r in recs if r["type"] == "round"]
    assert len(rounds) == 3 and [r["type"] for r in recs][-1] == "end"
    if "sim.drop_prob=0.5" in extra:
        expected = SimSection(n_workers=2, local_steps=3, rounds=3, equal_compute_steps=18, drop_prob=0.5)
        for r in rounds:  # the logged drops are exactly the seeded draws, and dropped workers are not merged
            assert r["dropped"] == dropped_workers(expected, r["round"], 7)
            assert r["merge"]["workers"] == [w for w in ("w0", "w1") if w not in r["dropped"]]
        assert all(len(r["workers"]) == 2 for r in rounds)


def test_resume_refuses_overrides(tmp_path):
    result = run_entry(tmp_path, "--resume", str(tmp_path), "--set", "sim.rounds=4")
    assert result.returncode != 0 and "cannot be combined" in result.stderr


def test_queue_runner(tmp_path):
    queue = tmp_path / "q.toml"
    queue.write_text(
        'stop_on_failure = true\n[[run]]\nmodule = "json.tool"\nargs = ["--help"]\n'
        '[[run]]\nmodule = "json.tool"\nargs = ["no-such-file.json"]\n[[run]]\nmodule = "json.tool"\nargs = ["--help"]\n',
        encoding="utf-8",
    )
    env = {**os.environ, RUNS_DIR_ENV: str(tmp_path / "runs")}
    result = subprocess.run([sys.executable, "-m", "scripts.ops.run_queue", str(queue)], cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 1  # the second entry fails and the queue stops
    (log,) = list((tmp_path / "runs" / "queue_logs").iterdir())
    events = [json.loads(line)["event"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert events == ["queue_start", "start", "finish", "start", "finish", "queue_stopped"]


def test_gate_a_queue_file_matches_the_decisions():
    from scripts.ops.run_queue import load_queue

    runs, stop = load_queue(REPO_ROOT / "configs" / "queues" / "gate_a_main.toml")
    assert stop and len(runs) == 6
    combos = {(next(a for a in r["args"] if a.startswith("run.seed=")), next(a for a in r["args"] if a.startswith("task_a.marked="))) for r in runs}
    assert combos == {(f"run.seed={s}", f"task_a.marked={m}") for s in (7, 11, 19) for m in ("true", "false")}
