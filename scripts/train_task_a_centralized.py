"""Centralized training on Task A (PLAN step 4; SRS S0-F-13, S0-F-20).

Usage, from the repository root::

    python -m scripts.train_task_a_centralized --config configs/task_a/centralized_pilot.toml [--set ...]

Trains the full model on freshly sampled Task A batches, evaluates on a fixed
"curve" set every ``eval_every`` steps, and at the end evaluates on the fixed
"gate" set (2,048 sequences per map, SRS D10), runs the routing-necessity
diagnostic, and saves the final weights in the run folder. Everything is
written to the run log; see docs/implementations/2026-10-07-centralized-task-a-pilot.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
import time
from pathlib import Path

import torch

from cassandra_chorus import paths, repro
from cassandra_chorus.config import RunSection, load_config
from cassandra_chorus.data import TaskA, TaskASection, bayes_optimal_accuracy
from cassandra_chorus.metrics import evaluate_task_a, routing_necessity
from cassandra_chorus.model import ModelSection, MoETransformer, count_parameters
from cassandra_chorus.ops import enforce, load_ops_settings, operations, run_preflight
from cassandra_chorus.runlog import RunLogger, make_run_id
from cassandra_chorus.train import OptimSection, check_schedule, make_optimizer, train_steps

PRECONDITION = 0.999  # SRS S0-A-01 precondition, per map (D10); meaningful for the marked variant


@dataclasses.dataclass(frozen=True)
class TrainSection:
    steps: int = 5000
    eval_every: int = 250
    curve_n_per_map: int = 256
    curve_seed: int = 1001
    gate_n_per_map: int = 2048
    gate_seed: int = 1002
    diagnostic_seed: int = 2001
    save_model: bool = True


@dataclasses.dataclass(frozen=True)
class CentralizedTaskAConfig:
    run: RunSection
    model: ModelSection
    task_a: TaskASection = dataclasses.field(default_factory=TaskASection)
    optim: OptimSection = dataclasses.field(default_factory=OptimSection)
    train: TrainSection = dataclasses.field(default_factory=TrainSection)


def check_consistency(cfg: CentralizedTaskAConfig, task: TaskA) -> None:
    problems = []
    if cfg.model.vocab_size != task.vocab_size:
        problems.append(f"model.vocab_size is {cfg.model.vocab_size} but Task A needs {task.vocab_size}")
    if cfg.model.context_length != cfg.task_a.seq_len:
        problems.append(f"model.context_length is {cfg.model.context_length} but task_a.seq_len is {cfg.task_a.seq_len}")
    if cfg.train.steps < 1 or cfg.train.eval_every < 1:
        problems.append("train.steps and train.eval_every must be positive")
    else:
        try:
            check_schedule(cfg.optim, cfg.train.steps)
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ValueError("inconsistent configuration: " + "; ".join(problems))


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the configuration asks for 'cuda' but CUDA is not available; set run.device=\"cpu\" deliberately")
    return torch.device(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, CentralizedTaskAConfig, args.overrides)
    task = TaskA(cfg.task_a)
    check_consistency(cfg, task)
    repro.prepare_process(cfg.run.determinism)  # before any CUDA work
    repro.set_cpu_threads(cfg.run.cpu_threads)
    device = resolve_device(cfg.run.device)
    n_maps = cfg.task_a.n_maps
    # Machine-level checks before any run folder exists (docs/implementations/2026-10-08-run-operations.md).
    ops_settings, ops_source = load_ops_settings()
    checks = run_preflight(ops_settings, paths.runs_root(), expected_seconds=cfg.train.steps * ops_settings.seconds_per_step)
    enforce(checks)

    with RunLogger.create(paths.runs_root(), make_run_id(cfg.run.name, cfg.run.seed)) as log:
        seeds = repro.seed_everything(cfg.run.seed)
        seeds["train_data"] = repro.derive_seed(cfg.run.seed, "task_a/train")
        determinism = repro.apply_determinism(cfg.run.determinism)
        model = MoETransformer(cfg.model).to(device)
        log.write_header(
            config=cfg, config_path=args.config, seeds=seeds,
            environment=repro.environment_info(), determinism=determinism,
        )
        log.write(
            "setup",
            parameters=count_parameters(model),
            vocab_size=task.vocab_size,
            maps_sha256=hashlib.sha256(task.maps.numpy().tobytes()).hexdigest(),
        )
        curve = task.evaluation_set(cfg.train.curve_n_per_map, cfg.train.curve_seed)
        gate = task.evaluation_set(cfg.train.gate_n_per_map, cfg.train.gate_seed)
        data_generator = torch.Generator().manual_seed(seeds["train_data"])
        batch_size = cfg.optim.batch_size

        def next_batch():
            batch = task.sample(batch_size, data_generator)
            return batch.inputs, batch.targets

        optimizer = make_optimizer(model, cfg.optim)
        with operations(log, ops_settings, ops_source, checks), log.capture_warnings():
            step, train_seconds = 0, 0.0
            first = evaluate_task_a(model, curve, device, n_maps)
            log.write("eval", step=0, sequences=0, **first)
            while step < cfg.train.steps:
                n = min(cfg.train.eval_every, cfg.train.steps - step)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                stats = train_steps(
                    model, optimizer, next_batch, n, cfg.optim, device, start_step=step, total_steps=cfg.train.steps
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                train_seconds += time.perf_counter() - t0
                step += n
                result = evaluate_task_a(model, curve, device, n_maps)
                log.write("eval", step=step, sequences=step * batch_size, train=dataclasses.asdict(stats),
                          train_seconds=train_seconds, **result)
                print(f"step {step:6d}  train ce {stats.mean_ce_loss:.4f}  curve acc {result['accuracy']:.4f}  "
                      f"min map {result['min_map_accuracy']:.4f}  ({train_seconds:.0f}s)", flush=True)

            gate_result = evaluate_task_a(model, gate, device, n_maps)
            diagnostic = routing_necessity(model, curve, device, n_maps, cfg.train.diagnostic_seed)
            bayes = bayes_optimal_accuracy(task, gate)
            precondition = gate_result["min_map_accuracy"] >= PRECONDITION if cfg.task_a.marked else None
            if cfg.train.save_model:
                torch.save(model.state_dict(), log.run_dir / "final_model.pt")
            log.write(
                "final", step=step, sequences=step * batch_size, train_seconds=train_seconds,
                gate=gate_result, bayes_optimal_gate=bayes, precondition_met=precondition,
                diagnostic=diagnostic, model_file="final_model.pt" if cfg.train.save_model else None,
            )

    print(f"run folder: {log.run_dir}")
    print(f"gate accuracy {gate_result['accuracy']:.5f}, min per map {gate_result['min_map_accuracy']:.5f}, "
          f"Bayes-optimal {bayes:.5f}, precondition met: {precondition}")
    print(f"diagnostic: trained {diagnostic['trained']['accuracy']:.4f}, "
          f"random routing {diagnostic['random_routing']['accuracy']:.4f}, "
          f"leave-one-out min {min(r['accuracy'] for r in diagnostic['leave_one_out']):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
