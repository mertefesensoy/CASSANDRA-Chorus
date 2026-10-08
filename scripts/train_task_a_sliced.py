"""Sliced and local-averaging training on Task A, simulated on one GPU (PLAN steps 6 and 7).

Usage, from the repository root::

    python -m scripts.train_task_a_sliced --config configs/task_a/sliced.toml [--set ...]
    python -m scripts.train_task_a_sliced --resume <run folder>

Each round assigns slices, trains the surviving workers one after another,
merges with the coordinator, evaluates the merged full model on the curve set
and writes a checkpoint, so a run resumes from its last completed round
(S0-N-04) and ends bitwise identical to an uninterrupted run. ``sim.mode``
selects sliced training, the partial-update arm (S0-F-26) or full-model local
averaging (S0-F-14). The final record has the same format as centralized runs.
See docs/implementations/2026-10-08-simulation-harness.md.
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
from cassandra_chorus.config import RunSection, config_hash, load_config
from cassandra_chorus.coordinator import WorkerResult, merge
from cassandra_chorus.data import TaskA, TaskASection, bayes_optimal_accuracy
from cassandra_chorus.metrics import evaluate_task_a, routing_necessity
from cassandra_chorus.model import ModelSection, MoETransformer, count_parameters
from cassandra_chorus.ops import enforce, load_ops_settings, operations, run_preflight
from cassandra_chorus.runlog import RunLogger, git_state, make_run_id
from cassandra_chorus.sim import (
    SimSection,
    data_generator,
    dropped_workers,
    load_checkpoint,
    make_outer,
    round_slices,
    save_checkpoint,
    skew_weights,
    train_worker,
    validate_sim,
    worker_specs,
)
from cassandra_chorus.train import OptimSection, check_schedule

PRECONDITION = 0.999


@dataclasses.dataclass(frozen=True)
class EvalSection:
    curve_n_per_map: int = 256
    curve_seed: int = 1001
    gate_n_per_map: int = 2048
    gate_seed: int = 1002
    diagnostic_seed: int = 2001
    save_model: bool = True
    worker_tables: bool = True  # each worker's map-by-expert routing table per round (for S0-F-27)


@dataclasses.dataclass(frozen=True)
class SlicedTaskAConfig:
    run: RunSection
    model: ModelSection
    task_a: TaskASection = dataclasses.field(default_factory=TaskASection)
    optim: OptimSection = dataclasses.field(default_factory=OptimSection)
    eval: EvalSection = dataclasses.field(default_factory=EvalSection)
    sim: SimSection = dataclasses.field(default_factory=SimSection)


class StopAfterRound(Exception):
    """Raised by --stop-after-round to simulate an interruption at a known point."""


def check(cfg: SlicedTaskAConfig, task: TaskA) -> None:
    problems = []
    if cfg.model.vocab_size != task.vocab_size:
        problems.append(f"model.vocab_size is {cfg.model.vocab_size} but Task A needs {task.vocab_size}")
    if cfg.model.context_length != cfg.task_a.seq_len:
        problems.append(f"model.context_length is {cfg.model.context_length} but task_a.seq_len is {cfg.task_a.seq_len}")
    for check_fn in (lambda: validate_sim(cfg.sim, cfg.model.n_experts, cfg.model.top_k),
                     lambda: check_schedule(cfg.optim, cfg.sim.rounds * cfg.sim.local_steps)):
        try:
            check_fn()
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ValueError("inconsistent configuration: " + "; ".join(problems))


def full_model(cfg: SlicedTaskAConfig, state, device) -> MoETransformer:
    with torch.random.fork_rng(devices=[]):
        model = MoETransformer(cfg.model)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--resume", type=Path, help="run folder to resume from its last completed round")
    parser.add_argument("--stop-after-round", type=int, default=None, help="stop cleanly after this round (testing)")
    args = parser.parse_args(argv)
    if (args.config is None) == (args.resume is None):
        parser.error("give exactly one of --config or --resume")
    if args.resume is not None and args.overrides:
        parser.error("--set cannot be combined with --resume; a resumed run keeps its configuration")

    config_path = args.config if args.config is not None else args.resume / "config.json"
    cfg = load_config(config_path, SlicedTaskAConfig, args.overrides)
    task = TaskA(cfg.task_a)
    check(cfg, task)
    sim = cfg.sim
    repro.prepare_process(cfg.run.determinism)  # before any CUDA work
    repro.set_cpu_threads(cfg.run.cpu_threads)
    device = torch.device(cfg.run.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the configuration asks for 'cuda' but CUDA is not available")
    total_steps = sim.n_workers * sim.local_steps * sim.rounds
    ops_settings, ops_source = load_ops_settings()
    checks = run_preflight(ops_settings, paths.runs_root() if args.resume is None else args.resume,
                           expected_seconds=total_steps * ops_settings.seconds_per_step)
    enforce(checks)

    n_maps, n_layers, n_experts, top_k = cfg.task_a.n_maps, cfg.model.n_layers, cfg.model.n_experts, cfg.model.top_k
    chash = config_hash(cfg)
    outer = make_outer(sim)
    seeds = repro.seed_everything(cfg.run.seed)
    if args.resume is not None:
        log = RunLogger.reopen(args.resume)
        determinism = repro.apply_determinism(cfg.run.determinism)
        start_round, global_state = load_checkpoint(args.resume, chash, outer, device)
        log.write("resume", from_round=start_round, environment=repro.environment_info(),
                  determinism=determinism, git=git_state()[0])
    else:
        log = RunLogger.create(paths.runs_root(), make_run_id(cfg.run.name, cfg.run.seed))
        determinism = repro.apply_determinism(cfg.run.determinism)
        global_state = {k: v.to(device) for k, v in MoETransformer(cfg.model).state_dict().items()}  # same init as centralized
        log.write_header(config=cfg, config_path=config_path, seeds=seeds,
                         environment=repro.environment_info(), determinism=determinism)
        log.write("setup", parameters=count_parameters(MoETransformer(cfg.model)), vocab_size=task.vocab_size,
                  maps_sha256=hashlib.sha256(task.maps.numpy().tobytes()).hexdigest(),
                  worker_steps=total_steps, sequences=total_steps * cfg.optim.batch_size)
        start_round = 0

    curve = task.evaluation_set(cfg.eval.curve_n_per_map, cfg.eval.curve_seed)
    gate = task.evaluation_set(cfg.eval.gate_n_per_map, cfg.eval.gate_seed)
    batch_size = cfg.optim.batch_size
    stopped = False
    with log:
        try:
            with operations(log, ops_settings, ops_source, checks), log.capture_warnings():
                for r in range(start_round, sim.rounds):
                    t0 = time.perf_counter()
                    slices = round_slices(sim, n_layers, n_experts, top_k, r, cfg.run.seed)
                    dropped = dropped_workers(sim, r, cfg.run.seed)
                    results, workers = [], []
                    for i, spec in enumerate(worker_specs(sim)):
                        held = {layer: list(e) for layer, e in slices[spec.name].experts.items()}
                        if spec.name in dropped:
                            workers.append({"worker": spec.name, "dropped": True, "held": held})
                            continue
                        generator = data_generator(cfg.run.seed, i, r)
                        weights = skew_weights(n_maps, sim.n_workers, i, sim.skew)

                        def next_batch(generator=generator, weights=weights):
                            batch = task.sample(batch_size, generator, map_weights=weights)
                            return batch.inputs, batch.targets

                        wr = train_worker(cfg.model, cfg.optim, global_state, slices[spec.name], sim.mode, next_batch,
                                          r, sim.local_steps, sim.rounds * sim.local_steps, device)
                        info = {"worker": spec.name, "dropped": False, "held": held, "train": dataclasses.asdict(wr.stats)}
                        if cfg.eval.worker_tables:
                            own = evaluate_task_a(wr.model, curve, device, n_maps)
                            info.update(curve_accuracy=own["accuracy"], map_expert_counts=own["map_expert_counts"])
                        workers.append(info)
                        results.append(WorkerResult(slices[spec.name], wr.state, wr.stats.sequences))
                        del wr
                    global_state, report = merge(global_state, results, sim.router_rule, outer)
                    merged = evaluate_task_a(full_model(cfg, global_state, device), curve, device, n_maps)
                    seconds = time.perf_counter() - t0
                    log.write("round", round=r, seconds=seconds, dropped=dropped, workers=workers,
                              merge=dataclasses.asdict(report), **merged)
                    save_checkpoint(log.run_dir, r, global_state, outer, chash)
                    print(f"round {r + 1:3d}/{sim.rounds}  merged curve acc {merged['accuracy']:.4f}  "
                          f"min map {merged['min_map_accuracy']:.4f}  dropped {len(dropped)}  ({seconds:.0f}s)", flush=True)
                    if args.stop_after_round is not None and r >= args.stop_after_round and r < sim.rounds - 1:
                        raise StopAfterRound(r)

                model = full_model(cfg, global_state, device)
                gate_result = evaluate_task_a(model, gate, device, n_maps)
                diagnostic = routing_necessity(model, curve, device, n_maps, cfg.eval.diagnostic_seed)
                bayes = bayes_optimal_accuracy(task, gate)
                precondition = gate_result["min_map_accuracy"] >= PRECONDITION if cfg.task_a.marked else None
                if cfg.eval.save_model:
                    torch.save(model.state_dict(), log.run_dir / "final_model.pt")
                log.write("final", rounds=sim.rounds, worker_steps=total_steps, sequences=total_steps * batch_size,
                          gate=gate_result, bayes_optimal_gate=bayes, precondition_met=precondition,
                          diagnostic=diagnostic, model_file="final_model.pt" if cfg.eval.save_model else None)
        except StopAfterRound as stop:
            log.close("stopped", after_round=stop.args[0])
            stopped = True

    print(f"run folder: {log.run_dir}")
    if stopped:
        print("stopped after round", args.stop_after_round, "- resume with --resume", log.run_dir)
        return 3
    print(f"gate accuracy {gate_result['accuracy']:.5f}, min per map {gate_result['min_map_accuracy']:.5f}, "
          f"Bayes-optimal {bayes:.5f}, precondition met: {precondition}")
    print(f"diagnostic: trained {diagnostic['trained']['accuracy']:.4f}, "
          f"random routing {diagnostic['random_routing']['accuracy']:.4f}, "
          f"leave-one-out min {min(r['accuracy'] for r in diagnostic['leave_one_out']):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
