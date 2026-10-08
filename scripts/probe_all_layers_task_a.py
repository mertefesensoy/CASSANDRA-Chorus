"""Post hoc check after the redundancy probe: zero every mixture layer at once (owner decision 2026-10-08).

Usage, from the repository root::

    python -m scripts.probe_all_layers_task_a [--runs-dir <runs folder>] [--device cuda]

The registered probe (scripts/probe_task_a.py) removed one mixture layer at a
time. In the marked variant no single layer of the sliced models was needed, so
this asks whether the mixture as a whole does any work: each saved Gate A model
is evaluated on its curve set with all of its mixture outputs set to zero (an
attention-only network). Not part of the registered reading; reported as a
labelled post hoc check in ``results/task_a_probe_all_layers_posthoc.md``.
See docs/implementations/2026-10-08-redundancy-probe.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from cassandra_chorus import repro
from cassandra_chorus.metrics.gate_a import collect
from cassandra_chorus.metrics.probe import bypass_layers
from cassandra_chorus.metrics.task_a import evaluate_task_a
from cassandra_chorus.paths import REPO_ROOT, runs_root
from scripts.probe_task_a import ARM_TITLES, load_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "task_a_probe_all_layers_posthoc.md")
    args = parser.parse_args(argv)
    repro.prepare_process("warn")
    repro.set_cpu_threads(1)
    repro.apply_determinism("warn")
    runs_dir = args.runs_dir or runs_root()
    rows = collect(p for p in runs_dir.iterdir() if p.is_dir())
    device = torch.device(args.device)
    results = []
    for (arm, variant, seed), row in sorted(rows.items()):
        model, task, batch = load_run(runs_dir / row.run_id, device)
        trained = evaluate_task_a(model, batch, device, task.cfg.n_maps)
        with bypass_layers(model, range(model.cfg.n_layers)):
            bypassed = evaluate_task_a(model, batch, device, task.cfg.n_maps)
        worst = 100 * max(t - b for t, b in zip(trained["per_map_accuracy"], bypassed["per_map_accuracy"]))
        results.append((arm, variant, seed, trained["accuracy"], bypassed["accuracy"], worst, row.run_id))
        print(f"{arm} {variant} s{seed}: trained {trained['accuracy']:.4f}, all mixtures zeroed {bypassed['accuracy']:.4f}", flush=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip():
        commit += " with uncommitted changes"
    lines = [
        "# Task A · Post hoc check: every mixture layer zeroed at once",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `python -m scripts.probe_all_layers_task_a` "
        f"(code `{commit}`) from the saved final models in `{runs_dir}`, {len(results)} models, curve sets.",
        "",
        "**Post hoc:** decided by the owner on 2026-10-08 after the registered probe's results were known. Not part of "
        "the registered reading and carries no verdict.",
        "",
        "| Arm | Variant | Seed | Trained accuracy | All mixtures zeroed | Worst-map cost (points) | Run ID |",
        "|---|---|---|---|---|---|---|",
    ]
    order = list(ARM_TITLES)
    for arm, variant, seed, acc, byp, worst, run_id in sorted(results, key=lambda r: (order.index(r[0]), r[1], r[2])):
        lines.append(f"| {ARM_TITLES[arm]} | {variant} | {seed} | {acc:.4f} | {byp:.4f} | {worst:.1f} | `{run_id}` |")
    lines.append("")
    args.out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"{len(results)} models; report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
