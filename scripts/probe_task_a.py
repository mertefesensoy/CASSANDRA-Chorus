"""Redundancy probe over the saved Gate A models (PLAN step 9a, SRS D24).

Usage, from the repository root::

    python -m scripts.probe_task_a [--runs-dir <runs folder>] [--device cuda]

Selects the completed Task A matrix runs exactly as the Gate A report does
(cassandra_chorus.metrics.gate_a.collect), loads each run's ``final_model.pt``,
and evaluates it on that run's curve set:

- the raw per-map accuracies go to ``<run>/probe.json`` (runs folder, not committed);
- ``results/task_a_redundancy_probe.md`` gets the tables and the registered reading;
- ``results/task_a_redundancy_probe.json`` gets every per-layer summary and
  competence table (for figures).

Two cross-checks per model are reported beside the probe (not part of the reading):
forcing each map's most-used pair, and single-layer random routing against the
mean forced accuracy. See docs/implementations/2026-10-08-redundancy-probe.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from cassandra_chorus import repro
from cassandra_chorus.config import build_config
from cassandra_chorus.data.task_a import TaskA, TaskASection
from cassandra_chorus.metrics.gate_a import collect
from cassandra_chorus.metrics.probe import (
    COMPETENT,
    NEEDED_POINTS,
    evaluate_reading,
    probe_model,
    retention,
    summarize,
)
from cassandra_chorus.metrics.task_a import evaluate_task_a, random_selector
from cassandra_chorus.model import ModelSection, MoETransformer
from cassandra_chorus.ops.power import keep_awake
from cassandra_chorus.paths import REPO_ROOT, runs_root

ARM_TITLES = {
    "centralized": "Centralized (reference)",
    "main": "Sliced, coverage (main arm)",
    "rolling": "Sliced, rolling assignment",
    "router_all": "Sliced, router averaged over all workers",
    "partial": "Partial update",
    "full": "Full-model local averaging",
}
CROSS_CHECK_SEED = 3001


def load_run(run_dir: Path, device):
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model_cfg = build_config(ModelSection, cfg["model"])
    task = TaskA(build_config(TaskASection, cfg["task_a"]))
    curve = cfg["eval"] if "eval" in cfg else cfg["train"]  # sliced runs keep it under eval, centralized under train
    batch = task.evaluation_set(curve["curve_n_per_map"], curve["curve_seed"])
    with torch.random.fork_rng(devices=[]):
        model = MoETransformer(model_cfg)
    model.load_state_dict(torch.load(run_dir / "final_model.pt", map_location="cpu", weights_only=True), strict=True)
    return model.to(device), task, batch


def single_layer_random(n_experts, top_k, layer, seed, device):
    rand = random_selector(n_experts, top_k, seed, device)

    def select(index, logits):
        return rand(index, logits) if index == layer else logits.topk(top_k, dim=-1).indices

    return select


def cross_checks(model, batch, device, n_maps, probe) -> list[dict]:
    """Per layer: retention of each map's most-used pair, and single-layer random routing vs mean forced accuracy."""
    cfg = model.cfg
    trained = evaluate_task_a(model, batch, device, n_maps)
    out = []
    for layer, data in enumerate(probe["layers"]):
        by_pair = {tuple(p["experts"]): p["per_map"] for p in data["pairs"]}
        top = []
        for m in range(n_maps):
            counts = trained["map_expert_counts"][layer][m]
            pair = tuple(sorted(sorted(range(cfg.n_experts), key=lambda e: -counts[e])[:2]))
            top.append(retention(by_pair[pair][m], trained["per_map_accuracy"][m]))
        rand = evaluate_task_a(model, batch, device, n_maps,
                               select=single_layer_random(cfg.n_experts, cfg.top_k, layer, CROSS_CHECK_SEED, device))
        forced_mean = sum(sum(p["per_map"]) / n_maps for p in data["pairs"]) / len(data["pairs"])
        out.append({"top_pair_retention_mean": sum(top) / n_maps, "top_pair_retention_min": min(top),
                    "random_routing_accuracy": rand["accuracy"], "mean_forced_accuracy": forced_mean})
    return out


def fmt_layers(values, digits):
    return ", ".join("n/a" if v is None else f"{v:.{digits}f}" for v in values)


def report(results, reading, runs_dir: Path, seconds: float) -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip():
        commit += " with uncommitted changes"
    lines = [
        "# Task A · Redundancy probe",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `python -m scripts.probe_task_a` (code `{commit}`) "
        f"from the saved final models in `{runs_dir}`, {len(results)} models, {seconds:.0f} s. The reading was registered "
        "before any probe number was computed (docs/implementations/2026-10-08-redundancy-probe.md, SRS D24).",
        "",
        f"## Reading: {reading['overall'].upper()}",
        "",
        f"A layer is needed if bypassing its mixture costs at least {NEEDED_POINTS:g} points on its worst map. A forced "
        f"(pair, map) cell is competent if it keeps at least {COMPETENT:g} of the trained accuracy on that map. Shares "
        "below are pooled over the main arm's needed layers.",
        "",
        "| Variant | Seed | Main arm's needed layers | Main share | Centralized | Partial | Full | (1) needed layer | (2) main share highest |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (variant, seed), row in sorted(reading["rows"].items()):
        if row.get("status") == "missing":
            lines.append(f"| {variant} | {seed} | missing | | | | | | |")
            continue
        o = row["other_share"]
        f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(f"| {variant} | {seed} | {row['needed_layers'] or 'none'} | {f(row['main_share'])} | {f(o['centralized'])} | "
                     f"{f(o['partial'])} | {f(o['full'])} | {'yes' if row['condition_1'] else 'no'} | {'yes' if row['condition_2'] else 'no'} |")
    lines += ["", "## By arm", ""]
    for arm, title in ARM_TITLES.items():
        rows = sorted((r for r in results if r["arm"] == arm), key=lambda r: (r["variant"], r["seed"]))
        if not rows:
            continue
        lines += [f"### {title}", "",
                  "| Variant | Seed | Bypass cost, points (layers 0 to 3) | Competent share | Specialization | Top-pair retention (mean) | Single-layer random routing | Mean forced accuracy | Run ID |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            s, c = r["summary"], r["cross_checks"]
            lines.append(
                f"| {r['variant']} | {r['seed']} | {fmt_layers([x['bypass_cost_points'] for x in s], 1)} | "
                f"{fmt_layers([x['competent_share'] for x in s], 3)} | {fmt_layers([x['specialization'] for x in s], 3)} | "
                f"{fmt_layers([x['top_pair_retention_mean'] for x in c], 3)} | {fmt_layers([x['random_routing_accuracy'] for x in c], 3)} | "
                f"{fmt_layers([x['mean_forced_accuracy'] for x in c], 3)} | `{r['run_id']}` |")
        lines.append("")
    lines += [
        "## Definitions",
        "",
        "- **Bypass cost:** trained accuracy minus accuracy with that layer's mixture output set to zero, worst map, in points.",
        "- **Competent share:** fraction of the 28 expert pairs × 8 maps whose forced accuracy keeps at least 0.99 of the trained accuracy on that map; 1 means any pair handles any map.",
        "- **Specialization:** per expert, the spread across maps of its mean retention over the 7 pairs that contain it; mean over experts. 0 means every expert serves every map equally well.",
        "- **Cross-checks (not part of the reading):** top-pair retention forces, for each map, the two experts the trained router uses most for it; single-layer random routing randomizes only that layer, to compare with the mean forced accuracy.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "task_a_redundancy_probe.md")
    args = parser.parse_args(argv)
    repro.prepare_process("warn")  # before any CUDA work, as in training
    repro.set_cpu_threads(1)
    repro.apply_determinism("warn")
    runs_dir = args.runs_dir or runs_root()
    rows = collect(p for p in runs_dir.iterdir() if p.is_dir())
    device = torch.device(args.device)
    t0 = time.perf_counter()
    results, summaries = [], {}
    with keep_awake() as awake:  # held from launch: standby froze earlier runs on this laptop
        print(f"keep-awake: {awake}", flush=True)
        for i, ((arm, variant, seed), row) in enumerate(sorted(rows.items()), 1):
            run_dir = runs_dir / row.run_id
            model, task, batch = load_run(run_dir, device)
            probe = probe_model(model, batch, device, task.cfg.n_maps)
            (run_dir / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
            summary = summarize(probe, model.cfg.n_experts)
            checks = cross_checks(model, batch, device, task.cfg.n_maps, probe)
            summaries[(arm, variant, seed)] = summary
            results.append({"arm": arm, "variant": variant, "seed": seed, "run_id": row.run_id,
                            "summary": summary, "cross_checks": checks})
            print(f"[{i}/{len(rows)}] {arm} {variant} s{seed}: needed {[x['needed'] for x in summary]}, "
                  f"competent {[round(x['competent_share'], 3) for x in summary]}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    reading = evaluate_reading(summaries)
    seconds = time.perf_counter() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report(results, reading, runs_dir, seconds), encoding="utf-8", newline="\n")
    data = {"reading": {"overall": reading["overall"],
                        "rows": [{"variant": v, "seed": s, **r} for (v, s), r in sorted(reading["rows"].items())]},
            "runs": results}
    args.out.with_suffix(".json").write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(results)} models; reading {reading['overall']}; {seconds:.0f} s; report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
