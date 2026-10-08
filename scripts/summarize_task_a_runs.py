"""Summarize centralized Task A runs from their logs, as Markdown (for RESULTS.md).

Usage, from the repository root::

    python -m scripts.summarize_task_a_runs RUN_FOLDER [RUN_FOLDER ...]

Reads only ``log.jsonl`` in each folder. Reports the configuration hash, git
commit, steps and time, the first evaluation step at which the curve set's
minimum per-map accuracy reached 0.999, the gate-set results, the
routing-necessity diagnostic, and a descriptive routing statistic per layer:
for each map, the share of its scored tokens' routing assignments that went to
the map's two most-used experts, averaged over maps. Under uniform random
routing over E = 8 experts with k = 2 this is about 0.25; a map that always
uses the same two experts gives 1.0. It is descriptive only; the formal
router-consistency metric is defined in PLAN step 8.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(folder: Path) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {}
    for line in (folder / "log.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records.setdefault(record["type"], []).append(record)
    return records


def top2_share(map_expert_layer: list[list[int]]) -> float:
    shares = []
    for row in map_expert_layer:
        total = sum(row)
        if total:
            shares.append(sum(sorted(row, reverse=True)[:2]) / total)
    return sum(shares) / len(shares)


def summarize(folder: Path) -> str:
    rec = load(folder)
    header, final = rec["header"][0], rec["final"][0]
    cfg = header["config"]
    reached = next((e["step"] for e in rec["eval"] if e["min_map_accuracy"] >= 0.999), None)
    gate, diag = final["gate"], final["diagnostic"]
    loo = diag["leave_one_out"]
    lines = [
        f"### {header['run_id']}",
        "",
        f"- Config `{header['config_path']}`, hash `{header['config_hash'][:12]}`, commit `{header['git']['commit'][:7]}`"
        f" (uncommitted changes: {header['git']['dirty']}), seed {cfg['run']['seed']}, determinism `{cfg['run']['determinism']}`",
        f"- GPU: {header['environment']['gpu']['name'] if header['environment']['gpu'] else 'none'}; "
        f"PyTorch {header['environment']['torch']}",
        f"- Model: d {cfg['model']['d_model']}, {cfg['model']['n_layers']} layers, h {cfg['model']['expert_hidden']}, "
        f"E {cfg['model']['n_experts']}, k {cfg['model']['top_k']}, balance_coef {cfg['model']['balance_coef']}, "
        f"dispatch {cfg['model']['dispatch']}; {rec['setup'][0]['parameters']['total']:,} parameters",
        f"- Task A: marked {cfg['task_a']['marked']}, maps sha256 `{rec['setup'][0]['maps_sha256'][:12]}`",
        f"- Training: {final['step']} steps of {cfg['optim']['batch_size']} ({final['sequences']:,} sequences), "
        f"{final['train_seconds']:.0f} s of training time",
        f"- First curve evaluation with min per-map accuracy >= 0.999: step {reached}",
        f"- Gate set ({gate['scored']:,} scored positions): accuracy {gate['accuracy']:.5f}, "
        f"min per map {gate['min_map_accuracy']:.5f}, Bayes-optimal {final['bayes_optimal_gate']:.5f}, "
        f"precondition met: {final['precondition_met']}",
        f"- Diagnostic (curve set): trained router {diag['trained']['accuracy']:.4f} "
        f"(min map {diag['trained']['min_map_accuracy']:.4f}); random routing {diag['random_routing']['accuracy']:.4f} "
        f"(min map {diag['random_routing']['min_map_accuracy']:.4f}); leave-one-expert-out accuracy "
        f"{min(r['accuracy'] for r in loo):.4f} to {max(r['accuracy'] for r in loo):.4f}",
        "- Top-2 expert share per layer (gate set, mean over maps): "
        + ", ".join(f"layer {i}: {top2_share(layer):.3f}" for i, layer in enumerate(gate["map_expert_counts"])),
        f"- Warnings recorded: {sum(c['count'] for c in rec['warnings_summary'][0]['counts']) if rec.get('warnings_summary') else 'n/a'}; "
        f"end status: {rec['end'][0]['status']}",
    ]
    if rec.get("power_summary"):  # runs since the run-operations change (2026-10-08)
        ps = rec["power_summary"][0]
        mon, win = ps.get("monitor") or {}, ps.get("windows_events") or {}
        win_text = (
            f"Windows events: {win.get('ac_offline')} mains drop-outs, {win.get('standby_enter')} standby entries"
            if win.get("available") else f"Windows events unavailable ({win.get('reason')})"
        )
        lines.append(
            f"- Operations: keep-awake held {ps['keep_awake'].get('held')}; monitor saw {mon.get('ac_changes')} mains changes, "
            f"{mon.get('stalls')} stalls ({mon.get('total_stall_seconds')} s in total); {win_text}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    folders = [Path(a) for a in (sys.argv[1:] if argv is None else argv)]
    if not folders:
        print(__doc__)
        return 2
    print("\n\n".join(summarize(f) for f in folders))
    return 0


if __name__ == "__main__":
    sys.exit(main())
