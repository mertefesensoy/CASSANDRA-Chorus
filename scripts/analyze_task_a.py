"""Write the Task A Gate A report from run logs (PLAN step 8; SRS S0-F-20 to S0-F-24, S0-F-27, section 4.4).

Usage, from the repository root::

    python -m scripts.analyze_task_a [--runs-dir <runs folder>] [--out results/task_a_gate_a.md]

Reads only completed matrix runs (by run name; pilots, checks and stopped runs
are ignored), evaluates the criteria registered before Gate A (SRS 4.4, D10,
D23) and writes one Markdown report with a results table per arm (S0-F-23).
Missing runs are reported as missing, never as passes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cassandra_chorus.metrics.gate_a import REDUNDANCY_FLAG, SEEDS, collect, evaluate_gate_a
from cassandra_chorus.paths import REPO_ROOT, runs_root

ARM_TITLES = {
    "centralized": "Centralized (reference)",
    "main": "Sliced, coverage (main arm)",
    "rolling": "Sliced, rolling assignment",
    "router_all": "Sliced, router averaged over all workers",
    "partial": "Partial update (S0-F-26)",
    "full": "Full-model local averaging (S0-F-14)",
}


def fmt(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


def span(values, digits):
    """'lo to hi' over seeds, one number if they agree at this precision, 'n/a' if empty."""
    if not values:
        return "n/a"
    lo, hi = fmt(min(values), digits), fmt(max(values), digits)
    return lo if lo == hi else f"{lo} to {hi}"


def report(rows, result, runs_dir: Path) -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        commit += " with uncommitted changes"
    lines = [
        "# Task A · Gate A report",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `python -m scripts.analyze_task_a` "
        f"(code `{commit}`) from the run logs in `{runs_dir}`. Criteria and their reading were registered before "
        "any Gate A run (SRS section 4.4, D10, D22, D23).",
        "",
        f"## Criteria verdict: {result['overall'].upper()}",
        "",
        "The gate decision itself is the owner's (PLAN step 9).",
        "",
        "| Criterion | " + " | ".join(f"Seed {s}" for s in SEEDS) + " | All seeds (S0-A-06) |",
        "|---|" + "---|" * (len(SEEDS) + 1),
    ]
    for name, by_seed in result["verdicts"].items():
        cells = [f"{by_seed.get(s)}: {result['details'][name].get(s, '')}" for s in SEEDS]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {result['all_seeds'][name]} |")
    lines += ["", "S0-A-04 (Task B) is not part of Gate A. The criteria apply to the main arm; the other arms are for diagnosis.", ""]

    lines += ["## Routing necessity against centralized (SRS D23)", ""]
    for variant, info in result["redundancy"].items():
        ratios = ", ".join(fmt(r, 2) for r in info["necessity_ratio_by_seed"])
        lines.append(f"- {variant}: main-arm necessity divided by centralized necessity, by seed: {ratios}.")
    ratios = [r for info in result["redundancy"].values() for r in info["necessity_ratio_by_seed"]]
    if any(info["flag"] for info in result["redundancy"].values()):
        below = sum(r is not None and r < REDUNDANCY_FLAG for r in ratios)
        lines += [
            "",
            f"**Recorded finding (D23).** In {below} of {len(ratios)} seed and variant pairs the main arm's routing "
            "necessity is below half of the centralized run's: its merged experts are much more interchangeable. By the reading registered before "
            "Gate A, this does not change the verdict above. It is reported as a limitation, its cause is to be "
            "diagnosed with the comparison arms and expert drift below, and Task B must show whether it costs capacity.",
        ]
    lines.append("")

    lines += [
        "## Arms side by side",
        "",
        "Ranges over seeds. Necessity and removal cost on the curve set; the ratio divides each run's necessity by the "
        "centralized run's with the same seed and variant.",
        "",
        "| Arm | Variant | Runs | Gate accuracy | Necessity | Necessity / centralized | Worst single-expert removal (points) |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, title in ARM_TITLES.items():
        for variant in ("marked", "unmarked"):
            arm_rows = [r for (a, v, _), r in sorted(rows.items()) if a == arm and v == variant]
            if not arm_rows:
                continue
            central = {s: r for (a, v, s), r in rows.items() if a == "centralized" and v == variant}
            ratios = [r.necessity / central[r.seed].necessity for r in arm_rows
                      if r.seed in central and central[r.seed].necessity > 0]
            costs = [100 * r.removal_cost for r in arm_rows if r.removal_cost is not None]
            lines.append(f"| {title} | {variant} | {len(arm_rows)} | {span([r.gate_accuracy for r in arm_rows], 5)} | "
                         f"{span([r.necessity for r in arm_rows], 3)} | {span(ratios, 2)} | {span(costs, 2)} |")
    lines += ["", "## Results by arm", ""]
    for arm, title in ARM_TITLES.items():
        arm_rows = sorted((r for (a, _, _), r in rows.items() if a == arm), key=lambda r: (r.variant, r.seed))
        if not arm_rows:
            continue
        lines += [f"### {title}", "",
                  "| Variant | Seed | Gate accuracy | Lowest map | Bayes | Necessity | Consistency by layer | Negligible experts | Drift, final round (mean by layer) | Run ID | Config | Commit |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in arm_rows:
            consistency = ", ".join(fmt(c, 3) for c in r.consistency)
            negligible = sum(len(v) for v in r.negligible.values())
            drift = ", ".join(fmt(d["mean"], 3) for d in r.drift_final) if r.drift_final else "n/a"
            commit = "unknown" if r.commit is None else r.commit[:7] + (" (dirty)" if r.dirty else "")
            lines.append(f"| {r.variant} | {r.seed} | {fmt(r.gate_accuracy, 5)} | {fmt(r.min_map_accuracy, 5)} | "
                         f"{fmt(r.bayes_optimal, 5)} | {fmt(r.necessity)} | {consistency} | {negligible} | {drift} | "
                         f"`{r.run_id}` | `{r.config_hash[:12]}` | `{commit}` |")
        lines.append("")

    lines += [
        "## Definitions (SRS D22)",
        "",
        "- **Consistency:** mutual information between map and chosen expert over scored gate-set positions, divided by "
        "the map entropy; 0 = routing ignores the map; ceiling 2/3 here.",
        "- **Necessity:** accuracy with the trained router minus accuracy with random routing, both on the curve set "
        "(the diagnostic does not re-score the gate set).",
        "- **Negligible:** experts whose token share in a layer is below 0.1 · k/E = 0.025.",
        "- **Drift:** per expert, mean Jensen-Shannon divergence (base 2) between its holders' map mixes in the final "
        "round, where holders are the workers whose copy is merged (all workers in full-model averaging) and a map mix is "
        "how the worker's model routes the curve set to that expert after its local steps; mean over experts with two "
        "or more holders.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "task_a_gate_a.md")
    args = parser.parse_args(argv)
    runs_dir = args.runs_dir or runs_root()
    rows = collect(p for p in runs_dir.iterdir() if p.is_dir())
    result = evaluate_gate_a(rows)
    text = report(rows, result, runs_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"{len(rows)} runs; criteria verdict {result['overall']}; report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
