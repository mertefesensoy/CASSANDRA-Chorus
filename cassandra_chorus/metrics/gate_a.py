"""Gate A analysis: Task A runs to result rows, and the criteria of SRS section 4.4.

Reads only run logs. Criteria (SRS 4.4, D10) and the reading of a pass with
redundant experts (D23) were fixed before any Gate A run. The comparison arms
are reported for diagnosis; the criteria apply to the main arm.
See docs/implementations/2026-10-08-metrics-and-gate-a-report.md.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cassandra_chorus.metrics.routing import (
    expert_drift,
    negligible_experts,
    router_consistency,
    routing_necessity_gap,
    top2_share,
)

SEEDS = (7, 11, 19)
ARMS = {
    "central": "centralized",
    "sliced-coverage": "main",
    "sliced-rolling": "rolling",
    "sliced-routerall": "router_all",
    "partial": "partial",
    "fullavg": "full",
}
_NAME = re.compile(r"^taskA-(" + "|".join(re.escape(k) for k in ARMS) + r")-(marked|unmarked)$")
PRECONDITION = 0.999      # S0-A-01 precondition, per map (D10)
A01_THRESHOLD = 0.99      # S0-A-01, per map (D10)
A02_MARGIN = 0.02         # S0-A-02, percentage points as a fraction
REDUNDANCY_FLAG = 0.5     # reporting aid only (D23): main-arm necessity below half of centralized


class AmbiguousRunsError(ValueError):
    """Two completed runs claim the same arm, variant and seed."""


@dataclasses.dataclass
class RunRow:
    run_id: str
    arm: str
    variant: str
    seed: int
    config_hash: str
    commit: str | None
    dirty: bool | None
    gate_accuracy: float
    min_map_accuracy: float
    bayes_optimal: float
    trained_curve: float
    random_routing: float
    necessity: float
    removal_cost: float | None
    consistency: list[float]
    top2: list[float]
    negligible: dict[int, list[int]]
    capacity: int | None
    n_experts: int
    drift_final: list[dict[str, Any]] | None
    drift_mean_over_rounds: list[float | None] | None


def read_records(run_dir: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for line in (Path(run_dir) / "log.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            grouped.setdefault(record["type"], []).append(record)
    return grouped


def _drift(rounds: list[dict], n_layers: int, n_experts: int) -> tuple[list[dict], list[float | None]]:
    per_round = []
    for r in rounds:
        alive = [w for w in r["workers"] if not w.get("dropped") and "map_expert_counts" in w]
        layer_values = []
        for layer in range(n_layers):
            tables = [w["map_expert_counts"][layer] for w in alive]
            held = [w["held"][str(layer)] if str(layer) in w["held"] else w["held"][layer] for w in alive]
            layer_values.append(expert_drift(tables, held, n_experts))
        per_round.append(layer_values)
    final = [{"mean": v["mean"], "max": v["max"]} for v in per_round[-1]] if per_round else None
    means = []
    for layer in range(n_layers):
        values = [rv[layer]["mean"] for rv in per_round if rv[layer]["mean"] is not None]
        means.append(sum(values) / len(values) if values else None)
    return final, means


def run_row(run_dir: Path) -> RunRow | None:
    """The result row of a completed Task A matrix run, or None if the folder is not one."""
    if not (Path(run_dir) / "log.jsonl").exists():
        return None
    rec = read_records(run_dir)
    if not rec.get("header") or not rec.get("final"):
        return None
    if not rec.get("end") or rec["end"][-1].get("status") != "completed":
        return None
    header, final = rec["header"][0], rec["final"][0]
    cfg = header["config"]
    match = _NAME.match(cfg["run"]["name"])
    if match is None:
        return None
    arm, variant = ARMS[match.group(1)], match.group(2)
    if (variant == "marked") != bool(cfg["task_a"]["marked"]):
        raise ValueError(f"{run_dir}: run name says {variant} but task_a.marked is {cfg['task_a']['marked']}")
    model = cfg["model"]
    tables = final["gate"]["map_expert_counts"]
    gate_acc = final["gate"]["accuracy"]
    # Necessity compares like with like: the diagnostic scores the curve set with the
    # trained router and with random routing; the gate set is not re-scored.
    trained_curve = final["diagnostic"]["trained"]["accuracy"]
    random_acc = final["diagnostic"]["random_routing"]["accuracy"]
    removed = [x["accuracy"] for x in final["diagnostic"].get("leave_one_out", [])]
    sim = cfg.get("sim")
    capacity = None
    if sim is not None:
        capacity = max(sim["capacities"]) if sim.get("capacities") else (model["n_experts"] if sim["mode"] == "full" else sim["capacity"])
    drift_final = drift_means = None
    if rec.get("round"):
        drift_final, drift_means = _drift(rec["round"], model["n_layers"], model["n_experts"])
    return RunRow(
        run_id=header["run_id"], arm=arm, variant=variant, seed=cfg["run"]["seed"],
        config_hash=header["config_hash"], commit=(header["git"] or {}).get("commit"), dirty=(header["git"] or {}).get("dirty"),
        gate_accuracy=gate_acc, min_map_accuracy=final["gate"]["min_map_accuracy"], bayes_optimal=final["bayes_optimal_gate"],
        trained_curve=trained_curve, random_routing=random_acc, necessity=routing_necessity_gap(trained_curve, random_acc),
        removal_cost=trained_curve - min(removed) if removed else None,
        consistency=[router_consistency(t) for t in tables], top2=[top2_share(t) for t in tables],
        negligible={layer: negligible_experts(c, model["top_k"]) for layer, c in enumerate(final["gate"]["expert_counts"])},
        capacity=capacity, n_experts=model["n_experts"], drift_final=drift_final, drift_mean_over_rounds=drift_means,
    )


def collect(run_dirs: Iterable[Path]) -> dict[tuple[str, str, int], RunRow]:
    """Rows keyed by (arm, variant, seed); refuses two completed runs for the same key."""
    rows: dict[tuple[str, str, int], RunRow] = {}
    for d in sorted(run_dirs):
        row = run_row(d)
        if row is None:
            continue
        key = (row.arm, row.variant, row.seed)
        if key in rows:
            raise AmbiguousRunsError(f"two completed runs for {key}: {rows[key].run_id} and {row.run_id}")
        rows[key] = row
    return rows


def evaluate_gate_a(rows: dict[tuple[str, str, int], RunRow]) -> dict[str, Any]:
    """Per-seed verdicts ("pass", "fail" or "missing") for the precondition and S0-A-01, 02, 03, 05; S0-A-06 and overall."""

    def get(arm, variant, seed):
        return rows.get((arm, variant, seed))

    verdicts: dict[str, dict[int, str]] = {k: {} for k in ("precondition", "S0-A-01", "S0-A-02", "S0-A-03", "S0-A-05")}
    details: dict[str, dict[int, str]] = {k: {} for k in verdicts}
    for s in SEEDS:
        cm, cu, mm, mu = get("centralized", "marked", s), get("centralized", "unmarked", s), get("main", "marked", s), get("main", "unmarked", s)
        if cm is None:
            verdicts["precondition"][s], details["precondition"][s] = "missing", "no centralized marked run"
        else:
            verdicts["precondition"][s] = "pass" if cm.min_map_accuracy >= PRECONDITION else "fail"
            details["precondition"][s] = f"lowest map {cm.min_map_accuracy:.5f}"
        if mm is None:
            verdicts["S0-A-01"][s], details["S0-A-01"][s] = "missing", "no sliced marked run"
        else:
            verdicts["S0-A-01"][s] = "pass" if mm.min_map_accuracy >= A01_THRESHOLD else "fail"
            details["S0-A-01"][s] = f"lowest map {mm.min_map_accuracy:.5f}"
        if mu is None or cu is None:
            verdicts["S0-A-02"][s], details["S0-A-02"][s] = "missing", "need sliced and centralized unmarked runs"
        else:
            gap = mu.gate_accuracy - cu.gate_accuracy
            verdicts["S0-A-02"][s] = "pass" if abs(gap) <= A02_MARGIN else "fail"
            details["S0-A-02"][s] = f"sliced minus centralized {100 * gap:+.3f} points"
        if mm is None or mu is None:
            verdicts["S0-A-03"][s], details["S0-A-03"][s] = "missing", "need both sliced variants"
        else:
            bad = {v: {l: e for l, e in r.negligible.items() if e} for v, r in (("marked", mm), ("unmarked", mu))}
            verdicts["S0-A-03"][s] = "pass" if not any(bad.values()) else "fail"
            details["S0-A-03"][s] = "no negligible expert" if not any(bad.values()) else f"negligible: {bad}"
        caps = [r.capacity for r in (mm, mu) if r is not None]
        if not caps:
            verdicts["S0-A-05"][s], details["S0-A-05"][s] = "missing", "no sliced run"
        else:
            n_e = (mm or mu).n_experts
            verdicts["S0-A-05"][s] = "pass" if all(c is not None and 2 * c <= n_e for c in caps) else "fail"
            details["S0-A-05"][s] = f"capacity {max(caps)} of {n_e} experts"
    all_seeds = {k: ("pass" if all(v.get(s) == "pass" for s in SEEDS) else
                     "missing" if any(v.get(s) == "missing" for s in SEEDS) else "fail") for k, v in verdicts.items()}
    overall = "pass" if all(v == "pass" for v in all_seeds.values()) else ("incomplete" if "missing" in all_seeds.values() else "fail")

    redundancy = {}
    for variant in ("marked", "unmarked"):
        pairs = [(get("main", variant, s), get("centralized", variant, s)) for s in SEEDS]
        pairs = [(m, c) for m, c in pairs if m is not None and c is not None]
        if pairs:
            ratios = [m.necessity / c.necessity if c.necessity > 0 else None for m, c in pairs]
            redundancy[variant] = {
                "necessity_ratio_by_seed": ratios,
                "flag": any(r is not None and r < REDUNDANCY_FLAG for r in ratios),
            }
    return {"verdicts": verdicts, "details": details, "all_seeds": all_seeds, "overall": overall, "redundancy": redundancy}
