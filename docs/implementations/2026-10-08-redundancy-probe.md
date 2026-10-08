# 2026-10-08 · Redundancy probe: what each expert learned (PLAN step 9a)

| | |
|---|---|
| PLAN step | 9a · Redundancy probe, before Task B |
| Branch | `stage0/09-redundancy-probe` (from `main` after the Gate A merge) |
| SRS | Decisions D23 (reading of a redundant pass) and D24 (Gate A decision and this probe) |
| Status | Implemented and run. Reading registered by the owner on 2026-10-08 before any probe number was computed (commits `6732714`, `dfaf437`); outcome **mixed** (`RESULTS.md`) |

## Problem / Motivation

Gate A passed with the D23 recorded finding:

- the main arm's experts are close to interchangeable (routing necessity 6% to 10% of centralized);
- the comparison arms locate the source in this setting: routing restricted to the worker's slice during local training.

Nothing has yet been measured inside the experts. Two explanations fit the Gate A numbers equally well:

- **Generalists.** Every expert learned every map, so any choice of experts works.
- **Bypassed.** The mixture layers matter little at all, and attention and the shared parts carry the task, so the choice of experts does not matter.

They predict different things for Task B. Generalists waste capacity that a larger task would need. Bypassed experts mean the mixture is not doing the work. The owner decided on 2026-10-08 (D24) to tell them apart before Task B.

## What Changed

| File | Description |
|---|---|
| `docs/SRS.md`, `docs/PLAN.md` | Decision D24 (the owner's Gate A decision) and step 9a. |
| `cassandra_chorus/metrics/probe.py` | Bypass and forced-pair evaluations of a model (`probe_model`), the derived quantities (`summarize`) and the registered reading (`evaluate_reading`). |
| `scripts/probe_task_a.py` | Selects the Gate A runs as the Gate A report does, loads each `final_model.pt`, writes `<run>/probe.json` (raw, not committed), `results/task_a_redundancy_probe.md` and `results/task_a_redundancy_probe.json` (every summary and competence table, for figures). Holds the keep-awake request while it runs. |
| `tests/test_probe.py` | 7 tests: identical experts, a silent layer, hook removal, probe shape, the summary arithmetic, and the reading's four outcomes. |
| `results/task_a_redundancy_probe.md`, `.json` | The probe over the 36 Gate A models. |
| `RESULTS.md` | The probe's entry and its interpretation. |

## Implementation Approach

Analysis only: no training. Every Gate A run saved its final model (`final_model.pt`), so the probe reads the 36 saved models: 6 centralized, 6 main arm and 24 comparison arms. It evaluates them on the curve set (256 sequences per map, seed 1001), the set the routing-necessity diagnostic uses. Evaluation is deterministic.

Two measurements per model:

1. **Bypass of a mixture layer (4 evaluations).** For each layer ℓ, the mixture's output in ℓ is replaced by zeros through a forward hook in the probe script; the residual stream passes through unchanged. Per-map accuracy is recorded. The model code is not changed.
2. **Forced pairs (112 evaluations).** For each layer ℓ and each of the 28 pairs {a, b} of the 8 experts, routing in ℓ is restricted to {a, b} with the existing `expert_mask`. With k = 2, every token in ℓ goes to both experts, weighted by the router's softmax over the pair. The other layers route as trained. Per-map accuracy is recorded.

Cost: 116 evaluations of 2,048 sequences per model, about 4,200 in all. Estimated under 20 minutes on the RTX 4070; to be measured.

## Mathematical / Statistical Details

Let a(m) be the trained model's accuracy on map m (curve set), and f(ℓ, P, m) the accuracy with layer ℓ forced to pair P.

- **Bypass cost** of layer ℓ: β(ℓ) = max over m of [a(m) − a_bypass(ℓ, m)], in points: the worst map's loss when ℓ's mixture is removed.
- **Retention** of a forced cell: r(ℓ, P, m) = f(ℓ, P, m) / a(m).
- **Competent cell:** r ≥ 0.99. The pair handles that map almost as well as the trained routing.
- **Competent share** of layer ℓ: the fraction of the 28 × 8 = 224 (pair, map) cells that are competent. 1 means any pair handles any map.
- **Expert competence:** c(ℓ, e, m) = the mean of the retention r(ℓ, P, m) over the 7 pairs P that contain e. A value of 1 means the expert, with any partner, is as good as trained routing on map m.
- **Specialization** of expert e in ℓ: max over m of c(ℓ, e, m) minus min over m of c(ℓ, e, m). 0 means it serves every map equally well. A layer's specialization is the mean over its 8 experts. Where a(m) = 0, the retention is taken as 1 (there was nothing to lose); in the Gate A models every a(m) is at least 0.94.

  *Correction before any probe run:* the first draft averaged raw accuracy f rather than retention. A unit test showed that this counts maps that are simply harder for the whole model as specialization: identical experts scored 0.07. Retention removes that confound. Specialization is descriptive and is not part of the registered reading.

## The reading, registered before running

Confirmed by the owner on 2026-10-08 as proposed, in the spirit of D23. Every number is reported whatever the outcome.

- A layer is **needed** if its bypass cost β(ℓ) is at least 5 points.
- **The D23 diagnosis is supported from inside the experts** if both hold on every seed and both variants:
  1. in the main arm, at least one layer is needed;
  2. in the needed layers, the main arm's competent share is higher than the centralized run's (same seed and variant), and also higher than the partial and full arms'.

  Operational detail, fixed with the code before any probe run: "in the needed layers" means the competent share pooled over the main arm's needed layers, compared with each other model's share pooled over the same layer indices. Per-layer values are reported as well.
- **"Bypassed" instead** if, in the main arm, no layer is needed on any seed. Then the redundancy is not generalist experts, and Task B's question changes: whether the mixture does any work under slicing.
- **Mixed outcomes** (for example, prediction 2 holds on some seeds only) are reported as such, without a verdict.

## Design Decisions

| Decision | Chosen | Alternatives | Decided by |
|---|---|---|---|
| Probe before Task B | Yes | Go straight to Task B | Owner, 2026-10-08 (D24) |
| Data | The 36 saved Gate A models | Retrain with extra logging | Engineering default: no training needed, and the probe describes exactly the models Gate A judged |
| Forcing unit | Expert pairs, one layer at a time | Single experts (needs k = 1, a routing the models never used); all layers at once (mixes layers) | Engineering default: k = 2 is how the models were trained |
| Bypass | Zero the mixture output of one layer | Replace it with its mean; bypass all layers | Engineering default: the cleanest test of "the layer is not needed" |
| Thresholds of the reading | 5 points for "needed", 0.99 retention for "competent" | Directional comparisons only; no registered reading | Owner, 2026-10-08, before any probe number was computed |

## Verification

**Unit tests** (CPU, reference laptop): `python -m pytest tests/test_probe.py -q`, 7 passed. Identical experts give a competent share of exactly 1 and specialization 0. A layer whose experts output zeros has a bypass cost of exactly 0. The hook is removed after use, and the summary arithmetic and all four reading outcomes match hand-computed cases. The first draft's specialization (raw accuracy) failed the identical-experts test, which led to the retention definition above, before any real model was probed.

**The probe itself:**

```bash
python -m scripts.probe_task_a
```

- Launched through `scripts/ops/launch_visible.ps1` at commit `6a02d4b` with a clean tree. It probed the 36 saved models (6 centralized at `723f072`, 30 sliced and comparison runs at `f8ac7ef`) on the RTX 4070 Laptop GPU in 1,820 s, against an estimate of under 20 minutes; the cross-checks account for most of the difference.
- A smoke test on one copied model ran first (32 s), after the reading was committed.

**Cross-checks on the real models** (in the report, not part of the reading):

- **Single-layer random routing against the mean forced-pair accuracy.**
  - These agree within about 0.04 everywhere except layer 0 of the unmarked centralized, partial and full models, where they differ by up to 0.083, 0.094 and 0.105.
  - There routing follows the input symbol, not the map: Gate A layer-0 consistency is about 0.002. Random routing per token is then not the same as one fixed pair for every token.
- **Forcing each map's two most-used experts.**
  - Outside layer 0 this keeps at least 0.968 of trained accuracy.
  - In layer 0 it falls to 0.928 in the centralized marked models, 0.942 in the rolling unmarked models, and 0.32 to 0.87 in the unmarked centralized, partial and full models.
  - There "the map's most-used pair" is not a meaningful quantity.
  - Both exceptions are the check doing its job: they mark where routing is not by map.

**Operational note:** the keep-awake request was made, but its status line went to the console, which the launcher transcript does not capture (PowerShell 5.1 `Start-Transcript` does not record native programs' output), so it is not on record. No standby or stall was observed: the run finished in one pass.

**Post hoc check (owner decision 2026-10-08, after the registered results were known; no verdict).**

- Command: `python -m scripts.probe_all_layers_task_a`, at commit `8a6160e` with a clean tree.
- It zeroes all four mixture layers at once and writes `results/task_a_probe_all_layers_posthoc.md`; one new unit test covers the multi-layer hook.
- Every model collapses: marked 0.25 to 0.40 accuracy, unmarked 0.02 to 0.08, against 1.00 and 0.95 trained, sliced models included.
- So in the marked sliced models the mixture as a whole is needed. No single layer is needed only because the four layers can stand in for one another.

**Not tested:**

- Bypassing two or three layers at once.
- What layer-0 experts specialize in, if not the map (presumably the input symbol; not measured).
- Other budgets, model sizes, and Task B models.

## Related Docs

- `RESULTS.md`: Gate A entry (main arm and comparison arms)
- `results/task_a_gate_a.md`
- `docs/implementations/2026-10-08-metrics-and-gate-a-report.md`
- `docs/SRS.md`: D23, D24
- `docs/PLAN.md`: step 9a, section 5
