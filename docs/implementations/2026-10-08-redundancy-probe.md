# 2026-10-08 · Redundancy probe: what each expert learned (PLAN step 9a)

| | |
|---|---|
| PLAN step | 9a · Redundancy probe, before Task B |
| Branch | `stage0/09-redundancy-probe` (from `main` after the Gate A merge) |
| SRS | Decisions D23 (reading of a redundant pass) and D24 (Gate A decision and this probe) |
| Status | Planned. Reading registered by the owner on 2026-10-08, in commit history before any probe number was computed |

## Problem / Motivation

Gate A passed with the D23 recorded finding:

- the main arm's experts are close to interchangeable (routing necessity 6% to 10% of centralized);
- the comparison arms locate the source in this setting: routing restricted to the worker's slice during local training.

Nothing has yet been measured inside the experts. Two explanations fit the Gate A numbers equally well:

- **Generalists.** Every expert learned every map, so any choice of experts works.
- **Bypassed.** The mixture layers matter little at all, and attention and the shared parts carry the task, so the choice of experts does not matter.

They predict different things for Task B. Generalists waste capacity that a larger task would need. Bypassed experts mean the mixture is not doing the work. The owner decided on 2026-10-08 (D24) to tell them apart before Task B.

## What Changed

To be completed after implementation. Planned:

| File | Description |
|---|---|
| `cassandra_chorus/metrics/probe.py` | Forced-pair and bypass evaluations of a saved model, and the derived quantities below. |
| `scripts/probe_task_a.py` | Runs the probe over the Gate A models and writes `results/task_a_redundancy_probe.md` and a JSON file with every matrix (for figures). |
| `tests/test_probe.py` | Tests on small synthetic models with known behaviour. |
| `RESULTS.md` | The probe's entry. |

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
- **Expert competence:** c(ℓ, e, m) = the mean of f(ℓ, P, m) over the 7 pairs P that contain e.
- **Specialization** of expert e in ℓ: max over m of c(ℓ, e, m) minus min over m of c(ℓ, e, m). 0 means it serves every map equally. A layer's specialization is the mean over its 8 experts.

## The reading, registered before running

Confirmed by the owner on 2026-10-08 as proposed, in the spirit of D23. Every number is reported whatever the outcome.

- A layer is **needed** if its bypass cost β(ℓ) is at least 5 points.
- **The D23 diagnosis is supported from inside the experts** if both hold on every seed and both variants:
  1. in the main arm, at least one layer is needed;
  2. in the needed layers, the main arm's competent share is higher than the centralized run's (same seed and variant), and also higher than the partial and full arms'.
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

To be completed after implementation. Planned:

- Unit tests with synthetic models where the answer is known: a model whose experts are identical (competent share 1), and one whose layer is bypassable (β = 0).
- Spot checks on real models:
  - forcing the pair the trained router picks most often for a map reproduces close to a(m) on that map;
  - for each model, the forced-pair accuracies averaged over pairs and maps sit near a single-layer random-routing figure, measured separately as a cross-check.
- **Not tested:** to be completed.

## Related Docs

- `RESULTS.md`: Gate A entry (main arm and comparison arms)
- `results/task_a_gate_a.md`
- `docs/implementations/2026-10-08-metrics-and-gate-a-report.md`
- `docs/SRS.md`: D23, D24
- `docs/PLAN.md`: step 9a, section 5
