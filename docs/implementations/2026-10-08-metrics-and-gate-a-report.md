# 2026-10-08 · Metrics and the Gate A report (PLAN step 8)

| | |
|---|---|
| PLAN step | 8 · Metrics: accuracy, router consistency, expert usage, expert drift; per-run report and results tables |
| Branch | `stage0/08-metrics` (stacked on `stage0/07-local-averaging`) |
| SRS requirements | S0-F-20 to S0-F-24, S0-F-27, S0-A-01 to S0-A-06; decisions D10, D22, D23 |
| Status | Planned |

## Problem / Motivation

Gate A needs every Task A run reduced to the quantities the SRS names, and the criteria evaluated mechanically against them:

- held-out accuracy per map;
- router consistency;
- expert usage, including negligible traffic;
- expert drift;
- routing necessity.

This removes any room for choosing numbers after seeing them. The definitions were fixed by the owner before any Gate A run (D22), and so was the reading of a pass with redundant experts (D23).

Every input is already in the run logs:

- the gate-set evaluation and diagnostic in the `final` record;
- the merged model's and each worker's map-by-expert routing tables in every `round` record.

So step 8 is an analysis module plus a report script, and no run needs repeating.

## What Changed

| File | Description |
|---|---|
| `cassandra_chorus/metrics/routing.py` | `router_consistency` (normalized mutual information), `top2_share`, `js_divergence`, `expert_drift`, `negligible_experts`, `routing_necessity`. |
| `cassandra_chorus/metrics/gate_a.py` | Reading runs from logs, choosing the run for each arm, variant and seed, building result rows, and evaluating S0-A-01 to S0-A-06. |
| `cassandra_chorus/metrics/__init__.py` | Public names. |
| `scripts/analyze_task_a.py` | Writes `results/task_a_gate_a.md`: one table per arm (S0-F-23) and the criteria verdicts. |
| `tests/test_metrics_routing.py`, `tests/test_gate_a.py` | Tests below. |

## Implementation Approach

### Choosing runs

A run belongs to the analysis when it has a `final` record and an `end` record with status `completed`. Its arm comes from its run name:

| Run name | Arm |
|---|---|
| `taskA-central-<variant>` | centralized |
| `taskA-sliced-coverage-<variant>` | main |
| `taskA-sliced-rolling-<variant>` | rolling |
| `taskA-sliced-routerall-<variant>` | router over all |
| `taskA-partial-<variant>` | partial update |
| `taskA-fullavg-<variant>` | full-model averaging |

Pilots, check runs and stopped runs never match. If two completed runs claim the same arm, variant and seed, the report refuses and names them, instead of picking one.

### Per-run row

For each run, the row records:

- **Accuracy:** gate accuracy and lowest map accuracy (S0-F-20).
- **Routing necessity:** gate accuracy minus random-routing accuracy (D22).
- **Router consistency:** per layer, normalized mutual information on the gate set, plus the top-2 share (S0-F-21).
- **Negligible experts:** per layer, the experts whose share of tokens over all positions is below 0.1 · k/E (S0-F-22, S0-A-03).
- **Expert drift, sliced arms only:** per layer, the mean and maximum over experts of the holder-pair Jensen-Shannon divergence in the final round, and its mean over rounds (S0-F-27).
- **Run identity:** run ID, configuration hash and commit.

### Criteria (SRS section 4.4, D10, D23)

| Criterion | Rule |
|---|---|
| Precondition | Every map of the centralized marked run at least 0.999, per seed |
| S0-A-01 | Marked variant: every map of the sliced run at least 0.99, per seed |
| S0-A-02 | Unmarked variant: sliced gate accuracy within 2 percentage points of the centralized run with the same seed |
| S0-A-03 | No negligible expert in any layer, per seed and variant |
| S0-A-05 | Satisfied by configuration: each worker holds at most half the experts. The report checks the logged capacity |
| S0-A-06 | Each criterion holds for every seed (7, 11 and 19) on its own |

The report gives a verdict per criterion and seed and an overall Gate A line. If redundancy is evident (routing necessity of the main arm far below centralized), it adds the D23 recorded-finding paragraph. Comparison arms are reported for diagnosis only; the criteria apply to the main arm.

## Mathematical / Statistical Details

Let C be a layer's map-by-expert count table over scored positions, where each of a token's k choices counts once. Let p(m, e) = C(m, e) / Σ C, with marginals p(m) and p(e).

- **Router consistency:**
  - NMI = I(M; E) / H(M);
  - I(M; E) = Σ_{m,e} p(m,e) log₂ [p(m,e) / (p(m) p(e))];
  - H(M) = -Σ_m p(m) log₂ p(m).

  With E = M = 8 and k = 2, a map's choices can at best be split evenly over a dedicated pair, so H(E | M) ≥ 1 bit. Hence I ≤ H(E) - 1 ≤ 2 bits and NMI ≤ 2/3.
- **Top-2 share:** the mean over maps of (count of the map's two most-chosen experts) / (all choices for that map).
- **Jensen-Shannon divergence** of two distributions P, Q over maps, base 2:
  - JS(P, Q) = ½ KL(P ‖ A) + ½ KL(Q ‖ A), with A = (P + Q) / 2;
  - it lies in [0, 1]: 0 when equal, 1 when their supports are disjoint.
- **Expert drift** of expert e in a round:
  1. for each holder h with a non-zero column, P_h(m) = C_h(m, e) / Σ_m C_h(m, e);
  2. drift(e) is the mean of JS(P_h, P_h') over all pairs of such holders, defined when there are at least two.
  3. A layer's drift is the mean and the maximum of drift(e) over the experts where it is defined.
- **Negligible traffic:** expert e's share is (tokens routed to e) / (all tokens, all positions). It is negligible when below 0.1 · k/E, which is 0.025 here.
- **Routing necessity:** gate accuracy minus random-routing accuracy (random routing evaluated on the curve set; recorded as such).

## Design Decisions

| Decision | Chosen | Alternatives | Decided by |
|---|---|---|---|
| Metric definitions | As above | Others | Owner, 2026-10-08 (D22) |
| Reading of a redundant pass | Pass with recorded finding | Fail on routing; margin criterion | Owner, 2026-10-08 (D23), before any Gate A run |
| Run selection | By run name and completed status; refuse ambiguity | Latest run wins | Engineering default: no silent choice between runs |
| Analysis only | Metrics computed from logs | Recompute from saved models | Engineering default: every needed quantity is logged; no reruns |

## Verification

To be completed after implementation.

**Not tested:** to be completed.

## Related Docs

- `docs/SRS.md`: section 4.4, D10, D22, D23
- `docs/PLAN.md`: section 5 (Gate A)
- `RESULTS.md`: centralized matrix and check runs
