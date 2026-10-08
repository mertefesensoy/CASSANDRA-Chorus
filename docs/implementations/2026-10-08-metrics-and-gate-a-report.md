# 2026-10-08 · Metrics and the Gate A report (PLAN step 8)

| | |
|---|---|
| PLAN step | 8 · Metrics: accuracy, router consistency, expert usage, expert drift; per-run report and results tables |
| Branch | `stage0/08-metrics` (stacked on `stage0/07-local-averaging`) |
| SRS requirements | S0-F-20 to S0-F-24, S0-F-27, S0-A-01 to S0-A-06; decisions D10, D22, D23 |
| Status | Implemented; report produced for the main arm (see `RESULTS.md`) |

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
| `cassandra_chorus/metrics/routing.py` | `router_consistency` (normalized mutual information), `top2_share`, `js_divergence`, `expert_drift`, `negligible_experts`, `routing_necessity_gap` (named apart from the step 4 diagnostic `metrics.task_a.routing_necessity`, which measures the accuracies it subtracts). |
| `cassandra_chorus/metrics/gate_a.py` | Reading runs from logs, choosing the run for each arm, variant and seed, building result rows, and evaluating S0-A-01 to S0-A-06. |
| `cassandra_chorus/metrics/__init__.py` | Public names. |
| `scripts/analyze_task_a.py` | Writes `results/task_a_gate_a.md`: the criteria verdicts per seed, the D23 necessity ratios, the arms side by side, one table per arm (S0-F-23) and the metric definitions. The header records the analysis commit and whether the tree had uncommitted changes. |
| `tests/test_metrics_routing.py`, `tests/test_gate_a.py` | Tests below; `test_gate_a.py` also runs the report script on synthetic run folders. |

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
- **Removal cost:** accuracy with the trained router minus the lowest accuracy with one expert removed from every layer, on the curve set (from the same diagnostic). Not a criterion; it sits beside necessity in the arms-side-by-side table.
- **Routing necessity:** accuracy with the trained router minus accuracy with random routing, both on the curve set, as the step 4 diagnostic measures them (D22). The gate set is not re-scored under random routing, so subtracting from gate accuracy would mix two sets.
- **Router consistency:** per layer, normalized mutual information on the gate set, plus the top-2 share (S0-F-21).
- **Negligible experts:** per layer, the experts whose share of tokens over all positions is below 0.1 · k/E (S0-F-22, S0-A-03).
- **Expert drift, sliced arms only:** per layer, the mean and maximum over experts of the holder-pair Jensen-Shannon divergence in the final round, and its mean over rounds (S0-F-27).
- **Run identity:** run ID, configuration hash and commit.

### What drift compares in each mode

The step 6 harness logs, for each worker and round, its slice (`held`) and the map-by-expert table of its own model at the end of its local steps, evaluated on the curve set (`eval.worker_tables`). Drift takes, for each expert, the tables of the workers whose copy of that expert is averaged in the merge:

| Mode | Holders of an expert (Gate A setting) | What a high drift would mean |
|---|---|---|
| `sliced` | the 2 workers assigned it | its two copies were trained on different maps before averaging |
| `partial` | the 2 workers assigned it (the other 2 route to it, but its copy there is frozen and not merged) | the same, for the copies that are merged |
| `full` | all 4 workers | the workers' routers disagree on which maps go to it |

The "map mix sent to an expert" is therefore a proxy: how the worker's model routes the curve set after training, not a count of the tokens it actually routed during its 125 steps. Training traffic is not logged.

### Criteria (SRS section 4.4, D10, D23)

| Criterion | Rule |
|---|---|
| Precondition | Every map of the centralized marked run at least 0.999, per seed |
| S0-A-01 | Marked variant: every map of the sliced run at least 0.99, per seed |
| S0-A-02 | Unmarked variant: sliced gate accuracy within 2 percentage points of the centralized run with the same seed |
| S0-A-03 | No negligible expert in any layer, per seed and variant |
| S0-A-05 | Satisfied by configuration: each worker holds at most half the experts. The report checks the logged capacity |
| S0-A-06 | Each criterion holds for every seed (7, 11 and 19) on its own |

The report gives a verdict per criterion and seed (pass, fail or missing) and an overall line: pass, fail, or incomplete while any run is missing. A missing run is never read as a pass. For D23 it always prints, per variant and seed, the main arm's necessity divided by the centralized run's; when any ratio is below 0.5 it adds the recorded-finding paragraph. Comparison arms are reported for diagnosis only; the criteria apply to the main arm.

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
- **Routing necessity:** a_trained − a_random, both pooled accuracies on the curve set (256 sequences per map in the Gate A configs, against 2,048 for the gate set; same scoring rule). The D23 ratio is necessity(main) / necessity(centralized) for the same variant and seed.

## Design Decisions

| Decision | Chosen | Alternatives | Decided by |
|---|---|---|---|
| Metric definitions | As above | Others | Owner, 2026-10-08 (D22) |
| Reading of a redundant pass | Pass with recorded finding | Fail on routing; margin criterion | Owner, 2026-10-08 (D23), before any Gate A run |
| Run selection | By run name and completed status; refuse ambiguity | Latest run wins | Engineering default: no silent choice between runs |
| Analysis only | Metrics computed from logs | Recompute from saved models | Engineering default: every needed quantity is logged; no reruns |
| Necessity sets | Trained and random routing on the same (curve) set | Gate accuracy minus curve-set random routing | Engineering default: a difference must compare like with like; caught while writing the report |
| Drift inputs | Holders are the workers whose copy is merged; map mixes are end-of-round routing on the curve set | Counting training traffic (not logged); counting every worker that routes to the expert in partial mode | Engineering default, recorded here so the comparison arms are read consistently. The owner may prefer another reading at the gate review |
| When the D23 paragraph appears | Any seed's necessity ratio below 0.5; ratios always printed | No threshold, prose judgment; a stricter ratio | Engineering default. D23 sets no number, and this threshold changes no verdict: it only decides whether the paragraph is printed. The owner may set another at the gate review |

## Verification

Run on the reference laptop (ASUS ROG Strix G614JI), CPU only (these tests need no GPU), from the worktree on `stage0/08-metrics`:

```bash
python -m pytest tests/test_metrics_routing.py tests/test_gate_a.py -q
```

21 passed. They cover:

- consistency at 0, at 1 and at the 2/3 ceiling, and its edge cases;
- top-2 share, JS divergence against a hand computation, drift with single holders and unused columns;
- the negligible threshold;
- a full passing matrix;
- S0-A-01 failing on one map;
- S0-A-02 compared within a seed;
- S0-A-03 failing on a starved expert;
- missing runs giving an incomplete verdict;
- the D23 flag raised without failing the gate;
- run selection: stopped runs, pilots and check runs are ignored, and duplicate runs are refused by name;
- a name and variant mismatch;
- drift from round records, with a dropped worker skipped;
- necessity taken from the curve set on both sides;
- the report script on a full matrix and on a partial one.

Full suite at commit `c2ed8ed`, same laptop: 235 CPU tests (`-m "not gpu"`) and the 11 GPU tests (`-m gpu`, RTX 4070 Laptop GPU, run after the main-arm queue freed the GPU) all passed. The report changes after that commit touch only `scripts/analyze_task_a.py` and `tests/test_gate_a.py`, whose 12 tests pass.

On real runs (6 centralized at commit `723f072`; 6 main-arm and 24 comparison-arm runs at `f8ac7ef`, all clean):

```bash
python -m scripts.analyze_task_a
```

This wrote `results/task_a_gate_a.md` from 36 runs, regenerated from a clean tree: criteria verdict pass, D23 paragraph raised in 6 of 6 seed and variant pairs. The arms-side-by-side ranges were also computed by a separate script reading the logs directly, and agree. Cross-checks:

- **Main-arm rows:** a separate script read the `final` records directly. Its trained and random-routing accuracies agree with the report's necessity, and its least-used expert shares agree with the zero negligible-expert count.
- **Centralized rows:** gate accuracy, lowest map, random routing and Bayes agree with the step 4 table in `RESULTS.md`.
- **Main-arm gate accuracy and lowest map:** not re-extracted independently. The report reads them straight from the logged fields.

The interpretation is in `RESULTS.md` (Gate A, main arm). The drift values (final-round layer means 0.001 to 0.139, maxima up to 0.242) were read from that report and compared with the synthetic cases for scale only.

**Not tested:**

- An independent recomputation of drift or consistency from saved models instead of logged tables.
- Runs that include dropped workers. Dropped workers are covered by a synthetic test only.

## Related Docs

- `docs/SRS.md`: section 4.4, D10, D22, D23
- `docs/PLAN.md`: section 5 (Gate A)
- `RESULTS.md`: centralized matrix and check runs
