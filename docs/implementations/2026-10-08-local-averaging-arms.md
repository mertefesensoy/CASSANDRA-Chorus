# 2026-10-08 · Local-averaging comparison arms (PLAN step 7)

| | |
|---|---|
| PLAN step | 7 · Local-averaging baselines: full model (S0-F-14) and partial update (S0-F-26) |
| Branch | `stage0/07-local-averaging` (stacked on `stage0/06-simulation`) |
| SRS requirements | S0-F-14, S0-F-26; decisions D11, D13, D20, D21 |
| Status | Implemented (code in step 6); runs are part of the Gate A matrix |

## Problem / Motivation

Gate A must separate three possible costs of sliced training:

- the cost of training locally for many steps between merges, measured by **full-model local averaging** (S0-F-14);
- the cost of each worker updating only some experts, measured by the **partial-update** arm (S0-F-26);
- the cost of routing masked to a worker's own experts, measured by **sliced** training itself.

Two further arms isolate decisions taken on 2026-10-08:

- **rolling** assignment instead of coverage (D20);
- the **all-workers router rule** instead of holder-only router averaging (D11).

The pre-registered reading of Gate A (SRS D23) relies on these arms to diagnose any expert redundancy found in sliced runs.

## What Changed

| File | Description |
|---|---|
| `configs/queues/gate_a_comparison.toml` | The 24 comparison runs: 4 arms × marked and unmarked × seeds 7, 11, 19, each a one-setting change from `configs/task_a/sliced.toml`. |
| `docs/implementations/2026-10-08-local-averaging-arms.md` | This document. |

The arms need no new training code. They are the `full` and `partial` worker modes and the `policy` and `router_rule` options of the step 6 harness (`docs/implementations/2026-10-08-simulation-harness.md`).

## Implementation Approach

Every arm uses the Gate A setting (D18): 4 workers, 125 local steps, 10 rounds, 5,000 worker steps of 64 sequences, the same model, data and optimizer as the centralized runs, and the same initial weights for a given seed. Each changes exactly one thing:

| Arm | Change from the main arm | Worker holds | Worker updates | Routing on the worker | Router merge |
|---|---|---|---|---|---|
| Main: sliced, coverage | none | 4 of 8 experts | its 4 experts and the shared part | masked to its 4 | holders |
| Sliced, rolling | `sim.policy=rolling` | 4 of 8 (rotating) | as main | masked | holders |
| Sliced, router over all | `sim.router_rule=all` | 4 of 8 | as main | masked | all workers |
| Partial update | `sim.mode=partial`, `sim.router_rule=all` | all 8 | its 4 assigned experts and the shared part | over all 8 | all workers (D21) |
| Full-model averaging | `sim.mode=full`, `sim.router_rule=all` | all 8 | everything | over all 8 | all workers |

## Mathematical / Statistical Details

None beyond the harness and merge rules of steps 5 and 6. The comparisons of interest are differences between arms, per seed and variant, in:

- gate accuracy;
- routing necessity (trained accuracy minus random-routing accuracy);
- router consistency (normalized mutual information, D22);
- expert drift (D22).

They are reported in step 8.

## Design Decisions

| Decision | Chosen | Alternatives | Decided by |
|---|---|---|---|
| Arms | Rolling, router over all, partial update, full-model averaging | Fewer arms | Owner, 2026-10-08 (D11, D13, D20) |
| Order | Main arm first, then these 24 runs | One queue of 30 | Owner, 2026-10-08 |
| One change per arm | Yes | Factorial combinations | Engineering default: keeps every difference attributable |

## Verification

The modes and options are covered by the step 6 tests:

- a partial-update worker freezes unassigned experts and trains every router row;
- a full worker returns everything;
- one full worker reproduces centralized training bitwise;
- every arm runs end to end on CPU.

All 24 entries of the queue file were loaded and validated against `configs/task_a/sliced.toml` (equal compute, router-rule consistency). The GPU runs themselves are the Gate A comparison matrix, recorded in `RESULTS.md` when complete.

**Not tested:** these arms at full budget on the GPU (the runs themselves).

## Related Docs

- `docs/implementations/2026-10-08-simulation-harness.md`, `2026-10-08-coordinator.md`
- `docs/SRS.md`: S0-F-14, S0-F-26, D11, D13, D20, D21, D23
