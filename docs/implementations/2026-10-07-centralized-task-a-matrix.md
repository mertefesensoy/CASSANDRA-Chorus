# 2026-10-07 · Centralized Task A matrix and final learning-rate decay (PLAN step 4, part 2)

| | |
|---|---|
| PLAN step | 4 · Centralized baseline on Task A, part 2: the full centralized matrix |
| Branch | `stage0/04-centralized-task-a` (PR #6) |
| SRS requirements | S0-F-13 (centralized baseline at the agreed budget), S0-F-20, S0-A-01 precondition (D10), S0-A-06 (seeds 7, 11, 19), S0-N-03, S0-N-07, S0-N-08 |
| Status | Planned |

## Problem / Motivation

The pilot (part 1) showed that centralized training meets the S0-A-01 precondition and that Task A can detect a routing failure, for seed 7 on the marked variant. The owner then decided, on 2026-10-07:

1. **Budget.** Every Task A comparison (centralized, full-model local averaging, sliced) gets **5,000 steps of 64 sequences, 320,000 sequences in total**. That is 5 to 6.7 times what centralized training needed to reach the precondition, leaving room for sliced training to converge more slowly. How per-step batch size is matched in sliced runs is decided at step 6.
2. **Full centralized matrix.** Marked and unmarked variants × seeds 7, 11 and 19, balance coefficient 0: six runs.
3. **Final learning-rate decay** for the matrix and all later Task A runs. The pilots, at a constant learning rate, showed brief accuracy dips of up to 1.4 points on one map after converging. A dip landing on the final measurement could fail a criterion by chance.

## What Changed

| File | Description |
|---|---|
| `cassandra_chorus/train/loop.py` | `OptimSection` gains `decay_fraction` (default 0, i.e. no decay) and `final_lr_ratio` (default 0.1); `lr_at` and `train_steps` take the total step count. |
| `scripts/train_task_a_centralized.py` | Passes the total step count; checks that warm-up and decay do not overlap. |
| `configs/task_a/centralized.toml` | Matrix configuration: the pilot's settings plus `decay_fraction = 0.2`. |
| `tests/test_train.py` | Tests of the decayed schedule and its validation. |
| `RESULTS.md` | The matrix entry. |

## Implementation Approach

**Schedule.** Linear warm-up as before, constant in the middle, and linear decay to `final_lr_ratio · lr` over the last `decay_fraction` of the run. The code default stays at `decay_fraction = 0`, so the generic loop and the recorded pilot configuration keep a constant rate. Every Task A configuration from now on sets 0.2 explicitly, per the owner's decision. The schedule depends only on the global step and the total step count, so for sliced runs (step 6) it can be driven by the cumulative step count.

**Matrix.** Six runs of `python -m scripts.train_task_a_centralized --config configs/task_a/centralized.toml` with:

- `--set run.seed=<7|11|19>`;
- `--set task_a.marked=<true|false>`;
- `--set run.name=taskA-central-<marked|unmarked>`.

The runs are in the foreground, one at a time, with the power watchdog armed: if the adapter drops out, the run is stopped (see the memory note on the laptop's power problem). For unmarked runs, the precondition does not apply; accuracy is reported against the Bayes-optimal ceiling (about 0.951).

## Mathematical / Statistical Details

Let η be the base rate, W the warm-up steps, T the total steps, f the decay fraction, r the final ratio, and D = round(f · T) the decay length, with the decay starting at s₀ = T - D. For global step s, counting from 0:

- if s < W: η_s = η · (s + 1) / W;
- if W ≤ s < s₀: η_s = η;
- if s ≥ s₀: η_s = η · (1 - (1 - r) · (s - s₀ + 1) / D).

The last step, s = T - 1, uses r · η. With η = 1e-3, W = 100, T = 5,000, f = 0.2 and r = 0.1: D = 1,000 and s₀ = 4,000, so the rate falls linearly from 1e-3 to 1e-4 over steps 4,000 to 4,999. A configuration where the decay would start before warm-up ends (s₀ < W) is rejected.

## Design Decisions

| Decision | Chosen | Alternatives | Decided by |
|---|---|---|---|
| Budget | 5,000 steps of 64 for every Task A comparison | 10,000 steps | Owner, 2026-10-07 |
| Matrix | Marked and unmarked × seeds 7, 11, 19, balance coefficient 0 | Also the balance-loss arm (12 runs) | Owner, 2026-10-07 |
| Schedule | Linear decay to 0.1 × lr over the final 20% | Constant rate | Owner, 2026-10-07 |
| Code default | `decay_fraction = 0`; Task A configurations set 0.2 explicitly | Default 0.2 | Engineering default: keeps the recorded pilot configuration's meaning |
| Pilots | Kept as recorded, not part of the matrix | Re-use pilot 1 as the seed-7 marked run | Engineering default: the matrix uses the decayed schedule |

## Verification

To be completed after implementation and the runs.

**Not tested:** to be completed.

## Related Docs

- `docs/implementations/2026-10-07-centralized-task-a-pilot.md` (part 1)
- `RESULTS.md`: 2026-10-07 pilot entry
- `docs/SRS.md`: S0-F-13, S0-A-01 (D10), S0-A-06
