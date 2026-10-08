# 2026-10-08 · Simulation harness: sequential workers, fault injection, skew, resumable rounds (PLAN step 6)

| | |
|---|---|
| PLAN step | 6 · Simulation harness (also supplies the worker modes that step 7 runs) |
| Branch | `stage0/06-simulation` (stacked on `stage0/05-coordinator`) |
| SRS requirements | S0-F-04, S0-F-07, S0-F-11, S0-F-12, S0-F-13 (D17), S0-F-14 and S0-F-26 (worker modes), S0-N-03, S0-N-04, S0-N-05; decisions D17 to D21 |
| Status | Planned |

## Problem / Motivation

The coordinator (step 5) decides who holds what and merges the results, but nothing yet trains workers. Step 6 simulates N workers on the one GPU (S0-F-04). Each round:

1. assign slices;
2. train each worker's slice for H local steps with routing masked to its experts (S0-F-07);
3. optionally drop some results (S0-F-11);
4. merge;
5. evaluate the merged full model.

A run must survive interruption and resume from its last completed round (S0-N-04). The overnight standby stalls of 2026-10-07/08 showed why that matters on this laptop. The same harness runs the full-model local-averaging arm (S0-F-14) and the partial-update arm (S0-F-26) by changing only what a worker trains and sends back, so step 7 is mostly running it.

## What Changed

| File | Description |
|---|---|
| `cassandra_chorus/sim/worker.py` | `train_worker`: builds a worker's model for its mode, trains H steps with a fresh optimizer, returns the parameters the coordinator expects. |
| `cassandra_chorus/sim/harness.py` | `SimSection` configuration; data skew, fault draws, per-round seeds; `run_rounds` with checkpoint and resume. |
| `cassandra_chorus/sim/__init__.py` | Public names. |
| `cassandra_chorus/runlog.py` | `RunLogger.reopen(run_dir)`: append to an existing run's log when resuming. |
| `cassandra_chorus/train/loop.py` | `train_steps` accepts a parameter list for the optimizer and an explicit schedule length, for workers. |
| `scripts/train_task_a_sliced.py` | Entry point for sliced, partial-update and full-model runs on Task A, with `--resume`. |
| `configs/task_a/sliced.toml` | The Gate A setting (D18, D20). |
| `tests/test_sim.py` | Tests below. |

## Implementation Approach

### Worker modes

| Mode | Worker model | Trained | Returns | Router merge |
|---|---|---|---|---|
| `sliced` (main) | Slice model holding only its experts: routing is masked by construction | All its parameters | Shared part and its experts | Holders (D11), or `all` as the comparison |
| `partial` (S0-F-26) | Full model, all experts routable | Shared part and assigned experts; other experts frozen (`requires_grad = False`, so the optimizer never sees them) | Shared part and assigned experts | All workers (D21) |
| `full` (S0-F-14) | Full model | Everything | Everything | All workers |

Each worker loads its starting parameters from the current global state, filtered by name, and trains with a **fresh AdamW each round** (D19).

### Learning-rate schedule (owner-confirmed 2026-10-08)

One schedule over each worker's trajectory: worker step index = round × H + local step, and schedule length R × H. Warm-up happens once, in round 0. The final decay covers the last 20% of rounds. These are the same `warmup_steps` and `decay_fraction` as the centralized configuration, now relative to a 1,250-step trajectory.

### Equal compute (D17)

Batch 64 per worker step; N × H × R must equal the centralized step count. The entry point checks this against `sim.equal_compute_steps` (5,000) and refuses a mismatch.

### Data

Every worker and round draws batches from its own generator, seeded with `derive_seed(seed, "sim/data/w<i>/r<r>")`. Batches are therefore independent of the order workers run in, and of whether other workers were dropped.

**Skew** (S0-F-12; default 0 for Gate A): worker i samples map m with weight (1 - s)/M + s · 1[m is one of worker i's preferred maps] / |preferred|, where the preferred maps are those with m mod N = i mod N. s = 0 is uniform; s = 1 means each worker sees only its own maps.

### Faults (S0-F-11; default 0 for Gate A)

Before each round, every worker is dropped independently with probability p, using a generator seeded with `derive_seed(seed, "sim/faults/r<r>")`. A dropped worker is not trained, since its result would be discarded and its data stream is independent of the others. It is logged with its lost compute. Equal compute then refers to steps attempted, and the report states steps actually merged.

### Rounds, checkpoints and resume (S0-N-04)

Each round:

1. Assign slices with the coordinator.
2. Train the surviving workers in order.
3. Merge (`router_rule` and outer optimizer from the configuration).
4. Evaluate the merged full model on the curve set.
5. Write a `round` record: metrics, merge report, workers dropped, each worker's training statistics, and each worker's map-by-expert routing table on the curve set (raw material for the drift diagnostic, S0-F-27, step 8).
6. Write `checkpoint.pt` atomically (temporary file, then `os.replace`) with the global state, the outer optimizer's velocity, the round index and the configuration hash.

`--resume <run folder>` reloads the configuration saved in the run folder, checks that its hash matches the checkpoint, reopens the log in append mode with a `resume` record, and continues from the next round.

Worker state is fresh each round (D19) and every random stream is derived from the seed and the round, so **a resumed run must end bitwise identical to an uninterrupted one**. A test checks this.

At the end: evaluate the gate set, run the routing-necessity diagnostic, compute the Bayes-optimal reference, save the final model, and write the `final` record. The format matches centralized runs, so the same summary script compares them.

### Operations

The same preflight, keep-awake, power and stall logging as centralized runs (`cassandra_chorus.ops`).

## Mathematical / Statistical Details

**Schedule.** For worker step j = r · H + i (round r, local step i) in a trajectory of length T_w = R · H, the learning rate is `lr_at(j, cfg, T_w)` as defined for centralized runs. That is linear warm-up over the first `warmup_steps`, then constant, then linear decay to `final_lr_ratio · lr` over the last round(`decay_fraction` · T_w) steps.

**Skew weights.** For M maps, N workers and skew s ∈ [0, 1], let P_i = {m : m mod N = i mod N}. Then w_{i,m} = (1 - s)/M + s · 1[m ∈ P_i] / |P_i|. These weights sum to 1 for each worker.

**Faults.** Each worker is dropped independently with probability p per round. The expected number of merged workers per round is N(1 - p). A round with no survivors leaves the global state unchanged (step 5 merge).

## Design Decisions

| Decision | Chosen | Alternatives | Decided by |
|---|---|---|---|
| Equal compute | Same batch; N × H × R = centralized steps, checked at start | Batch 64/N per worker | Owner, 2026-10-08 (D17) |
| Gate A setting | N 4, capacity 4 of 8, H 125, R 10 | H 50 (25 rounds); N 2 | Owner, 2026-10-08 (D18) |
| Worker optimizer state | Fresh each round | Kept per worker | Owner, 2026-10-08 (D19) |
| Gate A assignment | Coverage, plus rolling as a second arm | Coverage only; rolling only | Owner, 2026-10-08 (D20) |
| Partial-update router | Averaged over all workers | Holders | Owner, 2026-10-08 (D21) |
| Worker schedule | One schedule over the trajectory | Re-warm-up every round | Owner-confirmed, 2026-10-08 |
| Dropped workers | Not trained (result would be discarded) | Train and discard | Engineering default: identical results, less time |
| Skew form | Preferred maps by index modulo N, mixed with uniform by s | Explicit per-worker weights | Engineering default; explicit weights can be added for sweeps |
| Resume | Per-round atomic checkpoint, bitwise-identical continuation | Restart from scratch | SRS S0-N-04 |

## Verification

To be completed after implementation.

**Not tested:** to be completed.

## Related Docs

- `docs/SRS.md`: S0-F-04, S0-F-07, S0-F-11 to S0-F-14, S0-F-26, S0-N-04, D17 to D21
- `docs/implementations/2026-10-08-coordinator.md`, `2026-10-07-centralized-task-a-pilot.md`, `2026-10-08-run-operations.md`
