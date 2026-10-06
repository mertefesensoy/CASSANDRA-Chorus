# 2026-10-07 · Task A generator: synthetic lookup maps (PLAN step 3)

| | |
|---|---|
| PLAN step | 3 · Task A generator (synthetic maps, marked and unmarked) |
| Branch | `stage0/03-task-a` (stacked on `stage0/02-moe-model`) |
| SRS requirements | S0-F-15 (M random maps, one map per sequence), S0-F-16 (marked and unmarked), S0-F-17 (score only map-determined positions), S0-F-18 (record the map), S0-F-12 (skewed data, prepared for), decision D5 (M = 8, V = 26, L = 64) |
| Status | Planned |

## Problem / Motivation

Task A is the fast test with a known right answer (PLAN section 2). Each sequence is produced by one of M random lookup maps, and the model must work out which map applies. Since the generator knows the map behind every sequence, the router can be checked directly: do sequences from the same map go to the same experts, and does that survive merging?

Two things make the task useful for that question:

- **Scored answers must come from the map stored in the weights.** If a scored answer can be copied from earlier in the same sequence, part of the score measures in-context copying instead. With V = 26 symbols and 32 pairs per sequence, keys must repeat. Only a key's first occurrence is therefore scored (owner decision 2026-10-07).
- **The task is fixed across seeds.** The maps come from a separate task seed, so differences between run seeds 7, 11 and 19 reflect training, not different tasks (owner decision 2026-10-07).

## What Changed

| File | Description |
|---|---|
| `cassandra_chorus/data/task_a.py` | `TaskASection` configuration, `TaskA` generator (maps, batches, fixed evaluation sets), `TaskABatch`, and the Bayes-optimal reference accuracy. |
| `cassandra_chorus/data/__init__.py` | Public names of the data package. |
| `tests/test_task_a.py` | Tests of the token layout, scoring rule, determinism, map weights and the reference accuracy. |
| `docs/implementations/2026-10-07-task-a-generator.md` | This document. |

## Implementation Approach

### Vocabulary

Symbols `0` to `V-1` are the alphabet. Tokens `V` to `V+M-1` are the map markers. Markers are reserved in both variants, so the marked and unmarked models have the same vocabulary size V + M (34 at the reference parameters) and the same architecture.

### Maps

`maps` is an `M × V` integer table: `maps[m, x]` is the value map m gives key x. Each entry is drawn independently and uniformly from the V symbols with a CPU `torch.Generator` seeded by `task_seed`. So each map is a random function, not necessarily a bijection, and two maps can agree on some keys. The table depends only on `task_seed`, M and V.

### One sequence

The model input has length L = `seq_len` (64), with P = L / 2 = 32 key-value pairs (L must be even). L + 1 tokens are generated, and input and target are the first and last L of them:

- **Marked:** `marker(m), k1, v1, k2, v2, ..., k32, v32` gives 65 tokens. The input is `marker, k1, v1, ..., k32` and the targets are `k1, v1, ..., v32`.
- **Unmarked:** `k1, v1, ..., k32, v32, k33` gives 65 tokens. The input is `k1, v1, ..., v32` and the targets are `v1, k2, ..., k33`.

Keys are drawn uniformly with replacement; v_j = maps[m, k_j]. Key 33 of the unmarked variant only completes the target row and is never scored.

**Scoring and training positions.** The target at the input position holding k_j is v_j. It is scored if and only if j ≤ 32 and k_j does not occur among k_1 to k_{j-1}. Every other target is set to -100, which the model's loss ignores. That excludes key targets (pure noise), repeated keys (copyable), and the unmarked trailing key. The same targets drive training and scoring (owner decision 2026-10-07).

### Batches

`TaskA.sample(n, generator, map_ids=None, map_weights=None) -> TaskABatch`. All randomness comes from the caller's CPU `torch.Generator`, so a batch depends only on that generator's state.

- Map ids: given explicitly, or drawn from `map_weights`, or from the configuration's `map_weights` (empty means uniform).
- Returns:
  - `inputs [n, L]` and `targets [n, L]` (int64, -100 where unscored);
  - `map_ids [n]`, the map that produced each sequence (S0-F-18);
  - `scored [n, L]` (bool);
  - `keys [n, P]` and `values [n, P]`, for analysis.
- No side effects except advancing the generator.

`TaskA.evaluation_set(n_per_map, seed)` gives a fixed held-out set with exactly `n_per_map` sequences per map, generated from a fresh generator seeded with `seed`. It is identical every time it is built, and balanced so that per-map accuracy is equally precise for every map.

`TaskA.map_weights` validation: either empty (uniform) or exactly M non-negative numbers with a positive sum, normalized internally. This is the hook for skewed data per worker (S0-F-12, step 6).

### Reference accuracy

`bayes_optimal_accuracy(task, batch, prior=None)` gives the expected accuracy, per scored position, of the best possible predictor that knows the maps. Marked sequences reach 1 by construction. Unmarked sequences cannot reach 1, because the first pairs of a sequence may not identify the map. This reference makes "within 2 percentage points of centralized" (S0-A-02) and the centralized results themselves interpretable. It is an analysis tool and does not change the task.

## Mathematical / Statistical Details

**Number of scored positions.** A sequence draws P keys independently and uniformly from V symbols, and the number of scored positions is the number of distinct keys. Its expectation is

E[distinct] = V · (1 - (1 - 1/V)^P),

which for V = 26, P = 32 is 26 · (1 - (25/26)^32) ≈ 26 · (1 - 0.2851) ≈ 18.59, out of 32 pairs. The tests compare the empirical mean over many sequences with this value.

**Bayes-optimal predictor (unmarked).** Let π be the prior over maps (the sampling weights) and f_m the function of map m. Before the scored position for key x_j, the sequence has shown pairs (x_1, y_1), ..., (x_{j-1}, y_{j-1}), repeats included. The posterior over maps is

post_j(m) ∝ π_m · Π_{i<j} 1[f_m(x_i) = y_i],

because keys carry no information about the map and the observed values are deterministic given it. The predictive distribution of the value is q_j(v) = Σ_m post_j(m) · 1[f_m(x_j) = v]. The best prediction is the value with the largest q_j, and its probability of being right is max_v q_j(v). The reported reference accuracy is the mean of max_v q_j(v) over all scored positions in the batch. It is an expectation, not a sample, so it has no extra sampling noise beyond the choice of sequences.

For the marked variant, the marker fixes m, so post_j is a point mass and the reference accuracy is exactly 1.

Rough intuition at the reference parameters, for checking the implementation rather than as a claim: before any pair, the first scored value is right with probability about the largest share of maps that agree on the key, roughly 1/8 for 8 random maps over 26 symbols. After one pair, about 1 + 7/26 ≈ 1.27 maps remain consistent on average. After two pairs, almost always one. So the unmarked reference should sit a few points below 1.

## Design Decisions

| Decision | Chosen | Alternatives | Decided by |
|---|---|---|---|
| Sequence format | 32 pairs, keys with replacement, only first occurrences scored | 26 distinct pairs padded to 64; 21 key-value-separator triples | Owner, 2026-10-07 |
| Training positions | Same as scored positions | All positions (ordinary language-model loss) | Owner, 2026-10-07 |
| Map source | Separate `task_seed`, same maps for all run seeds | Maps from the run seed | Owner, 2026-10-07 |
| Map type | Random functions (independent uniform entries) | Random permutations | Engineering default: the most generic lookup map; stated for review |
| Markers in the unmarked variant | Reserved but unused, so both variants share vocabulary size | Smaller vocabulary for unmarked | Engineering default: one architecture for both variants |
| Training data | Sampled on the fly from the run's generator, never repeated | Fixed training set with epochs | Engineering default: the maps are what is learned, so sequences are free; sampling fresh ones avoids memorizing sequences |
| Evaluation set | Fixed by its own seed, exactly balanced across maps | Random map mix | Engineering default, so per-map accuracy (open item O1) is equally precise for each map |
| Bayes-optimal reference | Included as an analysis function | Leave interpretation of unmarked accuracy to centralized comparison only | Engineering default: costs little and makes the unmarked numbers interpretable |

## Verification

To be completed after implementation.

**Not tested:** to be completed after implementation.

## Related Docs

- `docs/SRS.md`: S0-F-12, S0-F-15 to S0-F-18, S0-A-01 to S0-A-03, decision D5, open item O1
- `docs/PLAN.md`: section 4 (step 3)
- `docs/implementations/2026-10-07-moe-transformer.md`: the model consumes `targets` with -100 for ignored positions
