# 2026-10-08 · Coordinator: slice assignment and merge (PLAN step 5)

| | |
|---|---|
| PLAN step | 5 · Coordinator logic: slice assignment and merge, with tests |
| Branch | `stage0/05-coordinator` (stacked on `stage0/ops-run-protocol`) |
| SRS requirements | S0-F-05 to S0-F-10, S0-F-25, S0-F-26 (merge side), S0-N-05, S0-N-06; decisions D6, D11, D12 |
| Status | Planned |

## Problem / Motivation

Sliced local training needs two coordinator operations:

1. Before a round, decide which experts each worker holds.
2. After the round, combine what the workers send back into one model.

Stage 1 reuses this code with real workers on other machines, so it must work on plain parameter dictionaries and know nothing about the single-machine simulation (S0-N-05). The layering test from step 1 enforces that mechanically.

The merge rules also decide whether the router can survive slicing at all, which is Stage 0's central question. On 2026-10-08 the owner decided that router rows are merged over their expert's holders by default (D11), after step 2's test showed that a worker gives exactly zero gradient to the router rows of experts it does not hold.

## What Changed

| File | Description |
|---|---|
| `cassandra_chorus/model/transformer.py` | `router_param_layer(name)`: the layer of a router weight, or None. Naming stays a model concern, beside `expert_param_owner`. |
| `cassandra_chorus/coordinator/assignment.py` | `WorkerSpec`, `Slice`, `assign_slices` with the random, coverage and rolling policies and the same-indices and independent cross-layer options. |
| `cassandra_chorus/coordinator/merge.py` | `WorkerResult`, `merge`, `MergeReport`, and the outer optimizers `PlainAverage` (default) and `NesterovOuter`. |
| `cassandra_chorus/coordinator/__init__.py` | Public names. |
| `tests/test_coordinator.py` | The S0-N-06 cases and more (below). |

## Implementation Approach

### Data types

- `WorkerSpec(name, capacity)`: the experts per mixture layer this worker can hold this round. Capacity can differ between workers (S0-F-05) and must satisfy k ≤ capacity ≤ E.
- `Slice(worker, experts)`: `experts` maps every layer to a sorted tuple of global expert indices (S0-F-25). It is exactly the `held` argument of `MoETransformer`, so a slice builds a slice model directly.
- `WorkerResult(slice, state, examples)`: the worker's parameter dictionary after local training, and the number of training examples it processed (S0-F-09). A worker whose result was lost (dropped, S0-F-11) is simply absent from the list.

### Assignment (S0-F-06, S0-F-25, D6, D12)

`assign_slices(workers, n_layers, n_experts, top_k, round_index, policy, cross_layer, seed)` is deterministic given its arguments. Randomness comes from a generator seeded with `derive_seed(seed, "assignment/round<r>")`.

- **random**: each worker independently draws `capacity` distinct experts uniformly.
- **coverage**: every expert is held by at least one worker. This requires the sum of capacities to be at least E, and raises an error otherwise. Shuffle the experts and deal them out one at a time to workers in turn, skipping full workers, until every expert is placed. Then fill each worker's remaining capacity with random experts it does not already hold.
- **rolling** (D12): workers tile a ring of E experts. Worker w starts at offset o_w (the sum of the capacities of the workers before it, modulo E) and holds the window of `capacity` consecutive indices starting at (o_w + r · shift) mod E in round r. **Engineering default: shift = 1**, so the window advances by one expert per round, as in FedRolex; the shift is configurable. If the capacities sum to at least E, every expert is covered in every round.
- **Cross-layer** (D6): `same` (default) uses one draw for every layer. `independent` draws separately per layer for random and coverage; for rolling, each layer's indices go through its own fixed random permutation, drawn from the seed.

### Merge (S0-F-08, S0-F-09, S0-F-10, D11)

`merge(global_state, results, router_rule="holders", outer=PlainAverage())` returns a new state dictionary and a `MergeReport`. It never modifies its inputs.

Each parameter is classified by name, using the model's naming helpers:

| Kind | Merged over | If no contributor |
|---|---|---|
| Expert parameter (layer l, expert e) | Returning workers whose slice holds e in layer l | Unchanged |
| Router weight of layer l, rule `holders` (default) | Row e: workers holding e in layer l | Row unchanged |
| Router weight, rule `all` | Every returning worker, like the shared part | Unchanged |
| Anything else (shared part) | Every returning worker | Unchanged (no worker returned) |

Averages are weighted by `examples` (S0-F-09). Each result's parameter names must be exactly the shared names plus the expert names of its slice, and each tensor must have the global shape. Any mismatch raises an error rather than merging silently.

**Outer optimizer** (S0-F-10). The merge first forms the averaged parameters, then hands the change Δ = average − global to an outer optimizer.

- `PlainAverage` (the default) applies Δ unchanged.
- `NesterovOuter(lr, momentum)` follows DiLoCo:
  - it keeps a velocity per parameter;
  - only parameters (or router rows) that had contributors this round are updated, and their velocity advances;
  - entries with no contributor keep both their value and their velocity (survey design implication 4: "left unchanged" means no outer step).

With lr = 1 and momentum = 0 it equals `PlainAverage` (a test checks this).

**Engineering defaults stated for review:**

- Sums are accumulated in float64 and cast back to the parameter's type, over workers in the order given, so the merge is deterministic and insensitive to rounding from many workers.
- The report lists, per layer, how many workers held each expert, which experts and router rows stayed unchanged, the returning workers, and the total examples.

### Reuse across the comparison arms

All arms use the same `merge`; only what each worker sends back differs:

| Arm | Requirement | Worker returns |
|---|---|---|
| Sliced | (main) | Its slice |
| Partial-update | S0-F-26 | The shared part and its assigned experts, although it held all of them |
| Full-model local averaging | S0-F-14 | Everything |

## Mathematical / Statistical Details

Let W be the set of returning workers and n_w the number of examples worker w processed.

**Shared parameter p:** θ_p ← Σ_{w∈W} n_w θ_{w,p} / Σ_{w∈W} n_w.

**Expert e of layer l:** with H_{l,e} = {w ∈ W : e ∈ slice_w(l)}, θ ← Σ_{w∈H} n_w θ_w / Σ_{w∈H} n_w if H is non-empty; otherwise unchanged.

**Router row e of layer l, rule `holders`:** the same as the expert, using H_{l,e}.

Why: worker w with e ∉ slice_w(l) never computes a gradient for that row, and with weight decay 0 it returns the row unchanged. Under rule `all`, the merged row is then

θ_old + (Σ_{w∈H} n_w (θ_w - θ_old)) / Σ_{w∈W} n_w,

that is, the holders' weighted mean update scaled by Σ_H n_w / Σ_W n_w. With equal example counts that factor is |H| / |W|.

**Nesterov outer step**, per contributing entry, with Δ = θ̄ - θ (the averaged value minus the current global value), learning rate η and momentum μ:

1. v ← μ v + Δ
2. θ ← θ + η (Δ + μ v)

With η = 1 and μ = 0 this gives θ ← θ̄.

**Rolling coverage:** with capacities c_w and Σ_w c_w ≥ E, the windows starting at o_w + r·s with o_w = Σ_{v<w} c_v cover consecutive arcs of the ring that together span all E positions. So every expert is held in every round. Over E rounds with s = 1, every worker has held every expert exactly c_w times.

## Design Decisions

| Decision | Chosen | Alternatives | Decided by |
|---|---|---|---|
| Router rows | Merged over holders by default; all workers as option | All workers only | Owner, 2026-10-08 (D11) |
| Assignment policies | random, coverage, rolling | random and coverage only | SRS S0-F-06; rolling by owner, 2026-10-08 (D12) |
| Cross-layer | same (default), independent | | Owner, 2026-10-07 (D6) |
| Rolling shift | 1 expert per round, configurable | Shift by the window size (disjoint successive windows) | Engineering default, stated for review |
| Coverage fill | Deal shuffled experts in turn, then fill randomly | Linear program as in FLEX-MoE | Engineering default: simple, deterministic; FLEX-MoE's program can be added if load imbalance appears |
| Averaging weights | Examples processed (S0-F-09) | Tokens routed to the expert | SRS as written |
| Outer optimizer | Plain average default; Nesterov option skipping entries without contributors | Heavy-ball momentum | SRS S0-F-10; form follows DiLoCo; skipping per survey implication 4 |
| Precision | float64 accumulation, cast back | float32 | Engineering default |
| Strictness | Mismatched names or shapes raise | Merge what matches | Engineering default: a silent partial merge would corrupt experiments |

## Verification

To be completed after implementation.

**Not tested:** to be completed.

## Related Docs

- `docs/SRS.md`: S0-F-05 to S0-F-10, S0-F-25 to S0-F-27, D11, D12
- `docs/implementations/2026-10-07-moe-transformer.md`: slices as filtered state dictionaries, router rows
- `docs/prior-work.md`: design implications 2, 4 to 7 (FedMoE, DiPaCo, DiLoCo, FedRolex, FLEX-MoE)
