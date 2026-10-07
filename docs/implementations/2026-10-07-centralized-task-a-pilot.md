# 2026-10-07 · Centralized Task A pilot, routing-necessity diagnostic, dense dispatch (PLAN step 4, part 1)

| | |
|---|---|
| PLAN step | 4 · Centralized baseline on Task A, part 1: pilot, diagnostic, budget proposal |
| Branch | `stage0/04-centralized-task-a` (stacked on `stage0/03-task-a`) |
| SRS requirements | S0-F-13 (centralized baseline; equal compute is set later), S0-F-20 (held-out accuracy of the full model), S0-F-24 (run log), S0-A-01 precondition (per map, D10), S0-N-03, S0-N-07, S0-N-08 |
| Status | Planned |

## Problem / Motivation

Step 4 produces the first trained models. The owner decided on 2026-10-07 that it starts with a pilot rather than the full matrix, for two reasons:

1. **Does Task A have the power to detect a routing failure?** Task A's content is 8 maps of 26 entries, about a thousand bits. One expert at the pilot size has about 98,000 parameters and could store all of it. If every expert learns every map, a router that sends a map's tokens to the "wrong" experts still gets the answer right. S0-A-01 to S0-A-03 could then pass with a broken router. Whether this happens depends on training dynamics, so it is measured: the trained centralized model is evaluated with its own router, with random routing, and with each expert removed. If random routing costs little accuracy, the owner chooses between keeping Task A as a plumbing check and changing decision D5.
2. **The equal-compute rule of S0-F-13 has no number yet.** It requires the centralized run to use the same total compute as the sliced run, which step 6 defines. The pilot produces learning curves from which a budget is proposed to the owner.

This step also adds the dense dispatch option approved by the owner, because it roughly halves the time of every Task A run.

## What Changed

| File | Description |
|---|---|
| `cassandra_chorus/model/moe.py` | Dense dispatch over available experts as an alternative to sparse; optional `select` hook replacing top-k selection (used by the random-routing diagnostic). |
| `cassandra_chorus/model/transformer.py` | `dispatch` field in `ModelSection` (default `"sparse"`); `select` passed through `forward`. |
| `cassandra_chorus/repro.py` | `derive_seed(seed, label)`: independent, reproducible seeds for separate random streams. |
| `cassandra_chorus/train/__init__.py`, `cassandra_chorus/train/loop.py` | `OptimSection`, optimizer construction, learning-rate schedule, and `train_steps`: a task-independent training loop for the centralized baseline, the step 6 workers and later Stage 1 clients. |
| `cassandra_chorus/metrics/task_a.py` | Task A evaluation: pooled and per-map accuracy, scored loss, expert counts, map-by-expert routing table, and the routing-necessity diagnostic. |
| `scripts/train_task_a_centralized.py` | Entry point: centralized training on Task A with periodic evaluation, final evaluation, diagnostic and checkpoint. |
| `configs/task_a/centralized_pilot.toml` | Pilot configuration. |
| `tests/test_model.py` | Dispatch-mode tests and `select` hook tests added; key tests run in both modes. |
| `tests/test_train.py`, `tests/test_metrics_task_a.py` | Tests of the loop, schedule, evaluation and diagnostic. |
| `RESULTS.md` | First entries: the two pilot runs. |
| `docs/implementations/2026-10-07-moe-transformer.md` | One-line pointer to the dispatch change described here. |

## Implementation Approach

### Dense dispatch (owner decision 2026-10-07)

`ModelSection.dispatch` is `"sparse"` (step 2 behaviour) or `"dense"`. Dense stacks the weights of the **available** experts, computes every available expert on every token with batched matrix products, and combines them with a gate matrix that is zero outside each token's selected k. The selection and gate weights are identical in both modes. The outputs agree to rounding: 4.8e-7 on the logits at Task A size, measured before this change.

Unavailable experts are never computed, so slices behave the same in both modes. One documented difference: in dense mode an available expert that no token selects still receives a gradient tensor, which is all zeros, where sparse mode gives none. AdamW skips parameters without a gradient but still applies its momentum to a zero gradient. At about 4,000 tokens per batch and 8 experts, an expert selected by no token is unlikely; the difference is recorded, not engineered away.

Dense memory grows with E × tokens × `expert_hidden`, so Task B will use sparse. Task A configurations set `dispatch = "dense"`.

### Selection hook

`MoETransformer.forward(..., select=None)`. If `select` is given, it is called as `select(layer, logits)` with the masked router logits `[N, E]` and must return `[N, k]` distinct available expert indices. The gate weights are then the softmax over the router's logits of those experts, as in normal routing. This exists only for diagnostics.

### Training loop (`cassandra_chorus/train/loop.py`)

Task-independent. It receives a model, an optimizer, a function returning the next `(inputs, targets)` batch, the number of steps and the `OptimSection`.

Each step does the following:

1. Set the learning rate for the step.
2. Run forward and backward on `loss` (cross-entropy plus `balance_coef` times the balance loss).
3. Clip the global gradient norm.
4. Take an AdamW step.

It returns the number of steps, sequences and scored targets processed and the mean losses. Stated defaults: AdamW with betas (0.9, 0.95), eps 1e-8, **weight decay 0**, learning rate 1e-3 with linear warm-up over the first 100 steps then constant, gradient-norm clip 1.0, batch 64.

Weight decay is 0 because decay would shrink the router rows of masked experts on sliced workers, which receive no gradient there (survey design implication 2). That would confound the step 6 comparison, so it is removed from the baseline as well.

### Evaluation (`cassandra_chorus/metrics/task_a.py`)

`evaluate_task_a(model, batch, device, ...)` runs the model in evaluation mode, without gradients, in chunks, and reports:

- **accuracy:** the share of scored positions where the argmax of the logits equals the target; pooled and per map;
- **minimum per-map accuracy;**
- **scored loss:** mean cross-entropy on scored positions;
- **expert counts:** routed assignments per layer and expert, over all positions;
- **map-by-expert table:** per layer, the number of times sequences of map m sent a scored position's token to expert e.

Step 8 builds the formal router-consistency and usage metrics on these counts.

`routing_necessity(model, batch, device, seed)` evaluates the same set three ways:

1. with the trained router;
2. with random routing: k distinct experts drawn uniformly per token and layer from a generator seeded by `seed`, with gates from the router's logits of those experts;
3. with each expert e removed in every layer (8 evaluations).

It reports pooled and minimum per-map accuracy for each.

### Entry point and run protocol

`python -m scripts.train_task_a_centralized --config configs/task_a/centralized_pilot.toml [--set ...]`

1. Load the configuration, then call `repro.prepare_process` before any CUDA work.
2. Seed everything with the run seed, which drives model initialization.
3. Draw training batches from a generator seeded with `derive_seed(seed, "task_a/train")`, so the data stream is independent of the initialization stream.
4. Every `eval_every` steps, evaluate on the **curve set** (256 sequences per map, evaluation seed 1001) and log an `eval` record.
5. At the end:
   - evaluate on the **gate set** (2,048 sequences per map, seed 1002; D10);
   - run the routing-necessity diagnostic on the curve set;
   - log both;
   - save `final_model.pt` in the run folder, which is outside the repository on the reference laptop.

The gate set is never used during training or for choosing anything during a run.

Pilot (owner decision 2026-10-07): seed 7, marked variant, 5,000 steps of 64 sequences, `balance_coef` 0 and then 0.01 (the Switch Transformer value). Pilot model: d = 128, 4 layers, 4 heads, `expert_hidden` 256, E = 8, k = 2, explicit attention, dense dispatch; 3,421,824 parameters, under the 5M bound of S0-N-02.

## Mathematical / Statistical Details

**Learning rate.** For step s (counting from 0) with base rate η and W warm-up steps: η_s = η · (s + 1) / W for s < W, and η_s = η afterwards.

**Accuracy.** Over a set of scored positions P, accuracy = (1 / |P|) Σ_{p ∈ P} 1[argmax_v logits_p(v) = target_p]. Per-map accuracy restricts P to sequences of that map. With about 38,000 scored positions per map in the gate set, one error moves a map's accuracy by about 0.0026 percentage points, and the 99.9% precondition allows about 38 errors per map.

**Scored loss.** The mean of -log softmax(logits_p)[target_p] over P, in nats.

**Derived seeds.** derive_seed(s, label) = the first 8 bytes of SHA-256 of the text "s/label", read as an unsigned integer, then reduced modulo 2^63. Different labels give independent-looking seeds, and the same inputs always give the same seed.

**Random routing.** For each token and layer, a uniform random score is drawn for every available expert, and the k experts with the highest scores are chosen. That is a uniformly random k-subset. The gate weights are the softmax over the router's logits of the chosen experts.

## Design Decisions

| Decision | Chosen | Alternatives | Decided by |
|---|---|---|---|
| Step 4 order | Pilot, diagnostic, budget proposal, then the full matrix after approval | Full matrix now with a guessed budget | Owner, 2026-10-07 |
| Routing-necessity diagnostic | Included | Proceed without knowing whether Task A can detect routing failure | Owner, 2026-10-07 |
| Per-map criterion and gate-set size | Per map, 2,048 sequences per map | Pooled; smaller set | Owner, 2026-10-07 (D10) |
| Dispatch | Configurable; dense for Task A, sparse for Task B | Sparse only | Owner, 2026-10-07 |
| Optimizer | AdamW, weight decay 0, betas (0.9, 0.95), 1e-3, 100-step warm-up then constant, clip 1.0, batch 64 | Cosine decay; weight decay 0.1 | Engineering defaults stated to the owner before implementation; weight decay 0 for the masked-router reason above |
| Loop location | `cassandra_chorus/train/`, task-independent | Inside the entry point | Engineering default stated to the owner; reused in steps 6 and 7 and Stage 1 |
| Random routing gates | Router's logits of the random experts | Equal weights 1/k | Engineering default: changes only which experts are used |
| Leave-one-out | One expert removed from every layer at a time (8 evaluations) | Each layer and expert separately (32 evaluations) | Engineering default; can be extended if the result calls for it |
| Data stream seed | Derived from the run seed with a label | The run seed itself | Engineering default: keeps initialization and data streams independent |

## Verification

To be completed after implementation and the pilot runs.

**Not tested:** to be completed.

## Related Docs

- `docs/SRS.md`: S0-F-13, S0-F-20, S0-A-01 (D10), S0-N-02, S0-N-08
- `docs/PLAN.md`: section 4 (step 4), section 10
- `docs/implementations/2026-10-07-moe-transformer.md`, `docs/implementations/2026-10-07-task-a-generator.md`
- `docs/prior-work.md` (draft, unreviewed): design implication 2 (weight decay on masked router rows)
