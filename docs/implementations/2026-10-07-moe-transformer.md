# 2026-10-07 · Mixture-of-experts transformer with maskable router (PLAN step 2)

| | |
|---|---|
| PLAN step | 2 · Mixture-of-experts transformer with maskable router |
| Branch | `stage0/02-moe-model` (stacked on `stage0/01-skeleton`) |
| SRS requirements | S0-F-01 (decoder-only transformer, mixture-of-experts feed-forward layers, configurable E and k), S0-F-02 (expert mask), S0-F-03 (sizes from configuration), S0-F-25 (slices per layer), S0-N-02 (size bounds), S0-N-05 (reusable by the coordinator) |
| Status | Verified on the reference laptop (RTX 4070 Laptop GPU), 2026-10-07; see Verification. No training run yet |

## Problem / Motivation

Stage 0 asks whether a router trained while only some experts are present still routes correctly once all experts are merged (PLAN section 2). Every later step needs a model in which "only some experts are present" is a precise, testable state, not an approximation. Specifically:

- the centralized baseline (step 4) trains the full model;
- sliced workers (step 6) train a model holding only their experts;
- the coordinator (step 5) merges parameters by name;
- the metrics (step 8) read routing decisions.

This step builds that model. It also fixes the gate normalization, so that a scale mismatch between sliced training and the merged model cannot be mistaken for a routing failure at Gate A.

## What Changed

| File | Description |
|---|---|
| `cassandra_chorus/model/layers.py` | Rotary position embedding, causal self-attention (explicit or fused), SwiGLU expert. |
| `cassandra_chorus/model/moe.py` | Mixture-of-experts layer: router, masking, top-k gate, sparse dispatch, load-balancing loss, routing statistics. |
| `cassandra_chorus/model/transformer.py` | `ModelSection` configuration, transformer block, `MoETransformer`, `ModelOutput`, parameter naming helper and parameter counts. |
| `cassandra_chorus/model/__init__.py` | Public names of the model package. |
| `tests/test_model.py` | Unit and equivalence tests on CPU, plus CUDA tests for determinism and slice equivalence. |
| `docs/implementations/2026-10-07-moe-transformer.md` | This document. |

## Implementation Approach

### Architecture (owner decision 2026-10-07: modern stack)

A decoder-only, pre-norm transformer. Token embedding, then `n_layers` blocks, then a final RMSNorm and an untied linear output head. Each block is:

```
x = x + Attention(RMSNorm(x))
x = x + MoE(RMSNorm(x))
```

Every feed-forward layer is a mixture-of-experts layer (S0-F-01). Attention uses rotary position embeddings, and the experts are SwiGLU feed-forward blocks. Linear layers have no bias. Engineering defaults, stated for review:

- all linear and embedding weights start as N(0, 0.02), and RMSNorm weights start at 1;
- dropout is configurable, with default 0;
- the output head is untied from the embedding.

### Slices are data (S0-F-25, S0-N-05)

Each mixture layer keeps its experts in a `ModuleDict` keyed by the expert's **global** index (`"0"` to `"E-1"`). `MoETransformer(cfg, held=None)` builds the full model; `held` is a mapping from every layer index to the global expert indices that layer holds, and builds a model containing only those experts. Parameter names carry the global index, for example `blocks.2.moe.experts.5.w1.weight`, so:

- a slice's parameters are exactly the full model's parameters filtered by name, and can be loaded with `load_state_dict(strict=True)`;
- `expert_param_owner(name)` returns `(layer, expert)` for an expert parameter and `None` for the shared part. The coordinator (step 5) will merge by name using this and needs no knowledge of how a slice was chosen.

The router (one `[E, d_model]` weight per layer, one row per expert) belongs to the shared part, as in SRS S0-F-08, and every slice model holds all E rows. Storing one row per expert keeps the survey's proposal (averaging router rows only over holders) possible if the owner adopts it at step 5.

### Routing (S0-F-02; owner decision 2026-10-07: top-k, then softmax over the selected k)

For each token, the router computes E logits. An expert is **available** if the layer holds it and, when the caller passes `expert_mask` (a mapping from layer to allowed global indices), the mask allows it. Unavailable experts get logit minus infinity. The top-k available experts are selected and their gate weights are the softmax over just those k logits (see Mathematical details). Rules:

- `top_k` must be at least 2. With the selected-k softmax and k = 1 the gate weight is always 1, so the router would get no gradient from the task loss. Configuring k = 1 raises an error explaining this.
- If a layer has fewer than k available experts, the model raises an error, at construction for `held` and at the forward pass for `expert_mask`. It never silently selects fewer.

### Dispatch

Tokens are flattened. For each available expert, in ascending global index, the tokens that selected it are gathered, passed through the expert, multiplied by their gate weight and added back with `index_add_`. Within one expert each token appears at most once, and experts are processed in a fixed order, so the result does not depend on scheduling. In the step 1 check, `index_add_` produced no non-determinism warning on CUDA in PyTorch 2.12.1. A dense reference (every expert on every token, weighted by a gate that is zero outside the selected k) exists only in the tests, to check the sparse path.

### Attention implementation

Configurable: `explicit` (scores, causal mask, softmax, product; the default) or `sdpa` (PyTorch's fused `scaled_dot_product_attention`). A check on the reference GPU on 2026-10-07 found that in fp32 the fused path uses the memory-efficient kernel. In `warn` mode, PyTorch reports that this kernel's backward pass "defaults to a non-deterministic algorithm"; in `strict` mode it ran deterministically without error. The explicit path ran with no warning in `warn` mode. Its forward output differed from the fused path by at most 1.8e-6 on random inputs. The explicit path is therefore the default, honouring the owner's choice of deterministic kernels wherever they exist; `sdpa` is available for speed in Task B, and its warning would be recorded in the run log.

### Load-balancing loss (owner decision 2026-10-07: implemented, coefficient default 0)

The Switch-Transformer form, computed per layer over the available experts and averaged over layers (formula below). It is added to the loss as `balance_coef` times that average. The default coefficient is 0, consistent with PLAN Gate A listing the loss as a remedy. At step 4 the centralized Task A baseline will be run with and without it before anything else is decided.

### Public contract

- `ModelSection`: configuration dataclass. Fields: `vocab_size`, `context_length`, `d_model`, `n_layers`, `n_heads`, `expert_hidden`, `n_experts` (default 8), `top_k` (2), `balance_coef` (0.0), `dropout` (0.0), `rope_base` (10000.0), `attention` (`"explicit"`), `init_std` (0.02). The constructor checks consistency: `d_model` divisible by `n_heads`, even head size, 2 ≤ k ≤ E, and so on.
- `MoETransformer.forward(idx, targets=None, expert_mask=None, return_routing=False) -> ModelOutput`:
  - inputs: token indices `[B, T]` with T ≤ `context_length`;
  - outputs: `logits [B, T, V]`; and, when `targets` is given, `ce_loss` and `loss = ce_loss + balance_coef · balance_loss`. Positions where the target is -100 are ignored, which step 3 needs to score only map-determined positions (S0-F-17). Always also returns `balance_loss`, `expert_counts [n_layers, E]` (tokens routed to each expert, by global index), and `routing` (per-layer top-k indices `[B, T, k]`) when `return_routing` is true, for the router analysis of step 8.
  - no side effects.
- `MoETransformer.held_experts() -> dict[int, tuple[int, ...]]`.
- `expert_param_owner(name) -> (layer, expert) | None`.
- `count_parameters(model) -> {"total", "shared", "experts"}` and `expected_parameter_count(cfg, held=None)`, the analytic formula below. A test checks that the two agree.

## Mathematical / Statistical Details

Notation: d = `d_model`, H = `n_heads`, D = d / H the head size, h = `expert_hidden`, E = `n_experts`, k = `top_k`, V = `vocab_size`, n_L = `n_layers`.

**RMSNorm.** For x ∈ R^d: RMSNorm(x) = w ⊙ x / sqrt(mean_i(x_i²) + ε), with learned w ∈ R^d and ε = 1e-6.

**Rotary position embedding.** For each head, split the D coordinates of a query or key at position p into halves (a, b) of size D/2. With θ_j = base^(-2j/D) for j = 0 to D/2 - 1 and base = `rope_base`, the rotated vector is (a ⊙ cos(pθ) - b ⊙ sin(pθ), a ⊙ sin(pθ) + b ⊙ cos(pθ)). This makes the score q_p · k_s depend on positions only through p - s.

**Attention.** For each head, scores S = Q Kᵀ / sqrt(D), with S_ps = -∞ for s > p (causal), then softmax over s, then multiply by V. The heads are concatenated and projected by W_o.

**SwiGLU expert.** Expert(x) = W₂ (SiLU(W₁ x) ⊙ (W₃ x)), with W₁, W₃ ∈ R^{h×d}, W₂ ∈ R^{d×h} and SiLU(u) = u · σ(u).

**Router and gate.** For token x_t, the logits are z_t = R x_t with R ∈ R^{E×d}. Let A be the set of available experts and n = |A|. Set z_{t,i} = -∞ for i ∉ A. Let S_t be the k indices in A with the largest z_{t,i}. The gate weights are

g_{t,i} = exp(z_{t,i}) / Σ_{j ∈ S_t} exp(z_{t,j}) for i ∈ S_t, and 0 otherwise,

so Σ_i g_{t,i} = 1 whatever A is. The layer output is y_t = Σ_{i ∈ S_t} g_{t,i} · Expert_i(x_t). Why not softmax over A first: the denominator would then contain n terms, so the same logits produce larger weights on a 4-expert slice than in the merged 8-expert model. That is a scale shift between sliced training and merged inference.

Gradient note: for i ∉ A the router row R_i receives zero gradient. For i ∈ A it receives gradient only through tokens that selected it (via g) and, if the balance loss is on, through P_i below.

**Load-balancing loss** (Switch Transformer form, restricted to A). Over the T tokens of a batch:

- f_i = (1 / (T k)) Σ_t 1[i ∈ S_t], the fraction of routing assignments that went to expert i, so Σ_i f_i = 1;
- P_i = (1/T) Σ_t p_{t,i}, where p_t = softmax over A of z_t, the mean router probability of expert i;
- L_bal = n · Σ_{i ∈ A} f_i P_i.

When routing is uniform (f_i = P_i = 1/n), L_bal = 1, its minimal value in that case. It grows as assignments and probabilities concentrate on the same experts. Only P_i carries gradient (f_i comes from a discrete selection). The model reports the mean of L_bal over layers. The training loss is CE + `balance_coef` · that mean.

**Parameter count** (no biases, untied head):

total = 2Vd (embedding and head) + d (final norm) + n_L · (4d² (attention) + 2d (two norms) + E·d (router) + E_held · 3dh (experts)),

where E_held is the number of experts the layer holds, summed per layer for a slice. Worked examples of where the S0-N-02 bounds fall, for illustration only (actual sizes are chosen at steps 4 and 10 and put to the owner):

| Example | V | d | n_L | H | h | E | Total parameters |
|---|---|---|---|---|---|---|---|
| Task A-like | 34 | 128 | 4 | 4 | 256 | 8 | 3,421,824 (under the 5M bound) |
| Task B-like | 27 | 384 | 6 | 6 | 768 | 8 | 46,050,432 (inside 20M to 50M) |

## Design Decisions

| Decision | Chosen | Alternatives | Decided by |
|---|---|---|---|
| Gate normalization | Top-k over available experts, then softmax over the selected k; k = 2 | Softmax then top-k (Switch, GShard): scale depends on how many experts are available | Owner, 2026-10-07 |
| k = 1 | Rejected with an error | Allow it (router would get no task gradient) | Follows from the gate decision |
| Load-balancing loss | Implemented, coefficient 0 by default; decided with evidence at step 4 | On everywhere now | Owner, 2026-10-07 |
| Base architecture | RMSNorm, rotary positions, SwiGLU experts, pre-norm | Match the parent's LayerNorm, learned positions, GELU | Owner, 2026-10-07 |
| What a slice is | A model containing only the held experts, keyed by global index | Full model plus a runtime mask: simpler, but then "slice" is a flag the coordinator cannot reason about, and the worker would hold the whole model | Engineering default stated to the owner before implementation |
| Router placement | Shared part, all E rows on every worker | Router rows travel with experts | SRS S0-F-08 as written; the survey's alternative is for the step 5 review |
| Attention kernel | Explicit by default, fused optional | Fused by default | Engineering default, following the owner's determinism decision |
| Dispatch | Sparse with `index_add_`; dense only as a test reference | Dense in the model (E/k times the expert compute) | Engineering default |
| Capacity limit | None: every token goes to its k experts | Capacity factor with token dropping | Engineering default: dropping would add a second effect to the experiment |
| Precision | fp32 in this step | Mixed precision | Deferred to Task B (step 10), where speed matters |
| Init, dropout, head | N(0, 0.02), dropout 0, untied head | Scaled residual init, tied head | Engineering defaults stated for review |

## Verification

Run on 2026-10-07 on the reference laptop: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB), Windows 11, Python 3.13.14, PyTorch 2.12.1+cu126.

**1. Test suite.** `python -m pytest -v`: 98 passed, 0 skipped. That is the 64 tests of step 1 plus 34 in `tests/test_model.py`. The model tests use a tiny configuration (V = 11, d = 32, 2 layers, 4 heads, h = 48, E = 4, k = 2) on random tokens; they check behaviour, not learning. What they establish:

- Output shapes, and that each layer routes exactly B·T·k assignments.
- Loss = CE + `balance_coef` · balance loss, with plain CE at the default coefficient 0. Targets of -100 are ignored.
- Causality. Routing decisions before a changed token are identical, and logits there agree to 3e-8, not bitwise. The difference comes from experts regrouping tokens, so a matrix product runs over a different number of rows; block-0 attention output was bitwise identical. Logits from the changed token on differ by up to 0.29.
- Every configuration error listed in the plan raises, including k = 1, k > E, and a slice or mask leaving fewer than k experts.
- Gate weights sum to 1, and only available experts are ever selected.
- A mask allowing every expert gives bitwise the same logits as no mask.
- Sparse dispatch matches the dense reference to 1e-6.
- Router rows of masked experts get exactly zero gradient, and masked experts get no gradient at all.
- The balance loss is 1 under uniform routing and 2 in a constructed collapse onto 2 of 4 experts.
- **A slice model loaded from the full model's filtered state dict gives bitwise the same logits, routing, counts, loss and gradients as the full model under the matching mask,** on CPU and on CUDA (`strict` mode).
- `expert_param_owner`, and parameter counts equal to the formula for full and slice models and for the two worked examples.
- The fused attention path agrees with the explicit path to 1e-5.
- On CUDA in `strict` mode, a forward and backward pass completes (so every operation has a deterministic kernel) and two repetitions give bit-identical logits and gradients.
- On CUDA in `warn` mode, no determinism warning is raised with explicit attention.

**2. Footprint at the sizes of S0-N-02.** A throwaway measurement, not committed as a script. Seed 7, `warn` mode, fp32, explicit attention, AdamW with learning rate 1e-3, 12 optimizer steps on one repeated random batch, median of the last 10 step times:

| Configuration | Parameters | Batch | Median step | Peak GPU memory |
|---|---|---|---|---|
| Task A-like (V 34, d 128, 4 layers, 4 heads, h 256, E 8, k 2) | 3,421,824 | 64 × 64 | 114 ms | 393 MiB |
| Task B-like (V 27, d 384, 6 layers, 6 heads, h 768, E 8, k 2) | 46,050,432 | 32 × 256 | 212 ms | 3,423 MiB |

Both fit easily in 8 GB. The Task A step time is high for the model size. The likely cause is that dispatch waits on the GPU once per expert per layer (finding which tokens chose the expert), 32 times per pass here; this was not profiled. If it limits the sweeps, it can be optimized without changing results; it was left as is.

**Not tested:**

- Learning. No model has been trained on any task; the losses in the footprint run come from repeatedly fitting one random batch and mean nothing.
- Dropout above 0: implemented but not exercised by any test.
- The fused attention path on CUDA in `warn` mode is known to be non-deterministic (step 2 check) and is not the default; it was not run end to end.
- Mixed precision: not implemented.
- Throughput at other batch shapes, and on any machine other than the reference laptop.
- The independent-per-layer slice policy specifically. Slices here are arbitrary per-layer subsets, which covers both policies, but no assignment policy exists until step 5.

## Related Docs

- `docs/SRS.md`: S0-F-01 to S0-F-03, S0-F-08, S0-F-25, S0-N-02, S0-N-05
- `docs/PLAN.md`: section 4 (step 2), section 5 (Gate A remedies)
- `docs/implementations/2026-10-07-repository-skeleton.md`: determinism modes and the `index_add_` observation
- `docs/prior-work.md` (draft, unreviewed): design implications 2, 3 and 10
