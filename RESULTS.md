# CASSANDRA Chorus · Results

Hand-written record of every run that informs a decision, failed runs included (SRS S0-N-08). Raw run logs and checkpoints stay local; each entry names the run IDs and configuration hashes it is based on, so the raw logs can be matched to it.

Each entry states (SRS S0-N-07): date, PLAN step, the exact command and configuration file, configuration hash, git commit, seeds, hardware, model size, dataset, the results, and what was not tested.

Raw logs on the reference laptop are in `C:\Users\senso\chorus-runs\<run ID>\log.jsonl`. Summaries below can be regenerated with `python -m scripts.summarize_task_a_runs <run folder> ...`.

## 2026-10-07 · PLAN step 4 pilot: centralized Task A, balance loss 0 versus 0.01

**Purpose.** First trained models. Answers three owner questions before the full centralized matrix is run: whether centralized training meets the S0-A-01 precondition per map, whether Task A can detect a routing failure (the routing-necessity diagnostic), and whether centralized training leaves experts with negligible traffic without the load-balancing loss (S0-A-03). Also provides the learning curve for the compute-budget proposal.

**Setup common to both runs.**

- Hardware: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB), Windows 11, Python 3.13.14, PyTorch 2.12.1+cu126, CUDA 12.6, determinism mode `warn`, 1 CPU thread.
- Model: 3,421,824 parameters. d 128, 4 layers, 4 heads, `expert_hidden` 256, E 8, k 2, RMSNorm, rotary positions, SwiGLU experts, explicit attention, dense dispatch.
- Data: Task A, marked variant, M 8, V 26, L 64, `task_seed` 20261007 (maps SHA-256 `1cf65b5600fa...`). Training batches sampled fresh from a stream seeded by `derive_seed(7, "task_a/train")`.
- Optimizer: AdamW, learning rate 1e-3 with 100-step warm-up then constant, betas (0.9, 0.95), weight decay 0, gradient clip 1.0, batch 64.
- Budget: 5,000 steps (320,000 sequences).
- Run seed 7 only.
- Gate set: 2,048 sequences per map (304,854 scored positions), evaluation seed 1002.
- Curve set: 256 per map, seed 1001; it is also used for the diagnostic (seed 2001).

| | Pilot 1 | Pilot 2 |
|---|---|---|
| Run ID | `20261007T035715Z_taskA-central-pilot-bal0_s7` | `20261007T144407Z_taskA-central-pilot-bal001_s7` |
| Command | `python -m scripts.train_task_a_centralized --config configs/task_a/centralized_pilot.toml` | same, plus `--set model.balance_coef=0.01 --set run.name=taskA-central-pilot-bal001` |
| Config hash, commit | `5984e7905249`, `277e61c` (clean) | `a09eec2a6d77`, `62f8148` (clean; differs from `277e61c` only by an added summary script) |
| `balance_coef` | 0 | 0.01 |
| First curve evaluation with every map at least 0.999 | step 750 | step 1,000 |
| Brief dips after that (lowest map on the curve set) | step 1,250: 0.9861; step 4,000: 0.9973 | step 3,000: 0.9951; 3,250: 0.9979; 4,750: 0.9983 |
| **Gate set, final model** | accuracy 1.00000, **every map 1.00000** | accuracy 1.00000, **every map 1.00000** |
| S0-A-01 precondition (every map at least 99.9%) | met | met |
| Diagnostic: trained router | 1.0000 | 1.0000 |
| Diagnostic: **random routing** (worst map) | **0.4604** (0.3808) | **0.4856** (0.4487) |
| Diagnostic: one expert removed from every layer | 0.8656 to 0.9896 | 0.8912 to 0.9808 |
| Least-used expert's share of tokens, per layer (negligible below 0.025) | 0.116, 0.137, 0.169, 0.191 | 0.236, 0.241, 0.241, 0.238 |
| Share of a map's routing that goes to its 2 most-used experts, per layer (0.25 = no preference) | 0.412, 0.533, 0.705, 0.409 | 0.379, 0.372, 0.370, 0.350 |
| Training time (excluding evaluation) | 631 s | 384 s |

**What this shows (seed 7 only, marked variant only):**

1. Centralized training meets the S0-A-01 precondition on every map, with or without the balance loss, at this model size and budget.
2. **Task A can detect a routing failure.** With routing randomized, accuracy falls from 1.0 to about 0.46 to 0.49, and removing a single expert costs up to 13 points. The trained model depends on its router choosing the right experts. The concern raised before the pilot, that every expert might store every map and make routing irrelevant, is not borne out in this run.
3. Without the balance loss, no expert has negligible traffic: the smallest share is 0.116 against the 0.025 threshold. Per the owner's decision of 2026-10-07, the balance loss therefore stays a Gate A remedy with default coefficient 0, and no plan change is proposed.
4. Without the balance loss, layer 2 specializes by map: each map sends half its routing assignments to one dominant expert. With it, usage is close to uniform and the map preference is weaker in every layer.
5. Accuracy is not perfectly stable after convergence at a constant learning rate: brief dips of up to 1.4 points on one map, which recover by the next evaluation.

**Not tested:** other seeds (11, 19); the unmarked variant; other model sizes or budgets; any sliced or local-averaging training. The routing-necessity numbers come from the 2,048-sequence curve set, not the gate set. The training times are not comparable between the pilots: pilot 1 ran while the laptop's power adapter was dropping out under load, and pilot 2 after the BIOS update (see below).

### Engineering runs on 2026-10-07 (inform no research decision; listed so every log is accounted for)

| Run ID | What it was | Outcome |
|---|---|---|
| `20261007T033745Z_taskA-central-pilot-bal0_s7` | First pilot attempt, 14 CPU threads, background job | Stopped by hand at about 2.5 s per step; no end record. Led to `run.cpu_threads = 1` (commit `277e61c`) |
| `20261007T035512Z_perfcheck-foreground_s7`, `20261007T035552Z_perfcheck-background_s7` | 300-step timing checks, same configuration, foreground and background | 16 s versus 48 s of training. **Gate accuracy, per-map accuracy, loss and diagnostic identical between the two processes** |
| `20261007T040840Z_taskA-central-pilot-bal001_s7` | First pilot 2 attempt | Stopped by hand at step 3,750 when the laptop's power adapter was found dropping out under load; no end record. Superseded by `20261007T144407Z` |
| `20261007T144107Z_powertest_s7` | 1,500-step load test after the BIOS update to G614JI.334, with a watchdog ready to stop it on any adapter drop-out | No drop-out during the test or the pilot 2 run that followed (morning baseline: 2 to 6 per minute under the same load) |
