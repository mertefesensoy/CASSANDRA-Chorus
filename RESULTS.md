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

## 2026-10-08 · PLAN step 4: full centralized Task A matrix

**Purpose.** The centralized reference for every later Task A comparison (S0-F-13), at the budget the owner approved on 2026-10-07: 5,000 steps of 64 sequences (320,000 sequences per run). Both variants, seeds 7, 11 and 19 (S0-A-06), balance coefficient 0.

**Setup.**

- Same hardware, software and model as the pilot entry below: RTX 4070 Laptop GPU, PyTorch 2.12.1+cu126, 3,421,824 parameters, dense dispatch, `warn` determinism, 1 CPU thread.
- One change from the pilot: the learning rate now decays linearly to 1e-4 over the last 1,000 steps (owner decision 2026-10-07).
- Configuration `configs/task_a/centralized.toml`, commit `723f072`, clean working tree for every run.
- Command: `python -m scripts.train_task_a_centralized --config configs/task_a/centralized.toml --set run.seed=<seed> --set task_a.marked=<true|false> --set run.name=taskA-central-<marked|unmarked>`.
- Gate set: 2,048 sequences per map (evaluation seed 1002). Diagnostic on the curve set (seed 1001).

| Variant | Seed | Run ID | Config hash | Gate accuracy | Lowest map | Best possible (Bayes) | Precondition (every map at least 99.9%) | Random routing (worst map) | One expert removed |
|---|---|---|---|---|---|---|---|---|---|
| marked | 7 | `20261007T210724Z_taskA-central-marked_s7` | `d7dc789feb23` | 1.00000 | 1.00000 | 1.00000 | met | 0.4909 (0.3835) | 0.8538 to 0.9766 |
| marked | 11 | `20261008T023619Z_taskA-central-marked_s11` | `54072d44f16f` | 1.00000 | 1.00000 | 1.00000 | met | 0.4857 (0.4041) | 0.9138 to 0.9922 |
| marked | 19 | `20261008T052122Z_taskA-central-marked_s19` | `647f4386685c` | 1.00000 | 1.00000 | 1.00000 | met | 0.4626 (0.3230) | 0.8190 to 0.9985 |
| unmarked | 7 | `20261008T062954Z_taskA-central-unmarked_s7` | `539abe295397` | 0.95074 | 0.94552 | 0.95106 | not applicable | 0.2563 (0.2104) | 0.8337 to 0.9111 |
| unmarked | 11 | `20261008T063833Z_taskA-central-unmarked_s11` | `1b15d30112a7` | 0.95111 | 0.94503 | 0.95106 | not applicable | 0.2830 (0.2170) | 0.8707 to 0.9137 |
| unmarked | 19 | `20261008T064621Z_taskA-central-unmarked_s19` | `bc4201220a7f` | 0.95095 | 0.94813 | 0.95106 | not applicable | 0.1677 (0.1307) | 0.8212 to 0.9024 |

Least-used expert's share of tokens, per layer, on the gate set (negligible below 0.025, S0-A-03):

| Run | Layers 0 to 3 |
|---|---|
| marked, seed 7 | 0.099, 0.098, 0.127, 0.197 |
| marked, seed 11 | 0.153, 0.179, 0.156, 0.185 |
| marked, seed 19 | 0.155, 0.134, 0.176, 0.153 |
| unmarked, seed 7 | 0.097, 0.182, 0.146, 0.062 |
| unmarked, seed 11 | 0.118, 0.208, 0.117, 0.096 |
| unmarked, seed 19 | 0.174, 0.162, 0.149, 0.115 |

**What this shows (centralized training only, this model size and budget):**

1. **Marked variant: the S0-A-01 precondition holds for every map, on all three seeds.** Each run reached every map at least 0.999 on the curve set by step 750 to 1,250, and finished at 1.00000 on every map of the gate set.
2. **Unmarked variant: accuracy reaches the Bayes-optimal ceiling.** The gate accuracies are 0.95074, 0.95111 and 0.95095, against an expected best of 0.95106: within 0.04 percentage points either way. One run lands slightly above it, which is possible because the ceiling is an expected value and the gate set is a fixed sample. This is the reference that S0-A-02 (sliced within 2 points of centralized) will be measured against.
3. **Routing is necessary in every run, and more so in the unmarked variant.** Random routing gives 0.46 to 0.49 on the marked variant, against 1.0 trained. On the unmarked variant it gives 0.17 to 0.28, against about 0.95. Removing one expert from every layer costs up to 18 points. Task A can therefore detect a routing failure at these parameters (the question raised before the pilot), with the caveat that this is a property of these trained models, not a guarantee for sliced ones.
4. **No expert has negligible traffic in any run without the balance loss.** The smallest share is 0.062 against the 0.025 threshold. The balance loss stays a Gate A remedy, as decided.
5. **With the final decay, every marked run ended at 1.00000.** Brief dips after step 1,500 still occurred (once per run, lowest map on the curve set 0.9962 to 0.9979), but none at the end.

**Operational notes (do not affect the numbers).**

- Training is deterministic given configuration and seed, so the stalls below changed only wall-clock time.
- Runs 1 to 3 were repeatedly frozen when the laptop fell into Modern Standby after mains drop-outs. Their recorded training times (19,684 s, 9,851 s and 4,073 s) are therefore not speeds.
- Runs 4 to 6, with the screen on and a keep-awake request held, took 496 s, 442 s and 570 s.

**Not tested:**

- Sliced and full-model local-averaging training (steps 5 to 7).
- The balance-loss arm on seeds 11 and 19.
- Other model sizes and budgets.
- Bit-exact reproduction of these runs on another machine.

## 2026-10-08 · Check runs before Gate A: expert redundancy after slicing (preliminary)

**Purpose.**

- Plumbing and timing checks of the simulation harness (PLAN step 6).
- A matched centralized check of an observation from the first one.
- Together they informed the Gate A reading registered in SRS D23 before any Gate A run.

**These are not Gate A evidence:** one seed, a fifth of the budget.

**Setup.**

- Both runs: seed 7, marked variant, 1,000 worker or centralized steps of 64 sequences.
- Same model, task and optimizer settings as the centralized matrix, on the RTX 4070.
- Learning-rate decay over the last 20% of each trajectory.

| Run ID | What | Gate accuracy | Random routing | One expert removed (worst) | Router consistency, layers 0 to 3 |
|---|---|---|---|---|---|
| `20261008T080021Z_simcheck_s7` | Sliced: 4 workers × 4 of 8 experts, 125 steps, **2 rounds**, coverage, router merged over holders. Config `configs/task_a/sliced.toml` with `sim.rounds=2`, `sim.equal_compute_steps=1000`; commit `3f869a3` | 1.00000 every map | **0.894** | 0.997 | 0.078, 0.342, 0.296, 0.032 |
| `20261008T080441Z_central-1000-check_s7` | Centralized, `train.steps=1000`; config `configs/task_a/centralized.toml`; commit `43f3958` | 1.00000 every map | **0.500** | 0.906 | 0.037, 0.139, 0.314, 0.058 |

Router consistency is the normalized mutual information between map and chosen expert (SRS D22; ceiling 2/3). For reference, the six full centralized runs give 0.04 to 0.05, 0.13 to 0.35, 0.22 to 0.30, 0.04 to 0.06 (marked) and about 0, 0.23 to 0.30, 0.12 to 0.37, 0.01 to 0.04 (unmarked).

**What this suggests (one seed, short budget):**

- At equal budget, the merged sliced model routes about as consistently by map as centralized training.
- But its experts are largely interchangeable. Random routing keeps 89% accuracy against 50%, and removing any one expert costs at most 0.3 points against 9.4.
- A plausible mechanism, not tested: on each worker, 4 experts must cover all 8 maps, so experts become generalists.
- Task A's accuracy criteria cannot detect this. A redundant mixture of experts wastes capacity, which matters for Stage 2.

**Not tested:**

- Other seeds, the unmarked variant, and the full budget (the Gate A matrix).
- The mechanism.
- Whether the redundancy costs anything on Task B.

### Engineering runs on 2026-10-08

| Run ID | What it was | Outcome |
|---|---|---|
| `20261008T073034Z_opscheck_s7` | 500-step centralized run through the visible launcher, testing the run-operations module | All preflight checks passed, keep-awake held, monitor matched Windows events; see `docs/implementations/2026-10-08-run-operations.md` |
| `20261007T205645Z_taskA-central-marked_s7` | First attempt at matrix run 1, with the power watchdog set to stop on any mains drop-out | Stopped by the watchdog at about step 2,250 after a drop-out at 23:59:50; no end record. After this the owner chose to log drop-outs only. Superseded by `20261007T210724Z` |

### Engineering runs on 2026-10-07 (inform no research decision; listed so every log is accounted for)

| Run ID | What it was | Outcome |
|---|---|---|
| `20261007T033745Z_taskA-central-pilot-bal0_s7` | First pilot attempt, 14 CPU threads, background job | Stopped by hand at about 2.5 s per step; no end record. Led to `run.cpu_threads = 1` (commit `277e61c`) |
| `20261007T035512Z_perfcheck-foreground_s7`, `20261007T035552Z_perfcheck-background_s7` | 300-step timing checks, same configuration, foreground and background | 16 s versus 48 s of training. **Gate accuracy, per-map accuracy, loss and diagnostic identical between the two processes** |
| `20261007T040840Z_taskA-central-pilot-bal001_s7` | First pilot 2 attempt | Stopped by hand at step 3,750 when the laptop's power adapter was found dropping out under load; no end record. Superseded by `20261007T144407Z` |
| `20261007T144107Z_powertest_s7` | 1,500-step load test after the BIOS update to G614JI.334, with a watchdog ready to stop it on any adapter drop-out | No drop-out during the test or the pilot 2 run that followed (morning baseline: 2 to 6 per minute under the same load) |

## 2026-10-08 · Gate A, main arm: sliced training against centralized on Task A

**Purpose.** The Gate A criteria of SRS section 4.4 on the main arm (D18, D20). Criteria, metric definitions (D22) and the reading of a pass with redundant experts (D23) were all registered before the first of these runs. Four comparison arms (rolling assignment, router over all workers, partial update, full-model averaging) diagnose the finding; they are not criteria.

**Setup.**

- Same hardware, software and model as the centralized matrix: RTX 4070 Laptop GPU, PyTorch 2.12.1+cu126, Python 3.13.14, 3,421,824 parameters (276,096 shared, 3,145,728 in experts), dense dispatch, `warn` determinism, 1 CPU thread.
- Gate A setting (D18): 4 workers, each holding 4 of the 8 experts in every layer, coverage assignment, 125 local steps per round, 10 rounds. Total 5,000 worker steps of 64 sequences, equal to one centralized run (D17). Router rows averaged over holders (D11), plain averaging, fresh AdamW each round with the learning-rate schedule along each worker's trajectory (D19). No drops, no skew.
- Configuration `configs/task_a/sliced.toml`, commit `f8ac7ef`, clean working tree for every run. Queue `configs/queues/gate_a_main.toml`; after the travel stop, the rest ran from a local queue outside the repository (`gate_a_main_resume.toml`: `--resume` of run 1, then the same five entries).
- Command: `python -m scripts.train_task_a_sliced --config configs/task_a/sliced.toml --set run.seed=<seed> --set task_a.marked=<true|false> --set run.name=taskA-sliced-coverage-<marked|unmarked>`.
- Analysis: `python -m scripts.analyze_task_a` (commit of the analysis code in the report header). Full tables, per-layer numbers and run IDs: `results/task_a_gate_a.md`.

**Criteria verdict: every Gate A criterion is met on the main arm, with the D23 recorded finding.** The gate decision itself is the owner's, at PLAN step 9.

| Criterion | Seed 7 | Seed 11 | Seed 19 |
|---|---|---|---|
| Precondition: every map of centralized marked at least 0.999 | 1.00000 | 1.00000 | 1.00000 |
| S0-A-01: every map of sliced marked at least 0.99 | 1.00000 | 1.00000 | 1.00000 |
| S0-A-02: sliced unmarked within 2 points of centralized, same seed | −0.018 points | −0.009 points | −0.037 points |
| S0-A-03: no expert below 0.025 of tokens in any layer (least-used share, both variants) | 0.136 | 0.113 | 0.162 |
| S0-A-05: experts held per worker, of 8 | 4 | 4 | 4 |

S0-A-06 (every seed on its own) holds for each row. S0-A-04 is Task B's.

**The recorded finding: the merged experts are close to interchangeable.**

| Variant | Arm | Gate accuracy | Random routing (worst map) | One expert removed from every layer | Necessity | Necessity, sliced / centralized |
|---|---|---|---|---|---|---|
| marked | centralized | 1.00000 (all seeds) | 0.463 to 0.491 (0.323 to 0.404) | 0.819 to 0.999 | 0.509 to 0.537 | |
| marked | sliced | 1.00000 (all seeds) | 0.950 to 0.959 (0.903 to 0.954) | 1.0000 for every expert, every seed | 0.041 to 0.050 | 0.08 to 0.10 |
| unmarked | centralized | 0.9507 to 0.9511 | 0.168 to 0.283 (0.131 to 0.217) | 0.821 to 0.914 | 0.668 to 0.784 | |
| unmarked | sliced | 0.9506 to 0.9510 | 0.907 to 0.913 (0.886 to 0.902) | 0.9497 to 0.9520 | 0.039 to 0.044 | 0.06 |

Random routing, removal and necessity are on the curve set (256 sequences per map); gate accuracy on the gate set (2,048 per map).

**What this shows (this model size, budget and setting; three seeds):**

1. **Every Gate A criterion is met on every seed.** Sliced training matches centralized accuracy: every map perfect on the marked variant, and within 0.04 points of centralized (0.05 of the Bayes ceiling 0.95106) on the unmarked one. No expert is starved.
2. **The experts it produces are much more interchangeable than centralized ones, on every seed and both variants.** Routing tokens at random keeps 95% to 96% accuracy (marked) and 91% (unmarked, against 95% trained). Removing any single expert from every layer costs at most 0.07 points, against up to 18 points centralized. This is the pattern seen in the one-seed check run, now at the full budget on three seeds. By D23 it does not change the verdict; it is a limitation to diagnose, and Task B must show whether it costs capacity.
3. **The router still separates maps, at least as strongly as centralized.** Router consistency in layers 1 and 2 is 0.27 to 0.46 for sliced against 0.12 to 0.37 centralized (ceiling 2/3), and the two most-chosen experts take 68% to 84% of each map's traffic there. So routing is map-dependent, but the experts it chooses between can each do the job.
4. **The two holders of each expert route similar maps to it.** Expert drift in the final round is at most 0.14 per layer on average and 0.24 at worst, on a scale where 1 means disjoint map mixes. Drift compares how each holder's model routes the curve set after its local steps, a proxy for what it trained the expert on (step 8 doc). The comparison arms below show that drift does not track the redundancy.

**Comparison arms (D20, D23): where the redundancy comes from.**

Each arm changes one thing from the main arm, with the same Gate A setting and the same number of worker steps and sequences (D17), seeds 7, 11 and 19, both variants. They are 24 runs from `configs/queues/gate_a_comparison.toml` (step 7), run from a byte-identical copy outside the repository on commit `f8ac7ef` (clean), the same code as the main arm.

| Arm | What changes from the main arm | Necessity / centralized (marked; unmarked) | Worst single-expert removal, points (marked; unmarked) |
|---|---|---|---|
| Centralized | (reference) | 1; 1 | 8.62 to 18.10; 8.00 to 13.04 |
| Sliced, coverage (main) | | 0.08 to 0.10; 0.06 | 0.00; 0.03 to 0.07 |
| Sliced, rolling | assignment shifts by one expert each round | 0.15 to 0.16; 0.17 to 0.18 | 0.00 to 0.01; 0.02 to 0.05 |
| Sliced, router over all | router rows averaged over all workers | 0.05 to 0.07; 0.05 | 0.00; 0.00 to 0.04 |
| Partial update | every worker routes over all 8 experts; still updates only its 4 (router over all, D21) | 0.56 to 0.70; 0.61 to 0.67 | 1.17 to 11.08; 3.51 to 6.05 |
| Full-model averaging | every worker holds, routes over and updates all 8 (router over all) | 0.64 to 0.76; 0.72 to 0.80 | 2.82 to 9.63; 5.02 to 5.75 |

Necessity and removal on the curve set, ranges over seeds; full table with run IDs in `results/task_a_gate_a.md` ("Arms side by side").

The arms are equal in steps and sequences, not in memory or computation. A partial or full worker holds all 8 experts, twice the expert parameters of a sliced worker. Each token still uses 2 experts, but the dense dispatch used for Task A evaluates every available expert, so here these arms also did about twice the expert computation per step. Partial update keeps optimizer state for its 4 experts; full averaging for all 8.

**What the comparison shows (this setting; three seeds per cell):**

5. **Every arm meets the accuracy criteria.** Lowest marked map 0.99987 over all 36 runs; unmarked gate accuracy within 0.06 points of the Bayes ceiling in every arm; no negligible expert anywhere (least-used share 0.039 or more).
6. **The arms fall into two groups, split by whether a worker routes over all experts.** Where routing during local training is limited to the worker's slice (the three sliced arms), necessity is 5% to 18% of centralized and removing an expert costs at most 0.07 points. Where every worker routes over all eight experts (partial update and full-model averaging), it is 56% to 80% of centralized and removing an expert costs 1.2 to 11.1 points.
7. **The cleanest contrast is partial update against sliced with the router over all.** These two arms differ in training only in whether a worker may route tokens to experts it does not update: same assignment, same four experts updated per worker, same router rule. They are not equal in memory or computation (above). Allowing it raises necessity from 5% to 7% of centralized to 56% to 70%. So in this setting, most of the redundancy comes from restricting each worker's routing to its slice, not from which experts each worker updates or how the router is merged. In PLAN section 5's terms, the cause is masked routing rather than partial updates. This supports the candidate explanation from the check run (each worker's four experts must serve all eight maps). What each expert actually learns was not measured directly.
8. **Periodic averaging alone costs some specialization too.** Full-model averaging, with no slicing at all, keeps 64% to 80% of centralized necessity. Partial update is close behind it (56% to 70%).
9. **The assignment policy and the router rule matter little next to that.** Rolling raises necessity two to three times over coverage (15% to 18% against 6% to 10%), and averaging the router over all workers instead of holders changes it little (5% to 7%). Both stay far below the partial and full arms.
10. **Drift does not track the redundancy.** Final-round drift is near zero in the rolling, partial and full arms (layer means at most 0.035) and up to 0.14 in the two coverage arms, so redundant and non-redundant arms both have low drift.

**Bearing on the gate review (for the owner; no decision taken here).** The criteria are met, and the D23 finding now has a measured source in this setting: routing restricted to a slice during local training. Partial update recovers most of the specialization while each worker still updates only its 4 experts. But every worker must then hold and run all 8, the memory cost slicing exists to avoid. It is a measured diagnostic arm, not one of the remedies registered in PLAN section 5; those are for a routing failure, which did not occur. Task B is where any capacity cost of the redundancy must show (D23).

**Operational notes (do not affect the numbers).**

- Run 1 (`20261008T081034Z_taskA-sliced-coverage-marked_s7`) was stopped when the laptop had to travel, after round 5, and finished with `--resume` from its checkpoint on the same commit. Resume is bitwise identical in the step 6 check (`43f3958`); that was not re-checked for this run.
- Every run held the keep-awake request. The power summaries (one per run) show no mains drop-outs, standby entries or stalls in either the power monitor or the Windows event log, and no runtime warnings. Run 1's first segment (rounds 0 to 5), stopped before travel, wrote no power summary, so its power history is not recorded.
- Runs 2 to 6 took 338 s to 504 s each through the visible launcher; run 1's resumed segment (rounds 6 to 9) took 175 s.
- Comparison queue, 09:59 to 13:05 UTC: all 24 runs completed with keep-awake held, no standby entries, no stalls and no runtime warnings. Mains drop-outs returned. 16 runs logged short drop-outs, 99 Windows "AC offline" events in all, up to 19 in one run, each bridged by the battery. Mains was also off for 16 minutes (11:51:05 to 12:07:26), the whole of `20261008T115103Z_taskA-partial-unmarked_s11`, which ran on battery (99% to 72%) and slowed from 29 s to 196 s per round. Training is deterministic, so these changed only wall-clock time.

**Not tested:**

- What each expert learns (for example, per-expert accuracy on each map); the source of the redundancy is located by the arms above, not measured inside the experts.
- Whether it costs anything on Task B, other model sizes, budgets, worker counts or capacities.
- Faults and skew in a Gate A run.
- Reproduction on another machine.
