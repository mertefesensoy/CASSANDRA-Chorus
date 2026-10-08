# CASSANDRA Chorus · Software Requirements Specification

| | |
|---|---|
| Status | Draft 0.5, not yet approved |
| Date | 2026-10-07 |
| Repository | https://github.com/mertefesensoy/CASSANDRA-Chorus |
| Owner | Mert Efe Şensoy |
| Scope of this version | Stage 0 in full; Stages 1 to 3 in outline |
| Changes since Draft 0.3 | Owner decisions of 2026-10-07 applied: see section 9 for the list, and sections 4.2 to 4.4 for the requirements they affect |
| Changes since Draft 0.4 | Owner decisions of 2026-10-08 after the prior-work survey (D11 to D16): router rows merged over their expert's holders (S0-F-08), rolling assignment (S0-F-06), a partial-update baseline (S0-F-26), an expert-drift diagnostic (S0-F-27), and section 10 rewritten |

## 1. Purpose

CASSANDRA Chorus is the next phase of the CASSANDRA program. Its goal is a language model trained by a pool of volunteer machines, where no machine holds the whole model during training, and where the full pipeline, logs and weights are open.

This document specifies what the software must do. Stage 0 is specified in detail because it decides whether the approach is viable. Later stages are outlined so that Stage 0 code is written with them in mind.

## 2. Background and decisions already taken

- The target model is a mixture-of-experts transformer: a shared part (embeddings, attention layers, normalization, router, output head) plus several expert blocks per mixture layer.
- Training uses **sliced local training**: each worker receives the shared part and a subset of the experts, trains locally for many steps without network traffic, and uploads its changes. A coordinator merges the results.
- First real workers will be trusted people. Defences against malicious workers are deferred to Stage 3.
- No token or cryptocurrency. Contribution is recorded in a plain credit ledger.
- Stage 0 runs on one machine that simulates several workers.

## 3. Definitions

| Term | Meaning |
|---|---|
| Shared part | All parameters that are not experts |
| Expert | One feed-forward block inside a mixture layer |
| Router | The component that chooses which experts process a token |
| Slice | The shared part plus a subset of experts, as given to one worker |
| Round | One cycle of: assign slices, train locally, upload, merge |
| Local steps (H) | Number of training steps a worker runs within one round |
| Coordinator | The process that holds the full model and performs merging |
| Centralized baseline | The same model trained normally on one machine |

## 4. Stage 0: single-machine simulation

### 4.1 Objective

Determine whether sliced local training of a mixture-of-experts model reaches quality close to the centralized baseline, and in particular whether a router trained with only some experts present still routes correctly once all experts are merged.

### 4.2 Functional requirements

**Model**

- **S0-F-01** The system shall implement a decoder-only transformer whose feed-forward layers are mixture-of-experts layers with a configurable number of experts (E) and a configurable number of experts selected per token (k).
- **S0-F-02** The router shall support an expert mask, so that routing can be restricted to a given subset of experts.
- **S0-F-03** Model size, depth, width, E and k shall be set by a configuration file.

**Sliced training**

- **S0-F-04** The system shall simulate N workers on one GPU by running them sequentially within a round.
- **S0-F-05** At the start of each round the coordinator shall assign each worker a slice. The number of experts per slice shall be configurable per worker, so that workers of different capacity can be simulated.
- **S0-F-06** The assignment policy shall be configurable. At minimum: random assignment, assignment that guarantees every expert is held by at least one worker per round, and rolling assignment, in which each worker's held experts shift in a fixed rotation from round to round so that every expert is trained evenly over rounds. (Rolling added in Draft 0.5, decision D12. Evidence: FedRolex, arXiv 2212.01548, found random sub-model extraction worse than rolling extraction for a transformer language model; that study sliced dense layers, not mixture-of-experts experts.)
- **S0-F-25** A slice shall list its experts separately for each mixture layer (a mapping from layer to expert indices), so that the merge does not depend on how the slice was chosen. Two cross-layer policies shall be supported: the same expert indices in every layer (the default) and an independent subset per layer. (Added in Draft 0.4; numbered after S0-F-24 so that existing references stay valid.)
- **S0-F-07** Each worker shall train its slice for H local steps with routing masked to the experts it holds.
- **S0-F-08** The coordinator shall merge by averaging the shared part across all workers that returned, and averaging each expert across the workers that held it. An expert held by no worker in a round shall be left unchanged.
  - **Router rows** (amended in Draft 0.5, decision D11). The router belongs to the shared part, but it has one row per expert, and a worker gives the rows of experts it does not hold no gradient. By default, each expert's router row shall therefore be averaged over the workers that held that expert, like the expert itself, and left unchanged if no worker held it.
  - Averaging the whole router over all returning workers (the rule of Draft 0.4) shall remain available as a configuration option and be run as a comparison on Task A.
  - Reason: with that rule, an expert held by n of N workers receives only n/N of its holders' mean router update.
- **S0-F-09** Averaging shall be weighted by the number of training examples each worker processed.
- **S0-F-10** The merge step shall accept an optional outer optimizer (for example momentum applied to the averaged update). The default shall be plain averaging.
- **S0-F-11** The system shall support fault injection: a configurable probability that a worker's result is dropped in a round.
- **S0-F-12** Data assignment to workers shall be configurable between uniform random and skewed (each worker sees a biased subset of the data).

**Baselines**

- **S0-F-13** The system shall train the same model architecture centrally, with total compute (steps × batch size) equal to the sliced run. Equal compute means: every worker step uses the same batch size as a centralized step, and the number of worker steps summed over all workers and rounds equals the number of centralized steps (N × H × rounds = steps; decision D17).
- **S0-F-14** The system shall train a full-model local-averaging baseline, in which every worker holds all experts. This separates the cost of local steps from the cost of slicing.
- **S0-F-26** The system shall train a partial-update local-averaging baseline, run on Task A next to the other arms.
  - Every worker holds the full model and routes over all experts, but updates only the experts assigned to it, keeping the others frozen locally.
  - Experts are merged over the workers they were assigned to. The router is averaged over all returning workers (decision D21, amending Draft 0.5): in this arm every worker trains every router row, so the reason for holder-only averaging in S0-F-08 does not apply.
  - Placed between S0-F-14 and sliced training, it separates the cost of masked routing from the cost of partial expert updates.
  - (Added in Draft 0.5, decision D13. Modelled on SPES, arXiv 2602.11543, which differs from Chorus in that its workers store the full model and route over all experts.)

**Task A: synthetic maps**

- **S0-F-15** The system shall generate M random lookup maps over a small symbol alphabet. Each training sequence is produced by exactly one map.
- **S0-F-16** Two variants shall be supported: marked (a symbol at the start identifies the map) and unmarked (the map must be inferred from context).
- **S0-F-17** Only positions whose next symbol is determined by the map shall be scored.
- **S0-F-18** The generator shall record which map produced each sequence, for router analysis.

Reference parameters (decided 2026-10-07): M = 8 maps over an alphabet of V = 26 symbols, sequences of length L = 64, and E = 8 experts per mixture layer in the Task A model. The marked variant adds one marker symbol per map to the vocabulary. The exact sequence format is specified with the generator (PLAN step 3).

**Task B: text8**

- **S0-F-19** The system shall train and evaluate on text8 at character level, with the conventional train, validation and test split.

**Measurement**

- **S0-F-20** The system shall report, per run: loss and accuracy (Task A) or bits per character (Task B) on held-out data, for the merged full model with all experts active.
- **S0-F-21** For Task A the system shall report router consistency: how consistently sequences from the same map are sent to the same experts.
  - **Definition** (decision D22, 2026-10-08): per mixture layer, the mutual information between the map and the chosen expert over the scored positions of the gate set, counting each of a token's k choices once, divided by the entropy of the map. 0 means routing ignores the map. With E = M = 8 and k = 2 the ceiling is 2/3.
  - The share of each map's choices going to its two most-used experts is reported beside it.
  - **Routing necessity** is also reported: trained accuracy minus accuracy under random routing.
- **S0-F-22** The system shall report expert usage balance, including the count of experts receiving negligible traffic.
- **S0-F-27** For Task A, the system shall report expert drift: per mixture layer and round, how differently the same expert index is used across the workers that held it, computed from each worker's map-by-expert routing table. A diagnostic used to explain a failure at Gate A, not a pass or fail criterion. (Added in Draft 0.5, decision D16. The failure mode was named "expert semantic blurring" in FedAlign-MoE, arXiv 2603.21276, which the survey read only as an abstract.)
- **S0-F-23** The system shall support sweeps over H, slice size, N and dropout probability, and produce one results table per sweep.
- **S0-F-24** Every run shall write its configuration, random seeds, software versions (Python, PyTorch, CUDA), GPU name, per-round metrics and final metrics to a structured log file.

### 4.3 Non-functional requirements

- **S0-N-01** All Stage 0 experiments shall run on a single GPU with 8 GB of memory (reference machine: laptop NVIDIA RTX 4070).
- **S0-N-02** Task A models shall stay under 5M parameters. Task B models shall be in the range 20M to 50M parameters.
- **S0-N-03** Runs shall be reproducible from a configuration file and seed. Where GPU non-determinism prevents bit-exact reproduction, the documentation shall say so. Multi-seed experiments use seeds 7, 11 and 19, the convention inherited from the parent CASSANDRA repository.
- **S0-N-04** A run shall be resumable from its last completed round.
- **S0-N-05** The coordinator logic (slice assignment, merge) shall be separated from the simulation harness, so that Stage 1 can reuse it with real workers.
- **S0-N-06** Slice assignment and merge shall have unit tests, including the cases: expert held by no worker, a worker dropped, unequal slice sizes.
- **S0-N-07** Reported results shall state their limits: hardware, model size, dataset, number of seeds, and what was not tested.
- **S0-N-08** Raw run logs and checkpoints shall stay local and out of version control. Results tables and a hand-written `RESULTS.md`, recording every run that informs a decision (failed runs included), shall be committed as the durable record.

### 4.4 Acceptance criteria (confirmed by the owner on 2026-10-07)

| ID | Criterion |
|---|---|
| S0-A-01 | Task A, marked variant: merged model reaches at least 99% scored accuracy on every map individually. Precondition: the centralized baseline reaches 99.9% or above on every map (see below) |
| S0-A-02 | Task A, unmarked variant: merged model accuracy within 2 percentage points of the centralized baseline |
| S0-A-03 | Task A: no expert receives negligible traffic in the merged model, in any mixture layer (defined below) |
| S0-A-04 | Task B: merged model bits per character within 5% (relative) of the centralized baseline at equal total compute |
| S0-A-05 | Criteria S0-A-01 to S0-A-04 hold with workers holding at most half of the experts each |
| S0-A-06 | Results hold across at least 3 seeds (7, 11 and 19 by default), in the sense defined below |

Definitions (added in Draft 0.4 and confirmed with the thresholds):

- **Precondition of S0-A-01.** The centralized baseline reaching 99.9% checks that the task and model size are learnable. It does not test the method. If the baseline falls short, S0-A-01 is not evaluated until the task or model is corrected.
- **Negligible traffic.** For each mixture layer separately, an expert's traffic share is the fraction of held-out tokens routed to it, where each token counts once for each of the k experts it is sent to. Under perfectly uniform routing every expert's share is k/E. An expert receives negligible traffic if its share is below 10% of that uniform share, that is below 0.1 · k/E.
- **Seeds.** A criterion holds across seeds only if every seed meets it on its own. A pass by the mean over seeds alone does not count.
- **Router consistency** (S0-F-21) is reported for every Task A run as a diagnostic, used at Gate A to explain a failure. It is not a pass or fail criterion, because the consistency a centralized model reaches is not yet known.
- **Per map** (decision D10, 2026-10-07). S0-A-01 and its precondition apply to each map's accuracy separately, not to accuracy pooled over maps, because a routing failure can break one map while the others pass. Accuracies used for these criteria are measured on a fixed held-out set of 2,048 sequences per map (about 38,000 scored positions per map), so that 99.9% is not decided by a handful of errors.

If Task A fails, the method is considered broken in its current form and Task B is not run until the cause is understood. If Task A passes and Task B fails by a moderate margin, the result is still reported, with the gap quantified.

## 5. Stage 1: trusted pool (outline)

- **S1-01** A coordinator service that holds the full model, assigns slices and data shards, receives updates and merges them.
- **S1-02** A client that reports its GPU capacity, receives a slice sized to it, trains, checkpoints locally and uploads its update.
- **S1-03** The client shall tolerate being stopped at any time without corrupting the round.
- **S1-04** Updates shall be compressed for upload over home connections.
- **S1-05** Accounts for known participants, and a credit ledger recording accepted work.
- **S1-06** Public per-round logs and checkpoints.
- **S1-07** The framework choice (Flower, Hivemind or custom networking) is made at the start of this stage.

## 6. Stage 2: flagship run (outline)

- **S2-01** A model too large for any single participant to train, sized so that the compressed result still runs on a consumer GPU.
- **S2-02** A tokenizer and a text corpus large enough for that model. Both are open decisions.
- **S2-03** A centralized reference point for comparison, at whatever smaller scale is affordable.

## 7. Stage 3: public pool (outline)

- **S3-01** Open sign-up.
- **S3-02** A written threat model stating what a malicious worker can do and what is defended against.
- **S3-03** Verification of submitted work by sampled re-execution and by scoring updates on held-out data.
- **S3-04** Merge rules that resist poisoned updates.
- **S3-05** A public leaderboard.

## 8. Out of scope for Stage 0

Networking, real client software, accounts, the credit ledger, update compression, verification, any defence against malicious workers, tokenizers and large corpora.

## 9. Decisions

Decided by the owner on 2026-10-07:

| # | Decision | Outcome |
|---|---|---|
| D1 | Repository and package name | `CASSANDRA-Chorus`, at https://github.com/mertefesensoy/CASSANDRA-Chorus. The Python package is `cassandra_chorus` |
| D2 | Licence | Apache-2.0, the same as the parent CASSANDRA repository |
| D3 | Language and framework for Stage 0 | Python with PyTorch, using the environment already installed on the reference laptop (Python 3.13.14, PyTorch 2.12.1 built for CUDA 12.6). Minimum versions are declared in the package metadata, and exact versions are written to every run log (S0-F-24) |
| D4 | Acceptance thresholds | Confirmed as proposed, with the definitions added in section 4.4 |
| D5 | Task A parameters | M = 8 maps, V = 26 symbols, L = 64, E = 8 experts (section 4.2) |
| D6 | Expert indices across layers | Both policies supported; the same indices in every layer is the default, and an independent subset per layer is run as an ablation (S0-F-25) |
| D7 | Prior-work survey timing | Alongside PLAN steps 1 to 4, reviewed by the owner before step 5 |
| D8 | Conventions inherited from CASSANDRA | Seeds 7, 11 and 19 (S0-N-03); no em or en dashes in project documents; raw run logs local and `RESULTS.md` committed (S0-N-08) |
| D9 | Version control | Default branch `main`. Each PLAN step is developed on its own branch and merged by pull request after owner review |

Decided by the owner later on 2026-10-07:

| # | Decision | Outcome |
|---|---|---|
| D10 | Open item O1: "across all maps" in S0-A-01 | Every map individually, for both the criterion and the centralized precondition, measured on 2,048 held-out sequences per map (section 4.4) |

Decided by the owner on 2026-10-08, after a walk-through of the prior-work survey (`docs/prior-work.md`):

| # | Decision | Outcome |
|---|---|---|
| D11 | Router merge | Router rows averaged over their expert's holders by default; averaging over all workers kept as an option and run as a comparison (S0-F-08) |
| D12 | Assignment policies | Rolling assignment added (S0-F-06) |
| D13 | Partial-update baseline | Added as a Task A arm (S0-F-26) |
| D14 | Prior work and framing | Section 10 rewritten. Chorus is described as a variant within existing work on partial-expert local training, never as novel |
| D15 | Gate A remedies | A short central router fit after merging, a frozen shared anchor on workers, and z-loss added to the list in PLAN section 5 |
| D16 | Expert drift | Reported as a Task A diagnostic (S0-F-27) |
| D17 | Equal compute | Same batch per worker step; N × H × rounds = centralized steps (S0-F-13) |
| D18 | Gate A setting | N = 4 workers, each holding 4 of 8 experts per layer, H = 125 local steps, 10 rounds (PLAN section 5) |
| D19 | Worker optimizer state | Fresh each round: workers keep no state between rounds |
| D20 | Gate A assignment | Coverage as the main run, rolling as a second arm |
| D21 | Partial-update router | Averaged over all workers (S0-F-26) |
| D22 | Metric definitions | Router consistency as normalized mutual information, with top-2 share and routing necessity beside it (S0-F-21). Expert drift as the Jensen-Shannon divergence, base 2, between the map mixes that different holders sent to the same expert in a round (S0-F-27). Negligible traffic over all positions, as logged (S0-A-03) |
| D23 | Gate A reading, registered before any Gate A run | If sliced runs pass S0-A-01 to S0-A-03 but their experts are far more interchangeable than centralized ones (much lower routing necessity), Gate A passes **with a recorded finding**. The redundancy is reported prominently as a limitation, and its cause is diagnosed with the full-model and partial-update arms and the drift metric. Task B's bits-per-character gap is where any cost in lost capacity must show (PLAN section 5) |
| D24 | Gate A decision (owner, 2026-10-08, after reading the 36-run Gate A record) | Gate A passes with the D23 recorded finding. Before Task B, a redundancy probe (PLAN step 9a) measures from the saved Gate A models whether each expert learned every map or whether the experts are bypassed |

Name availability, checked on 2026-10-07 by read-only lookups: PyPI has no project named `cassandra-chorus`, and Hugging Face has no user or organisation named `cassandra-chorus` and no model repository `mertefesensoy/cassandra-chorus` (or that repository is private). Neither check reserves the name.

Still open:

| # | Decision | Needed by |
|---|---|---|
| O2 | Flower, Hivemind or custom networking | Start of Stage 1 |
| O3 | Flagship model size, tokenizer and corpus | Start of Stage 2 |

## 10. Prior work and how Chorus is described

Rewritten in Draft 0.5 (decision D14) after the prior-work survey of 2026-10-07 (`docs/prior-work.md`).

**Provenance of the survey.** It was a bounded web survey, written by a research subagent. The key abstracts and the SPES repository were checked against raw arXiv and GitHub data. Claims drawn from inside the papers were read through a summarizing tool and were not re-checked; they must be verified against the papers before any external use.

The two assumptions of Draft 0.4 are **contradicted as worded**:

- **"No complete open-source implementation of sliced local training for mixture-of-experts language models exists."** SPES (arXiv 2602.11543, open code under Apache-2.0) pretrains mixture-of-experts language models of 2B and 7B parameters. Each node trains its own subset of experts over local steps between synchronizations.
- **"Published sub-network training results are limited to small vision and feed-forward models."** Counterexamples include:
  - HeteroFL: a transformer on WikiText2;
  - FedRolex: a transformer on Stack Overflow;
  - TwIST: GPT-2 at 124M parameters;
  - SDP: a dense LLaMA-style model up to 1B parameters, synchronized every step;
  - SPES and MoE-DisCo: mixture-of-experts models at billion scale;
  - FedMoE, which already uses the S0-F-08 merge rule, for fine-tuning.

**What the bounded search did not find.** Open code combining all of the following:

- workers that store only the shared part and their own experts;
- routing masked to those experts during local training;
- experts held by several workers averaged only among them;
- slice sizes that differ per worker;
- volunteer hardware as the target.

That is not proof that no such code exists. The search did no code search on GitHub, no citation tracing, and covered no non-English venues. Papers citing SPES are the most likely place for a closer match, and must be checked before any public claim.

**How Chorus is described.** As a variant within existing work on partial-expert local training of mixture-of-experts language models, stating the specific differences above. Never as "first" or "novel".

**Networking frameworks (for O2).** Flower was actively developed (release 1.39.0, 2026-09-28). Hivemind was in low-activity maintenance (1.1.12, 2026-01-03). Petals had no push since 2024-09-07. These figures date from 2026-10-07 and should be re-checked at the start of Stage 1.
