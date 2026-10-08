# CASSANDRA Chorus · Initial Plan Memo

| | |
|---|---|
| Status | Draft 0.4. The owner's decisions of 2026-10-07 are applied (section 10). Everything else is a proposal until the owner confirms it. |
| Date | 2026-10-07 |
| Repository | https://github.com/mertefesensoy/CASSANDRA-Chorus |
| Owner | Mert Efe Şensoy |
| Companion document | `SRS.md` (requirement IDs below refer to it) |

## 1. Summary

The next phase of CASSANDRA aims at a language model trained by a pool of volunteer machines, with the pipeline, logs and weights all open. The model is a mixture-of-experts transformer. Each worker trains the shared part plus a subset of the experts locally and syncs occasionally, so no worker ever holds the whole model and slow or unreliable machines can still take part.

Whether this training method works is not known. The plan therefore starts with a cheap single-machine experiment (Stage 0) that can fail fast, and builds no pool infrastructure until it passes.

## 2. The question Stage 0 answers

> If a router is trained while only some experts are present on each worker, does the merged model, with all experts present, still route and predict correctly?

A second question follows from it: how much quality is lost against normal training at equal total compute, and how does that loss change as workers get weaker (smaller slices, longer gaps between syncs, more dropouts)?

## 3. Stages

| Stage | What it is | Exit condition |
|---|---|---|
| 0 | Single-machine simulation of sliced training | Acceptance criteria in SRS section 4.4 met, or a documented negative result |
| 1 | Trusted pool: real machines of known people, small model | A full training run completed over the network, matching Stage 0 results |
| 2 | Flagship: a model too large for any one member to train | Model released with full public logs |
| 3 | Public pool: open sign-up with verification | Threat model published and defences tested |

## 4. Stage 0 work breakdown

Steps are listed in dependency order. No dates are given: the pace depends on the owner's availability, which this memo does not assume.

| # | Step | Output | SRS reference |
|---|---|---|---|
| 1 | Repository skeleton: layout, licence, configuration format, logging, test setup | Empty but runnable repo | S0-N-03, S0-N-05, S0-N-08, S0-F-24 |
| 2 | Mixture-of-experts transformer with maskable router | Model code and unit tests | S0-F-01 to S0-F-03 |
| 3 | Task A generator (synthetic maps, marked and unmarked) | Dataset code with map labels | S0-F-15 to S0-F-18 |
| 4 | Centralized baseline on Task A | First reference numbers | S0-F-13 |
| 5 | Coordinator logic: slice assignment and merge, with tests | Reusable module | S0-F-05 to S0-F-10, S0-F-25, S0-N-06 |
| 6 | Simulation harness: N sequential workers, fault injection, resumable rounds | Sliced training runs | S0-F-04, S0-F-11, S0-F-12, S0-N-04 |
| 7 | Full-model local-averaging baseline | Separates cost of local steps from cost of slicing | S0-F-14 |
| 8 | Metrics: accuracy, router consistency, expert usage | Per-run report | S0-F-20 to S0-F-24 |
| 9 | **Decision gate A**: Task A results against criteria | Go, fix, or stop | S0-A-01 to S0-A-03, under the conditions of S0-A-05 and S0-A-06 |
| 10 | Task B on text8: centralized, local-averaging and sliced runs | Bits-per-character comparison | S0-F-19, S0-A-04 |
| 11 | Sweeps over local steps, slice size, worker count, dropout | Results tables | S0-F-23 |
| 12 | **Decision gate B**: write-up of Stage 0 | Short technical report | S0-N-07 |

Steps 2 and 3 are independent and can be done in either order. Step 5 is the piece that carries over to Stage 1, so it should be kept free of simulation-specific code.

How each step is carried out (decided 2026-10-07):

- Each step is developed on its own branch (`stage0/NN-slug`), described in a document under `docs/implementations/`, and merged into `main` by pull request after the owner has reviewed it. Nothing is pushed without the owner's agreement.
- Design choices that neither this memo nor the SRS settles are put to the owner before the step is implemented.
- The prior-work survey (section 9) runs alongside steps 1 to 4 and is reviewed by the owner before step 5 starts.

## 5. Decision gates

**Gate A (after Task A)**

- Passes: continue to Task B.
- Fails on routing (router inconsistent or experts unused after merging): try the known remedies before giving up, one at a time and each recorded: guaranteed expert coverage per round, a load-balancing loss, averaging the router more often than the experts, freezing the router after a centrally trained warm-up.
- Still fails: the method is not viable in this form. Fall back to modular paths (each worker trains one complete route through a grid of modules), which reuses the coordinator, harness and metrics.

**Gate B (after Task B)**

- Gap within the criterion: proceed to Stage 1.
- Moderate gap: report it, and decide whether the flagship is still worth building at that cost.
- Large gap: treat as a negative result, publish it, and reconsider the architecture.

A negative result is still a deliverable. It should be written up with the same care as a positive one.

## 6. Proposed repository layout

```
/
├── README.md
├── LICENSE
├── RESULTS.md          hand-written record of every run that informs a decision
├── docs/
│   ├── SRS.md
│   ├── PLAN.md
│   ├── prior-work.md   prior-work survey (section 9)
│   └── implementations/  one document per change
├── configs/            experiment configuration files
├── cassandra_chorus/
│   ├── model/          mixture-of-experts transformer, router
│   ├── data/           synthetic maps, text8
│   ├── coordinator/    slice assignment, merge (reused in Stage 1)
│   ├── sim/            single-machine worker simulation
│   └── metrics/        accuracy, router consistency, expert usage
├── scripts/            run and sweep entry points
├── tests/
└── results/            results tables (committed)
```

The Python package is named `cassandra_chorus` to match the repository. Raw run logs and checkpoints are kept out of version control (SRS S0-N-08); where they are written is settled in step 1.

## 7. Resources

- **Hardware:** one laptop with an NVIDIA RTX 4070 (8 GB). Task A runs should take minutes; Task B runs are expected to take hours each. These are estimates and have not been measured.
- **Software:** Python with PyTorch, using the environment already installed on the laptop: Python 3.13.14 and PyTorch 2.12.1 built for CUDA 12.6 (decided 2026-10-07).
- **People:** the owner. Stage 0 needs no other participants.

## 8. Risks

| Risk | Likelihood | Response |
|---|---|---|
| Router does not survive sliced training | Unknown; this is the central risk | Gate A remedies, then the modular-paths fallback |
| Task A passes but does not predict behaviour on text | Moderate | Task B exists for this reason; no claims are made from Task A alone |
| Sequential simulation hides problems that real networks expose | Moderate | Fault injection and skewed data in Stage 0; Stage 1 tests the rest |
| Sweeps take longer than the laptop can reasonably run | Moderate | Reduce the grid; run the most informative settings first |
| The idea already exists in published or open-source form | Unknown | Prior-work survey before any novelty claim (see section 9) |

## 9. Prior-work survey

Before the Stage 0 write-up, and ideally before step 5, check the current state of: sub-network and federated-dropout training applied to transformers, DiPaCo and any open implementations, federated or decentralized mixture-of-experts training, and the maintenance status of Flower and Hivemind. The survey may change the design or show that part of the work can be reused.

Timing (decided 2026-10-07): the survey runs alongside steps 1 to 4, is written to `docs/prior-work.md` with a source for every claim, and is reviewed by the owner before step 5 starts. Until then the assumptions in SRS section 10 remain unverified, and no novelty claim is made.

## 10. Owner decisions

Decided on 2026-10-07 (the full list, with details, is in SRS section 9):

1. Licence: Apache-2.0.
2. Stage 0 uses Python with PyTorch, in the environment already installed on the laptop.
3. The acceptance thresholds in SRS section 4.4 are confirmed, with added definitions of negligible traffic, of passing across seeds, and of router consistency as a diagnostic.
4. Task A parameters: 8 maps, 26 symbols, sequences of 64, and 8 experts per mixture layer.
5. The prior-work survey runs in parallel with steps 1 to 4 and is reviewed before step 5.
6. Both cross-layer slice policies are supported, with the same expert indices in every layer as the default.
7. Conventions inherited from CASSANDRA: seeds 7, 11 and 19; no em or en dashes in documents; raw run logs local and `RESULTS.md` committed.
8. Version control: `main` plus one branch and pull request per step.

Also decided earlier: the repository, `CASSANDRA-Chorus`, at https://github.com/mertefesensoy/CASSANDRA-Chorus. On 2026-10-07 no PyPI project and no Hugging Face namespace named `cassandra-chorus` existed. That does not reserve the name.

Still needed before Task A runs: whether "across all maps" in SRS S0-A-01 means every map individually or accuracy pooled over maps.
