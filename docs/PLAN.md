# CASSANDRA Chorus — Initial Plan Memo

| | |
|---|---|
| Status | Draft 0.3, proposal only. Nothing here is committed until the owner confirms it. |
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
| 1 | Repository skeleton: layout, licence, configuration format, logging, test setup | Empty but runnable repo | S0-N-03, S0-N-05 |
| 2 | Mixture-of-experts transformer with maskable router | Model code and unit tests | S0-F-01 to S0-F-03 |
| 3 | Task A generator (synthetic maps, marked and unmarked) | Dataset code with map labels | S0-F-15 to S0-F-18 |
| 4 | Centralized baseline on Task A | First reference numbers | S0-F-13 |
| 5 | Coordinator logic: slice assignment and merge, with tests | Reusable module | S0-F-05 to S0-F-10, S0-N-06 |
| 6 | Simulation harness: N sequential workers, fault injection, resumable rounds | Sliced training runs | S0-F-04, S0-F-11, S0-F-12, S0-N-04 |
| 7 | Full-model local-averaging baseline | Separates cost of local steps from cost of slicing | S0-F-14 |
| 8 | Metrics: accuracy, router consistency, expert usage | Per-run report | S0-F-20 to S0-F-24 |
| 9 | **Decision gate A**: Task A results against criteria | Go, fix, or stop | S0-A-01 to S0-A-03 |
| 10 | Task B on text8: centralized, local-averaging and sliced runs | Bits-per-character comparison | S0-F-19, S0-A-04 |
| 11 | Sweeps over local steps, slice size, worker count, dropout | Results tables | S0-F-23 |
| 12 | **Decision gate B**: write-up of Stage 0 | Short technical report | S0-N-07 |

Steps 2 and 3 are independent and can be done in either order. Step 5 is the piece that carries over to Stage 1, so it should be kept free of simulation-specific code.

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
├── docs/
│   ├── SRS.md
│   └── PLAN.md
├── configs/            experiment configuration files
├── cassandra_chorus/
│   ├── model/          mixture-of-experts transformer, router
│   ├── data/           synthetic maps, text8
│   ├── coordinator/    slice assignment, merge (reused in Stage 1)
│   ├── sim/            single-machine worker simulation
│   └── metrics/        accuracy, router consistency, expert usage
├── scripts/            run and sweep entry points
├── tests/
└── results/            run logs and results tables
```

The Python package is named `cassandra_chorus` to match the repository.

## 7. Resources

- **Hardware:** one laptop with an NVIDIA RTX 4070 (8 GB). Task A runs should take minutes; Task B runs are expected to take hours each. These are estimates and have not been measured.
- **Software:** Python with PyTorch is assumed, pending confirmation.
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

## 10. What is needed from the owner

1. Licence (Apache-2.0 suggested).
2. Confirmation of Python with PyTorch for Stage 0.
3. Confirmation or revision of the acceptance thresholds in SRS section 4.4.
4. Task A parameters: number of maps, alphabet size, sequence length. Suggested starting point: 8 maps, 26 symbols, sequences of 64.
5. Whether the prior-work survey should be done before coding starts or in parallel with steps 1 to 4.

Already decided: the repository, `CASSANDRA-Chorus`, at https://github.com/mertefesensoy/CASSANDRA-Chorus. Availability of the `cassandra-chorus` name on PyPI and Hugging Face still needs checking before any package or model is published under it.
