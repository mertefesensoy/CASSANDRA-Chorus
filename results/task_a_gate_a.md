# Task A · Gate A report

Generated 2026-10-08 10:01 UTC by `python -m scripts.analyze_task_a` (code `c2ed8ed`) from the run logs in `C:\Users\senso\chorus-runs`. Criteria and their reading were registered before any Gate A run (SRS section 4.4, D10, D22, D23).

## Verdict: PASS

| Criterion | Seed 7 | Seed 11 | Seed 19 | All seeds (S0-A-06) |
|---|---|---|---|---|
| precondition | pass: lowest map 1.00000 | pass: lowest map 1.00000 | pass: lowest map 1.00000 | pass |
| S0-A-01 | pass: lowest map 1.00000 | pass: lowest map 1.00000 | pass: lowest map 1.00000 | pass |
| S0-A-02 | pass: sliced minus centralized -0.018 points | pass: sliced minus centralized -0.009 points | pass: sliced minus centralized -0.037 points | pass |
| S0-A-03 | pass: no negligible expert | pass: no negligible expert | pass: no negligible expert | pass |
| S0-A-05 | pass: capacity 4 of 8 experts | pass: capacity 4 of 8 experts | pass: capacity 4 of 8 experts | pass |

S0-A-04 (Task B) is not part of Gate A. The criteria apply to the main arm; the other arms are for diagnosis.

## Routing necessity against centralized (SRS D23)

- marked: main-arm necessity divided by centralized necessity, by seed: 0.10, 0.09, 0.08.
- unmarked: main-arm necessity divided by centralized necessity, by seed: 0.06, 0.06, 0.06.

**Recorded finding (D23).** In 6 of 6 seed and variant pairs the main arm's routing necessity is below half of the centralized run's: its merged experts are much more interchangeable. By the reading registered before Gate A, this does not change the verdict above. It is reported as a limitation, its cause is to be diagnosed with the comparison arms and expert drift below, and Task B must show whether it costs capacity.

## Results by arm

### Centralized (reference)

| Variant | Seed | Gate accuracy | Lowest map | Bayes | Necessity | Consistency by layer | Negligible experts | Drift, final round (mean by layer) | Run ID | Config | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| marked | 7 | 1.00000 | 1.00000 | 1.00000 | 0.5091 | 0.035, 0.129, 0.302, 0.051 | 0 | n/a | `20261007T210724Z_taskA-central-marked_s7` | `d7dc789feb23` | `723f072` |
| marked | 11 | 1.00000 | 1.00000 | 1.00000 | 0.5143 | 0.052, 0.299, 0.218, 0.038 | 0 | n/a | `20261008T023619Z_taskA-central-marked_s11` | `54072d44f16f` | `723f072` |
| marked | 19 | 1.00000 | 1.00000 | 1.00000 | 0.5374 | 0.054, 0.346, 0.227, 0.064 | 0 | n/a | `20261008T052122Z_taskA-central-marked_s19` | `647f4386685c` | `723f072` |
| unmarked | 7 | 0.95074 | 0.94552 | 0.95106 | 0.6950 | 0.002, 0.269, 0.373, 0.025 | 0 | n/a | `20261008T062954Z_taskA-central-unmarked_s7` | `539abe295397` | `723f072` |
| unmarked | 11 | 0.95111 | 0.94503 | 0.95106 | 0.6676 | 0.001, 0.230, 0.116, 0.038 | 0 | n/a | `20261008T063833Z_taskA-central-unmarked_s11` | `1b15d30112a7` | `723f072` |
| unmarked | 19 | 0.95095 | 0.94813 | 0.95106 | 0.7839 | 0.002, 0.301, 0.359, 0.014 | 0 | n/a | `20261008T064621Z_taskA-central-unmarked_s19` | `bc4201220a7f` | `723f072` |

### Sliced, coverage (main arm)

| Variant | Seed | Gate accuracy | Lowest map | Bayes | Necessity | Consistency by layer | Negligible experts | Drift, final round (mean by layer) | Run ID | Config | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| marked | 7 | 1.00000 | 1.00000 | 1.00000 | 0.0498 | 0.116, 0.436, 0.321, 0.045 | 0 | 0.019, 0.095, 0.071, 0.012 | `20261008T081034Z_taskA-sliced-coverage-marked_s7` | `6dbd9c32e3e2` | `f8ac7ef` |
| marked | 11 | 1.00000 | 1.00000 | 1.00000 | 0.0465 | 0.130, 0.456, 0.366, 0.033 | 0 | 0.030, 0.121, 0.127, 0.007 | `20261008T092338Z_taskA-sliced-coverage-marked_s11` | `5bb908982158` | `f8ac7ef` |
| marked | 19 | 1.00000 | 1.00000 | 1.00000 | 0.0411 | 0.078, 0.435, 0.347, 0.041 | 0 | 0.020, 0.139, 0.070, 0.014 | `20261008T093202Z_taskA-sliced-coverage-marked_s19` | `7017a855a565` | `f8ac7ef` |
| unmarked | 7 | 0.95056 | 0.94531 | 0.95106 | 0.0435 | 0.007, 0.270, 0.390, 0.018 | 0 | 0.001, 0.036, 0.069, 0.007 | `20261008T093924Z_taskA-sliced-coverage-unmarked_s7` | `496e3abc8ca3` | `f8ac7ef` |
| unmarked | 11 | 0.95102 | 0.94788 | 0.95106 | 0.0386 | 0.018, 0.327, 0.419, 0.026 | 0 | 0.002, 0.070, 0.072, 0.006 | `20261008T094554Z_taskA-sliced-coverage-unmarked_s11` | `a9cb2b5553e0` | `f8ac7ef` |
| unmarked | 19 | 0.95058 | 0.94499 | 0.95106 | 0.0431 | 0.005, 0.333, 0.377, 0.026 | 0 | 0.001, 0.045, 0.106, 0.010 | `20261008T095215Z_taskA-sliced-coverage-unmarked_s19` | `3f1941ddc9c9` | `f8ac7ef` |

## Definitions (SRS D22)

- **Consistency:** mutual information between map and chosen expert over scored gate-set positions, divided by the map entropy; 0 = routing ignores the map; ceiling 2/3 here.
- **Necessity:** accuracy with the trained router minus accuracy with random routing, both on the curve set (the diagnostic does not re-score the gate set).
- **Negligible:** experts whose token share in a layer is below 0.1 · k/E = 0.025.
- **Drift:** per expert, mean Jensen-Shannon divergence (base 2) between the map mixes its holders sent it in the final round; mean over experts with two or more holders.
