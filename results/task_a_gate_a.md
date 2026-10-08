# Task A · Gate A report

Generated 2026-10-08 13:06 UTC by `python -m scripts.analyze_task_a` (code `e44d0a3`) from the run logs in `C:\Users\senso\chorus-runs`. Criteria and their reading were registered before any Gate A run (SRS section 4.4, D10, D22, D23).

## Criteria verdict: PASS

The gate decision itself is the owner's (PLAN step 9).

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

### Sliced, rolling assignment

| Variant | Seed | Gate accuracy | Lowest map | Bayes | Necessity | Consistency by layer | Negligible experts | Drift, final round (mean by layer) | Run ID | Config | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| marked | 7 | 1.00000 | 1.00000 | 1.00000 | 0.0837 | 0.062, 0.393, 0.389, 0.026 | 0 | 0.001, 0.003, 0.004, 0.000 | `20261008T095932Z_taskA-sliced-rolling-marked_s7` | `07e57d1cd544` | `f8ac7ef` |
| marked | 11 | 0.99998 | 0.99987 | 1.00000 | 0.0766 | 0.108, 0.517, 0.199, 0.054 | 0 | 0.002, 0.004, 0.008, 0.006 | `20261008T100404Z_taskA-sliced-rolling-marked_s11` | `8202ad30d266` | `f8ac7ef` |
| marked | 19 | 1.00000 | 1.00000 | 1.00000 | 0.0866 | 0.095, 0.443, 0.386, 0.033 | 0 | 0.000, 0.002, 0.002, 0.000 | `20261008T101053Z_taskA-sliced-rolling-marked_s19` | `8277daa3c453` | `f8ac7ef` |
| unmarked | 7 | 0.95091 | 0.94531 | 0.95106 | 0.1201 | 0.003, 0.404, 0.434, 0.016 | 0 | 0.000, 0.007, 0.005, 0.004 | `20261008T101805Z_taskA-sliced-rolling-unmarked_s7` | `e344ac5b22f5` | `f8ac7ef` |
| unmarked | 11 | 0.95103 | 0.94853 | 0.95106 | 0.1172 | 0.006, 0.376, 0.436, 0.010 | 0 | 0.000, 0.003, 0.003, 0.005 | `20261008T102547Z_taskA-sliced-rolling-unmarked_s11` | `d21c9a68ac95` | `f8ac7ef` |
| unmarked | 19 | 0.95061 | 0.94718 | 0.95106 | 0.1314 | 0.003, 0.423, 0.357, 0.024 | 0 | 0.000, 0.008, 0.004, 0.004 | `20261008T103333Z_taskA-sliced-rolling-unmarked_s19` | `bf8a7864f79d` | `f8ac7ef` |

### Sliced, router averaged over all workers

| Variant | Seed | Gate accuracy | Lowest map | Bayes | Necessity | Consistency by layer | Negligible experts | Drift, final round (mean by layer) | Run ID | Config | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| marked | 7 | 1.00000 | 1.00000 | 1.00000 | 0.0326 | 0.124, 0.379, 0.359, 0.044 | 0 | 0.013, 0.089, 0.100, 0.010 | `20261008T104031Z_taskA-sliced-routerall-marked_s7` | `01dcb8614522` | `f8ac7ef` |
| marked | 11 | 1.00000 | 1.00000 | 1.00000 | 0.0380 | 0.106, 0.489, 0.347, 0.042 | 0 | 0.020, 0.118, 0.120, 0.022 | `20261008T104642Z_taskA-sliced-routerall-marked_s11` | `016991385dec` | `f8ac7ef` |
| marked | 19 | 1.00000 | 1.00000 | 1.00000 | 0.0284 | 0.120, 0.421, 0.295, 0.042 | 0 | 0.018, 0.125, 0.055, 0.012 | `20261008T105231Z_taskA-sliced-routerall-marked_s19` | `06042db9d3ed` | `f8ac7ef` |
| unmarked | 7 | 0.95070 | 0.94437 | 0.95106 | 0.0372 | 0.007, 0.286, 0.409, 0.013 | 0 | 0.002, 0.045, 0.072, 0.007 | `20261008T105919Z_taskA-sliced-routerall-unmarked_s7` | `d89ca23ac195` | `f8ac7ef` |
| unmarked | 11 | 0.95054 | 0.94408 | 0.95106 | 0.0328 | 0.019, 0.329, 0.360, 0.024 | 0 | 0.002, 0.068, 0.067, 0.008 | `20261008T110611Z_taskA-sliced-routerall-unmarked_s11` | `4172251c5de1` | `f8ac7ef` |
| unmarked | 19 | 0.95073 | 0.94515 | 0.95106 | 0.0401 | 0.007, 0.325, 0.382, 0.021 | 0 | 0.001, 0.058, 0.098, 0.009 | `20261008T111218Z_taskA-sliced-routerall-unmarked_s19` | `192562430496` | `f8ac7ef` |

### Partial update (S0-F-26)

| Variant | Seed | Gate accuracy | Lowest map | Bayes | Necessity | Consistency by layer | Negligible experts | Drift, final round (mean by layer) | Run ID | Config | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| marked | 7 | 1.00000 | 1.00000 | 1.00000 | 0.2872 | 0.034, 0.273, 0.507, 0.046 | 0 | 0.000, 0.000, 0.000, 0.000 | `20261008T111836Z_taskA-partial-marked_s7` | `249c67d21631` | `f8ac7ef` |
| marked | 11 | 1.00000 | 1.00000 | 1.00000 | 0.3239 | 0.098, 0.394, 0.407, 0.033 | 0 | 0.000, 0.000, 0.000, 0.000 | `20261008T112621Z_taskA-partial-marked_s11` | `0e0a212366ba` | `f8ac7ef` |
| marked | 19 | 1.00000 | 1.00000 | 1.00000 | 0.3786 | 0.063, 0.403, 0.356, 0.041 | 0 | 0.000, 0.000, 0.000, 0.000 | `20261008T113449Z_taskA-partial-marked_s19` | `f4b41d3c9ecd` | `f8ac7ef` |
| unmarked | 7 | 0.95053 | 0.94793 | 0.95106 | 0.4484 | 0.005, 0.237, 0.251, 0.031 | 0 | 0.000, 0.008, 0.035, 0.010 | `20261008T114234Z_taskA-partial-unmarked_s7` | `c23e81025f62` | `f8ac7ef` |
| unmarked | 11 | 0.95116 | 0.94561 | 0.95106 | 0.4058 | 0.009, 0.289, 0.248, 0.071 | 0 | 0.001, 0.008, 0.013, 0.012 | `20261008T115103Z_taskA-partial-unmarked_s11` | `7c2632acee0a` | `f8ac7ef` |
| unmarked | 19 | 0.95055 | 0.94857 | 0.95106 | 0.5249 | 0.001, 0.292, 0.258, 0.033 | 0 | 0.000, 0.009, 0.004, 0.006 | `20261008T120737Z_taskA-partial-unmarked_s19` | `49ea56818f49` | `f8ac7ef` |

### Full-model local averaging (S0-F-14)

| Variant | Seed | Gate accuracy | Lowest map | Bayes | Necessity | Consistency by layer | Negligible experts | Drift, final round (mean by layer) | Run ID | Config | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| marked | 7 | 1.00000 | 1.00000 | 1.00000 | 0.3488 | 0.017, 0.181, 0.308, 0.033 | 0 | 0.000, 0.003, 0.007, 0.002 | `20261008T121708Z_taskA-fullavg-marked_s7` | `8d4ed606bfb8` | `f8ac7ef` |
| marked | 11 | 1.00000 | 1.00000 | 1.00000 | 0.3299 | 0.024, 0.296, 0.178, 0.028 | 0 | 0.000, 0.000, 0.000, 0.000 | `20261008T122608Z_taskA-fullavg-marked_s11` | `59bc764798db` | `f8ac7ef` |
| marked | 19 | 1.00000 | 1.00000 | 1.00000 | 0.4092 | 0.007, 0.319, 0.149, 0.021 | 0 | 0.000, 0.001, 0.000, 0.000 | `20261008T123449Z_taskA-fullavg-marked_s19` | `a70f9453dfb4` | `f8ac7ef` |
| unmarked | 7 | 0.95066 | 0.94547 | 0.95106 | 0.5538 | 0.003, 0.239, 0.286, 0.024 | 0 | 0.001, 0.005, 0.007, 0.006 | `20261008T124312Z_taskA-fullavg-unmarked_s7` | `c4561d9562f0` | `f8ac7ef` |
| unmarked | 11 | 0.95113 | 0.94767 | 0.95106 | 0.4993 | 0.005, 0.243, 0.122, 0.044 | 0 | 0.001, 0.006, 0.014, 0.006 | `20261008T125053Z_taskA-fullavg-unmarked_s11` | `2325a4192f41` | `f8ac7ef` |
| unmarked | 19 | 0.95066 | 0.94658 | 0.95106 | 0.5676 | 0.002, 0.320, 0.297, 0.052 | 0 | 0.001, 0.007, 0.007, 0.006 | `20261008T125804Z_taskA-fullavg-unmarked_s19` | `8ab9e0f26cd7` | `f8ac7ef` |

## Definitions (SRS D22)

- **Consistency:** mutual information between map and chosen expert over scored gate-set positions, divided by the map entropy; 0 = routing ignores the map; ceiling 2/3 here.
- **Necessity:** accuracy with the trained router minus accuracy with random routing, both on the curve set (the diagnostic does not re-score the gate set).
- **Negligible:** experts whose token share in a layer is below 0.1 · k/E = 0.025.
- **Drift:** per expert, mean Jensen-Shannon divergence (base 2) between its holders' map mixes in the final round, where holders are the workers whose copy is merged (all workers in full-model averaging) and a map mix is how the worker's model routes the curve set to that expert after its local steps; mean over experts with two or more holders.
