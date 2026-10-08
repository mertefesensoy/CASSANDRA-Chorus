# Task A · Redundancy probe

Generated 2026-10-08 14:50 UTC by `python -m scripts.probe_task_a` (code `6a02d4b`) from the saved final models in `C:\Users\senso\chorus-runs`, 36 models, 1820 s. The reading was registered before any probe number was computed (docs/implementations/2026-10-08-redundancy-probe.md, SRS D24).

## Reading: MIXED

A layer is needed if bypassing its mixture costs at least 5 points on its worst map. A forced (pair, map) cell is competent if it keeps at least 0.99 of the trained accuracy on that map. Shares below are pooled over the main arm's needed layers.

| Variant | Seed | Main arm's needed layers | Main share | Centralized | Partial | Full | (1) needed layer | (2) main share highest |
|---|---|---|---|---|---|---|---|---|
| marked | 7 | none | n/a | n/a | n/a | n/a | no | no |
| marked | 11 | none | n/a | n/a | n/a | n/a | no | no |
| marked | 19 | none | n/a | n/a | n/a | n/a | no | no |
| unmarked | 7 | [0] | 0.629 | 0.000 | 0.000 | 0.000 | yes | yes |
| unmarked | 11 | [0] | 0.710 | 0.000 | 0.000 | 0.000 | yes | yes |
| unmarked | 19 | [0] | 0.643 | 0.000 | 0.000 | 0.000 | yes | yes |

## By arm

### Centralized (reference)

| Variant | Seed | Bypass cost, points (layers 0 to 3) | Competent share | Specialization | Top-pair retention (mean) | Single-layer random routing | Mean forced accuracy | Run ID |
|---|---|---|---|---|---|---|---|---|
| marked | 7 | 14.0, 27.8, 37.8, 0.0 | 0.089, 0.317, 0.330, 0.848 | 0.098, 0.247, 0.432, 0.018 | 0.928, 0.982, 1.000, 0.999 | 0.931, 0.941, 0.784, 0.996 | 0.929, 0.941, 0.783, 0.995 | `20261007T210724Z_taskA-central-marked_s7` |
| marked | 11 | 9.6, 31.2, 17.0, 0.0 | 0.366, 0.295, 0.487, 0.960 | 0.088, 0.373, 0.195, 0.004 | 0.967, 0.995, 1.000, 1.000 | 0.960, 0.827, 0.946, 0.999 | 0.961, 0.821, 0.947, 0.999 | `20261008T023619Z_taskA-central-marked_s11` |
| marked | 19 | 16.7, 72.9, 17.9, 0.0 | 0.241, 0.393, 0.696, 0.938 | 0.093, 0.677, 0.262, 0.008 | 0.969, 1.000, 1.000, 1.000 | 0.957, 0.730, 0.944, 0.998 | 0.956, 0.725, 0.944, 0.998 | `20261008T052122Z_taskA-central-marked_s19` |
| unmarked | 7 | 76.1, 4.3, 6.5, 2.8 | 0.000, 0.625, 0.424, 0.705 | 0.259, 0.085, 0.096, 0.022 | 0.464, 1.006, 1.007, 0.993 | 0.433, 0.947, 0.918, 0.943 | 0.355, 0.931, 0.919, 0.944 | `20261008T062954Z_taskA-central-unmarked_s7` |
| unmarked | 11 | 69.1, 1.9, 12.5, 3.8 | 0.000, 0.464, 0.071, 0.656 | 0.328, 0.039, 0.160, 0.038 | 0.583, 1.004, 0.968, 0.992 | 0.546, 0.947, 0.872, 0.938 | 0.463, 0.935, 0.876, 0.938 | `20261008T063833Z_taskA-central-unmarked_s11` |
| unmarked | 19 | 88.2, 4.9, 11.5, 2.5 | 0.000, 0.638, 0.482, 0.897 | 0.198, 0.084, 0.331, 0.010 | 0.324, 1.005, 1.008, 0.996 | 0.268, 0.946, 0.864, 0.947 | 0.212, 0.934, 0.865, 0.947 | `20261008T064621Z_taskA-central-unmarked_s19` |

### Sliced, coverage (main arm)

| Variant | Seed | Bypass cost, points (layers 0 to 3) | Competent share | Specialization | Top-pair retention (mean) | Single-layer random routing | Mean forced accuracy | Run ID |
|---|---|---|---|---|---|---|---|---|
| marked | 7 | 0.0, 0.0, 0.0, 0.0 | 1.000, 0.973, 1.000, 1.000 | 0.000, 0.005, 0.000, 0.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 0.999, 1.000, 1.000 | 1.000, 0.999, 1.000, 1.000 | `20261008T081034Z_taskA-sliced-coverage-marked_s7` |
| marked | 11 | 0.0, 0.0, 0.0, 0.0 | 1.000, 0.996, 1.000, 1.000 | 0.000, 0.001, 0.000, 0.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | `20261008T092338Z_taskA-sliced-coverage-marked_s11` |
| marked | 19 | 0.0, 0.4, 0.0, 0.0 | 0.991, 0.996, 1.000, 1.000 | 0.002, 0.001, 0.000, 0.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | `20261008T093202Z_taskA-sliced-coverage-marked_s19` |
| unmarked | 7 | 51.2, 0.9, 1.4, 1.2 | 0.629, 1.000, 0.911, 0.969 | 0.011, 0.005, 0.012, 0.010 | 0.991, 1.001, 1.005, 0.999 | 0.946, 0.951, 0.949, 0.950 | 0.943, 0.951, 0.950, 0.950 | `20261008T093924Z_taskA-sliced-coverage-unmarked_s7` |
| unmarked | 11 | 44.3, 1.6, 3.1, 1.3 | 0.710, 0.996, 0.924, 0.987 | 0.013, 0.007, 0.016, 0.007 | 0.992, 1.002, 1.002, 0.998 | 0.946, 0.950, 0.949, 0.951 | 0.944, 0.951, 0.950, 0.950 | `20261008T094554Z_taskA-sliced-coverage-unmarked_s11` |
| unmarked | 19 | 63.4, 0.8, 1.5, 1.3 | 0.643, 1.000, 0.915, 0.960 | 0.016, 0.008, 0.014, 0.012 | 0.995, 1.002, 1.003, 0.999 | 0.944, 0.950, 0.949, 0.949 | 0.941, 0.950, 0.949, 0.949 | `20261008T095215Z_taskA-sliced-coverage-unmarked_s19` |

### Sliced, rolling assignment

| Variant | Seed | Bypass cost, points (layers 0 to 3) | Competent share | Specialization | Top-pair retention (mean) | Single-layer random routing | Mean forced accuracy | Run ID |
|---|---|---|---|---|---|---|---|---|
| marked | 7 | 0.0, 5.4, 0.0, 0.0 | 1.000, 0.911, 0.978, 1.000 | 0.000, 0.032, 0.005, 0.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 0.995, 0.999, 1.000 | 1.000, 0.995, 0.999, 1.000 | `20261008T095932Z_taskA-sliced-rolling-marked_s7` |
| marked | 11 | 0.0, 5.4, 0.0, 0.0 | 0.978, 0.902, 1.000, 1.000 | 0.005, 0.053, 0.000, 0.000 | 0.998, 1.000, 1.000, 1.000 | 0.999, 0.992, 1.000, 1.000 | 0.999, 0.992, 1.000, 1.000 | `20261008T100404Z_taskA-sliced-rolling-marked_s11` |
| marked | 19 | 0.0, 0.0, 0.0, 0.0 | 0.987, 1.000, 0.978, 1.000 | 0.003, 0.000, 0.004, 0.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 0.999, 1.000 | 1.000, 1.000, 0.999, 1.000 | `20261008T101053Z_taskA-sliced-rolling-marked_s19` |
| unmarked | 7 | 63.2, 3.5, 4.9, 1.4 | 0.116, 0.888, 0.871, 0.942 | 0.031, 0.016, 0.034, 0.012 | 0.942, 1.002, 1.004, 0.997 | 0.928, 0.950, 0.944, 0.949 | 0.921, 0.950, 0.946, 0.950 | `20261008T101805Z_taskA-sliced-rolling-unmarked_s7` |
| unmarked | 11 | 60.7, 1.0, 3.9, 1.0 | 0.192, 0.969, 0.763, 0.969 | 0.049, 0.010, 0.033, 0.008 | 0.970, 1.001, 1.004, 0.998 | 0.933, 0.951, 0.943, 0.950 | 0.924, 0.951, 0.945, 0.950 | `20261008T102547Z_taskA-sliced-rolling-unmarked_s11` |
| unmarked | 19 | 65.3, 1.0, 1.8, 0.9 | 0.049, 0.982, 0.839, 1.000 | 0.039, 0.008, 0.030, 0.008 | 0.945, 1.004, 1.005, 0.998 | 0.921, 0.951, 0.945, 0.949 | 0.911, 0.950, 0.946, 0.949 | `20261008T103333Z_taskA-sliced-rolling-unmarked_s19` |

### Sliced, router averaged over all workers

| Variant | Seed | Bypass cost, points (layers 0 to 3) | Competent share | Specialization | Top-pair retention (mean) | Single-layer random routing | Mean forced accuracy | Run ID |
|---|---|---|---|---|---|---|---|---|
| marked | 7 | 0.0, 0.0, 0.0, 0.0 | 1.000, 1.000, 1.000, 1.000 | 0.000, 0.000, 0.000, 0.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | `20261008T104031Z_taskA-sliced-routerall-marked_s7` |
| marked | 11 | 0.0, 0.0, 0.0, 0.0 | 1.000, 0.982, 1.000, 1.000 | 0.000, 0.004, 0.000, 0.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 0.999, 1.000, 1.000 | `20261008T104642Z_taskA-sliced-routerall-marked_s11` |
| marked | 19 | 0.0, 0.8, 0.0, 0.0 | 0.996, 0.991, 1.000, 1.000 | 0.001, 0.002, 0.000, 0.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | `20261008T105231Z_taskA-sliced-routerall-marked_s19` |
| unmarked | 7 | 48.6, 0.7, 1.8, 0.9 | 0.719, 0.996, 0.924, 0.996 | 0.011, 0.005, 0.015, 0.009 | 0.994, 1.002, 1.005, 1.000 | 0.946, 0.951, 0.950, 0.950 | 0.944, 0.951, 0.950, 0.951 | `20261008T105919Z_taskA-sliced-routerall-unmarked_s7` |
| unmarked | 11 | 46.1, 3.1, 2.4, 0.8 | 0.777, 1.000, 0.942, 0.987 | 0.010, 0.008, 0.021, 0.008 | 0.994, 1.001, 1.004, 0.999 | 0.947, 0.951, 0.948, 0.951 | 0.945, 0.950, 0.949, 0.950 | `20261008T110611Z_taskA-sliced-routerall-unmarked_s11` |
| unmarked | 19 | 66.4, 0.4, 0.9, 1.5 | 0.661, 1.000, 0.933, 0.960 | 0.016, 0.008, 0.010, 0.010 | 0.996, 1.001, 1.004, 0.999 | 0.944, 0.951, 0.950, 0.949 | 0.942, 0.950, 0.949, 0.949 | `20261008T111218Z_taskA-sliced-routerall-unmarked_s19` |

### Partial update

| Variant | Seed | Bypass cost, points (layers 0 to 3) | Competent share | Specialization | Top-pair retention (mean) | Single-layer random routing | Mean forced accuracy | Run ID |
|---|---|---|---|---|---|---|---|---|
| marked | 7 | 3.1, 6.0, 5.1, 0.0 | 0.888, 0.781, 0.719, 1.000 | 0.020, 0.122, 0.091, 0.000 | 0.994, 1.000, 1.000, 1.000 | 0.996, 0.981, 0.980, 1.000 | 0.996, 0.976, 0.978, 1.000 | `20261008T111836Z_taskA-partial-marked_s7` |
| marked | 11 | 0.3, 10.8, 9.8, 0.0 | 0.996, 0.728, 0.464, 1.000 | 0.002, 0.121, 0.191, 0.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 0.978, 0.942, 1.000 | 1.000, 0.979, 0.947, 1.000 | `20261008T112621Z_taskA-partial-marked_s11` |
| marked | 19 | 0.0, 51.5, 27.5, 0.0 | 0.996, 0.522, 0.804, 1.000 | 0.000, 0.492, 0.261, 0.000 | 1.000, 1.000, 1.000, 1.000 | 1.000, 0.875, 0.964, 1.000 | 1.000, 0.858, 0.964, 1.000 | `20261008T113449Z_taskA-partial-marked_s19` |
| unmarked | 7 | 60.7, 12.2, 3.7, 0.4 | 0.000, 0.576, 0.420, 0.982 | 0.256, 0.091, 0.079, 0.009 | 0.871, 1.004, 1.004, 1.001 | 0.816, 0.933, 0.917, 0.949 | 0.754, 0.930, 0.921, 0.949 | `20261008T114234Z_taskA-partial-unmarked_s7` |
| unmarked | 11 | 62.9, 11.9, 7.9, 1.7 | 0.000, 0.420, 0.652, 0.830 | 0.155, 0.056, 0.082, 0.020 | 0.909, 1.004, 1.003, 1.002 | 0.853, 0.935, 0.936, 0.948 | 0.797, 0.932, 0.937, 0.948 | `20261008T115103Z_taskA-partial-unmarked_s11` |
| unmarked | 19 | 79.1, 40.2, 3.6, 1.1 | 0.000, 0.469, 0.750, 0.960 | 0.310, 0.324, 0.066, 0.010 | 0.844, 1.008, 1.005, 0.998 | 0.756, 0.923, 0.936, 0.948 | 0.662, 0.882, 0.938, 0.948 | `20261008T120737Z_taskA-partial-unmarked_s19` |

### Full-model local averaging

| Variant | Seed | Bypass cost, points (layers 0 to 3) | Competent share | Specialization | Top-pair retention (mean) | Single-layer random routing | Mean forced accuracy | Run ID |
|---|---|---|---|---|---|---|---|---|
| marked | 7 | 3.9, 6.1, 11.4, 0.0 | 0.482, 0.625, 0.585, 0.982 | 0.035, 0.052, 0.117, 0.003 | 0.978, 0.994, 1.000, 0.996 | 0.982, 0.984, 0.976, 0.999 | 0.983, 0.985, 0.976, 0.999 | `20261008T121708Z_taskA-fullavg-marked_s7` |
| marked | 11 | 1.3, 11.2, 0.3, 0.0 | 0.817, 0.491, 0.951, 1.000 | 0.020, 0.222, 0.008, 0.000 | 0.991, 1.000, 1.000, 1.000 | 0.994, 0.945, 0.999, 1.000 | 0.995, 0.942, 0.998, 1.000 | `20261008T122608Z_taskA-fullavg-marked_s11` |
| marked | 19 | 0.0, 62.2, 3.8, 0.0 | 0.955, 0.500, 0.830, 1.000 | 0.006, 0.625, 0.054, 0.000 | 0.999, 1.000, 1.000, 1.000 | 0.998, 0.758, 0.990, 1.000 | 0.999, 0.750, 0.990, 1.000 | `20261008T123449Z_taskA-fullavg-marked_s19` |
| unmarked | 7 | 73.8, 5.2, 6.5, 0.9 | 0.000, 0.683, 0.496, 0.951 | 0.253, 0.050, 0.133, 0.015 | 0.736, 1.004, 1.006, 0.999 | 0.702, 0.940, 0.910, 0.950 | 0.620, 0.941, 0.915, 0.950 | `20261008T124312Z_taskA-fullavg-unmarked_s7` |
| unmarked | 11 | 69.1, 8.2, 7.0, 5.7 | 0.000, 0.558, 0.393, 0.817 | 0.256, 0.049, 0.086, 0.021 | 0.830, 0.999, 1.000, 1.000 | 0.775, 0.934, 0.916, 0.947 | 0.683, 0.936, 0.921, 0.947 | `20261008T125053Z_taskA-fullavg-unmarked_s11` |
| unmarked | 19 | 84.6, 19.4, 3.0, 2.5 | 0.000, 0.710, 0.629, 0.938 | 0.305, 0.097, 0.049, 0.012 | 0.696, 1.004, 1.005, 1.001 | 0.629, 0.945, 0.936, 0.949 | 0.524, 0.933, 0.938, 0.949 | `20261008T125804Z_taskA-fullavg-unmarked_s19` |

## Definitions

- **Bypass cost:** trained accuracy minus accuracy with that layer's mixture output set to zero, worst map, in points.
- **Competent share:** fraction of the 28 expert pairs × 8 maps whose forced accuracy keeps at least 0.99 of the trained accuracy on that map; 1 means any pair handles any map.
- **Specialization:** per expert, the spread across maps of its mean retention over the 7 pairs that contain it; mean over experts. 0 means every expert serves every map equally well.
- **Cross-checks (not part of the reading):** top-pair retention forces, for each map, the two experts the trained router uses most for it; single-layer random routing randomizes only that layer, to compare with the mean forced accuracy.
