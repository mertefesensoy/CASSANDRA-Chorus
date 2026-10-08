# Task A · Post hoc check: every mixture layer zeroed at once

Generated 2026-10-08 15:00 UTC by `python -m scripts.probe_all_layers_task_a` (code `8a6160e`) from the saved final models in `C:\Users\senso\chorus-runs`, 36 models, curve sets.

**Post hoc:** decided by the owner on 2026-10-08 after the registered probe's results were known. Not part of the registered reading and carries no verdict.

| Arm | Variant | Seed | Trained accuracy | All mixtures zeroed | Worst-map cost (points) | Run ID |
|---|---|---|---|---|---|---|
| Centralized (reference) | marked | 7 | 1.0000 | 0.2717 | 78.6 | `20261007T210724Z_taskA-central-marked_s7` |
| Centralized (reference) | marked | 11 | 1.0000 | 0.2866 | 80.6 | `20261008T023619Z_taskA-central-marked_s11` |
| Centralized (reference) | marked | 19 | 1.0000 | 0.2748 | 85.5 | `20261008T052122Z_taskA-central-marked_s19` |
| Centralized (reference) | unmarked | 7 | 0.9513 | 0.0326 | 93.1 | `20261008T062954Z_taskA-central-unmarked_s7` |
| Centralized (reference) | unmarked | 11 | 0.9506 | 0.0307 | 93.9 | `20261008T063833Z_taskA-central-unmarked_s11` |
| Centralized (reference) | unmarked | 19 | 0.9515 | 0.0245 | 93.9 | `20261008T064621Z_taskA-central-unmarked_s19` |
| Sliced, coverage (main arm) | marked | 7 | 1.0000 | 0.3943 | 84.4 | `20261008T081034Z_taskA-sliced-coverage-marked_s7` |
| Sliced, coverage (main arm) | marked | 11 | 1.0000 | 0.3654 | 83.6 | `20261008T092338Z_taskA-sliced-coverage-marked_s11` |
| Sliced, coverage (main arm) | marked | 19 | 1.0000 | 0.3196 | 84.3 | `20261008T093202Z_taskA-sliced-coverage-marked_s19` |
| Sliced, coverage (main arm) | unmarked | 7 | 0.9515 | 0.0673 | 90.9 | `20261008T093924Z_taskA-sliced-coverage-unmarked_s7` |
| Sliced, coverage (main arm) | unmarked | 11 | 0.9513 | 0.0596 | 91.3 | `20261008T094554Z_taskA-sliced-coverage-unmarked_s11` |
| Sliced, coverage (main arm) | unmarked | 19 | 0.9504 | 0.0622 | 90.6 | `20261008T095215Z_taskA-sliced-coverage-unmarked_s19` |
| Sliced, rolling assignment | marked | 7 | 1.0000 | 0.3506 | 80.8 | `20261008T095932Z_taskA-sliced-rolling-marked_s7` |
| Sliced, rolling assignment | marked | 11 | 1.0000 | 0.3524 | 77.2 | `20261008T100404Z_taskA-sliced-rolling-marked_s11` |
| Sliced, rolling assignment | marked | 19 | 1.0000 | 0.3890 | 71.1 | `20261008T101053Z_taskA-sliced-rolling-marked_s19` |
| Sliced, rolling assignment | unmarked | 7 | 0.9515 | 0.0539 | 90.8 | `20261008T101805Z_taskA-sliced-rolling-unmarked_s7` |
| Sliced, rolling assignment | unmarked | 11 | 0.9517 | 0.0640 | 91.3 | `20261008T102547Z_taskA-sliced-rolling-unmarked_s11` |
| Sliced, rolling assignment | unmarked | 19 | 0.9504 | 0.0480 | 91.4 | `20261008T103333Z_taskA-sliced-rolling-unmarked_s19` |
| Sliced, router averaged over all workers | marked | 7 | 1.0000 | 0.3728 | 77.1 | `20261008T104031Z_taskA-sliced-routerall-marked_s7` |
| Sliced, router averaged over all workers | marked | 11 | 1.0000 | 0.3701 | 74.7 | `20261008T104642Z_taskA-sliced-routerall-marked_s11` |
| Sliced, router averaged over all workers | marked | 19 | 1.0000 | 0.4031 | 78.8 | `20261008T105231Z_taskA-sliced-routerall-marked_s19` |
| Sliced, router averaged over all workers | unmarked | 7 | 0.9515 | 0.0633 | 91.3 | `20261008T105919Z_taskA-sliced-routerall-unmarked_s7` |
| Sliced, router averaged over all workers | unmarked | 11 | 0.9504 | 0.0642 | 90.4 | `20261008T110611Z_taskA-sliced-routerall-unmarked_s11` |
| Sliced, router averaged over all workers | unmarked | 19 | 0.9504 | 0.0607 | 91.1 | `20261008T111218Z_taskA-sliced-routerall-unmarked_s19` |
| Partial update | marked | 7 | 1.0000 | 0.3591 | 77.2 | `20261008T111836Z_taskA-partial-marked_s7` |
| Partial update | marked | 11 | 1.0000 | 0.3373 | 76.9 | `20261008T112621Z_taskA-partial-marked_s11` |
| Partial update | marked | 19 | 1.0000 | 0.3292 | 81.8 | `20261008T113449Z_taskA-partial-marked_s19` |
| Partial update | unmarked | 7 | 0.9511 | 0.0715 | 90.5 | `20261008T114234Z_taskA-partial-unmarked_s7` |
| Partial update | unmarked | 11 | 0.9515 | 0.0782 | 90.0 | `20261008T115103Z_taskA-partial-unmarked_s11` |
| Partial update | unmarked | 19 | 0.9501 | 0.0688 | 90.6 | `20261008T120737Z_taskA-partial-unmarked_s19` |
| Full-model local averaging | marked | 7 | 1.0000 | 0.2526 | 81.6 | `20261008T121708Z_taskA-fullavg-marked_s7` |
| Full-model local averaging | marked | 11 | 1.0000 | 0.3252 | 76.7 | `20261008T122608Z_taskA-fullavg-marked_s11` |
| Full-model local averaging | marked | 19 | 1.0000 | 0.2642 | 82.3 | `20261008T123449Z_taskA-fullavg-marked_s19` |
| Full-model local averaging | unmarked | 7 | 0.9514 | 0.0503 | 91.3 | `20261008T124312Z_taskA-fullavg-unmarked_s7` |
| Full-model local averaging | unmarked | 11 | 0.9514 | 0.0454 | 92.2 | `20261008T125053Z_taskA-fullavg-unmarked_s11` |
| Full-model local averaging | unmarked | 19 | 0.9506 | 0.0425 | 91.8 | `20261008T125804Z_taskA-fullavg-unmarked_s19` |
