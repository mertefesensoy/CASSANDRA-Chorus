# CASSANDRA Chorus

CASSANDRA Chorus is the next phase of the [CASSANDRA](https://github.com/mertefesensoy/CASSANDRA) low-hardware language-model research program. The long-term goal is a language model trained by a pool of volunteer machines, where no machine holds the whole model during training, and where the full pipeline, logs and weights are open. No token or cryptocurrency is involved; contribution is tracked in a plain credit ledger.

The model is a mixture-of-experts transformer. Each worker trains the shared part of the model plus a subset of the experts locally, and a coordinator merges the results ("sliced local training"). Whether this works for a language model is the open question that Stage 0 tests, on a single machine, before any pool infrastructure is built.

## Status

Stage 0, PLAN step 1 (repository skeleton): configuration, run logging, seeding and test setup. No model, data or training code exists yet, and there are no results.

## Documents

- [`docs/SRS.md`](docs/SRS.md): software requirements specification
- [`docs/PLAN.md`](docs/PLAN.md): plan memo, stages and Stage 0 work breakdown
- [`docs/implementations/`](docs/implementations/): one document per change
- [`RESULTS.md`](RESULTS.md): hand-written record of runs that inform decisions

## Running

Requirements: Python 3.13, PyTorch 2.12 with CUDA, NumPy 2.4, and pytest for the tests. Everything runs from the repository root without installing the package.

Raw run logs and checkpoints are not committed. They go to the folder named by the `CHORUS_RUNS_DIR` environment variable, or to `runs/` in the repository if it is not set. Keep that folder outside OneDrive and, with the Microsoft Store build of Python, outside `AppData`. To set it permanently for your Windows user (PowerShell, then open a new terminal):

```powershell
[Environment]::SetEnvironmentVariable("CHORUS_RUNS_DIR", "C:\Users\senso\chorus-runs", "User")
```

Run the tests (the GPU tests are skipped when CUDA is not available):

```powershell
python -m pytest
```

Machine-level run settings (keep-awake, power logging, pre-start checks such as "runs folder not in OneDrive", "GPU idle" and scheduled tasks to avoid) live in an optional, gitignored `ops.local.toml`. Copy `ops.example.toml` to `ops.local.toml` and adjust it. Long runs should be started in a visible window, which also keeps a transcript under `<runs folder>\launcher_logs`:

```powershell
powershell -File scripts\ops\launch_visible.ps1 -Module scripts.train_task_a_centralized -Arguments "--config configs/task_a/centralized.toml --set run.seed=11" -RunsDir C:\Users\senso\chorus-runs
```

Keep the screen on when starting a long run. The run holds a keep-awake request from launch, which stops Windows from going into standby, but cannot wake the machine once it is already in standby.

Smoke check of the run pipeline (prints the run folder and a checksum; the same seed gives the same checksum):

```powershell
python -m scripts.smoke --config configs/smoke.toml
python -m scripts.smoke --config configs/smoke.toml --set run.seed=11
```

## Licence

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
