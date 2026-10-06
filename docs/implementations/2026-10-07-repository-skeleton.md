# 2026-10-07 · Repository skeleton (PLAN step 1)

| | |
|---|---|
| PLAN step | 1 · Repository skeleton: layout, licence, configuration format, logging, test setup |
| Branch | `stage0/01-skeleton` (stacked on `docs/owner-decisions-2026-10-07`) |
| SRS requirements | S0-N-03 (reproducibility), S0-N-05 (coordinator separated from simulation), S0-N-08 (raw runs local), S0-F-24 (structured run log) |
| Status | Planned |

## Problem / Motivation

Every later Stage 0 step needs the same plumbing: a configuration file that fully describes a run, seeding, a structured log that records what was run and on what, and a test setup. If each step invents its own, runs stop being comparable and S0-N-03 ("reproducible from a configuration file and seed") becomes a matter of discipline rather than code. Step 1 builds that plumbing once, with nothing model-specific in it, and proves it end to end with a smoke run on the reference GPU.

It also puts in place a mechanical guard for the owner's rule that coordinator code (slice assignment and merge) stays free of simulation code, because Stage 1 reuses the coordinator with real workers (S0-N-05).

## What Changed

| File | Description |
|---|---|
| `.gitignore` | Ignores Python caches, the default local runs folder `/runs/`, and checkpoint files. |
| `.gitattributes` | Stores and checks out text files with LF line endings on every machine. |
| `NOTICE` | Apache-2.0 notice in the parent repository's style, with text8 provenance (data not redistributed). |
| `pyproject.toml` | Package metadata, minimum versions (the tested ones), and pytest configuration. |
| `README.md` | Status updated to step 1; how to set the runs folder, run the tests and run the smoke check. |
| `RESULTS.md` | Stub for the hand-written record of runs that inform decisions (S0-N-08). |
| `results/README.md` | Explains that this folder holds committed results tables. |
| `configs/smoke.toml` | Configuration of the smoke run. |
| `cassandra_chorus/__init__.py` | Package marker and version. |
| `cassandra_chorus/config.py` | Typed configuration loading from TOML or JSON, command-line overrides, validation, hashing. |
| `cassandra_chorus/paths.py` | Resolves the runs folder from `CHORUS_RUNS_DIR`, falling back to `./runs`. |
| `cassandra_chorus/repro.py` | Seeding, determinism modes, and capture of the software and hardware environment. |
| `cassandra_chorus/runlog.py` | Run folder creation and the JSONL run log, including git state and warning capture. |
| `cassandra_chorus/{model,data,coordinator,sim,metrics}/__init__.py` | Empty sub-packages from PLAN section 6, each stating its purpose. |
| `scripts/__init__.py`, `scripts/smoke.py` | Smoke entry point: config, seeds, determinism, log, a tiny CUDA computation and its checksum. |
| `tests/test_config.py`, `tests/test_runlog.py`, `tests/test_repro.py` | Unit tests of the three modules. |
| `tests/test_layering.py` | Fails if any module under `coordinator/` imports from `sim/`. |
| `tests/test_smoke_gpu.py` | Runs the smoke entry point in separate processes and compares checksums; skipped without CUDA. |
| `docs/implementations/_TEMPLATE.md` | Template for implementation documents (owner's global rule). |
| `docs/implementations/2026-10-07-repository-skeleton.md` | This document. |

## Implementation Approach

The code runs from the repository root without installation (`python -m scripts.smoke ...`, `python -m pytest`); pytest finds the package through `pythonpath = ["."]`.

### `config.py`

Configurations are frozen dataclasses. Each entry point declares its own top-level dataclass composed of sections; step 1 defines only `RunSection` (`name`, `seed`, `device`, `determinism`, `notes`), and later steps add their own sections.

- `load_config(path, schema, overrides=()) -> schema instance`. Reads a `.toml` (via the standard library `tomllib`) or `.json` file into a dictionary, applies overrides, then builds the dataclass with strict validation. Raises `ConfigError` naming the offending key on: an unknown key at any depth, a missing required field, a value of the wrong type (a boolean is not accepted as an integer; an integer is accepted as a float and converted), or a value outside a `Literal` set. No side effects.
- Overrides have the form `section.key=value`. The value is parsed as a TOML literal (`11`, `0.5`, `true`, `"text"`); if that fails it is taken as a plain string, so `run.name=abc` works. Setting a key the schema does not have is an error, as above.
- `config_to_dict(cfg)` gives a plain dictionary; `config_hash(cfg)` gives the SHA-256 of its canonical JSON form (see Mathematical details). Loading a saved resolved `config.json` reproduces an equal configuration with the same hash.

### `paths.py`

`runs_root() -> Path` returns `CHORUS_RUNS_DIR` if it is set and non-empty, otherwise `<repo>/runs`. It does not create the folder.

### `repro.py`

- `prepare_process(mode)`: for modes `warn` and `strict`, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` unless already set. cuBLAS reads this when the first GPU matrix multiply creates its handle, so it must be called before any CUDA work; if CUDA is already initialized it raises rather than silently having no effect.
- `apply_determinism(mode) -> dict`: `off` disables deterministic algorithms; `warn` calls `torch.use_deterministic_algorithms(True, warn_only=True)`; `strict` calls it without `warn_only`, so an operation that has no deterministic kernel raises. Both `warn` and `strict` also set cuDNN to deterministic and disable cuDNN benchmarking. Returns the settings actually in force, for the log.
- `seed_everything(seed) -> dict`: seeds Python's `random`, NumPy's global generator and PyTorch (CPU and every CUDA device). Returns the seeds used.
- `environment_info() -> dict`: Python version, operating system, PyTorch, CUDA, cuDNN and NumPy versions, CUDA availability, GPU name and total memory, and the `CUBLAS_WORKSPACE_CONFIG` value. No side effects.

### `runlog.py`

- `make_run_id(name, seed, now)` gives `YYYYMMDDTHHMMSSZ_<name>_s<seed>` in UTC. Names are limited to letters, digits, `.`, `_` and `-`.
- `RunLogger.create(runs_root, run_id)` creates `<runs_root>/<run_id>/` and refuses if it already exists: a run never overwrites another.
- `write(record_type, **fields)` appends one JSON object per line with `type` and `time_utc`, and flushes after every record so a crash loses at most the record being written. Non-finite floats are written as the strings `"nan"`, `"inf"` and `"-inf"`, since standard JSON has no representation for them. NumPy and PyTorch scalars are converted to Python numbers.
- The header record holds: the resolved configuration, its hash, the configuration file path, the seeds, the environment, the determinism settings and the git state. The git state is the commit, the branch, whether there are uncommitted changes, and the list of untracked files. When tracked files have uncommitted changes, `git diff HEAD` is saved as `uncommitted.patch` in the run folder, so that the run can still be reproduced.
- The resolved configuration is also saved as `config.json` in the run folder, loadable by `load_config`.
- `capture_warnings()` is a context manager that records every distinct warning (category and message) once as a `warning` record and still shows it on the console. On exit it writes a `warnings_summary` record with the count of each. This is how `warn` mode documents where bit-exact reproduction is not guaranteed (S0-N-03).
- `close(status)` writes an `end` record. As a context manager, an exception produces `status = "failed"` with the error message before the exception propagates.

### `scripts/smoke.py`

Loads the config, calls `prepare_process`, creates the run folder and header, then inside `capture_warnings()`: seeds, applies determinism, resolves the device (a request for CUDA when CUDA is unavailable is an error, never a silent fall-back to CPU), draws two random matrices, computes a forward product, a scalar loss and one backward pass, and logs the SHA-256 checksum of the resulting tensors. It prints the run folder and checksum. This is a toy computation; no model is trained.

### `tests/test_layering.py`

Parses every Python file under `cassandra_chorus/coordinator/` with `ast`, resolves each import (absolute and relative) to a full module name, and fails if any resolves to `cassandra_chorus.sim` or below. The checker is also tested on synthetic source that violates the rule, so it cannot pass vacuously.

## Mathematical / Statistical Details

No statistical method is involved. Two hashes are used, both SHA-256:

- **Configuration hash.** Let *c* be the resolved configuration as a nested dictionary. Its canonical form is the JSON text with keys sorted at every level, no whitespace, and UTF-8 encoding. The configuration hash is SHA-256 of that text, written as hexadecimal. Two configurations get the same hash exactly when every field has the same value, independent of key order in the source file. A configuration that relies on a default and one that states the same value explicitly get the same hash, because hashing happens after defaults are filled.
- **Smoke checksum.** SHA-256 over the raw bytes of the result tensors (copied to the CPU, contiguous, in a fixed order). Equal checksums mean bit-identical results; any single-bit difference changes the checksum.

## Design Decisions

| Decision | Chosen | Alternatives | Decided by |
|---|---|---|---|
| Config format | TOML | YAML (PyYAML reads `1e-3` as a string), JSON (no comments) | Owner, 2026-10-07 |
| Run log format | JSONL, one record per line | CSV metrics; a database | Owner, 2026-10-07 (matches parent `runs/*.jsonl`) |
| Raw runs location | `CHORUS_RUNS_DIR`, else `./runs`; owner sets `C:\Users\senso\chorus-runs` on the laptop | Always `./runs` (OneDrive syncs checkpoints and may lock files mid-write) | Owner, 2026-10-07 |
| Determinism default | `warn`, with warnings captured into the log | `strict` (may force design changes in step 2), `off` | Owner, 2026-10-07 |
| How code is run | From the repo root, no install | Editable install into the Microsoft Store Python | Owner, 2026-10-07 |
| Minimum versions | Python 3.13, PyTorch 2.12, NumPy 2.4 (the tested ones) | Lower floors | Engineering default: only tested versions are claimed. Revisit at Stage 1, when volunteer machines vary |
| Typed config | Frozen dataclasses with a small validator | pydantic, Hydra, OmegaConf | Engineering default: no new dependency |
| Saving uncommitted changes | `uncommitted.patch` beside the log | Refuse to run with uncommitted changes | Engineering default: keeps iteration fast without losing reproducibility |
| Non-finite floats in logs | Strings `"nan"`, `"inf"`, `"-inf"` | Python's non-standard `NaN` token (breaks strict JSON readers); `null` (loses information) | Engineering default |
| Layering guard | Static import check in a test | Convention only | Engineering default, serving the owner's rule (S0-N-05) |
| CI, linter | Not in step 1 | GitHub Actions on CPU | Deferred; can be added later on request |

## Verification

To be completed after implementation.

**Not tested:** to be completed after implementation.

## Related Docs

- `docs/SRS.md`: S0-N-03, S0-N-05, S0-N-08, S0-F-24, decisions D1 to D9
- `docs/PLAN.md`: section 4 (step 1), section 6 (layout)
