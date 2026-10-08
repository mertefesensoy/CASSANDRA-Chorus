# 2026-10-08 · Run operations: keep-awake, power and standby logging, preflight, visible launch

| | |
|---|---|
| PLAN step | none: run infrastructure for steps 4 onward, and for Stage 1 worker clients |
| Branch | `stage0/ops-run-protocol` (stacked on `stage0/04-centralized-task-a`) |
| SRS requirements | S0-N-03 (reproducible, documented runs), S0-N-07 (report limits), S0-F-24 (run log) |
| Status | Planned |

## Problem / Motivation

On the night of 2026-10-07/08 the centralized Task A matrix stalled for hours, and nothing in the run logs showed it:

- Each 2-second mains drop-out made ASUS Armoury Crate switch Windows from the "Turbo" plan to "Performance", which on battery turned the display off and slept after 180 seconds.
- With nobody at the laptop, the brief battery spell sent it into Modern Standby (Kernel-Power 506).
- A drop-out during standby froze the training process until the next wake-up (507). That produced stalls of 21 minutes, about 5 hours, 62 minutes and about 1 hour.
- Results were unaffected, because training is deterministic; the cost was wall-clock time.

The owner has since changed the plan timeouts. The mains drop-outs are a hardware fault still to be checked.

The parent CASSANDRA project had already solved the sleep half of this, with ADR 0012 and `keep_awake_while_pid.ps1`. The owner decided on 2026-10-08 that its lessons be **integrated natively into Chorus rather than copying legacy code**, and that **power events are logged only and never stop a run**.

## Legacy lesson to Chorus implementation

| Legacy CASSANDRA practice | Lesson | Chorus implementation |
|---|---|---|
| `keep_awake_while_pid.ps1`: SetThreadExecutionState(CONTINUOUS, SYSTEM_REQUIRED, DISPLAY_REQUIRED) every 30 s while the launcher lives | Windows must be told a long run is in progress | `ops.keep_awake()` context manager inside the training process: one call from the main thread at run start, released on exit, return value logged. **New lesson: it prevents entering standby only. It must be held from launch with the screen on, and it cannot bring the laptop out of standby.** |
| ADR 0012: every run in a visible PowerShell window | Runs must be supervisable | `scripts/ops/launch_visible.ps1` opens a visible window at the repository root and runs one command. **New lesson: jobs the coding agent started in the background ran about 3 times slower;** a visible window avoids that |
| Gate: C: free space at least 15 GiB | Runs must not die on a full disk | Preflight check, threshold from the local settings file (default 15 GiB) |
| Gate: no other GPU compute process | Concurrent GPU jobs distort timing and can run out of memory | Preflight check via `nvidia-smi` |
| Checkpoints in `C:\cassandra_runs`; "OneDrive kills checkpoint renames" | Never write run outputs into OneDrive | Preflight refuses a runs folder inside OneDrive |
| Launcher created MUSAHIT's `SKIP_NEXT_RUN.flag` | The nightly task takes the GPU at 02:00 | Preflight refuses a run expected to overlap the next start of any listed scheduled task. It **never** creates a skip flag, because that changes another of the owner's systems |
| Pit-stop checkpoints and resume | Long runs must survive interruption | Not in this change (owner: log only). Resumable rounds (S0-N-04) arrive with step 6 |
| (no legacy equivalent) | Power events and stalls were invisible in run logs | `ops.PowerMonitor`, a thread polling `GetSystemPowerStatus` once a second: `power` records on mains/battery changes, and `stall` records when its own one-second wait took far longer (the process was frozen). After the run, the Windows event log for the run's window (Kernel-Power 105, 506, 507, the authoritative record) is summarized into the log |

## What Changed

| File | Description |
|---|---|
| `cassandra_chorus/ops/__init__.py` | Public names of the package. |
| `cassandra_chorus/ops/power.py` | `keep_awake`, `power_status`, `PowerMonitor`, `windows_power_events`. |
| `cassandra_chorus/ops/preflight.py` | `OpsSettings`, `load_ops_settings`, `run_preflight`, `enforce`. |
| `cassandra_chorus/runlog.py` | A lock around `write`, so the monitor thread and the main thread can both log. |
| `scripts/train_task_a_centralized.py`, `scripts/smoke.py` | Preflight before the run folder exists. Keep-awake and the power monitor for the whole run. An `ops` record after the header, and a `power_summary` record at the end. |
| `scripts/summarize_task_a_runs.py` | Reports power changes, stalls and Windows events. |
| `scripts/ops/launch_visible.ps1` | Starts one command in a visible PowerShell window at the repository root. |
| `ops.example.toml`, `.gitignore` | Documented example of the local settings file; `ops.local.toml` is gitignored. |
| `README.md` | How to launch runs and set local settings. |
| `tests/test_ops.py`, `tests/test_runlog.py`, `tests/test_train.py` | Tests below. |

## Implementation Approach

### Machine-level settings (owner decision 2026-10-08)

Settings that describe the machine rather than the experiment live in a **gitignored** `ops.local.toml` at the repository root. `CHORUS_OPS_FILE` can point elsewhere; code defaults apply when the file is absent. They are validated strictly, like experiment configurations, and written to each run log in an `ops` record. They are **not** part of the configuration hash, so an experiment's identity does not depend on the machine.

`OpsSettings` fields and defaults:

| Field | Default |
|---|---|
| `keep_awake` | true |
| `power_monitor` | true |
| `poll_seconds` | 1.0 |
| `stall_seconds` | 10.0 |
| `min_free_gib` | 15.0 |
| `refuse_onedrive` | true |
| `require_gpu_idle` | true |
| `avoid_scheduled_tasks` | empty |
| `seconds_per_step` | 0.15 (used only to estimate run length for the scheduled-task check) |

On the reference laptop the local file lists `MUSAHIT_Nightly`.

### Preflight (owner decision 2026-10-08: refuse hard checks, warn soft)

`run_preflight(settings, runs_dir, expected_seconds)` returns one result per check: name, passed, hard or soft, detail. `enforce` raises `PreflightError` naming every failed hard check, before any run folder exists.

| Check | Kind | Rule |
|---|---|---|
| `runs_dir_outside_onedrive` | hard | The resolved runs folder is not under any OneDrive root (environment variables `OneDrive`, `OneDriveConsumer`, `OneDriveCommercial`) and has no path component starting with "OneDrive" |
| `free_disk` | hard | Free space on the runs folder's drive is at least `min_free_gib` |
| `gpu_idle` | hard | `nvidia-smi` lists no compute process. If `nvidia-smi` is unavailable, the check passes with the detail "not checked" |
| `scheduled_task:<name>` | hard | The task's next start (read via `Get-ScheduledTaskInfo` as an ISO time) is not before now + 1.5 × expected duration + 5 minutes. A task that does not exist passes with that detail |
| `on_mains` | soft | The laptop is on mains at start; on battery only warns |

The expected duration is steps × `seconds_per_step`.

### Keep-awake

`keep_awake(enabled)` is a context manager. On Windows it calls `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)` once, from the thread that runs the training (the main thread). The setting lasts as long as that thread unless replaced. On exit it calls `SetThreadExecutionState(ES_CONTINUOUS)`. It yields a status: requested, held (true if the call returned non-zero), the previous state, and a reason when not held (disabled, not Windows, or the call failed).

### Power monitor

`PowerMonitor(on_event, poll_seconds, stall_seconds)` is a daemon thread. Each poll:

1. Reads `GetSystemPowerStatus`: mains state, battery percentage, battery saver.
2. Reports a `power` event when the mains state changes; the first poll always reports.
3. Measures the wall-clock time the previous wait actually took. If it exceeded `stall_seconds`, reports a `stall` event with its length and wall-clock bounds.

`summary()` gives the number of polls, mains changes, stalls and the longest stall.

Limitation, stated in the log: one-second polling can miss a mains drop-out shorter than a second, and can see a 2-second one late. The Windows events are the authoritative count.

### Windows events

`windows_power_events(start_utc, end_utc)` runs `wevtutil qe System` with a query for Kernel-Power events 105, 506 and 507 in the window, and parses the XML. It returns counts (mains off, mains on, standby entries, standby exits) and the first 50 events. If unavailable, it returns `{"available": false, "reason": ...}`.

### Visible launcher

`scripts/ops/launch_visible.ps1 -Module <python module> [-Arguments <string>] [-RunsDir <path>]` opens a new `powershell.exe` window titled with the module name. It changes to the repository root, sets `CHORUS_RUNS_DIR` if given, runs `python -m <module> <arguments>`, and leaves the window open with the exit code shown.

## Mathematical / Statistical Details

**Stall rule.** Let p be the nominal poll interval (1 s) and Δ the wall-clock time a single wait actually took. A stall is recorded when Δ > s, with s = `stall_seconds` = 10 s. The record gives Δ and its wall-clock start and end. Wall-clock time is used because it keeps running while the process is frozen. A clock adjustment larger than s could cause a false stall; such an adjustment is visible in the record's times.

**Scheduled-task overlap.** With expected duration D = steps × `seconds_per_step`, the run is refused if the task's next start lies before now + 1.5·D + 300 s.

## Design Decisions

| Decision | Chosen | Alternatives | Decided by |
|---|---|---|---|
| Integrate or copy legacy tooling | Native Python in the training process, plus a small launcher | Reuse `keep_awake_while_pid.ps1` and the legacy launchers | Owner, 2026-10-08 |
| Power events | Logged only | Stop the run; stop and resume from checkpoint | Owner, 2026-10-08 |
| Machine settings location | Gitignored `ops.local.toml`, outside the configuration hash | `[ops]` section in experiment configs | Owner, 2026-10-08 |
| Preflight failures | Refuse hard checks, warn on soft | Warn only | Owner, 2026-10-08 |
| MUSAHIT skip flag | Never created automatically | Create it as the legacy launcher did | Engineering default: it changes another of the owner's systems |
| Keep-awake mechanism | One call on the main thread | A helper process re-calling every 30 s | Engineering default: same API; no extra process to orphan |
| Mains detection | `GetSystemPowerStatus` polling plus Windows events afterwards | Event-log subscription in real time | Engineering default: no new dependency (pywin32), and the event log remains the authoritative cross-check |

## Verification

To be completed after implementation.

**Not tested:** to be completed.

## Related Docs

- Parent repository: `docs/decisions/0012-visible-terminal-experiment-launches.md`, `experiments/tiny_language_lab/keep_awake_while_pid.ps1`, `INSTRUCTIONS.md` section 5
- `RESULTS.md`: operational notes of the 2026-10-08 matrix entry
- `docs/implementations/2026-10-07-repository-skeleton.md`: the run log
