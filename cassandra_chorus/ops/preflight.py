"""Machine-level run settings and pre-start checks.

Settings describe the machine, not the experiment (owner decision
2026-10-08): they come from a gitignored ``ops.local.toml`` at the repository
root (or the file named by ``CHORUS_OPS_FILE``), are written to each run log,
and are not part of the configuration hash.

Checks (owner decision 2026-10-08): a failed hard check refuses the run before
any run folder exists; a failed soft check is only logged. Rules are in
docs/implementations/2026-10-08-run-operations.md.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cassandra_chorus.config import ConfigError, build_config, read_config_file
from cassandra_chorus.ops.power import power_status
from cassandra_chorus.paths import REPO_ROOT

OPS_FILE_ENV = "CHORUS_OPS_FILE"
DEFAULT_OPS_FILE = REPO_ROOT / "ops.local.toml"


@dataclasses.dataclass(frozen=True)
class OpsSettings:
    """Machine-level settings (``ops.local.toml``); see the run-operations doc."""

    keep_awake: bool = True
    power_monitor: bool = True
    poll_seconds: float = 1.0
    stall_seconds: float = 10.0
    min_free_gib: float = 15.0
    refuse_onedrive: bool = True
    require_gpu_idle: bool = True
    avoid_scheduled_tasks: tuple[str, ...] = ()
    seconds_per_step: float = 0.15


@dataclasses.dataclass(frozen=True)
class _OpsFile:
    ops: OpsSettings = dataclasses.field(default_factory=OpsSettings)


def load_ops_settings(path: str | Path | None = None) -> tuple[OpsSettings, str | None]:
    """Settings and the file they came from (None when defaults were used).

    The file has one ``[ops]`` table and is validated strictly: unknown keys
    or wrong types raise :class:`ConfigError`.
    """
    if path is None:
        env = os.environ.get(OPS_FILE_ENV, "").strip()
        path = Path(env) if env else DEFAULT_OPS_FILE
    path = Path(path)
    if not path.exists():
        return OpsSettings(), None
    return build_config(_OpsFile, read_config_file(path)).ops, str(path)


@dataclasses.dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    hard: bool
    detail: str


class PreflightError(RuntimeError):
    """One or more hard pre-start checks failed; the run must not start."""


# --------------------------------------------------------------------------
# Probes (replaceable in tests)


def onedrive_roots() -> list[Path]:
    roots = []
    for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        value = os.environ.get(name, "").strip()
        if value:
            roots.append(Path(value))
    return roots


def free_gib(path: Path) -> float:
    """Free space on the drive holding ``path`` (or its nearest existing ancestor)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 2**30


def gpu_compute_processes() -> int | None:
    """Number of GPU compute processes reported by nvidia-smi; None if it cannot be run."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return len([line for line in result.stdout.splitlines() if line.strip()])


def task_next_run(name: str) -> datetime | None | str:
    """Next start of a Windows scheduled task as an aware datetime.

    Returns None if the task has no next start, the string ``"missing"`` if
    it does not exist, and raises ``RuntimeError`` if it cannot be queried.
    """
    script = (
        "$ErrorActionPreference='Stop'; try { $i = Get-ScheduledTaskInfo -TaskName '" + name.replace("'", "''")
        + "' } catch { 'MISSING'; exit 0 }; if ($i.NextRunTime) { $i.NextRunTime.ToUniversalTime().ToString('o') } else { 'NONE' }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot query scheduled task {name!r}: {exc}") from exc
    out = result.stdout.strip()
    if result.returncode != 0 or not out:
        raise RuntimeError(f"cannot query scheduled task {name!r}: {result.stderr.strip()[:200]}")
    if out == "MISSING":
        return "missing"
    if out == "NONE":
        return None
    return datetime.fromisoformat(out.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


@dataclasses.dataclass
class Probes:
    onedrive_roots: Callable[[], list[Path]] = onedrive_roots
    free_gib: Callable[[Path], float] = free_gib
    gpu_compute_processes: Callable[[], int | None] = gpu_compute_processes
    task_next_run: Callable[[str], Any] = task_next_run
    power_status: Callable[[], dict | None] = power_status


# --------------------------------------------------------------------------
# Checks


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def run_preflight(
    settings: OpsSettings,
    runs_dir: Path,
    expected_seconds: float,
    now: datetime | None = None,
    probes: Probes | None = None,
) -> list[Check]:
    """Run every pre-start check and return the results; never raises for a failed check."""
    probes = probes or Probes()
    now = now or datetime.now(timezone.utc)
    runs_dir = Path(runs_dir)
    checks: list[Check] = []

    if settings.refuse_onedrive:
        roots = [r for r in probes.onedrive_roots() if _inside(runs_dir, r)]
        named = [p for p in runs_dir.resolve().parts if p.lower().startswith("onedrive")]
        inside = bool(roots or named)
        checks.append(Check("runs_dir_outside_onedrive", not inside, True,
                            f"{runs_dir} is inside OneDrive" if inside else f"{runs_dir} is outside OneDrive"))

    free = probes.free_gib(runs_dir)
    checks.append(Check("free_disk", free >= settings.min_free_gib, True,
                        f"{free:.1f} GiB free, minimum {settings.min_free_gib:.1f} GiB"))

    if settings.require_gpu_idle:
        busy = probes.gpu_compute_processes()
        if busy is None:
            checks.append(Check("gpu_idle", True, True, "not checked: nvidia-smi unavailable"))
        else:
            checks.append(Check("gpu_idle", busy == 0, True, f"{busy} other GPU compute process(es)"))

    latest_safe_start = now + timedelta(seconds=1.5 * expected_seconds + 300)
    for name in settings.avoid_scheduled_tasks:
        try:
            nxt = probes.task_next_run(name)
        except RuntimeError as exc:
            checks.append(Check(f"scheduled_task:{name}", False, True, str(exc)))
            continue
        if nxt == "missing":
            checks.append(Check(f"scheduled_task:{name}", True, True, "task does not exist"))
        elif nxt is None:
            checks.append(Check(f"scheduled_task:{name}", True, True, "no next start scheduled"))
        else:
            ok = nxt >= latest_safe_start
            checks.append(Check(
                f"scheduled_task:{name}", ok, True,
                f"next start {nxt.isoformat()}; run expected to need about {expected_seconds / 60:.0f} min"
                + ("" if ok else "; the run would overlap it"),
            ))

    status = probes.power_status()
    if status is not None:
        on_mains = status.get("ac_online") is not False
        checks.append(Check("on_mains", on_mains, False,
                            "on mains" if on_mains else f"on battery ({status.get('battery_percent')}%)"))
    return checks


def enforce(checks: list[Check]) -> None:
    """Raise :class:`PreflightError` naming every failed hard check."""
    failed = [c for c in checks if c.hard and not c.passed]
    if failed:
        raise PreflightError("pre-start checks failed: " + "; ".join(f"{c.name}: {c.detail}" for c in failed))


__all__ = [
    "Check", "ConfigError", "OpsSettings", "PreflightError", "Probes",
    "enforce", "load_ops_settings", "run_preflight",
]
