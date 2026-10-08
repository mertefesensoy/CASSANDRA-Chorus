"""Run folders and the structured JSONL run log (SRS S0-F-24, S0-N-03, S0-N-08).

A run folder ``<runs_root>/<run_id>/`` holds:

``log.jsonl``
    One JSON object per line, each with ``type`` and ``time_utc``. Record
    types used so far: ``header`` (first, once), ``warning``,
    ``warnings_summary``, ``round``, ``final``, ``end`` (last, once).
``config.json``
    The resolved configuration, loadable again with
    :func:`cassandra_chorus.config.load_config`.
``uncommitted.patch``
    Present only if tracked files had uncommitted changes when the run
    started: ``git diff HEAD``, so the exact code can be rebuilt.

A run folder is never reused: creating one that exists is an error.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import subprocess
import threading
import warnings
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cassandra_chorus.config import config_hash, config_to_dict
from cassandra_chorus.paths import REPO_ROOT

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_RESERVED_FIELDS = ("type", "time_utc")
_MAX_UNTRACKED_LISTED = 200


def make_run_id(name: str, seed: int, now: datetime | None = None) -> str:
    """``YYYYMMDDTHHMMSSZ_<name>_s<seed>``, with the time in UTC."""
    if not _NAME_RE.match(name):
        raise ValueError(f"run name {name!r} may only contain letters, digits, '.', '_' and '-'")
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return f"{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}_{name}_s{seed}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def to_jsonable(obj: Any) -> Any:
    """Convert ``obj`` to values that standard JSON can represent.

    Non-finite floats become the strings ``"nan"``, ``"inf"`` and ``"-inf"``.
    NumPy scalars and one-element PyTorch tensors become Python numbers. Paths
    become strings. Anything else that is not JSON-native raises ``TypeError``
    rather than being silently stringified.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    # NumPy scalars (np.generic) and one-element tensors, detected without importing either library.
    if hasattr(obj, "item") and (getattr(obj, "ndim", None) == 0 or getattr(obj, "shape", None) == (1,)):
        return to_jsonable(obj.item())
    raise TypeError(f"cannot write {type(obj).__name__} to the run log")


def git_state(repo_dir: Path = REPO_ROOT) -> tuple[dict[str, Any], bytes]:
    """Describe the git checkout at ``repo_dir``.

    Returns ``(state, patch)``. ``state`` has ``available`` and, when git is
    usable, ``commit``, ``branch``, ``dirty`` (any tracked change or untracked
    file), ``tracked_changes`` (count) and ``untracked`` (paths, capped).
    ``patch`` is ``git diff HEAD --binary`` when tracked files changed, else
    empty. Never raises for a missing git or a folder outside a repository.
    """

    def git(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, timeout=60, check=False)

    def text(result: subprocess.CompletedProcess[bytes]) -> str:
        return result.stdout.decode("utf-8", errors="replace").strip()

    try:
        top = git("rev-parse", "--show-toplevel")
    except FileNotFoundError:
        return {"available": False, "reason": "git executable not found"}, b""
    if top.returncode != 0:
        return {"available": False, "reason": top.stderr.decode("utf-8", errors="replace").strip()}, b""

    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = text(git("status", "--porcelain=v1", "--untracked-files=all"))
    lines = [line for line in status.splitlines() if line]
    untracked = [line[3:] for line in lines if line.startswith("??")]
    tracked = [line for line in lines if not line.startswith("??")]
    patch = git("diff", "HEAD", "--binary").stdout if tracked and commit.returncode == 0 else b""
    state = {
        "available": True,
        "commit": text(commit) if commit.returncode == 0 else None,
        "branch": text(branch) if branch.returncode == 0 else None,
        "dirty": bool(lines),
        "tracked_changes": len(tracked),
        "untracked": untracked[:_MAX_UNTRACKED_LISTED],
        "untracked_count": len(untracked),
    }
    return state, patch


class RunLogger:
    """Append-only JSONL log for one run folder.

    Use :meth:`create`, not the constructor. As a context manager it writes the
    ``end`` record on exit: ``completed``, ``interrupted`` (Ctrl-C) or
    ``failed`` with the error message, and never swallows the exception.
    :meth:`write` is thread-safe, so a monitor thread can log alongside the
    training loop.
    """

    LOG_NAME = "log.jsonl"
    CONFIG_NAME = "config.json"
    PATCH_NAME = "uncommitted.patch"

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self._fh = (run_dir / self.LOG_NAME).open("x", encoding="utf-8", newline="\n")
        self._closed = False
        self._lock = threading.Lock()
        self._warning_counts: dict[tuple[str, str], int] = {}

    @classmethod
    def create(cls, runs_root: Path, run_id: str) -> RunLogger:
        """Create ``runs_root/run_id`` and its log. Raises ``FileExistsError`` if it exists."""
        run_dir = Path(runs_root) / run_id
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)
        return cls(run_dir, run_id)

    @property
    def log_path(self) -> Path:
        return self.run_dir / self.LOG_NAME

    def write(self, record_type: str, **fields: Any) -> dict[str, Any]:
        """Append one record and flush it to disk. Returns the record as written. Thread-safe."""
        clash = [name for name in _RESERVED_FIELDS if name in fields]
        if clash:
            raise ValueError(f"field names {clash} are reserved")
        with self._lock:
            if self._closed:
                raise RuntimeError("run log is closed")
            record = to_jsonable({"type": record_type, "time_utc": _utc_now_iso(), **fields})
            self._fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._fh.flush()
        return record

    def write_header(
        self,
        *,
        config: Any,
        config_path: str | Path,
        seeds: dict[str, int],
        environment: dict[str, Any],
        determinism: dict[str, Any],
        repo_dir: Path = REPO_ROOT,
    ) -> dict[str, Any]:
        """Write ``config.json``, any ``uncommitted.patch``, and the header record."""
        resolved = config_to_dict(config)
        (self.run_dir / self.CONFIG_NAME).write_text(
            json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        git, patch = git_state(repo_dir)
        if patch:
            (self.run_dir / self.PATCH_NAME).write_bytes(patch)
        git["patch_file"] = self.PATCH_NAME if patch else None
        return self.write(
            "header",
            run_id=self.run_id,
            schema=type(config).__qualname__,
            config_path=str(config_path),
            config_hash=config_hash(config),
            config=resolved,
            seeds=seeds,
            environment=environment,
            determinism=determinism,
            git=git,
        )

    @contextlib.contextmanager
    def capture_warnings(self) -> Iterator[None]:
        """Record each distinct warning once in the log, and still show it.

        Inside the block every warning is seen (filter ``always``); the first
        occurrence of each (category, message) pair is written as a ``warning``
        record and shown on the console, later repeats are only counted. On
        exit a ``warnings_summary`` record gives the count of each, and is
        written even when there were none.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            original_show = warnings.showwarning

            def show(message, category, filename, lineno, file=None, line=None):
                key = (category.__name__, str(message))
                seen = self._warning_counts.get(key, 0)
                self._warning_counts[key] = seen + 1
                if seen == 0:
                    self.write(
                        "warning",
                        category=category.__name__,
                        message=str(message),
                        filename=str(filename),
                        lineno=lineno,
                    )
                    original_show(message, category, filename, lineno, file, line)

            warnings.showwarning = show
            try:
                yield
            finally:
                if not self._closed:
                    self.write(
                        "warnings_summary",
                        counts=[
                            {"category": c, "message": m, "count": n}
                            for (c, m), n in self._warning_counts.items()
                        ],
                    )

    def close(self, status: str = "completed", **fields: Any) -> None:
        """Write the ``end`` record and close the file, as one locked step. Idempotent.

        Any later :meth:`write` (for example from a monitor thread that was not
        stopped) raises ``RuntimeError`` instead of appending after ``end``.
        """
        with self._lock:
            if self._closed:
                return
            record = to_jsonable({"type": "end", "time_utc": _utc_now_iso(), "status": status, **fields})
            self._fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._fh.flush()
            self._fh.close()
            self._closed = True

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close("completed")
        elif issubclass(exc_type, KeyboardInterrupt):
            self.close("interrupted")
        else:
            self.close("failed", error=f"{exc_type.__name__}: {exc}")
        return False
