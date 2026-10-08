"""Run a queue of Chorus commands one after another (for a matrix of runs).

Usage, from the repository root (normally through scripts/ops/launch_visible.ps1)::

    python -m scripts.ops.run_queue configs/queues/<queue>.toml

The queue file lists entries::

    stop_on_failure = true   # optional, default true

    [[run]]
    module = "scripts.train_task_a_sliced"
    args = ["--config", "configs/task_a/sliced.toml", "--set", "run.seed=11"]

Each entry runs as ``python -m <module> <args>`` from the repository root, with
its output shown live. Every start and finish (with exit code and duration) is
appended to ``<runs folder>/queue_logs/<time>_<queue name>.jsonl``. Runs do
their own pre-start checks and power logging (cassandra_chorus.ops).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from cassandra_chorus.paths import REPO_ROOT, runs_root


def load_queue(path: Path) -> tuple[list[dict], bool]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    unknown = set(data) - {"run", "stop_on_failure"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    runs = data.get("run", [])
    for i, entry in enumerate(runs):
        if set(entry) - {"module", "args"} or not isinstance(entry.get("module"), str):
            raise ValueError(f"{path}: entry {i} needs 'module' (string) and optional 'args' (list of strings)")
        if not all(isinstance(a, str) for a in entry.get("args", [])):
            raise ValueError(f"{path}: entry {i} args must all be strings")
    return runs, bool(data.get("stop_on_failure", True))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    queue_path = Path(args[0])
    runs, stop_on_failure = load_queue(queue_path)
    log_dir = runs_root() / "queue_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{queue_path.stem}.jsonl"

    def record(**fields) -> None:
        fields = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(fields) + "\n")
        print(f"[queue] {json.dumps(fields)}", flush=True)

    record(event="queue_start", queue=str(queue_path), entries=len(runs))
    failures = 0
    for i, entry in enumerate(runs):
        command = [sys.executable, "-m", entry["module"], *entry.get("args", [])]
        record(event="start", index=i, command=command[1:])
        t0 = time.time()
        code = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        record(event="finish", index=i, exit_code=code, seconds=round(time.time() - t0, 1))
        if code != 0:
            failures += 1
            if stop_on_failure:
                record(event="queue_stopped", reason=f"entry {i} exited with {code}")
                return 1
    record(event="queue_complete", failures=failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
