"""End-to-end smoke runs in separate processes: same seed, same checksum.

Each run is a fresh Python process, so this checks reproducibility across
processes, which is what re-running an experiment means. The CPU variant runs
anywhere; the CUDA variant is skipped without a GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cassandra_chorus.paths import REPO_ROOT, RUNS_DIR_ENV

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def run_smoke(runs_dir: Path, *overrides: str) -> list[dict]:
    """Run the smoke entry point once; return the records of the run it created."""
    before = set(runs_dir.iterdir()) if runs_dir.exists() else set()
    args = [sys.executable, "-m", "scripts.smoke", "--config", "configs/smoke.toml"]
    for override in overrides:
        args += ["--set", override]
    env = {**os.environ, RUNS_DIR_ENV: str(runs_dir)}
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)  # let the entry point set it, as in real use
    result = subprocess.run(args, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    (run_dir,) = set(runs_dir.iterdir()) - before
    return [json.loads(line) for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()]


def checksum(records: list[dict]) -> str:
    (final,) = [r for r in records if r["type"] == "final"]
    return final["metrics"]["smoke_checksum"]


def check_log_structure(records: list[dict], device: str) -> None:
    types = [r["type"] for r in records]
    assert types[0] == "header" and types[-1] == "end" and records[-1]["status"] == "completed"
    assert "warnings_summary" in types
    header = records[0]
    assert header["determinism"]["mode"] == "warn"
    assert header["determinism"]["cublas_workspace_config"] == ":4096:8"
    assert header["environment"]["torch"] == torch.__version__
    assert header["git"]["available"] is True
    (final,) = [r for r in records if r["type"] == "final"]
    assert final["device"] == device


@pytest.mark.parametrize(
    "device",
    [pytest.param("cuda", marks=[requires_cuda, pytest.mark.gpu]), "cpu"],
)
def test_same_seed_same_checksum_across_processes(tmp_path, device):
    device_override = f'run.device="{device}"'
    # Distinct names keep run IDs distinct even within one second; the name
    # does not enter the computation, so the checksums are still comparable.
    first = run_smoke(tmp_path, device_override, "run.name=smoke-a")
    second = run_smoke(tmp_path, device_override, "run.name=smoke-b")
    other_seed = run_smoke(tmp_path, device_override, "run.name=smoke-c", "run.seed=11")
    for records in (first, second, other_seed):
        check_log_structure(records, device)
    assert first[0]["seeds"]["torch"] == 7 and other_seed[0]["seeds"]["torch"] == 11
    assert checksum(first) == checksum(second)
    assert checksum(first) != checksum(other_seed)
    if device == "cuda":
        assert first[0]["environment"]["gpu"]["name"]
