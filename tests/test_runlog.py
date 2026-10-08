from __future__ import annotations

import dataclasses
import json
import subprocess
import warnings
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from cassandra_chorus.config import RunSection, config_hash, load_config
from cassandra_chorus.paths import REPO_ROOT
from cassandra_chorus.runlog import RunLogger, git_state, make_run_id, to_jsonable


@dataclasses.dataclass(frozen=True)
class Cfg:
    run: RunSection


def read_log(logger: RunLogger) -> list[dict]:
    return [json.loads(line) for line in logger.log_path.read_text(encoding="utf-8").splitlines()]


def git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        cwd=cwd, check=True, capture_output=True,
    )


def test_make_run_id():
    now = datetime(2026, 10, 7, 1, 2, 3, tzinfo=timezone.utc)
    assert make_run_id("smoke", 7, now) == "20261007T010203Z_smoke_s7"
    with pytest.raises(ValueError, match="may only contain"):
        make_run_id("bad name", 7, now)
    with pytest.raises(ValueError, match="timezone-aware"):
        make_run_id("smoke", 7, datetime(2026, 10, 7))


def test_create_never_overwrites(tmp_path):
    RunLogger.create(tmp_path, "r1").close()
    with pytest.raises(FileExistsError):
        RunLogger.create(tmp_path, "r1")


def test_records_are_json_lines_with_type_and_time(tmp_path):
    with RunLogger.create(tmp_path, "r") as log:
        log.write("round", round=1, loss=0.25)
        log.write("round", round=2, loss=float("nan"), big=float("inf"), small=float("-inf"))
        log.write("round", round=3, np_value=np.float32(0.5), torch_value=torch.tensor(2.0))
    records = read_log(log)
    assert [r["type"] for r in records] == ["round", "round", "round", "end"]
    assert all(r["time_utc"].endswith("Z") for r in records)
    assert records[1]["loss"] == "nan" and records[1]["big"] == "inf" and records[1]["small"] == "-inf"
    assert records[2]["np_value"] == 0.5 and records[2]["torch_value"] == 2.0
    assert records[-1]["status"] == "completed"


def test_reserved_fields_and_unsupported_values(tmp_path):
    with RunLogger.create(tmp_path, "r") as log:
        with pytest.raises(ValueError, match="reserved"):
            log.write("round", type="x")
        with pytest.raises(TypeError, match="cannot write"):
            log.write("round", value=object())
    with pytest.raises(TypeError):
        to_jsonable(torch.zeros(3))  # only one-element tensors are converted


def test_write_after_close_raises(tmp_path):
    log = RunLogger.create(tmp_path, "r")
    log.close()
    log.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        log.write("round")


def test_failure_and_interrupt_are_recorded(tmp_path):
    with pytest.raises(ZeroDivisionError):
        with RunLogger.create(tmp_path, "failed") as log:
            1 / 0
    end = read_log(log)[-1]
    assert end["status"] == "failed" and end["error"] == "ZeroDivisionError: division by zero"
    with pytest.raises(KeyboardInterrupt):
        with RunLogger.create(tmp_path, "interrupted") as log2:
            raise KeyboardInterrupt
    assert read_log(log2)[-1]["status"] == "interrupted"


def test_capture_warnings_records_each_distinct_warning_once(tmp_path):
    with RunLogger.create(tmp_path, "r") as log:
        with log.capture_warnings():
            for _ in range(3):
                warnings.warn("same message", UserWarning)
            warnings.warn("other message", RuntimeWarning)
    records = read_log(log)
    logged = [(r["category"], r["message"]) for r in records if r["type"] == "warning"]
    assert logged == [("UserWarning", "same message"), ("RuntimeWarning", "other message")]
    (summary,) = [r for r in records if r["type"] == "warnings_summary"]
    assert {(c["message"], c["count"]) for c in summary["counts"]} == {("same message", 3), ("other message", 1)}


def test_capture_warnings_writes_empty_summary(tmp_path):
    with RunLogger.create(tmp_path, "r") as log:
        with log.capture_warnings():
            pass
    (summary,) = [r for r in read_log(log) if r["type"] == "warnings_summary"]
    assert summary["counts"] == []


def test_header_and_resolved_config(tmp_path):
    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text('[run]\nname = "t"\nseed = 11\n', encoding="utf-8")
    cfg = load_config(cfg_path, Cfg)
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    with RunLogger.create(tmp_path / "runs", "r") as log:
        log.write_header(
            config=cfg, config_path=cfg_path, seeds={"torch": 11},
            environment={"python": "x"}, determinism={"mode": "warn"}, repo_dir=not_a_repo,
        )
    (header, end) = read_log(log)
    assert header["type"] == "header" and end["type"] == "end"
    assert header["run_id"] == "r" and header["schema"] == "Cfg"
    assert header["config_hash"] == config_hash(cfg)
    assert header["config"]["run"]["seed"] == 11
    assert header["git"]["available"] is False and header["git"]["patch_file"] is None
    # The saved resolved configuration reloads to the same configuration.
    assert load_config(log.run_dir / "config.json", Cfg) == cfg


def test_git_state_of_this_repository():
    state, _ = git_state(REPO_ROOT)
    assert state["available"] is True
    assert len(state["commit"]) == 40 and state["branch"]


def test_uncommitted_changes_are_saved_as_a_patch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-q", "-m", "init")
    state, patch = git_state(repo)
    assert state["dirty"] is False and patch == b""

    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    (repo / "new.txt").write_text("x\n", encoding="utf-8")
    state, patch = git_state(repo)
    assert state["dirty"] is True and state["tracked_changes"] == 1 and state["untracked"] == ["new.txt"]
    assert b"-one" in patch and b"+two" in patch

    cfg = Cfg(run=RunSection(name="t"))
    with RunLogger.create(tmp_path / "runs", "r") as log:
        header = log.write_header(
            config=cfg, config_path="c.toml", seeds={}, environment={}, determinism={}, repo_dir=repo,
        )
    assert header["git"]["patch_file"] == "uncommitted.patch"
    assert (log.run_dir / "uncommitted.patch").read_bytes() == patch
