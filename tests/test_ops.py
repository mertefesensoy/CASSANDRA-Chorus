from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cassandra_chorus.config import ConfigError
from cassandra_chorus.ops import (
    OpsSettings,
    PowerMonitor,
    PreflightError,
    Probes,
    enforce,
    keep_awake,
    load_ops_settings,
    operations,
    power_status,
    run_preflight,
    windows_power_events,
)
from cassandra_chorus.ops.power import parse_kernel_power_events, summarize_events
from cassandra_chorus.paths import REPO_ROOT
from cassandra_chorus.runlog import RunLogger

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only API")
NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)


def fake_probes(**overrides) -> Probes:
    base = dict(
        onedrive_roots=lambda: [Path("C:/Users/x/OneDrive")],
        free_gib=lambda path: 100.0,
        gpu_compute_processes=lambda: 0,
        task_next_run=lambda name: NOW + timedelta(hours=10),
        power_status=lambda: {"ac_online": True, "battery_percent": 90, "battery_saver": False},
    )
    base.update(overrides)
    return Probes(**base)


def by_name(checks):
    return {c.name: c for c in checks}


# ---------------------------------------------------------------- settings


def test_settings_default_when_file_absent(tmp_path):
    settings, source = load_ops_settings(tmp_path / "missing.toml")
    assert settings == OpsSettings() and source is None


def test_settings_file_is_validated(tmp_path):
    good = tmp_path / "ops.toml"
    good.write_text('[ops]\navoid_scheduled_tasks = ["A", "B"]\nmin_free_gib = 5\n', encoding="utf-8")
    settings, source = load_ops_settings(good)
    assert settings.avoid_scheduled_tasks == ("A", "B") and settings.min_free_gib == 5.0 and source == str(good)
    bad = tmp_path / "bad.toml"
    bad.write_text("[ops]\nkeep_awak = true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="ops.keep_awak: unknown key"):
        load_ops_settings(bad)


def test_settings_file_from_environment(tmp_path, monkeypatch):
    path = tmp_path / "env.toml"
    path.write_text("[ops]\nrequire_gpu_idle = false\n", encoding="utf-8")
    monkeypatch.setenv("CHORUS_OPS_FILE", str(path))
    assert load_ops_settings()[0].require_gpu_idle is False


def test_example_file_matches_the_defaults():
    assert load_ops_settings(REPO_ROOT / "ops.example.toml")[0] == OpsSettings()


# ---------------------------------------------------------------- preflight


def test_all_checks_pass():
    checks = run_preflight(OpsSettings(avoid_scheduled_tasks=("T",)), Path("C:/runs"), 600, NOW, fake_probes())
    assert all(c.passed for c in checks)
    assert set(by_name(checks)) == {"runs_dir_outside_onedrive", "free_disk", "gpu_idle", "scheduled_task:T", "on_mains"}
    enforce(checks)  # does not raise


@pytest.mark.parametrize(
    ("runs_dir", "roots"),
    [("C:/Users/x/OneDrive/proj/runs", [Path("C:/Users/x/OneDrive")]), ("D:/OneDrive - Org/runs", [])],
)
def test_onedrive_runs_folder_is_refused(runs_dir, roots):
    checks = run_preflight(OpsSettings(), Path(runs_dir), 60, NOW, fake_probes(onedrive_roots=lambda: roots))
    assert by_name(checks)["runs_dir_outside_onedrive"].passed is False
    with pytest.raises(PreflightError, match="runs_dir_outside_onedrive"):
        enforce(checks)


def test_low_disk_and_busy_gpu_are_refused():
    checks = run_preflight(OpsSettings(), Path("C:/runs"), 60, NOW,
                           fake_probes(free_gib=lambda p: 3.0, gpu_compute_processes=lambda: 2))
    failed = {c.name for c in checks if not c.passed}
    assert failed == {"free_disk", "gpu_idle"}
    with pytest.raises(PreflightError) as exc:
        enforce(checks)
    assert "free_disk" in str(exc.value) and "gpu_idle" in str(exc.value)


def test_gpu_check_without_nvidia_smi_is_not_a_failure():
    checks = run_preflight(OpsSettings(), Path("C:/runs"), 60, NOW, fake_probes(gpu_compute_processes=lambda: None))
    assert by_name(checks)["gpu_idle"].passed and "not checked" in by_name(checks)["gpu_idle"].detail


def test_scheduled_task_overlap():
    settings = OpsSettings(avoid_scheduled_tasks=("Nightly",))
    # Expected 20 minutes -> refused if the task starts within 1.5 x 20 min + 5 min = 35 min.
    soon = run_preflight(settings, Path("C:/runs"), 1200, NOW, fake_probes(task_next_run=lambda n: NOW + timedelta(minutes=34)))
    later = run_preflight(settings, Path("C:/runs"), 1200, NOW, fake_probes(task_next_run=lambda n: NOW + timedelta(minutes=36)))
    assert by_name(soon)["scheduled_task:Nightly"].passed is False
    assert by_name(later)["scheduled_task:Nightly"].passed is True
    missing = run_preflight(settings, Path("C:/runs"), 1200, NOW, fake_probes(task_next_run=lambda n: "missing"))
    assert by_name(missing)["scheduled_task:Nightly"].passed is True


def test_unqueryable_task_is_a_hard_failure():
    def broken(name):
        raise RuntimeError("cannot query")

    checks = run_preflight(OpsSettings(avoid_scheduled_tasks=("X",)), Path("C:/runs"), 60, NOW, fake_probes(task_next_run=broken))
    assert by_name(checks)["scheduled_task:X"].passed is False


def test_on_battery_only_warns():
    checks = run_preflight(OpsSettings(), Path("C:/runs"), 60, NOW,
                           fake_probes(power_status=lambda: {"ac_online": False, "battery_percent": 55, "battery_saver": False}))
    on_mains = by_name(checks)["on_mains"]
    assert on_mains.passed is False and on_mains.hard is False and "55%" in on_mains.detail
    enforce(checks)  # soft failures never refuse


# ---------------------------------------------------------------- power monitor


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_monitor_reports_mains_changes_and_stalls():
    events = []
    states = iter([True, True, False, True, True])
    clock = FakeClock()
    monitor = PowerMonitor(lambda kind, f: events.append((kind, f)), poll_seconds=1.0, stall_seconds=10.0,
                           status_fn=lambda: {"ac_online": next(states), "battery_percent": 80, "battery_saver": False},
                           wall=clock)
    for gap in (0, 1, 1, 1, 25):  # the last wait took 25 s: a stall
        clock.t += gap
        monitor.poll_once()
    kinds = [k for k, _ in events]
    assert kinds == ["power", "power", "power", "stall"]
    assert [f["ac_online"] for k, f in events if k == "power"] == [True, False, True]
    (stall,) = [f for k, f in events if k == "stall"]
    assert stall["seconds"] == 25.0
    summary = monitor.summary()
    assert summary["ac_changes"] == 2 and summary["stalls"] == 1 and summary["longest_stall_seconds"] == 25.0
    assert summary["polls"] == 5


def test_monitor_without_power_status_reports_unknown_once():
    events = []
    monitor = PowerMonitor(lambda k, f: events.append((k, f)), status_fn=lambda: None, wall=FakeClock())
    monitor.poll_once()
    monitor.poll_once()
    assert events == [("power", {"source": "GetSystemPowerStatus", "ac_online": None})]


def test_monitor_thread_starts_and_stops():
    events = []
    monitor = PowerMonitor(lambda k, f: events.append(k), poll_seconds=0.01,
                           status_fn=lambda: {"ac_online": True, "battery_percent": 1, "battery_saver": False})
    with monitor:
        time.sleep(0.2)
    assert monitor.polls >= 3 and events[0] == "power"
    assert not any(t.name == "chorus-power-monitor" for t in threading.enumerate())


# ---------------------------------------------------------------- Windows APIs (live, read-only)


@windows_only
def test_keep_awake_is_held_and_released():
    with keep_awake(True) as status:
        assert status["held"] is True and status["previous_state"] is not None
    assert status.get("released") is True


def test_keep_awake_disabled():
    with keep_awake(False) as status:
        assert status["held"] is False and "disabled" in status["reason"]


@windows_only
def test_power_status_reads():
    status = power_status()
    assert status is not None and status["ac_online"] in (True, False, None)


@windows_only
def test_windows_power_events_query_runs():
    now = datetime.now(timezone.utc)
    result = windows_power_events(now - timedelta(minutes=5), now)
    assert result["available"] is True and result["ac_offline"] >= 0


def test_parse_kernel_power_events():
    ns = 'xmlns="http://schemas.microsoft.com/win/2004/08/events/event"'
    xml = (
        "<Events>"
        f"<Event {ns}><System><EventID>105</EventID><TimeCreated SystemTime='2026-10-08T03:41:27Z'/></System>"
        "<EventData><Data Name='AcOnline'>false</Data><Data Name='RemainingCapacity'>64424</Data></EventData></Event>"
        f"<Event {ns}><System><EventID>105</EventID><TimeCreated SystemTime='2026-10-08T03:41:29Z'/></System>"
        "<EventData><Data Name='AcOnline'>true</Data></EventData></Event>"
        f"<Event {ns}><System><EventID>506</EventID><TimeCreated SystemTime='2026-10-08T03:41:35Z'/></System></Event>"
        "</Events>"
    )
    events = parse_kernel_power_events(xml)
    assert [e["kind"] for e in events] == ["power_source_change", "power_source_change", "standby_enter"]
    summary = summarize_events(events)
    assert (summary["ac_offline"], summary["ac_online"], summary["standby_enter"], summary["standby_exit"]) == (1, 1, 1, 0)


# ---------------------------------------------------------------- operations context and logger thread-safety


def test_operations_writes_ops_and_summary_records(tmp_path):
    settings = OpsSettings(keep_awake=False, poll_seconds=0.01)
    checks = run_preflight(settings, tmp_path, 60, NOW, fake_probes())
    with RunLogger.create(tmp_path, "r") as log:
        with operations(log, settings, None, checks):
            time.sleep(0.1)
    records = [json.loads(line) for line in log.log_path.read_text(encoding="utf-8").splitlines()]
    types = [r["type"] for r in records]
    assert types[0] == "ops" and types[-2] == "power_summary" and types[-1] == "end"
    assert records[0]["keep_awake"]["held"] is False and len(records[0]["preflight"]) == len(checks)
    summary = records[-2]
    assert summary["monitor"]["polls"] >= 3 and "windows_events" in summary


def test_run_logger_is_thread_safe(tmp_path):
    with RunLogger.create(tmp_path, "r") as log:
        def burst(tag):
            for i in range(300):
                log.write("x", tag=tag, i=i)
        threads = [threading.Thread(target=burst, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    lines = log.log_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]  # every line is complete JSON
    assert sum(1 for r in records if r["type"] == "x") == 1200 and records[-1]["type"] == "end"


def test_write_after_close_raises(tmp_path):
    log = RunLogger.create(tmp_path, "r")
    log.close()
    with pytest.raises(RuntimeError, match="closed"):
        log.write("late")
