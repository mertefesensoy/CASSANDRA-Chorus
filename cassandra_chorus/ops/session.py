"""One context manager that wraps a run in the operations protocol.

Usage in an entry point, after the run log header is written::

    settings, source = load_ops_settings()
    checks = run_preflight(settings, runs_dir, expected_seconds)
    enforce(checks)                       # before the run folder exists
    with RunLogger.create(...) as log:
        log.write_header(...)
        with operations(log, settings, source, checks):
            ...  # training

Writes an ``ops`` record (settings, their source, preflight results, keep-awake
status), ``power`` and ``stall`` records while the run lasts, and a
``power_summary`` record at the end with the monitor's counts and the Windows
Kernel-Power events for the run's time window. Must be entered from the main
thread, which holds the keep-awake request. Never stops the run.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from cassandra_chorus.ops.power import PowerMonitor, keep_awake, windows_power_events
from cassandra_chorus.ops.preflight import Check, OpsSettings


@contextlib.contextmanager
def operations(log: Any, settings: OpsSettings, source: str | None, checks: list[Check]) -> Iterator[None]:
    start = datetime.now(timezone.utc)
    with keep_awake(settings.keep_awake) as awake:
        log.write(
            "ops",
            settings=dataclasses.asdict(settings),
            settings_source=source,
            preflight=[dataclasses.asdict(c) for c in checks],
            keep_awake=dict(awake),
        )
        monitor = None
        if settings.power_monitor:

            def on_event(kind: str, fields: dict[str, Any]) -> None:
                try:
                    log.write(kind, **fields)
                except RuntimeError:  # the log was already closed
                    pass

            monitor = PowerMonitor(on_event, settings.poll_seconds, settings.stall_seconds).start()
        try:
            yield
        finally:
            if monitor is not None:
                monitor.stop()
            end = datetime.now(timezone.utc)
            log.write(
                "power_summary",
                window_utc=[start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")],
                keep_awake=dict(awake),
                monitor=monitor.summary() if monitor is not None else None,
                windows_events=windows_power_events(start, end) if settings.power_monitor else None,
            )
