"""Keep-awake, power-state polling, stall detection and Windows power events.

Lessons integrated from the parent CASSANDRA project (keep_awake_while_pid.ps1)
and from the 2026-10-07/08 Chorus matrix, where mains drop-outs sent the
laptop into Modern Standby and froze training for hours without any trace in
the run log. See docs/implementations/2026-10-08-run-operations.md.

Everything here degrades to a recorded "not available" on systems other than
Windows; nothing here ever stops a run (owner decision 2026-10-08).
"""

from __future__ import annotations

import contextlib
import ctypes
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

IS_WINDOWS = sys.platform == "win32"


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Keep-awake


@contextlib.contextmanager
def keep_awake(enabled: bool = True) -> Iterator[dict[str, Any]]:
    """Ask Windows to keep the system and display on while the block runs.

    Call it from the thread that lives for the whole run (the main thread):
    the request belongs to that thread. It prevents entering sleep or standby
    through idle timeouts; it cannot bring a machine out of standby, so it must
    be held from launch with the screen on. Yields a status dictionary for the
    run log: ``requested``, ``held``, ``previous_state`` and ``reason``.
    """
    status: dict[str, Any] = {"requested": enabled, "held": False, "previous_state": None, "reason": None}
    if not enabled:
        status["reason"] = "disabled in the ops settings"
        yield status
        return
    if not IS_WINDOWS:
        status["reason"] = f"not supported on {sys.platform}"
        yield status
        return
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state.argtypes = [ctypes.c_uint32]
    set_state.restype = ctypes.c_uint32
    previous = set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    status["previous_state"] = hex(previous)
    status["held"] = previous != 0
    if previous == 0:
        status["reason"] = "SetThreadExecutionState returned 0"
    try:
        yield status
    finally:
        if status["held"]:
            set_state(ES_CONTINUOUS)
            status["released"] = True


# --------------------------------------------------------------------------
# Power status


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_uint32),
        ("BatteryFullLifeTime", ctypes.c_uint32),
    ]


def power_status() -> dict[str, Any] | None:
    """Mains and battery state from ``GetSystemPowerStatus``; None if unavailable.

    ``ac_online`` is True, False or None (unknown); ``battery_percent`` is
    0 to 100 or None; ``battery_saver`` is the battery-saver flag.
    """
    if not IS_WINDOWS:
        return None
    status = _SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    ac = {0: False, 1: True}.get(status.ACLineStatus)
    percent = None if status.BatteryLifePercent == 255 else int(status.BatteryLifePercent)
    return {"ac_online": ac, "battery_percent": percent, "battery_saver": bool(status.SystemStatusFlag & 1)}


# --------------------------------------------------------------------------
# Monitor


class PowerMonitor:
    """Background thread logging mains changes and stalls of this process.

    Each poll reads the power status and reports a ``power`` event when the
    mains state differs from the previous poll (the first poll always
    reports). It also measures how long the previous wait really took on the
    wall clock; more than ``stall_seconds`` means the process was frozen (for
    example in Modern Standby) and is reported as a ``stall`` event.

    ``on_event(kind, fields)`` must be thread-safe. One-second polling can miss
    drop-outs shorter than the interval; the Windows event log is the
    authoritative count (see :func:`windows_power_events`).
    """

    def __init__(
        self,
        on_event: Callable[[str, dict[str, Any]], Any],
        poll_seconds: float = 1.0,
        stall_seconds: float = 10.0,
        status_fn: Callable[[], dict[str, Any] | None] = power_status,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.on_event = on_event
        self.poll_seconds = poll_seconds
        self.stall_seconds = stall_seconds
        self.status_fn = status_fn
        self.wall = wall
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_ac: Any = "unset"
        self._last_wake: float | None = None
        self.polls = 0
        self.ac_changes = 0
        self.stalls = 0
        self.longest_stall = 0.0
        self.total_stall = 0.0

    def poll_once(self) -> None:
        """One poll: compare the mains state, then check the gap since the last poll."""
        now = self.wall()
        if self._last_wake is not None:
            gap = now - self._last_wake
            if gap > self.stall_seconds:
                self.stalls += 1
                self.total_stall += gap
                self.longest_stall = max(self.longest_stall, gap)
                self.on_event("stall", {"seconds": round(gap, 1), "from_utc": _utc(self._last_wake), "to_utc": _utc(now)})
        status = self.status_fn()
        ac = None if status is None else status["ac_online"]
        if ac != self._last_ac:
            if self._last_ac != "unset":
                self.ac_changes += 1
            self.on_event("power", {"source": "GetSystemPowerStatus", **(status or {"ac_online": None})})
            self._last_ac = ac
        self.polls += 1
        self._last_wake = now

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.poll_seconds)

    def start(self) -> PowerMonitor:
        self._thread = threading.Thread(target=self._run, name="chorus-power-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def summary(self) -> dict[str, Any]:
        return {
            "polls": self.polls,
            "poll_seconds": self.poll_seconds,
            "stall_threshold_seconds": self.stall_seconds,
            "ac_changes": self.ac_changes,
            "stalls": self.stalls,
            "longest_stall_seconds": round(self.longest_stall, 1),
            "total_stall_seconds": round(self.total_stall, 1),
            "note": "1 s polling can miss mains drop-outs shorter than the interval; Windows events are authoritative",
        }

    def __enter__(self) -> PowerMonitor:
        return self.start()

    def __exit__(self, *exc: Any) -> bool:
        self.stop()
        return False


# --------------------------------------------------------------------------
# Windows event log (authoritative record)

_KERNEL_POWER = "Microsoft-Windows-Kernel-Power"
_EVENT_NAMES = {"105": "power_source_change", "506": "standby_enter", "507": "standby_exit"}


def parse_kernel_power_events(xml_text: str) -> list[dict[str, Any]]:
    """Parse ``wevtutil qe /f:xml`` output (wrapped in one root) into event dictionaries."""
    ns = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}
    events = []
    for event in ET.fromstring(xml_text).findall("e:Event", ns):
        system = event.find("e:System", ns)
        event_id = system.find("e:EventID", ns).text
        when = system.find("e:TimeCreated", ns).get("SystemTime")
        record: dict[str, Any] = {"time_utc": when, "event_id": int(event_id), "kind": _EVENT_NAMES.get(event_id, event_id)}
        if event_id == "105":
            for data in event.findall("e:EventData/e:Data", ns):
                if data.get("Name") == "AcOnline":
                    record["ac_online"] = data.text == "true"
        events.append(record)
    return events


def summarize_events(events: list[dict[str, Any]], keep: int = 50) -> dict[str, Any]:
    return {
        "available": True,
        "ac_offline": sum(1 for e in events if e["kind"] == "power_source_change" and e.get("ac_online") is False),
        "ac_online": sum(1 for e in events if e["kind"] == "power_source_change" and e.get("ac_online") is True),
        "standby_enter": sum(1 for e in events if e["kind"] == "standby_enter"),
        "standby_exit": sum(1 for e in events if e["kind"] == "standby_exit"),
        "events": events[:keep],
        "events_truncated": len(events) > keep,
    }


def windows_power_events(start: datetime, end: datetime) -> dict[str, Any]:
    """Kernel-Power events 105, 506 and 507 between ``start`` and ``end`` (aware datetimes)."""
    if not IS_WINDOWS:
        return {"available": False, "reason": f"not supported on {sys.platform}"}

    def stamp(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    query = (
        f"*[System[Provider[@Name='{_KERNEL_POWER}'] and (EventID=105 or EventID=506 or EventID=507) "
        f"and TimeCreated[@SystemTime>='{stamp(start)}' and @SystemTime<='{stamp(end)}']]]"
    )
    try:
        result = subprocess.run(
            ["wevtutil", "qe", "System", f"/q:{query}", "/f:xml", "/e:Events"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if result.returncode != 0:
        return {"available": False, "reason": result.stderr.strip()[:300]}
    try:
        return summarize_events(parse_kernel_power_events(result.stdout))
    except ET.ParseError as exc:
        return {"available": False, "reason": f"could not parse wevtutil output: {exc}"}
