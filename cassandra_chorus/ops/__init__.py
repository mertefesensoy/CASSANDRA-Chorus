"""Run operations: keep-awake, power and stall logging, pre-start checks.

Integrates the parent CASSANDRA project's run protocol (ADR 0012 and its
keep-awake helper) natively, plus lessons from the 2026-10-07/08 matrix.
Also intended for Stage 1 worker clients. See
docs/implementations/2026-10-08-run-operations.md.
"""

from cassandra_chorus.ops.power import PowerMonitor, keep_awake, power_status, windows_power_events
from cassandra_chorus.ops.preflight import (
    Check,
    OpsSettings,
    PreflightError,
    Probes,
    enforce,
    load_ops_settings,
    run_preflight,
)
from cassandra_chorus.ops.session import operations

__all__ = [
    "Check",
    "OpsSettings",
    "PowerMonitor",
    "PreflightError",
    "Probes",
    "enforce",
    "keep_awake",
    "load_ops_settings",
    "operations",
    "power_status",
    "run_preflight",
    "windows_power_events",
]
