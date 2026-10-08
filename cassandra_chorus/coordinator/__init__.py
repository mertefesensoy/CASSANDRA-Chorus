"""Slice assignment and merge (PLAN step 5; SRS S0-F-05 to S0-F-10, S0-F-25).

Stage 1 reuses this package unchanged in the coordinator service for real
workers, so it must not depend on the single-machine simulation: nothing here
may import ``cassandra_chorus.sim`` (SRS S0-N-05, enforced by
``tests/test_layering.py``).
"""

from cassandra_chorus.coordinator.assignment import Slice, WorkerSpec, assign_slices, holders
from cassandra_chorus.coordinator.merge import MergeReport, NesterovOuter, PlainAverage, WorkerResult, merge

__all__ = [
    "MergeReport",
    "NesterovOuter",
    "PlainAverage",
    "Slice",
    "WorkerResult",
    "WorkerSpec",
    "assign_slices",
    "holders",
    "merge",
]
