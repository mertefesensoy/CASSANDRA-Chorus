"""Single-machine simulation of N workers (PLAN step 6; SRS S0-F-04, S0-F-11, S0-F-12, S0-N-04).

The simulation may use the coordinator; the coordinator must never use the simulation.
"""

from cassandra_chorus.sim.harness import (
    CHECKPOINT_NAME,
    SimSection,
    data_generator,
    dropped_workers,
    load_checkpoint,
    make_outer,
    round_slices,
    save_checkpoint,
    skew_weights,
    validate_sim,
    worker_specs,
)
from cassandra_chorus.sim.worker import WorkerRound, build_worker_model, returned_state, train_worker

__all__ = [
    "CHECKPOINT_NAME",
    "SimSection",
    "WorkerRound",
    "build_worker_model",
    "data_generator",
    "dropped_workers",
    "load_checkpoint",
    "make_outer",
    "returned_state",
    "round_slices",
    "save_checkpoint",
    "skew_weights",
    "train_worker",
    "validate_sim",
    "worker_specs",
]
