"""Seeding, GPU determinism and environment capture (SRS S0-N-03).

Determinism modes (owner decision 2026-10-07: default ``warn``):

``strict``
    Deterministic kernels everywhere; an operation that has none raises an
    error and stops the run.
``warn``
    Deterministic kernels wherever they exist; an operation without one emits
    a warning, which :meth:`cassandra_chorus.runlog.RunLogger.capture_warnings`
    records in the run log. The log therefore states where bit-exact
    reproduction is not guaranteed.
``off``
    Seeded, but no deterministic kernels requested. Fastest; same-seed runs
    may differ slightly on a GPU.

Call order in an entry point: :func:`prepare_process` before any CUDA work,
then :func:`seed_everything` and :func:`apply_determinism`.
"""

from __future__ import annotations

import os
import platform
import random
from typing import Any

import numpy as np
import torch

MODES = ("strict", "warn", "off")
CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_DETERMINISTIC_VALUE = ":4096:8"
MAX_SEED = 2**32 - 1  # NumPy's global generator accepts seeds up to this value

# Environment variables that affect reproducibility and are recorded in logs.
_RECORDED_ENV = ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"determinism mode must be one of {MODES}, got {mode!r}")


def prepare_process(mode: str) -> None:
    """Set process-wide settings that must precede any CUDA work.

    For ``warn`` and ``strict``, sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
    (unless it is already set), which cuBLAS needs for deterministic matrix
    multiplication and reads when it is first used. Raises ``RuntimeError`` if
    CUDA is already initialized, because the setting might then have no effect.
    """
    _check_mode(mode)
    if mode == "off":
        return
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "prepare_process() must be called before any CUDA work: cuBLAS reads "
            f"{CUBLAS_ENV} when it is first used"
        )
    os.environ.setdefault(CUBLAS_ENV, CUBLAS_DETERMINISTIC_VALUE)


def apply_determinism(mode: str) -> dict[str, Any]:
    """Configure PyTorch for ``mode`` and return the settings now in force.

    Changes global PyTorch state. The returned dictionary is written to the
    run log header.
    """
    _check_mode(mode)
    if mode == "off":
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True, warn_only=(mode == "warn"))
    torch.backends.cudnn.deterministic = mode != "off"
    torch.backends.cudnn.benchmark = False
    return {
        "mode": mode,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get(CUBLAS_ENV),
    }


def seed_everything(seed: int) -> dict[str, int]:
    """Seed Python's ``random``, NumPy's global generator and PyTorch.

    ``torch.manual_seed`` also seeds every CUDA device. Python's string hashing
    seed (``PYTHONHASHSEED``) cannot be changed once the interpreter has
    started; its value is recorded by :func:`environment_info` instead.
    Returns the seed given to each generator.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {MAX_SEED}], got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return {"python_random": seed, "numpy_global": seed, "torch": seed}


def environment_info() -> dict[str, Any]:
    """Software versions and GPU description, for the run log (SRS S0-F-24).

    Reading GPU properties initializes CUDA, so call this after
    :func:`prepare_process`.
    """
    cuda_available = torch.cuda.is_available()
    gpu = None
    if cuda_available:
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "count": torch.cuda.device_count(),
            "name": props.name,
            "total_memory_mib": props.total_memory // 2**20,
            "compute_capability": f"{props.major}.{props.minor}",
        }
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
        "cuda_available": cuda_available,
        "gpu": gpu,
        "env": {name: os.environ.get(name) for name in _RECORDED_ENV},
    }
