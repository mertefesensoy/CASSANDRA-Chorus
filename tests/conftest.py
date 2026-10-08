from __future__ import annotations

import os

# cuBLAS reads this when first used, so it must be set before any test touches
# CUDA; entry points do the same through repro.prepare_process().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest  # noqa: E402
import torch  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def test_ops_settings(tmp_path_factory):
    """Machine-level ops settings for the whole test session.

    Entry points run as subprocesses while this test process may hold the GPU,
    so the GPU-idle check is off; keep-awake is off so tests change nothing on
    the machine; no scheduled tasks are checked. Tests never read the real
    ops.local.toml. Subprocesses inherit the variable through os.environ.
    """
    path = tmp_path_factory.mktemp("ops") / "ops.toml"
    path.write_text("[ops]\nrequire_gpu_idle = false\nkeep_awake = false\n", encoding="utf-8")
    previous = os.environ.get("CHORUS_OPS_FILE")
    os.environ["CHORUS_OPS_FILE"] = str(path)
    yield path
    if previous is None:
        os.environ.pop("CHORUS_OPS_FILE", None)
    else:
        os.environ["CHORUS_OPS_FILE"] = previous


@pytest.fixture
def restore_torch_determinism():
    """Undo global PyTorch determinism changes made by a test."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if cublas is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas
