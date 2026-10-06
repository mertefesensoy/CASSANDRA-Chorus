from __future__ import annotations

import os

# cuBLAS reads this when first used, so it must be set before any test touches
# CUDA; entry points do the same through repro.prepare_process().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest  # noqa: E402
import torch  # noqa: E402


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
