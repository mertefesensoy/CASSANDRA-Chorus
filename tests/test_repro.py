from __future__ import annotations

import random
import warnings

import numpy as np
import pytest
import torch

from cassandra_chorus import repro

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def draw():
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_seed_everything_is_reproducible():
    assert repro.seed_everything(7) == {"python_random": 7, "numpy_global": 7, "torch": 7}
    first = draw()
    repro.seed_everything(7)
    assert draw() == first
    repro.seed_everything(11)
    assert draw() != first


@pytest.mark.parametrize("seed", [-1, 2**32, True, 7.0, "7"])
def test_invalid_seeds(seed):
    with pytest.raises(ValueError, match="seed must be"):
        repro.seed_everything(seed)


@pytest.mark.usefixtures("restore_torch_determinism")
@pytest.mark.parametrize(
    ("mode", "enabled", "warn_only", "cudnn_det"),
    [("strict", True, False, True), ("warn", True, True, True), ("off", False, False, False)],
)
def test_apply_determinism_modes(mode, enabled, warn_only, cudnn_det):
    settings = repro.apply_determinism(mode)
    assert settings["mode"] == mode
    assert settings["deterministic_algorithms"] is enabled
    assert settings["warn_only"] is warn_only
    assert settings["cudnn_deterministic"] is cudnn_det
    assert settings["cudnn_benchmark"] is False


def test_invalid_mode():
    with pytest.raises(ValueError, match="determinism mode"):
        repro.apply_determinism("sometimes")
    with pytest.raises(ValueError, match="determinism mode"):
        repro.prepare_process("sometimes")


@pytest.mark.usefixtures("restore_torch_determinism")
def test_prepare_process_sets_cublas_before_cuda(monkeypatch):
    monkeypatch.setattr(repro.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.delenv(repro.CUBLAS_ENV, raising=False)
    repro.prepare_process("off")
    assert repro.CUBLAS_ENV not in repro.os.environ
    repro.prepare_process("warn")
    assert repro.os.environ[repro.CUBLAS_ENV] == ":4096:8"
    monkeypatch.setenv(repro.CUBLAS_ENV, ":16:8")
    repro.prepare_process("strict")
    assert repro.os.environ[repro.CUBLAS_ENV] == ":16:8"  # an existing value is kept


def test_prepare_process_refuses_after_cuda_init(monkeypatch):
    monkeypatch.setattr(repro.torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before any CUDA work"):
        repro.prepare_process("warn")


def test_environment_info():
    info = repro.environment_info()
    assert info["torch"] == torch.__version__ and info["numpy"] == np.__version__
    assert set(info["env"]) == {"CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES"}
    if info["cuda_available"]:
        assert info["gpu"]["name"] and info["gpu"]["total_memory_mib"] > 0
    else:
        assert info["gpu"] is None


@requires_cuda
@pytest.mark.gpu
@pytest.mark.usefixtures("restore_torch_determinism")
def test_warn_mode_reports_a_nondeterministic_cuda_op():
    # torch.histc on a CUDA float tensor has no deterministic kernel (checked on PyTorch 2.12.1).
    x = torch.rand(100, device="cuda")
    repro.apply_determinism("warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.histc(x, bins=10)
    assert any("does not have a deterministic implementation" in str(w.message) for w in caught)
    repro.apply_determinism("strict")
    with pytest.raises(RuntimeError, match="does not have a deterministic implementation"):
        torch.histc(x, bins=10)
