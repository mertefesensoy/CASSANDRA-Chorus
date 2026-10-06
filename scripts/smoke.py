"""Smoke check of the run pipeline (PLAN step 1).

Usage, from the repository root::

    python -m scripts.smoke --config configs/smoke.toml [--set run.seed=11 ...]

Loads the configuration, prepares determinism before any CUDA work, creates a
run folder with a JSONL log, seeds every generator, runs one tiny forward and
backward computation on the configured device, and logs the SHA-256 checksum of
the results. Two runs with the same configuration and seed should print the
same checksum. Nothing is trained: this checks the plumbing, not a model.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
from pathlib import Path

import torch

from cassandra_chorus import paths, repro
from cassandra_chorus.config import RunSection, load_config
from cassandra_chorus.runlog import RunLogger, make_run_id


@dataclasses.dataclass(frozen=True)
class SmokeSection:
    size: int = 256


@dataclasses.dataclass(frozen=True)
class SmokeConfig:
    run: RunSection
    smoke: SmokeSection = dataclasses.field(default_factory=SmokeSection)


def resolve_device(name: str) -> torch.device:
    """The requested device. Never falls back from CUDA to CPU silently."""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "the configuration asks for device 'cuda' but CUDA is not available; "
            "set run.device=\"cpu\" to run on the CPU deliberately"
        )
    return torch.device(name)


def smoke_computation(size: int, device: torch.device) -> tuple[str, float]:
    """Random matrices, a product, a scalar loss and one backward pass.

    Returns the SHA-256 of the raw bytes of (product, gradient, loss), in that
    order, and the loss value. Uses the global PyTorch generator, so the result
    depends only on the seed, the device and the determinism settings.
    """
    a = torch.randn(size, size, device=device)
    w = torch.randn(size, size, device=device, requires_grad=True)
    h = a @ w
    loss = torch.logsumexp(h, dim=1).mean()
    loss.backward()
    digest = hashlib.sha256()
    for tensor in (h.detach(), w.grad, loss.detach()):
        digest.update(tensor.contiguous().cpu().numpy().tobytes())
    return digest.hexdigest(), loss.item()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="TOML or JSON configuration file")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="override one configuration value; may be repeated",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, SmokeConfig, args.overrides)
    repro.prepare_process(cfg.run.determinism)  # before any CUDA work
    device = resolve_device(cfg.run.device)  # fail before creating a run folder
    run_id = make_run_id(cfg.run.name, cfg.run.seed)

    with RunLogger.create(paths.runs_root(), run_id) as log:
        seeds = repro.seed_everything(cfg.run.seed)
        determinism = repro.apply_determinism(cfg.run.determinism)
        log.write_header(
            config=cfg,
            config_path=args.config,
            seeds=seeds,
            environment=repro.environment_info(),
            determinism=determinism,
        )
        with log.capture_warnings():
            checksum, loss = smoke_computation(cfg.smoke.size, device)
            log.write("final", device=str(device), metrics={"smoke_checksum": checksum, "smoke_loss": loss})

    print(f"run folder: {log.run_dir}")
    print(f"checksum:   {checksum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
