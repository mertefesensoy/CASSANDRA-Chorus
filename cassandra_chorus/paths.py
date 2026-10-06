"""Where run outputs go.

Raw run logs and checkpoints stay out of version control (SRS S0-N-08). Their
location comes from the ``CHORUS_RUNS_DIR`` environment variable, so each
machine can keep them wherever suits it; without it they go to ``runs/`` in
the repository, which is gitignored.

On the reference laptop the repository sits inside OneDrive, and Python is the
Microsoft Store build, which silently redirects writes under ``AppData`` to a
private folder. ``CHORUS_RUNS_DIR`` should therefore point somewhere outside
both (owner decision, 2026-10-07: ``C:\\Users\\senso\\chorus-runs``).
"""

from __future__ import annotations

import os
from pathlib import Path

RUNS_DIR_ENV = "CHORUS_RUNS_DIR"
REPO_ROOT = Path(__file__).resolve().parent.parent


def runs_root() -> Path:
    """The folder that holds one sub-folder per run. Not created here."""
    value = os.environ.get(RUNS_DIR_ENV, "").strip()
    if value:
        return Path(value).expanduser()
    return REPO_ROOT / "runs"
