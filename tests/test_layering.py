"""The coordinator must not depend on the simulation (SRS S0-N-05).

Stage 1 reuses ``cassandra_chorus.coordinator`` with real workers, so a static
check scans every module in it and fails on any import that resolves to
``cassandra_chorus.sim``. Limitation: imports built from strings at run time
(``importlib.import_module("...")``) are not seen.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from cassandra_chorus.paths import REPO_ROOT

GUARDED_PACKAGE = REPO_ROOT / "cassandra_chorus" / "coordinator"
FORBIDDEN = ("cassandra_chorus.sim",)


def module_name(path: Path) -> tuple[str, bool]:
    """Dotted module name of a file under the repository, and whether it is a package."""
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    return ".".join(parts), is_package


def imported_modules(source: str, module: str, is_package: bool) -> Iterator[str]:
    """Every module name an import statement in ``source`` refers to, made absolute."""
    package = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                parts = package.split(".")
                keep = len(parts) - (node.level - 1)
                if keep <= 0:
                    raise ValueError(f"relative import beyond the top-level package in {module}")
                base = ".".join(parts[:keep] + ([node.module] if node.module else []))
            yield base
            for alias in node.names:  # `from cassandra_chorus import sim` imports a module too
                yield f"{base}.{alias.name}"


def violations(source: str, module: str, is_package: bool) -> list[str]:
    return sorted(
        {
            name
            for name in imported_modules(source, module, is_package)
            if any(name == f or name.startswith(f + ".") for f in FORBIDDEN)
        }
    )


def test_coordinator_does_not_import_sim():
    files = sorted(GUARDED_PACKAGE.rglob("*.py"))
    assert files, "no coordinator modules found; the check would pass vacuously"
    found = {}
    for path in files:
        name, is_package = module_name(path)
        bad = violations(path.read_text(encoding="utf-8"), name, is_package)
        if bad:
            found[str(path.relative_to(REPO_ROOT))] = bad
    assert not found, f"coordinator modules import the simulation: {found}"


@pytest.mark.parametrize(
    "source",
    [
        "import cassandra_chorus.sim",
        "import cassandra_chorus.sim.worker as w",
        "from cassandra_chorus.sim import worker",
        "from cassandra_chorus.sim.worker import run",
        "from cassandra_chorus import sim",
        "from ..sim import worker",
        "from .. import sim",
        "def f():\n    from ..sim.worker import run\n",
    ],
)
def test_checker_catches_violations(source):
    assert violations(source, "cassandra_chorus.coordinator.merge", is_package=False)


@pytest.mark.parametrize(
    "source",
    [
        "import torch",
        "from ..model import moe",
        "from . import assign",
        "from cassandra_chorus.metrics import usage",
        "import cassandra_chorus.simulation_free_name",
    ],
)
def test_checker_allows_other_imports(source):
    assert not violations(source, "cassandra_chorus.coordinator.merge", is_package=False)
