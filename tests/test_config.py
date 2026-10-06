from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Literal

import pytest

from cassandra_chorus.config import (
    ConfigError,
    RunSection,
    apply_overrides,
    config_hash,
    config_to_dict,
    load_config,
    parse_override,
)
from cassandra_chorus.paths import REPO_ROOT


@dataclasses.dataclass(frozen=True)
class Inner:
    a: int
    b: float = 0.5
    mode: Literal["x", "y"] = "x"
    tags: tuple[str, ...] = ()
    limit: int | None = None


@dataclasses.dataclass(frozen=True)
class Schema:
    run: RunSection
    inner: Inner = dataclasses.field(default_factory=lambda: Inner(a=1))


def write(tmp_path: Path, text: str, name: str = "cfg.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_filled(tmp_path):
    cfg = load_config(write(tmp_path, '[run]\nname = "t"\n[inner]\na = 3\n'), Schema)
    assert cfg.run == RunSection(name="t", seed=7, device="cuda", determinism="warn", notes="")
    assert cfg.inner == Inner(a=3)


def test_missing_optional_section_uses_its_default(tmp_path):
    cfg = load_config(write(tmp_path, '[run]\nname = "t"\n'), Schema)
    assert cfg.inner == Inner(a=1)


def test_repository_smoke_config_loads():
    from scripts.smoke import SmokeConfig

    cfg = load_config(REPO_ROOT / "configs" / "smoke.toml", SmokeConfig)
    assert cfg.run.name == "smoke" and cfg.run.seed == 7 and cfg.run.determinism == "warn"


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ('[run]\nname = "t"\n[bogus]\nx = 1\n', "bogus"),
        ('[run]\nname = "t"\nnme = "typo"\n', "run.nme"),
        ('[run]\nname = "t"\n[inner]\na = 1\nc = 2\n', "inner.c"),
    ],
)
def test_unknown_keys_are_rejected(tmp_path, text, key):
    with pytest.raises(ConfigError, match=rf"^{key}: unknown key"):
        load_config(write(tmp_path, text), Schema)


def test_missing_required_key(tmp_path):
    with pytest.raises(ConfigError, match=r"^run\.name: required key is missing"):
        load_config(write(tmp_path, "[run]\nseed = 7\n"), Schema)


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ('seed = "7"', "run.seed"),
        ("seed = true", "run.seed"),  # a boolean is not an integer
        ("seed = 7.0", "run.seed"),
        ("name = 5", "run.name"),
        ('determinism = "sometimes"', "run.determinism"),
        ('device = "cuda:1"', "run.device"),
    ],
)
def test_wrong_types_and_values_are_rejected(tmp_path, line, key):
    text = f'[run]\nname = "t"\n{line}\n' if not line.startswith("name") else f"[run]\n{line}\n"
    with pytest.raises(ConfigError, match=rf"^{key}:"):
        load_config(write(tmp_path, text), Schema)


def test_int_is_accepted_for_float_and_converted(tmp_path):
    cfg = load_config(write(tmp_path, '[run]\nname = "t"\n[inner]\na = 1\nb = 2\n'), Schema)
    assert cfg.inner.b == 2.0 and isinstance(cfg.inner.b, float)


def test_tuples_and_optionals(tmp_path):
    cfg = load_config(write(tmp_path, '[run]\nname = "t"\n[inner]\na = 1\ntags = ["p", "q"]\nlimit = 4\n'), Schema)
    assert cfg.inner.tags == ("p", "q") and cfg.inner.limit == 4
    with pytest.raises(ConfigError, match=r"^inner\.tags\[1\]:"):
        load_config(write(tmp_path, '[run]\nname = "t"\n[inner]\na = 1\ntags = ["p", 3]\n'), Schema)


def test_overrides(tmp_path):
    path = write(tmp_path, '[run]\nname = "t"\n')
    cfg = load_config(path, Schema, ["run.seed=11", "run.notes=hello world", 'run.name="x7"', "inner.a=5"])
    assert cfg.run.seed == 11
    assert cfg.run.notes == "hello world"  # not a TOML literal, so taken as a string
    assert cfg.run.name == "x7"
    assert cfg.inner.a == 5  # section absent from the file is created
    assert load_config(path, Schema, ["run.seed=11", "run.seed=19"]).run.seed == 19  # last wins


def test_bad_overrides(tmp_path):
    path = write(tmp_path, '[run]\nname = "t"\n')
    with pytest.raises(ConfigError, match=r"^run\.nope: unknown key"):
        load_config(path, Schema, ["run.nope=1"])
    with pytest.raises(ConfigError, match="not of the form"):
        load_config(path, Schema, ["runseed11"])
    with pytest.raises(ConfigError, match="empty key segment"):
        load_config(path, Schema, ["run..seed=1"])
    with pytest.raises(ConfigError, match="is a value, not a table"):
        load_config(path, Schema, ["run.name.x=1"])  # run.name is a value in the file
    with pytest.raises(ConfigError, match=r"^run\.seed: expected int, got dict"):
        load_config(path, Schema, ["run.seed.x=1"])  # absent from the file, so caught by validation
    with pytest.raises(ConfigError, match=r"^run\.notes: expected str"):
        load_config(path, Schema, ["run.notes=7"])  # numbers must be quoted to become strings


def test_parse_override_values():
    assert parse_override("a.b=0.5") == (["a", "b"], 0.5)
    assert parse_override("a=true") == (["a"], True)
    assert parse_override("a=[1, 2]") == (["a"], [1, 2])
    assert parse_override("a = cuda ") == (["a"], "cuda")


def test_apply_overrides_does_not_mutate_input():
    data = {"run": {"name": "t"}}
    apply_overrides(data, ["run.seed=3"])
    assert data == {"run": {"name": "t"}}


def test_hash_ignores_key_order_and_explicit_defaults(tmp_path):
    a = load_config(write(tmp_path, '[run]\nname = "t"\nseed = 7\n', "a.toml"), Schema)
    b = load_config(write(tmp_path, '[run]\nseed = 7\nname = "t"\n', "b.toml"), Schema)
    c = load_config(write(tmp_path, '[run]\nname = "t"\n', "c.toml"), Schema)
    d = load_config(write(tmp_path, '[run]\nname = "t"\nseed = 11\n', "d.toml"), Schema)
    assert config_hash(a) == config_hash(b) == config_hash(c)
    assert config_hash(a) != config_hash(d)
    assert len(config_hash(a)) == 64 and int(config_hash(a), 16) >= 0


def test_json_round_trip_preserves_config_and_hash(tmp_path):
    cfg = load_config(write(tmp_path, '[run]\nname = "t"\n[inner]\na = 2\ntags = ["p"]\n'), Schema)
    saved = tmp_path / "config.json"
    saved.write_text(json.dumps(config_to_dict(cfg)), encoding="utf-8")
    again = load_config(saved, Schema)
    assert again == cfg and config_hash(again) == config_hash(cfg)


def test_unsupported_file_type(tmp_path):
    with pytest.raises(ConfigError, match="unsupported configuration file type"):
        load_config(write(tmp_path, "run: {}", "cfg.yaml"), Schema)


def test_malformed_toml(tmp_path):
    with pytest.raises(ConfigError, match="cannot parse"):
        load_config(write(tmp_path, "[run\nname = 't'\n"), Schema)
