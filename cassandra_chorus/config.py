"""Typed run configuration: load from TOML or JSON, apply overrides, validate, hash.

Each entry point declares its own top-level frozen dataclass made of sections
(every run has a ``run`` section of type :class:`RunSection`). Loading is
strict: an unknown key, a missing required key, a value of the wrong type or a
value outside a ``Literal`` set raises :class:`ConfigError` naming the key, so a
typo in a configuration file can never silently fall back to a default.

Supported field types: ``bool``, ``int``, ``float``, ``str``, ``Literal[...]``,
``X | None``, ``tuple[X, ...]``, fixed-length ``tuple[X, Y]``, ``list[X]``,
``dict[str, X]`` and nested dataclasses.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import tomllib
import types
import typing
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar

T = TypeVar("T")

DeterminismMode = Literal["strict", "warn", "off"]
Device = Literal["cuda", "cpu"]


class ConfigError(ValueError):
    """A configuration file or override does not match its schema."""


@dataclasses.dataclass(frozen=True)
class RunSection:
    """Settings every run has, whatever it trains.

    ``name`` becomes part of the run ID, so it is limited to letters, digits,
    ``.``, ``_`` and ``-`` (checked when the run folder is created).
    ``determinism`` is explained in :mod:`cassandra_chorus.repro`.
    ``cpu_threads`` is PyTorch's CPU thread count; 1 by default because the
    CPU work of a CUDA run is tiny and extra threads slowed GPU steps on the
    reference laptop (see the step 4 implementation doc). On a CUDA run it does
    not change results (the CPU only samples and counts integers); on device
    ``cpu`` the model's float arithmetic may round differently with it.
    """

    name: str
    seed: int = 7
    device: Device = "cuda"
    determinism: DeterminismMode = "warn"
    cpu_threads: int = 1
    notes: str = ""


# --------------------------------------------------------------------------
# Reading files and overrides


def read_config_file(path: str | Path) -> dict[str, Any]:
    """Read a ``.toml`` or ``.json`` configuration file into a dictionary."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".toml":
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise ConfigError(f"{path}: unsupported configuration file type {suffix!r}; use .toml or .json")
    except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path}: cannot parse: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a table")
    return data


def parse_override(text: str) -> tuple[list[str], Any]:
    """Parse ``section.key=value`` into a key path and a value.

    The value is read as a TOML literal (``11``, ``0.5``, ``true``, ``"text"``,
    ``[1, 2]``). If it is not a valid TOML literal it is taken as a plain
    string, so ``run.name=abc`` works; quote it (``run.notes="7"``) to force a
    string that would otherwise parse as a number.
    """
    if "=" not in text:
        raise ConfigError(f"override {text!r} is not of the form section.key=value")
    key, raw = text.split("=", 1)
    parts = key.strip().split(".")
    if any(not part for part in parts):
        raise ConfigError(f"override {text!r} has an empty key segment")
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return parts, value


def apply_overrides(data: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Return a deep copy of ``data`` with each override applied in order.

    Missing intermediate tables are created; whether the final key exists in
    the schema is checked later by :func:`build_config`.
    """
    result = copy.deepcopy(dict(data))
    for text in overrides:
        parts, value = parse_override(text)
        node = result
        for i, part in enumerate(parts[:-1]):
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                where = ".".join(parts[: i + 1])
                raise ConfigError(f"override {text!r}: {where} is a value, not a table")
            node = child
        node[parts[-1]] = value
    return result


# --------------------------------------------------------------------------
# Validation


def _join(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp)


def _coerce(tp: Any, value: Any, key: str) -> Any:
    """Check ``value`` against type ``tp`` and return it in canonical form."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if tp is Any:
        return value
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return _build_dataclass(tp, value, key)
    if origin is Literal:
        for allowed in args:
            if type(value) is type(allowed) and value == allowed:
                return value
        options = ", ".join(repr(a) for a in args)
        raise ConfigError(f"{key}: {value!r} is not one of {options}")
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        for option in args:
            if option is type(None):
                continue
            try:
                return _coerce(option, value, key)
            except ConfigError:
                continue
        raise ConfigError(f"{key}: {value!r} does not match {tp}")
    if origin in (tuple, list):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{key}: expected a list, got {type(value).__name__}")
        if origin is list:
            (item_tp,) = args
            return [_coerce(item_tp, v, f"{key}[{i}]") for i, v in enumerate(value)]
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, f"{key}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{key}: expected {len(args)} items, got {len(value)}")
        return tuple(_coerce(t, v, f"{key}[{i}]") for i, (t, v) in enumerate(zip(args, value)))
    if origin is dict:
        key_tp, value_tp = args
        if key_tp is not str:
            raise ConfigError(f"{key}: schema dictionaries must have str keys")
        if not isinstance(value, Mapping):
            raise ConfigError(f"{key}: expected a table, got {type(value).__name__}")
        return {str(k): _coerce(value_tp, v, _join(key, str(k))) for k, v in value.items()}
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif tp is str:
        if isinstance(value, str):
            return value
    else:
        raise ConfigError(f"{key}: schema uses unsupported type {tp!r}")
    raise ConfigError(f"{key}: expected {_type_name(tp)}, got {type(value).__name__} {value!r}")


def _build_dataclass(cls: type[T], data: Any, where: str) -> T:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where or 'config'}: expected a table, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        allowed = ", ".join(sorted(fields))
        raise ConfigError(f"{_join(where, unknown[0])}: unknown key (allowed here: {allowed})")
    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        key = _join(where, name)
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], key)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{key}: required key is missing")
    return cls(**kwargs)


def build_config(schema: type[T], data: Mapping[str, Any]) -> T:
    """Validate ``data`` against ``schema`` and return the schema instance."""
    if not (dataclasses.is_dataclass(schema) and isinstance(schema, type)):
        raise TypeError("schema must be a dataclass type")
    return _build_dataclass(schema, data, "")


def load_config(path: str | Path, schema: type[T], overrides: Iterable[str] = ()) -> T:
    """Read ``path``, apply ``overrides`` in order, and validate against ``schema``.

    Raises :class:`ConfigError` on any mismatch. Has no side effects.
    """
    return build_config(schema, apply_overrides(read_config_file(path), overrides))


# --------------------------------------------------------------------------
# Serialization and hashing


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """The resolved configuration (defaults filled in) as a plain dictionary."""
    if not dataclasses.is_dataclass(cfg) or isinstance(cfg, type):
        raise TypeError("expected a dataclass instance")
    return dataclasses.asdict(cfg)


def canonical_json(cfg: Any) -> str:
    """Canonical JSON of the resolved configuration: sorted keys, no whitespace."""
    return json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(cfg: Any) -> str:
    """SHA-256 (hexadecimal) of :func:`canonical_json`.

    Equal for two configurations exactly when every resolved field is equal,
    whatever the key order in the source file and whether a default was
    written out or left implicit.
    """
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()
