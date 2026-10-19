"""Small, strict JSON contracts without an application-framework dependency."""

from __future__ import annotations

import math
import types
from dataclasses import asdict, fields, is_dataclass
from typing import Literal, Union, get_args, get_origin, get_type_hints


class ContractError(ValueError):
    """An unsupported schema or invalid application operation value."""


def _value(value, annotation, path: str, *, decoding: bool):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        if not any(type(value) is type(choice) and value == choice for choice in args):
            raise ContractError(f"{path}: expected one of {args!r}")
        return value
    if origin in (Union, types.UnionType):
        for choice in args:
            try:
                return _value(value, choice, path, decoding=decoding)
            except ContractError:
                pass
        raise ContractError(f"{path}: invalid value type or enum")
    if origin is tuple:
        allowed = (list, tuple) if decoding else (tuple,)
        if not isinstance(value, allowed):
            raise ContractError(f"{path}: expected an array")
        return tuple(_value(item, args[0], f"{path}[{i}]", decoding=decoding)
                     for i, item in enumerate(value))
    if isinstance(annotation, type) and issubclass(annotation, WireModel):
        if isinstance(value, annotation):
            return value
        if decoding and isinstance(value, dict):
            return annotation.from_dict(value)
        raise ContractError(f"{path}: expected {annotation.__name__}")
    if annotation is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ContractError(f"{path}: expected a finite number")
        return float(value)
    if annotation in (str, int, bool, type(None)):
        if type(value) is not annotation:
            raise ContractError(f"{path}: expected {annotation.__name__}")
        return value
    raise TypeError(f"unsupported contract annotation: {annotation!r}")


class WireModel:
    """Dataclass mixin; validates direct construction as well as JSON input.

    Requests reject unknown keys, rather than silently dropping an option.
    Versioned readers reject unsupported schema versions. Paths are strings on
    the wire; filesystem/provider validation belongs at the execution boundary.
    """

    def __post_init__(self):
        hints = get_type_hints(type(self))
        for field in fields(self):
            _value(getattr(self, field.name), hints[field.name], field.name, decoding=False)

    def to_dict(self) -> dict:
        def json_value(value):
            if is_dataclass(value):
                return {k: json_value(v) for k, v in asdict(value).items()}
            if isinstance(value, dict):
                return {k: json_value(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [json_value(v) for v in value]
            return value
        return json_value(self)

    @classmethod
    def from_dict(cls, data: dict):
        if not isinstance(data, dict):
            raise ContractError(f"{cls.__name__}: expected an object")
        if any(not isinstance(key, str) for key in data):
            raise ContractError(f"{cls.__name__}: object keys must be strings")
        hints = get_type_hints(cls)
        known = {field.name for field in fields(cls)}
        if set(data) - known:
            raise ContractError(f"{cls.__name__}: unknown fields {sorted(set(data) - known)}")
        values = {key: _value(value, hints[key], key, decoding=True)
                  for key, value in data.items()}
        try:
            return cls(**values)
        except TypeError as exc:
            raise ContractError(f"{cls.__name__}: missing required fields") from exc


def nonempty(value: str, label: str) -> None:
    if not value.strip():
        raise ContractError(f"{label}: must not be empty")


def absolute_path(value: str, label: str) -> None:
    from pathlib import Path
    if not Path(value).is_absolute():
        raise ContractError(f"{label}: must be an absolute path")
