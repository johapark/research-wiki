"""Strict, deterministic JSON for archive identity; never serialize NaN implicitly."""

import hashlib
import json
import math

from ..operations._wire import ContractError


def _check(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _check(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _check(item)
        return
    raise ContractError("archive payload must contain only finite JSON values and string keys")


def canonical_bytes(value):
    try:
        _check(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ContractError("archive JSON must be encodable as UTF-8") from exc


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ContractError("duplicate archive JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ContractError("non-finite archive JSON number")


def decode_json(data):
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
        canonical_bytes(value)  # Also rejects numeric overflow and unpaired surrogates.
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ContractError("invalid strict archive JSON") from exc


def digest(data):
    return hashlib.sha256(data).hexdigest()
