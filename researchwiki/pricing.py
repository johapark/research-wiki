"""Model-rate lookup and cost estimation, from `config/pricing.yaml`.

Rates live in data, not code, so correcting one is a YAML edit rather than a
patch. Every caller that prints a dollar figure should also print `as_of()` —
a pricing table nobody has re-checked is the normal state, and showing the date
makes that visible instead of silently wrong.

**Model IDs are matched by longest prefix**, because the recorded `model_used`
is whatever the SDK echoed back and that is often a dated build ID. This is not
cosmetic: the table this replaced was keyed on `claude-haiku-4-5`, the API
returns `claude-haiku-4-5-20251001`, `dict.get` missed, and 429 calls over 2.7M
input tokens priced at $0.00 in every report.

Unknown model → `None` → callers render $0.00. That's correct for a local model
(LM Studio, Ollama, gemma-*, qwen-*), which has no per-token price. It is also
what an unpriced cloud model looks like, so `unpriced_models()` exists to tell
the two apart.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_FILENAME = "pricing.yaml"
#: Sentinels `ingest_iterations` uses for "no API call happened".
NON_MODEL_SENTINELS = frozenset({
    "(local)", "(skipped)", "(no calls)", "(failed)", "(missing)", "stub", "",
})

_cache: dict | None = None
_cache_path: Path | None = None


@dataclass(frozen=True)
class Rate:
    """USD per million tokens for one model."""
    model_key: str
    input_per_mtok: float
    output_per_mtok: float
    provider: str = ""
    note: str = ""

    def usd(self, in_tok: int, out_tok: int) -> float:
        return (in_tok / 1_000_000) * self.input_per_mtok + \
               (out_tok / 1_000_000) * self.output_per_mtok


def pricing_path() -> Path:
    """`$RW_PRICING_CONFIG` if set, else `config/pricing.yaml` beside the package.

    Resolved relative to the package rather than the cwd so the table is found
    from any working directory — unlike `paths.py`, which is deliberately
    cwd-relative because it locates the *user's wiki*. This is shipped data.
    """
    override = os.environ.get("RW_PRICING_CONFIG")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() or p.parent != Path(".") else Path("config") / p
    return Path(__file__).resolve().parent.parent / "config" / _DEFAULT_FILENAME


def _load() -> dict:
    global _cache, _cache_path
    path = pricing_path()
    if _cache is not None and _cache_path == path:
        return _cache
    try:
        import yaml
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        # A missing or malformed table must not break `status` — it degrades to
        # "no cost estimate", which is honest, rather than raising mid-report.
        data = {}
    if not isinstance(data, dict):
        data = {}
    _cache, _cache_path = data, path
    return data


def reset_cache() -> None:
    """Drop the memoized table. For tests that point at a different file."""
    global _cache, _cache_path
    _cache, _cache_path = None, None


def as_of() -> str:
    """The date the rates were last verified, e.g. `2026-08-03`. '' if absent."""
    return str(_load().get("as_of") or "")


def sources() -> dict:
    return dict(_load().get("sources") or {})


def modifiers() -> dict:
    """Cache/batch multipliers, for reference. Never applied automatically —
    the schema records no cache-read/write split to apply them to."""
    return dict(_load().get("modifiers") or {})


def _pick_dated(entries: list, today: _dt.date) -> dict | None:
    """Choose among time-boxed rate entries for one model.

    An entry with `until: YYYY-MM-DD` applies through that date inclusive; the
    entry with no `until` is the fallback. Sonnet 5's introductory rate is the
    live case — it lapses 2026-08-31, and a table that couldn't express that
    would start overstating or understating on a specific calendar day with
    nothing to indicate why.
    """
    fallback: dict | None = None
    for e in entries:
        if not isinstance(e, dict):
            continue
        until = e.get("until")
        if not until:
            fallback = fallback or e
            continue
        try:
            end = _dt.date.fromisoformat(str(until))
        except ValueError:
            continue
        if today <= end:
            return e
    return fallback


def resolve(model: str, *, today: _dt.date | None = None) -> Rate | None:
    """Rate for `model`, or None when it isn't priced.

    Exact key first, then **longest matching prefix**, so a dated build ID
    (`claude-haiku-4-5-20251001`) resolves to its family (`claude-haiku-4-5`)
    while `claude-haiku-3-5` can't be shadowed by a shorter `claude-haiku`.
    """
    if not model or model in NON_MODEL_SENTINELS:
        return None
    table = _load().get("models") or {}
    if not isinstance(table, dict):
        return None
    today = today or _dt.date.today()

    key: str | None = None
    if model in table:
        key = model
    else:
        candidates = [k for k in table if model.startswith(k)]
        if candidates:
            key = max(candidates, key=len)
    if key is None:
        return None

    entry = table[key]
    if isinstance(entry, list):
        entry = _pick_dated(entry, today)
    if not isinstance(entry, dict):
        return None
    try:
        return Rate(
            model_key=key,
            input_per_mtok=float(entry["in"]),
            output_per_mtok=float(entry["out"]),
            provider=str(entry.get("provider") or ""),
            note=str(entry.get("note") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


def estimate_usd(model: str, in_tok: int, out_tok: int, *,
                 today: _dt.date | None = None) -> float:
    """Estimated USD for one model's token totals. 0.0 when unpriced.

    Upper bound: prompt-cache hits cost 0.1x base input and the schema has no
    cache breakdown, so a run with `cache_prompt=True` really cost less.
    """
    rate = resolve(model, today=today)
    return rate.usd(in_tok, out_tok) if rate else 0.0


def unpriced_models(models) -> list[str]:
    """Which of `models` have no rate, excluding the not-a-model sentinels.

    Distinguishes "local model, $0.00 is right" from "cloud model missing from
    the table, $0.00 is a lie". `status` surfaces the second case so a stale
    table gets noticed.
    """
    return sorted({
        m for m in models
        if m and m not in NON_MODEL_SENTINELS and resolve(m) is None
    })
