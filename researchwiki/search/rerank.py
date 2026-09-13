"""Optional one-call listwise reranking for difficult page queries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..wiki import extract_section, read_pages


MAX_CANDIDATES = 12


@dataclass(frozen=True)
class RerankResult:
    hits: list
    usage: dict
    applied: bool


_SCHEMA = {
    "type": "object",
    "properties": {
        "order": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["order"],
    "additionalProperties": False,
}

_SYSTEM = """You rerank wiki search candidates for relevance to one query.
Return each supplied candidate identifier exactly once, most relevant first.
Judge the requested information, not shared vocabulary. Use only supplied
content. Output JSON: {"order": ["category/stem", ...]}."""


def _key(hit) -> str:
    return hit.key if hasattr(hit, "key") else str(hit["key"])


def _prompt(query: str, hits: list) -> str:
    pages = {page.key: page for page in read_pages()}
    from .tools import claims_by_stem

    parts = [f"# Query\n{query}", "", "# Candidates"]
    for hit in hits:
        key = _key(hit)
        page = pages.get(key)
        if page is None:
            parts.extend([f"\n## {key}", "(page unavailable)"])
            continue
        title = page.str_field("title")
        summary = extract_section(page.body, "Summary").strip()[:900]
        claims = [
            claim for claim in claims_by_stem(page.stem)
            if claim.get("section") in {"key_contributions", "results"}
        ]
        claims.sort(
            key=lambda claim: (
                -(claim.get("semantic_score") or 0.0),
                -(claim.get("bm25_top1") or 0.0),
            )
        )
        parts.extend([f"\n## {key}", f"Title: {title}", f"Summary: {summary}"])
        for claim in claims[:2]:
            parts.append(f"Claim: {(claim.get('text') or '')[:500]}")
    parts.append("\nReturn JSON only.")
    return "\n".join(parts)


def _parse_order(text: str, allowed: list[str]) -> list[str] | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    order = data.get("order") if isinstance(data, dict) else None
    if not isinstance(order, list):
        return None
    seen: set[str] = set()
    cleaned = []
    allowed_set = set(allowed)
    for value in order:
        if isinstance(value, str) and value in allowed_set and value not in seen:
            cleaned.append(value)
            seen.add(value)
    return cleaned or None


def rerank_hits(query: str, hits: list) -> RerankResult:
    """Rerank at most twelve candidates, falling back unchanged on bad output."""
    candidates = list(hits[:MAX_CANDIDATES])
    if len(candidates) < 2:
        return RerankResult(candidates, {}, False)
    allowed = [_key(hit) for hit in candidates]
    try:
        from ..agents import llm

        response = llm.call(
            # A bounded ordering task, not source-fidelity adjudication; keep it
            # on the inexpensive classifier role rather than the ingest judge.
            phase="classifier",
            prompt=_prompt(query, candidates),
            system=_SYSTEM,
            max_tokens=300,
            temperature=0.0,
            reasoning_effort="low",
            schema=_SCHEMA,
        )
        order = _parse_order(response.text, allowed)
    except Exception:
        return RerankResult(candidates, {}, False)
    if order is None:
        return RerankResult(candidates, llm.response_usage(response), False)
    by_key = {_key(hit): hit for hit in candidates}
    complete_order = order + [key for key in allowed if key not in set(order)]
    return RerankResult(
        [by_key[key] for key in complete_order],
        llm.response_usage(response),
        True,
    )
