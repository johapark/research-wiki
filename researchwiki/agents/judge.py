"""Shared LLM-judge plumbing.

Both the cross-paper contradiction lint (`tasks/lint/cross_paper.py`) and the
claim-overlap cross-linker (`tasks/claim_overlap.py`) ask an LLM a small
classification question and parse a strict-JSON verdict. This module holds the
two pieces they share: the tolerant response→JSON parse and the call wrapper.
Keeping them here avoids each judge re-implementing ```-fence stripping.
"""

from __future__ import annotations

import json


def parse_json_response(raw: str) -> dict | None:
    """Parse an LLM response into a dict, tolerating ```-fences and stray prose.

    Returns None when nothing JSON-shaped can be recovered.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.lstrip("`")
        if "\n" in raw:                       # drop optional language tag line
            raw = raw.split("\n", 1)[1]
        raw = raw.rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        i, j = raw.find("{"), raw.rfind("}")   # fallback: first {...} block
        if 0 <= i < j:
            try:
                return json.loads(raw[i : j + 1])
            except json.JSONDecodeError:
                return None
        return None


def run_llm_judge(*, phase: str, system: str, prompt: str, schema: dict | None = None) -> dict | None:
    """Call the configured LLM for `phase` and return the parsed JSON verdict.

    Returns None on any failure (LLM unreachable, bad JSON) so callers can treat
    "no verdict" uniformly rather than distinguishing error kinds.
    """
    try:
        from . import llm
    except Exception:
        return None
    try:
        resp = llm.call(phase=phase, system=system, prompt=prompt, schema=schema)
    except Exception:
        return None
    return parse_json_response(resp.text or "")
