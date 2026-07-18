"""Per-claim *support* (entailment) check — the qualitative analogue of the
numeric-drift veto.

The fidelity grader already scores each claim against the source PDF by BM25
overlap (`top1_score`) and embedding similarity (`semantic_score`). Those are
*similarity* signals: a fabricated-but-on-topic claim scores high on both, so
it neither trips the per-claim weak-claim flag nor moves the aggregate mean
the promotion gate reads. This module adds a *distinct* signal — does the
retrieved chunk actually **support** the claim? — modeled on Self-RAG's
segment-level `ISSUP` critique (see wiki/ai/asai-2023-self-rag-...).

Design notes:
  - Reuses `ClaimScore.supporting_text` (the top-1 chunk already retrieved at
    grade time), so the marginal cost is one entailment classification per
    claim, not any new retrieval.
  - The classifier is dependency-injected (`Classifier`): the default batches
    all (claim, chunk) pairs into a single judge-role LLM call, but tests pass
    a stub and a future cheap NLI cross-encoder can drop in unchanged.
  - Opt-in. `count_unsupported` feeds `n_unsupported` into the grade aggregate
    only when a classifier is supplied; the promotion veto treats a missing /
    zero `n_unsupported` as "check not run", so existing behaviour is unchanged
    until the gate is explicitly enabled and calibrated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Literal

SupportVerdict = Literal["supported", "partial", "unsupported"]

# A classifier maps a batch of (claim_text, source_chunk) pairs to one verdict
# each, in the same order. Injected so the LLM path and tests share the core.
Classifier = Callable[[list[tuple[str, str]]], list[SupportVerdict]]

_VALID: frozenset[str] = frozenset({"supported", "partial", "unsupported"})


@dataclass
class ClaimSupport:
    """Support verdict for one graded claim, keyed by its section+position so
    it can be joined back to the `ClaimScore` it came from."""
    section: str
    position: int
    text: str
    verdict: SupportVerdict


def check_support(
    claims: list[tuple[str, int, str, str]],
    classify: Classifier,
) -> list[ClaimSupport]:
    """Judge support for each `(section, position, claim_text, chunk_text)`.

    Claims whose `chunk_text` is empty (e.g. no PDF hit) are skipped rather
    than sent to the classifier — with no retrieved evidence there is nothing
    to entail against, and a fabricated "unsupported" verdict on missing
    evidence would be noise. The caller sees them absent from the result.
    """
    scored = [c for c in claims if (c[3] or "").strip()]
    if not scored:
        return []
    pairs = [(c[2], c[3]) for c in scored]
    verdicts = classify(pairs)
    if len(verdicts) != len(scored):
        raise ValueError(
            f"classifier returned {len(verdicts)} verdicts for {len(scored)} claims"
        )
    out: list[ClaimSupport] = []
    for (section, position, text, _chunk), v in zip(scored, verdicts):
        v = v if v in _VALID else "partial"  # unknown → treat as soft, not a veto
        out.append(ClaimSupport(section=section, position=position, text=text, verdict=v))
    return out


def unsupported_claims(supports: list[ClaimSupport]) -> list[ClaimSupport]:
    """The claims judged flatly `unsupported`, keeping section/position/text so
    a reviewer of a sandboxed page sees *which* claims the source doesn't
    entail — not just how many. `partial` is excluded for the same reason it
    doesn't veto: only a flat-unsupported claim is actionable."""
    return [s for s in supports if s.verdict == "unsupported"]


def count_unsupported(supports: list[ClaimSupport]) -> int:
    """Number of claims the check judged flatly `unsupported`. `partial` does
    not count toward the veto — only a claim the source does not support at
    all should block promotion, mirroring the zero-tolerance numeric-drift
    veto's "hard failure only" posture."""
    return len(unsupported_claims(supports))


# --------------------------------------------------------------------------
# Default LLM classifier — one batched judge-role call for the whole page.
# --------------------------------------------------------------------------

_SUPPORT_SYSTEM = (
    "You verify whether a SOURCE passage supports a CLAIM. Judge only against "
    "the SOURCE text given — not outside knowledge. 'supported' = the source "
    "states or directly entails the claim; 'partial' = the source is related "
    "and partly backs it but not fully; 'unsupported' = the source does not "
    "back the claim (including fabricated or contradicted content). Return "
    "strict JSON only."
)

_SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {"type": "string",
                                "enum": ["supported", "partial", "unsupported"]},
                },
                "required": ["id", "verdict"],
            },
        }
    },
    "required": ["verdicts"],
}


def _build_support_prompt(pairs: list[tuple[str, str]]) -> str:
    blocks = []
    for i, (claim, chunk) in enumerate(pairs):
        blocks.append(
            f"[{i}]\nCLAIM: {claim}\nSOURCE: {chunk.strip()[:1200]}"
        )
    return (
        "For each numbered item, judge whether SOURCE supports CLAIM.\n"
        'Return JSON: {"verdicts": [{"id": N, "verdict": '
        '"supported|partial|unsupported"}, ...]} with one entry per item.\n\n'
        + "\n\n".join(blocks)
    )


def llm_support_classifier(pairs: list[tuple[str, str]]) -> list[SupportVerdict]:
    """Batched judge-role entailment over all (claim, chunk) pairs on a page.

    Imported lazily so `check_support` / `count_unsupported` stay import-light
    for tests and offline use (no `agents.llm` / provider deps until the real
    classifier actually runs)."""
    from ..agents import llm

    resp = llm.call(
        phase="claim_support",
        prompt=_build_support_prompt(pairs),
        system=_SUPPORT_SYSTEM,
        schema=_SUPPORT_SCHEMA,
    )
    verdicts = _parse_verdicts(resp.text, n=len(pairs))
    return verdicts


def _parse_verdicts(text: str, *, n: int) -> list[SupportVerdict]:
    """Parse the judge's JSON into an ordered verdict list of length `n`.
    Robust to fenced code blocks and to missing ids (defaults to 'partial',
    which never vetoes — an unparseable response can't fabricate a failure)."""
    out: list[SupportVerdict] = ["partial"] * n
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return out
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    for item in data.get("verdicts", []):
        try:
            i = int(item["id"])
            v = str(item["verdict"]).strip().lower()
        except (KeyError, ValueError, TypeError):
            continue
        if 0 <= i < n and v in _VALID:
            out[i] = v  # type: ignore[assignment]
    return out
