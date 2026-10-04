"""Dangling `[[stem#slug]]` claim-anchor lint.

Scans every wiki page for claim anchors and flags any whose (stem, slug)
pair no longer resolves against `claims(paper_stem, claim_slug)` in
state.db. This is the claim-level analogue of `broken_wikilinks` —
anchors are content-addressed, so a "broken" one means either the target
paper was removed OR the target claim's text has changed (regenerating
its slug).

Emits per-page hit lists; the JSON key `dangling_claim_anchors` lets CI
gate on zero-drift.

`find_undefined_footnote_refs` is the footnote-level sibling, keyed
`undefined_footnote_refs`. A `[^id]` with no `[^id]: …` definition line is
worse than it looks: `check-grounding` only reports a unit whose *sole*
citation is that footnote, so a paragraph citing `[^a][^b]` where only `b` is
undefined passes the structural gate, and `grade synthesis` then silently
grades it against `a` alone — reporting an empty `unresolved_citations`, with
no trace that half the citation went nowhere. A page can clear both gates with
a paper's worth of claims never checked. Unlike the anchor check this needs no
DB, and it covers every page type (the synthesis contract checks the same thing
for pages under `wiki/synthesis/`).
"""

from __future__ import annotations

import re
from pathlib import Path

from ...grade.grounding import (
    ClaimDBUnavailable,
    extract_claim_anchors,
    _resolve_claim_anchors,
)


def find_dangling_claim_anchors(
    pages_body: dict[Path, str],
) -> list[dict]:
    """Return [{page, stem, slug, dangling}] for every unresolved anchor.

    Batch-resolves against state.db in one query rather than per-anchor.
    Empty list on any DB failure — we don't want the lint to gate on an
    unavailable DB.
    """
    # Collect all (page, stem, slug) references across the corpus.
    per_page: dict[Path, list[tuple[str, str]]] = {}
    all_pairs: set[tuple[str, str]] = set()
    for md, body in pages_body.items():
        pairs = extract_claim_anchors(body)
        if pairs:
            per_page[md] = pairs
            all_pairs.update(pairs)

    if not all_pairs:
        return []

    # _resolve_claim_anchors now raises ClaimDBUnavailable on a DB failure
    # (instead of returning an empty set indistinguishable from genuine drift),
    # so we can lean permissive cleanly: skip the check when the DB is down.
    try:
        resolved = _resolve_claim_anchors(all_pairs)
    except ClaimDBUnavailable:
        return []

    dangling: list[dict] = []
    for md, pairs in per_page.items():
        for stem, slug in pairs:
            if (stem, slug) not in resolved:
                dangling.append({
                    "page": md,
                    "stem": stem,
                    "slug": slug,
                })
    return dangling


_FOOTNOTE_REF_RE = re.compile(r"\[\^([^\]\s]+)\](?!:)")
_FOOTNOTE_DEF_RE = re.compile(r"^[ \t]*\[\^([^\]\s]+)\]:", re.MULTILINE)
_FENCED_CODE_RE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)


def find_undefined_footnote_refs(
    pages_body: dict[Path, str],
) -> list[dict]:
    """Return [{page, kind, detail}] for pages citing an undefined `[^id]`.

    Root meta pages are skipped: `log.md` quotes footnote ids in prose about
    citations rather than carrying citations of its own, the same reason
    `broken_wikilinks` excludes them.
    """
    out: list[dict] = []
    for md, body in sorted(pages_body.items(), key=lambda kv: str(kv[0])):
        if md.parent.name == "wiki":
            continue
        text = _FENCED_CODE_RE.sub("", body)
        refs = {m.group(1) for m in _FOOTNOTE_REF_RE.finditer(text)}
        if not refs:
            continue
        defs = {m.group(1) for m in _FOOTNOTE_DEF_RE.finditer(text)}
        missing = sorted(refs - defs)
        if not missing:
            continue
        shown = ", ".join(f"`[^{r}]`" for r in missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        out.append({
            "page": md,
            "kind": "undefined_footnote_ref",
            "detail": f"{len(missing)} footnote ref(s) with no definition: {shown}{more}"
                      " — grade synthesis will skip or under-grade the citing units",
        })
    return out
