"""Surface unreferenced wiki pages with high BM25 affinity to a synthesis page's topic_seed.

The third gate alongside `check-grounding` (structural) and `grade synthesis`
(fidelity). Coverage answers the recall question: are there papers in the wiki
that the topic_seed says belong on this page, but which the page doesn't cite?

Usage:
  researchwiki check-coverage <file.md>
  researchwiki check-coverage <file.md> --top-n 20
  researchwiki check-coverage <file.md> --json

Why this exists. Pattern from Starling (jones-2026-self-driving-datasets-from-20):
its query-construction phase iterates until estimated recall gap ≤ 15% — i.e., the
agent itself notices when its filters are missing relevant content and refines.
For a synthesis or idea page in this wiki, the analogue is: after authoring,
re-rank wiki pages against the page's own `topic_seed`, and surface the top-N
that aren't already cited. The author then decides whether to cite them or note
their exclusion.

`lint`'s `stale_by_content` runs the same retrieval logic post-hoc across the
whole corpus; this CLI lifts the signal into the per-page authoring loop, before
the page is committed.

**Two retrieval signals, not one.** The page-level pass above ranks whole pages
against the seed, which mixes a paper's contribution with its intro, discussion
and title vocabulary — so a genuinely relevant paper can rank below noise, and a
hit's score says nothing about *why* it was retrieved. A second pass therefore
ranks **contribution claims** against the seed and reports the matching claim as
evidence.

This exists because of a concrete miss. Building `wiki/concepts/mixture-model.md`,
this gate ranked `van-iterson-2017` second (score 4.64) among hits that also
included two unrelated papers scoring 4.47 and 4.24. It was the page's single
most valuable omission — it carries the disagreement the hub is built around —
and the only way to know that was to go query its claims by hand. The claim pass
puts that evidence in the report and sorts claim-backed hits first. It only
annotates rows the page-level pass already produced — it never adds one.

Exit codes:
  0  No unreferenced top-N hits — coverage looks complete for this seed.
  1  ≥1 unreferenced hit — review and decide cite-or-exclude.
  2  Bad input: missing path, missing topic_seed, search backend unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .lint.staleness import unreferenced_top_hits
from .lint.walk import all_pages, page_key
from ..log import log


# Cosine floor for the claim pass. Stricter than `semantic_members.DEFAULT_FLOOR`
# (0.70) because this report already carries the page-level hit list, and a long
# advisory tail trains the reader to skip the whole thing. 0.74 sits in the gap
# the calibration case measured: its lowest true member scored 0.744 and its
# highest false positive 0.733.
_CLAIM_FLOOR = 0.74

# How deep to rank claim matches. Only used to annotate rows the page-level pass
# already produced, so this bounds work, not output — it must comfortably exceed
# `--top-n` or annotation becomes a function of unrelated papers' scores.
_CLAIM_RANK_DEPTH = 200

def _read_frontmatter(md: Path) -> dict | None:
    """Mirrors db/rebuild.py:_parse_frontmatter — same forgiving YAML parse."""
    try:
        text = md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    try:
        fm = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError:
        return None
    return fm if isinstance(fm, dict) else {}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki check-coverage",
        description=(__doc__ or "").splitlines()[0],
    )
    parser.add_argument("path", help="Synthesis or idea page to check.")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Retrieve top-N more_like_text hits and report any "
                             "not already cited. Default 20.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit a structured report instead of prose.")
    args = parser.parse_args(argv)

    md = Path(args.path).resolve()
    if not md.is_file():
        print(f"check-coverage: not a file: {md}", file=sys.stderr)
        return 2

    fm = _read_frontmatter(md)
    if fm is None:
        print(f"check-coverage: malformed frontmatter in {md}", file=sys.stderr)
        return 2
    seed = (fm.get("topic_seed") or "").strip() if isinstance(fm.get("topic_seed"), str) else ""
    if not seed:
        print(f"check-coverage: {md} has no `topic_seed` in frontmatter — nothing to check.",
              file=sys.stderr)
        return 2

    try:
        from ..search import get_default_backend, SearchBackendUnavailable
    except ImportError as e:
        print(f"check-coverage: search backend import failed: {e}", file=sys.stderr)
        return 2
    try:
        backend = get_default_backend()
        backend.query("__probe__", limit=1)
    except SearchBackendUnavailable:
        print("check-coverage: search index not built — run `researchwiki reindex`.",
              file=sys.stderr)
        return 2
    # No broad `except Exception` below: SearchBackendUnavailable (an
    # EnvironmentFailure) is the one recoverable failure mode for a probe query
    # and it's caught above with an actionable message. Anything else is a bug
    # in the query path — let it reach the CLI funnel for a traceback and code 3
    # rather than reporting a real bug as "search backend probe failed".

    known = {page_key(p) for p in all_pages()}
    unreferenced = unreferenced_top_hits(backend, md, seed, known, top_n=args.top_n)

    # Claim-level pass. Same machinery the concept scaffolder uses, pointed at
    # the page's seed instead of a concept term: papers whose *contribution*
    # claims sit near the seed, excluding everything the page already cites.
    from .lint.walk import extract_links
    linked = extract_links(md.read_text(encoding="utf-8"), known)
    cited_stems = {k.split("/")[-1] for k in linked} | {md.stem}
    try:
        from ..concepts.semantic_members import semantic_member_candidates
        # Limit generously rather than to top_n. This ranks every paper in
        # the corpus above the floor, but is only ever used to annotate rows
        # already in `unreferenced` — so a tight cap doesn't shorten the
        # report, it just drops the annotation whenever N unrelated papers
        # happen to outscore a genuine match against the seed.
        claim_hits = semantic_member_candidates(
            seed, exclude_stems=cited_stems, limit=_CLAIM_RANK_DEPTH,
            floor=_CLAIM_FLOOR,
        )
    except Exception:
        claim_hits = []      # advisory enrichment — never fail the gate on it

    # Annotation only: a claim hit adds evidence to a page the report already
    # lists, never a new row. That keeps the pass incapable of introducing a
    # false positive, which is why it needs no lexical guard of its own.
    by_stem = {c.stem: c for c in claim_hits}
    for h in unreferenced:
        c = by_stem.get(h["stem"])
        if c is not None:
            h["claim_score"] = round(c.score, 3)
            h["claim_slug"] = c.claim_slug
            h["claim_section"] = c.section
            h["claim_text"] = c.text

    if args.as_json:
        report = {
            "path": str(md),
            "topic_seed": seed,
            "top_n": args.top_n,
            "unreferenced": unreferenced,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if not unreferenced else 1

    log(f"{md}", tag="check-coverage")
    print(f"  topic_seed: {seed!r}")
    print(f"  top_n     : {args.top_n}")
    if not unreferenced:
        print("  ✓ no unreferenced top-N hits — coverage looks complete for this seed.")
        return 0

    n_backed = sum(1 for h in unreferenced if "claim_score" in h)
    note = f" — {n_backed} with a matching contribution claim" if n_backed else ""
    print(f"  ⚠ {len(unreferenced)} unreferenced hit(s){note}; "
          f"review and decide cite-or-exclude:")
    # Claim-backed hits first: a page that ranks *and* has a contribution claim
    # near the seed is a stronger candidate than a page-text match.
    for h in sorted(unreferenced, key=lambda x: -x.get("claim_score", 0.0)):
        flag = "  ← claim match" if "claim_score" in h else ""
        print(f"    score={h['score']:.2f}  [[{h['key']}]]{flag}")
        if h.get("title"):
            print(f"      › {h['title']}")
        if "claim_score" in h:
            print(f"      {h['claim_score']:.3f} [[{h['stem']}#{h['claim_slug']}]] "
                  f"({h['claim_section']})")
            print(f"        › {h['claim_text'][:120]}")

    return 1
