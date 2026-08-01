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
    print(f"  ⚠ {len(unreferenced)} unreferenced hit(s) — review and decide cite-or-exclude:")
    for h in unreferenced:
        print(f"    score={h['score']:.2f}  [[{h['key']}]]")
        if h.get("title"):
            print(f"      › {h['title']}")
    return 1
