"""Propose a category taxonomy from inbox/ papers.

Usage:
  researchwiki bootstrap-categories
  researchwiki bootstrap-categories --apply

Reads `inbox/*.pdf`, pulls title + first-page text, and asks an LLM to propose
a 5–10 category taxonomy grounded in the actual papers. Always ensures `other`
is in the final list as the catch-all bucket — papers that the per-paper
classifier abstains on land there, and the `suggest-splits` tool surfaces
splits once `other` accumulates enough papers.

A content category is defined solely by the existence of its `wiki/<slug>/`
directory — there is no shipped frozenset or CLAUDE.md table to sync. So this
tool's job is to propose slugs grounded in the papers and (with `--apply`)
create the directories.

Default mode: prints the proposed categories to stdout with the `mkdir`
commands to create them.

`--apply`: creates the `wiki/<slug>/` directories. Use after reviewing the
proposal.

Cold-start UX: if `inbox/` has fewer than 5 PDFs, exits early — too few
papers to ground a taxonomy. The user can ingest those first (they'll land
in `other`) and re-run when the corpus is bigger, or wait for `suggest-splits`
to fire automatically.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..categories import PAGE_TYPE_DIRS, content_categories
from ..log import log
from ..paths import inbox_dir, wiki_root

MIN_INBOX_FOR_BOOTSTRAP = 3   # below this, manual `--category` is fine
MIN_CATEGORIES = 2            # always at least 1 real + `other`
MAX_CATEGORIES = 10           # absolute ceiling regardless of inbox size
SLUG_RE = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$")


def _adaptive_max_categories(n_papers: int) -> int:
    """Cap the taxonomy size by inbox size.

    A 4-paper inbox shouldn't get a 7-category sprawl just because the LLM
    can dream them up; it should get a minimal taxonomy (often 2: the
    cohesive topic + `other`) unless the papers genuinely span distinct
    areas. The cap grows with corpus size to give the LLM room to find
    real seams when the evidence supports them.
    """
    if n_papers < 6:
        return 3        # cohesive small inbox: up to 2 real + other
    if n_papers < 16:
        return 5        # mid-size: up to 4 real + other
    if n_papers < 30:
        return 7        # bigger: up to 6 real + other
    return MAX_CATEGORIES   # large corpora: up to 9 real + other


def _load_prompt_system() -> str:
    path = wiki_root() / "prompts" / "bootstrap-categories-system.md"
    return path.read_text()


def _build_user_prompt(bag: list[dict], max_cats: int) -> str:
    """Format the staged-paper bag as the user message for the proposer."""
    n = len(bag)
    parts = [f"Staged papers ({n} total):", ""]
    for i, paper in enumerate(bag, 1):
        title = paper.get("title") or "(no title)"
        year = paper.get("year") or "?"
        venue = paper.get("venue") or "?"
        excerpt = (paper.get("excerpt") or "").strip()
        if len(excerpt) > 1500:
            excerpt = excerpt[:1500].rstrip() + "..."
        parts.append(f"{i}. {title} ({year}, {venue})")
        if excerpt:
            parts.append(f"   {excerpt}")
        parts.append("")

    # Adaptive cap guidance — small inbox should get a minimal taxonomy.
    parts.append("---")
    parts.append("")
    parts.append(f"**Proposal cap for this corpus**: between 2 and {max_cats} categories "
                 f"(inclusive of `other`). With {n} papers, favor the minimal taxonomy that "
                 f"actually fits the evidence:")
    if max_cats <= 3:
        parts.append("- If all papers cluster around one coherent theme, propose 2 categories: "
                     "that theme + `other`.")
        parts.append("- If they split into 2 distinct areas, propose 3.")
        parts.append("- Don't pad to fill the cap — fewer is better for a small corpus.")
    else:
        parts.append("- Cohesive corpora can still be 2 categories. Don't pad.")
        parts.append("- Diverse corpora may justify the full cap; let the seams in the evidence drive it.")
    return "\n".join(parts)


def _gather_inbox_metadata(pdfs: list[Path]) -> list[dict]:
    """For each PDF, run a lightweight reconcile (no LLM) to get title/year/
    venue, plus a first-page text excerpt as the abstract proxy.

    Falls through gracefully on per-paper failures — bootstrap doesn't need
    every paper to succeed, just enough to ground a taxonomy.
    """
    # `reconcile` is re-exported as a function via phases/__init__, not as a
    # module — import the function directly from its file to dodge the shadow.
    from ..agents.phases.reconcile import reconcile

    bag: list[dict] = []
    for pdf in pdfs:
        try:
            meta = reconcile(pdf, use_llm=False)
        except Exception as e:
            print(f"  skipped {pdf.name}: {e}", file=sys.stderr)
            continue
        excerpt = (meta.get("abstract") or meta.get("pdf_text") or "")[:2000]
        bag.append({
            "title": meta.get("title"),
            "year": meta.get("year"),
            "venue": meta.get("venue"),
            "excerpt": excerpt,
        })
    return bag


# JSON Schema for the bootstrap-categories proposer envelope. Honored by
# chat-relay; ignored by other providers. Each category needs a slug + scope;
# extra fields (rationale at the top level) are allowed but unconstrained.
_PROPOSER_SCHEMA = {
    "type": "object",
    "required": ["categories"],
    "properties": {
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["slug", "scope"],
                "properties": {
                    "slug":  {"type": "string"},
                    "scope": {"type": "string"},
                },
            },
        },
    },
}


def _call_proposer(bag: list[dict], max_cats: int) -> dict | None:
    """One Sonnet call to propose a taxonomy. Returns parsed dict or None."""
    from ..agents import llm

    system = _load_prompt_system()
    user = _build_user_prompt(bag, max_cats)
    try:
        # The `classifier` role's default max_tokens=200 is sized for the
        # per-paper classifier (single category + confidence). Bootstrap
        # output is much larger — up to MAX_CATEGORIES entries with scopes
        # plus a rationale — so override to 2000.
        resp = llm.call(
            phase="classifier",
            prompt=user,
            system=system,
            max_tokens=2000,
            schema=_PROPOSER_SCHEMA,
        )
    except Exception as e:
        print(f"ERROR: classifier call failed: {e}", file=sys.stderr)
        return None

    text = resp.text.strip()
    # Tolerate ```json fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    print("ERROR: classifier returned unparsable JSON:", file=sys.stderr)
    print(resp.text, file=sys.stderr)
    return None


def _validate_categories(parsed: dict, max_cats: int) -> list[dict] | None:
    """Validate the LLM's proposal. Returns a normalized list or None on fatal
    issues. Always ensures `other` is in the final list (appends if missing)
    and trims to `max_cats` (the adaptive ceiling for the current corpus).
    """
    cats = parsed.get("categories")
    if not isinstance(cats, list):
        print("ERROR: 'categories' field missing or not a list", file=sys.stderr)
        return None

    seen_slugs: set[str] = set()
    out: list[dict] = []
    for entry in cats:
        if not isinstance(entry, dict):
            continue
        slug = (entry.get("slug") or "").strip().lower()
        scope = (entry.get("scope") or "").strip()
        if not slug or not scope:
            continue
        if not SLUG_RE.match(slug):
            print(f"WARN: dropping invalid slug `{slug}` (must be lowercase, "
                  f"alphanumeric/hyphen)", file=sys.stderr)
            continue
        if slug in seen_slugs:
            print(f"WARN: dropping duplicate slug `{slug}`", file=sys.stderr)
            continue
        seen_slugs.add(slug)
        out.append({"slug": slug, "scope": scope})

    if "other" not in seen_slugs:
        out.append({"slug": "other", "scope": "Cross-cutting / not-yet-classified backlog. Auto-monitored: `suggest-splits` proposes promotions when this category passes its threshold."})

    if len(out) < MIN_CATEGORIES:
        print(f"ERROR: only {len(out)} valid categories — need at least "
              f"{MIN_CATEGORIES}", file=sys.stderr)
        return None
    if len(out) > max_cats:
        print(f"WARN: {len(out)} categories proposed; trimming to {max_cats} "
              f"(corpus size cap)", file=sys.stderr)
        # Always keep `other`. Drop from the tail of the others.
        non_other = [c for c in out if c["slug"] != "other"][:max_cats - 1]
        other_entry = next(c for c in out if c["slug"] == "other")
        out = non_other + [other_entry]

    return out


def _content_slugs(cats: list[dict]) -> list[str]:
    """Proposed slugs that are real content categories to create — excludes
    `other` and the page-type scaffold dirs, which already exist."""
    return [c["slug"] for c in cats
            if c["slug"] != "other" and c["slug"] not in PAGE_TYPE_DIRS]


def _print_proposal(cats: list[dict], rationale: str, n_papers: int) -> None:
    print()
    print(f"Proposed taxonomy for {n_papers} paper(s) in inbox/:")
    print()
    for c in cats:
        print(f"  {c['slug']:<16} {c['scope'][:90]}")
    print()
    if rationale:
        print(f"Rationale: {rationale}")
        print()
    slugs = _content_slugs(cats)
    print("A category is valid once its directory exists. Create them with:")
    print("  mkdir -p " + " ".join(f"wiki/{s}" for s in slugs))
    print()
    print("Or re-run with `--apply` to create the directories automatically.")
    print()


def _apply_taxonomy(cats: list[dict], rationale: str) -> int:
    """Create the content-category directories under wiki/. A content category
    is defined solely by the existence of `wiki/<slug>/` — there is no
    frozenset or CLAUDE.md table to rewrite. `other` and the page-type scaffold
    dirs are skipped (they already exist)."""
    root = wiki_root()
    wiki = root / "wiki"
    slugs = _content_slugs(cats)
    created: list[str] = []
    for s in slugs:
        d = wiki / s
        if d.exists():
            print(f"exists  {d.relative_to(root)}/")
        else:
            d.mkdir(parents=True, exist_ok=True)
            print(f"created {d.relative_to(root)}/")
            created.append(s)

    log_msg = (f"bootstrap-categories | created {len(created)} categor(ies): "
               f"{', '.join(created) or '(none new)'}")
    if rationale:
        log_msg += f"\nRationale: {rationale[:200]}"
    log(log_msg, tag="bootstrap-categories")
    print()
    print("Done — categories are valid now that their wiki/<slug>/ dirs exist.")
    print("Run `researchwiki reindex` to pick up the new directory structure.")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki bootstrap-categories",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--apply", action="store_true",
                        help="Create the proposed wiki/<slug>/ directories (which "
                             "is what makes each category valid). Without this flag, "
                             "prints the proposal and the mkdir commands to stdout.")
    args = parser.parse_args(argv)

    inbox = inbox_dir()
    pdfs = sorted(inbox.glob("*.pdf"))
    if len(pdfs) < MIN_INBOX_FOR_BOOTSTRAP:
        print(f"only {len(pdfs)} PDF(s) in inbox/ — need ≥{MIN_INBOX_FOR_BOOTSTRAP} "
              f"to ground a taxonomy.")
        print()
        print("Tip: drop more PDFs into inbox/ first. Until then, ingest individually —")
        print("papers without a confident classification land in `other`, and")
        print("`suggest-splits` will propose promotions once that bucket grows.")
        print()
        print(f"Current categories: {sorted(content_categories())}")
        return 0

    print(f"Reconciling metadata for {len(pdfs)} PDF(s)...", file=sys.stderr)
    bag = _gather_inbox_metadata(pdfs)
    if len(bag) < MIN_INBOX_FOR_BOOTSTRAP:
        print(f"only {len(bag)} PDF(s) had extractable metadata — too few "
              f"to ground a taxonomy.", file=sys.stderr)
        return 1

    max_cats = _adaptive_max_categories(len(bag))
    print(f"Calling classifier with {len(bag)} paper(s) "
          f"(taxonomy cap: {max_cats})...", file=sys.stderr)
    parsed = _call_proposer(bag, max_cats)
    if parsed is None:
        return 1

    cats = _validate_categories(parsed, max_cats)
    if cats is None:
        return 1

    rationale = (parsed.get("rationale") or "").strip()
    if args.apply:
        return _apply_taxonomy(cats, rationale)
    _print_proposal(cats, rationale, len(bag))
    return 0
