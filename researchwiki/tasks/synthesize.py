"""Scaffold a synthesis wiki page.

✅ Use when: a cross-paper question produced a non-trivial answer and you
   want to persist it (Query → File loop). Synthesis pages cover anything
   multi-paper that isn't itself a paper — trajectory of a technique,
   side-by-side comparison, status-of-the-field snapshot, recurring concept
   aggregated across papers.
❌ Don't use: to create paper pages (those come from `ingest`). Don't
   scaffold a synthesis from zero papers — list the actual referenced
   wiki stems via `--papers`, otherwise the page violates Rule 1.

Creates a YAML-compliant stub page in `wiki/synthesis/` laid out in the fixed
synthesis structure (CLAUDE.md §2; procedure in
`prompts/synthesis-page-author.md`). The LLM then fills the body, grounding
every claim in `[[wikilinks]]` to real wiki papers (Rule 1).

The pre-pulled claim evidence goes to `.ingest/synthesis/<slug>/evidence.md`,
not into the page: a field-scale page cites dozens of papers, and inlining
every claim would bury the structure the author is meant to fill.

Exit codes: 0 = stub written; 1 = user-input error (target page exists
without `--force`, or title produces an empty slug).

Usage:
  researchwiki synthesize --title "DNA foundation models" \\
      --papers smith-2024-... jones-2025-...

  researchwiki synthesize --title "CRISPR off-target strategies" \\
      --papers smith-2024-... jones-2025-... lee-2026-...
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date

from ..categories import PAGE_TYPE_DIRS
from ..fsatomic import write_text_atomic
from ..log import append_log_md, log
from ..paths import ingest_dir, wiki_dir
from ..stems import slugify_phrase
from ..wiki import commit_page, find_stem_collision, read_pages


def _dominant_category(referenced: list[str]) -> str:
    """Content category for a synthesis page: the most common category prefix
    among its referenced papers (page-type dirs excluded).

    A synthesis lives in `wiki/synthesis/` (its page *type*, carried by
    `type:` and the directory), so its YAML `category:` should name the
    *content* field it surveys — classified the same way a paper page is,
    which for a cross-paper page is just the field its papers belong to.
    Falls back to `TODO` when no referenced paper carries a content-category
    prefix (the author fills it), mirroring the paper-page auto-suggest
    fallback. Ties break toward the first-referenced category (Counter keeps
    insertion order)."""
    cats: Counter[str] = Counter()
    for r in referenced:
        prefix = r.split("/", 1)[0] if "/" in r else ""
        if prefix and prefix not in PAGE_TYPE_DIRS:
            cats[prefix] += 1
    return cats.most_common(1)[0][0] if cats else "TODO"


def _slugify(title: str) -> str:
    """Page slug for a synthesis title. Thin alias over the shared helper —
    `concepts.candidates._term_slug` must produce byte-identical output for the
    same input, or a scaffolded hub's filename stops matching the edge that
    points at it. Sharing one implementation is what enforces that."""
    return slugify_phrase(title)


def _resolve_paper(stem_or_link: str, known: dict[str, str]) -> str | None:
    """Resolve `stem` or `category/stem` to `category/stem`, else None."""
    s = stem_or_link.strip().strip("[]").strip()
    if "/" in s:
        return s if s in set(known.values()) else None
    return known.get(s)


def _format_claim_bullet(hit: dict) -> str:
    """Render one claim_lookup/claims_by_stem hit as a synthesis evidence bullet.

    Uses the durable `[[stem#slug]]` citation form (falls back to bare
    `[[stem]]` when a claim predates the slug migration). Keeps the section
    anchor for traceability, surfaces the grounding scores so weak units are
    visible. Truncates claim text to ~240 chars — enough to make sense,
    short enough to keep the stub scannable.
    """
    from ..search import format_claim_ref
    cite = format_claim_ref(hit)
    section = hit.get("section", "")
    pos = hit.get("position", 0)
    text = (hit.get("text") or "").strip()
    if len(text) > 240:
        text = text[:237].rstrip() + "..."
    sem = hit.get("semantic_score")
    bm = hit.get("bm25_top1")
    score_parts = []
    if sem is not None:
        score_parts.append(f"sem={sem:.2f}")
    if bm is not None:
        score_parts.append(f"bm25={bm:.1f}")
    if hit.get("graded") is False:
        score_parts.append("ungraded")
    score_tail = f" *({', '.join(score_parts)})*" if score_parts else ""
    return (
        f"- {cite} ({section}#{pos}){score_tail}\n"
        f"  > {text}"
    )


def _evidence_block(
    topic_seed: str | None,
    referenced: list[str],
    seed_k: int,
    per_paper_k: int | None,
) -> str:
    """Build the pre-grounded claim evidence the author drafts from.

    Two sub-sections, both populated from the structured claims table so
    every line is cite-ready by `[[stem#claim_slug]]`:

      1. Topic-seed hits: top-K claim_lookup matches against `--topic-seed`.
         These cross paper boundaries — the LLM curates which to keep.
      2. Per-referenced-paper claims: for each `--papers` stem, dump its
         claims (capped at `per_paper_k` to keep the stub scannable;
         `None` = uncapped).

    Returns markdown for `.ingest/synthesis/<slug>/evidence.md`.
    """
    from ..search import claim_lookup, claims_by_stem

    def _safe_comment(text: str) -> str:
        # HTML comments close at `-->`. If interpolated text (an exception
        # message, or the user-supplied topic_seed) contains that sequence it
        # would close the diagnostic early and spill the tail into the rendered
        # stub. Neutralize by inserting a zero-width space between the dashes.
        return text.replace("-->", "--​>")

    out: list[str] = []

    if topic_seed:
        out.append("### Topic-seed hits")
        out.append(
            f"<!-- claim_lookup({_safe_comment(repr(topic_seed))}, k={seed_k}). "
            "Curate: keep the units relevant to the question, drop the rest. "
            "Cite by [[stem#slug]] in the page prose (durable content-"
            "addressed anchor; see the bullet heads below). -->"
        )
        try:
            hits = claim_lookup(topic_seed, k=seed_k)
        except Exception as e:
            hits = []
            out.append(f"<!-- claim_lookup failed: {_safe_comment(str(e))} -->")
        if hits:
            out.append("")
            for h in hits:
                out.append(_format_claim_bullet(h))
                out.append("")
        else:
            out.append("")
            out.append("<!-- (no claim_lookup hits — broaden topic_seed or gather claims by hand) -->")
            out.append("")

    if referenced:
        out.append("### Per-paper claims")
        out.append(
            "<!-- The full citable surface of each --papers entry. "
            "Pulled directly from the graded claims table. -->"
        )
        for r in referenced:
            stem = r.split("/", 1)[1] if "/" in r else r
            try:
                paper_hits = claims_by_stem(stem)
            except Exception as e:
                paper_hits = []
                out.append("")
                out.append(f"#### [[{r}]]")
                out.append(f"<!-- claims_by_stem failed: {_safe_comment(str(e))} -->")
                continue
            if per_paper_k is not None:
                paper_hits = paper_hits[:per_paper_k]
            out.append("")
            out.append(f"#### [[{r}]]")
            if not paper_hits:
                out.append("<!-- no claims indexed for this paper "
                           "(run `researchwiki grade` to populate) -->")
                continue
            for h in paper_hits:
                out.append(_format_claim_bullet(h))
                out.append("")

    if not out:
        return ""
    return "\n".join(out).rstrip() + "\n"


# The fixed synthesis structure: (H2, purpose comment). Names are exact —
# `Outlook` is the section the grounding gate labels and `Short answer` is what
# the semantic index embeds first — and lint's `synthesis_contract` checks them.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("Question", "Scope: what the page covers, what it leaves out and why, and "
                 "links to neighbouring synthesis pages."),
    ("Short answer", "The argument in miniature: one framing sentence, then 4–6 "
                     "takeaways. Write last."),
    ("Background", "What a newcomer needs first: the problem, key terms, the "
                   "shared experimental setup."),
    ("Organizing framework", "How this page sorts the evidence and why that "
                             "explains the field, then a table placing every "
                             "cited paper (bare [[stem]] links)."),
    ("Findings", "One H3 per theme the framework defines, never one per paper: "
                 "point → evidence → conditions → limits."),
    ("Cross-cutting insights", "What appears only when the themes are read "
                               "together: agreements, conflicts, shared "
                               "mechanisms, trade-offs."),
    ("Tensions / open questions", "Genuine disagreements, with the evidence on "
                                  "each side."),
    ("Outlook", "Where the field is heading. Cite the papers' own next steps; "
                "label your conclusions *(inference)* with the papers they follow "
                "from, and background knowledge *(model prior)*."),
    ("References", "[^id]: [[category/stem]], one per source paper."),
)


def _template(
    title: str,
    referenced: list[str],
    topic_seed: str | None = None,
    *,
    evidence_path: str | None = None,
) -> str:
    today = date.today().isoformat()
    seed_line = (
        f'topic_seed: "{topic_seed.replace(chr(34), chr(39))}"'
        if topic_seed else None
    )
    category = _dominant_category(referenced)
    cat_line = (
        "category: [TODO]  # content field this page surveys; set to a valid "
        "content category (the type is carried by type:/the synthesis/ dir)"
        if category == "TODO" else f"category: [{category}]"
    )

    # No `referenced_papers:` YAML — synthesis pages cite via the body
    # (## References footnotes + inline [[wikilink]]s), which is the single
    # source of truth the framework reads. (Concept pages still use the field.)
    yaml = [
        f'title: "{title}"',
        "type: synthesis",
        cat_line,
    ]
    if seed_line:
        yaml.append(seed_line)
    yaml.append(f"generated_at: {today}")
    # `author_model:` is set by the conversational author when filling in
    # prose. Stamped as TODO here so lint forces the author to replace it with
    # the exact model id; a scaffold cannot know which model will complete it.
    yaml.append('author_model: "TODO"  # the LLM model that authored this synthesis (e.g. claude-sonnet-4-6, claude-opus-4-7); fill on save')
    yaml.append("tags: [synthesis]")

    parts = []
    if evidence_path:
        parts.append(f"<!-- Pre-pulled claim evidence: {evidence_path} -->\n\n")
    for heading, purpose in SECTIONS:
        parts.append(f"## {heading}\n<!-- {purpose} -->\n\n")
    body = "".join(parts).rstrip() + "\n"

    yaml_block = "---\n" + "\n".join(yaml) + "\n---\n\n"
    return yaml_block + body


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki synthesize",
        description="Scaffold a synthesis wiki page.",
    )
    parser.add_argument("--title", required=True, help='Human-readable title, e.g. "DNA foundation models"')
    parser.add_argument("--slug", default=None,
                        help="Filename stem. Defaults to a slugified title.")
    parser.add_argument("--papers", nargs="*", default=[],
                        help="Referenced wiki paper stems (bare stem or category/stem).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing page with the same slug.")
    parser.add_argument("--topic-seed", default=None,
                        help="Search-index seed string describing the page's topic "
                             "(e.g., 'CRISPR off-target prediction'). `lint` uses this "
                             "to detect when new papers topically-relevant to the page "
                             "enter the wiki without being cited — i.e., when the page "
                             "becomes stale-by-content rather than stale-by-mtime. "
                             "Also seeds the evidence file: top-K claim_lookup hits "
                             "as cite-ready [[stem#slug]] bullets.")
    parser.add_argument("--seed-k", type=int, default=10,
                        help="Number of topic-seed claim_lookup hits to pull (default 10). "
                             "Ignored when --topic-seed is omitted.")
    parser.add_argument("--per-paper-k", type=int, default=None,
                        help="Max claims to pull per --papers entry. Default: all "
                             "(typically 10–25 per paper).")
    args = parser.parse_args(argv)

    slug = args.slug or _slugify(args.title)
    if not slug:
        log(f"could not derive slug from title `{args.title}`", tag="synthesize")
        return 1

    # Use the unfiltered page walk — `read_wiki_papers()` filters out pages
    # missing a `doi:` field, which silently drops valid stems on `--papers`
    # (e.g. liao-2025-dualmpnn-... was in 4/4 NEW syntheses' --papers list,
    # hit `missing_doi`, got dropped without surfacing the WARN unless
    # stderr was visible). The synthesize CLI cares about stem-existence,
    # not DOI completeness.
    existing_papers = {p.stem: f"{p.category}/{p.stem}" for p in read_pages()}

    resolved: list[str] = []
    missing: list[str] = []
    for p in args.papers:
        r = _resolve_paper(p, existing_papers)
        (resolved if r else missing).append(r or p)
    if missing:
        log(f"WARN: papers not found in wiki: {', '.join(missing)}", tag="synthesize")

    target_dir = wiki_dir() / "synthesis"
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / f"{slug}.md"

    if out.exists() and not args.force:
        log(f"page already exists: {out} (use --force to overwrite)", tag="synthesize")
        return 1

    if (hit := find_stem_collision(slug)) is not None and hit != out:
        log(f"WARN: stem `{slug}` already used at {hit} (different category)", tag="synthesize")

    evidence_block = _evidence_block(
        topic_seed=args.topic_seed,
        referenced=resolved,
        seed_k=args.seed_k,
        per_paper_k=args.per_paper_k,
    )

    evidence_rel: str | None = None
    if evidence_block:
        evidence_file = ingest_dir() / "synthesis" / slug / "evidence.md"
        evidence_file.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(evidence_file, f"# Evidence for {args.title}\n\n{evidence_block}")
        evidence_rel = f".ingest/synthesis/{slug}/evidence.md"
        log(f"wrote {evidence_file}", tag="synthesize")

    content = _template(
        title=args.title,
        referenced=resolved,
        topic_seed=args.topic_seed,
        evidence_path=evidence_rel,
    )
    write_text_atomic(out, content)
    commit_page(out)
    if missing:
        log(f"wrote {out}  ⚠ {len(missing)}/{len(args.papers)} paper(s) missing "
            f"from the requested --papers list (see WARN above)", tag="synthesize")
    else:
        log(f"wrote {out}  ({len(resolved)}/{len(args.papers)} paper(s) resolved)",
            tag="synthesize")

    headline = f"synthesis: {args.title} → wiki/synthesis/{slug}.md"
    details = (
        f"Referenced papers: {', '.join(r.split('/', 1)[1] for r in resolved) if resolved else '(none)'}."
    )
    if missing:
        details += f" Unresolved: {', '.join(missing)}."
    append_log_md("synthesize", headline, details)

    # Advisory grounding summary — synthesis stubs are skeletal at write time
    # (every section is a placeholder comment), so this report is
    # informational, not a gate. The point is to remind the LLM author that
    # any prose they add later will be checked against this surface.
    try:
        from ..grade import grounding
        report = grounding.check(content)
        if report.total_claims:
            log(
                f"grounding: {report.grounded_claims}/{report.total_claims} "
                f"claim-units cited ({report.coverage * 100:.0f}%) — "
                f"run `researchwiki check-grounding {out}` after authoring",
                tag="synthesize",
            )
    except Exception as e:
        # Advisory only — never fail synthesis creation over the preview.
        # Log rather than swallow silently so a real bug in grounding.check
        # surfaces instead of vanishing.
        log(f"grounding preview skipped ({type(e).__name__}: {e})", tag="synthesize")

    print(out)
    return 0
