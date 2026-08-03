"""Ingest one or more PDFs into wiki-ready digests.

✅ Use when: a PDF has been dropped into `inbox/` and you want to add it to
   the wiki. Also the normal path for replacing a failed-parsing entry — move
   the new PDF into `inbox/` and re-run.
❌ Don't use: to rewrite a wiki page for a paper that's already ingested
   (edit the page directly, or delete + re-ingest). Not a search tool.

For each PDF the pipeline runs:
  1. pypdfium2 text extraction
  2. Detect DOI from PDF metadata or first-page regex
  3. Provider lookup (default: Semantic Scholar) by DOI or title fallback
  4. Canonical stem derivation per CLAUDE.md rules
  5. Outgoing + incoming cross-link intersection with existing wiki DOIs
     (with fallback chain: S2 → Crossref → PDF-text DOI harvest)
  6. Section anchoring (Introduction / Methods / Results / Discussion / References)
  7. Dedup guards (DOI already in wiki OR stem collides OR papers/{stem}.pdf exists)
  8. Move inbox/<raw>.pdf → papers/{stem}.pdf
  9. Category auto-suggestion via the search index when --category is omitted
  10. Write `.ingest/{stem}-digest.md` as the LLM's single input

Exit codes: 0 = all PDFs produced digests or were skipped as duplicates;
1 = one or more PDFs hit an unrecoverable error (see per-paper logs).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ..fsatomic import write_text_atomic
from ..log import append_log_md, log
from ..paths import ingest_dir, papers_dir
from ..pdf.text import detect_doi, dois_as_s2_refs, extract_pdf, extract_ref_dois
from ..providers import ScholarlyArticle, get_default_provider
from ..pdf.sections import anchor_sections
from ..stems import derive_stem, first_author_surname
from ..wiki import (
    classify_pdf_collision,
    find_stem_collision,
    intersect_crosslinks,
    intersect_incoming,
    read_wiki_dois,
)


# ---------- digest writer ----------

def _write_digest(
    stem: str,
    pdf_dest: Path,
    article: ScholarlyArticle,
    refs: list[ScholarlyArticle],
    crosslinks: list[dict],
    incoming: list[dict],
    recommendations: list[ScholarlyArticle],
    pdf_text: str,
    sections: dict[str, str],
    category_hint: str | None,
    raw_source_pdf: Path,
    category_source: str = "user-supplied",
    staged_supp: list[dict] | None = None,
) -> Path:
    out_dir = ingest_dir()
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{stem}-digest.md"

    title = article.title
    authors_list = article.authors
    year = article.year
    venue = article.venue
    doi = (article.doi_lower or "")
    abstract = article.abstract
    tldr = article.tldr
    ref_count = article.reference_count
    cite_count = article.citation_count

    yaml_authors = ", ".join(authors_list) if authors_list else "(unknown)"
    if category_hint:
        provenance = "" if category_source == "user-supplied" else f"  # {category_source}"
        category_line = f"category: [{category_hint}]{provenance}"
    else:
        # Abstention fallback. Lands the page in the structured "uncategorized
        # backlog" bucket; `suggest-splits` proposes promotions later.
        category_line = "category: [other]  # auto-fallback (no category hint)"
    supp_yaml = ""
    if staged_supp:
        lines = ["supplementary:"]
        for s in staged_supp:
            lines.append(f"  - file: {s['filename']}")
            lines.append(f"    kind: {s['kind']}")
        supp_yaml = "\n" + "\n".join(lines)

    yaml_block = f"""---
title: "{title}"
authors: {yaml_authors}
year: {year if year else 'TODO'}
doi: {doi or 'TODO'}
{category_line}
pdf_path: "[[{pdf_dest.name}]]"
source_collection: external{supp_yaml}
author_model: "TODO"  # which LLM authored this page (e.g. claude-sonnet-4-6, claude-opus-4-7); fill on save
tags: []
---"""

    # Outgoing cross-links (wiki papers this paper cites)
    crosslink_lines = [
        f"- {c['wikilink']} — DOI {c['doi']} ({c['year']}): {c['title'][:120]}"
        for c in crosslinks
    ]
    crosslink_block = "\n".join(crosslink_lines) if crosslink_lines else "(none — S2 reference intersection with wiki empty)"

    # Incoming citations (wiki papers that cite THIS paper)
    incoming_lines = [
        f"- {c['wikilink']} — DOI {c['doi']} ({c['year']}): {c['title'][:120]}"
        for c in incoming
    ]
    incoming_block = "\n".join(incoming_lines) if incoming_lines else "(none — no wiki paper in the audit currently cites this one)"

    # Recommendations (skip papers already in wiki)
    wiki_dois = read_wiki_dois()
    rec_lines: list[str] = []
    for r in recommendations[:20]:
        rdoi = (r.doi_lower or "")
        if not rdoi or rdoi in wiki_dois:
            continue
        rec_lines.append(f"- `{rdoi}` ({r.year}) — {r.title[:140]}")
    rec_block = "\n".join(rec_lines) if rec_lines else "(none)"

    # References listing (titles + DOIs)
    ref_lines: list[str] = []
    for r in refs[:60]:
        rdoi = (r.doi_lower or "")
        ref_lines.append(f"- {r.title[:130]} ({r.year}) `{rdoi or 'no-doi'}`")
    ref_block = "\n".join(ref_lines) if ref_lines else "(none)"

    # Section excerpts
    def _section_block(name: str, content: str) -> str:
        if not content:
            return f"### {name.title()}\n_(not anchored in PDF text)_"
        return f"### {name.title()} (≤4000 chars)\n```\n{content}\n```"
    sections_rendered = "\n\n".join(
        _section_block(n, sections.get(n, ""))
        for n in ("introduction", "methods", "results", "discussion")
    )

    preamble = pdf_text[:3500]

    # Draft index.md line
    hook_source = tldr or abstract or ""
    hook = hook_source.split(". ")[0].strip()
    if hook and not hook.endswith("."):
        hook += "."
    if len(hook) > 240:
        hook = hook[:237].rstrip() + "…"
    if not hook:
        hook = "_(add one-sentence hook from Summary)_"
    category_slug = category_hint or "TODO-category"
    venue_display = f"*{venue}* " if venue else ""
    year_display = year if year else "YYYY"
    index_line_draft = f"- [[{category_slug}/{stem}]] — **TODO: short name** ({venue_display}{year_display}): {hook}"

    body = f"""# Ingest digest: {stem}

_Generated by `researchwiki ingest` from `{raw_source_pdf}`._

This digest is the LLM's single input for writing `wiki/{{category}}/{stem}.md`.
All fields below are pre-computed so the LLM does not need to re-extract them from the PDF.

## S2 metadata

- **Title**: {title}
- **Authors**: {yaml_authors}
- **Year**: {year}
- **Venue**: {venue}
- **DOI**: {doi}
- **S2 reference count**: {ref_count}, **citation count**: {cite_count}
- **Suggested stem**: `{stem}`
- **PDF moved to**: `{pdf_dest}`

## YAML frontmatter (copy directly into wiki page)

```yaml
{yaml_block[4:-4]}
```

## Summary seed — S2 abstract (verbatim, authors' text; Rule 1 abstract exception)

{abstract or '_(no S2 abstract available)_'}

## TLDR — S2 SciTLDR output (AI-generated, verify against PDF before keeping)

{tldr or '_(no TLDR)_'}

## Outgoing cross-links (wiki papers CITED BY this paper, verified via S2 /references)

{crosslink_block}

**Action for LLM**: add each of these as `[[wikilink]]` in the wiki page's Related Papers section with a one-line relationship. Direction: THIS paper → cited paper.

## Incoming cross-links (wiki papers that CITE this paper, verified via S2 /citations)

{incoming_block}

**Action for LLM**: for each incoming citation, add a `[[wikilink]]` in the NEW wiki page's Related Papers section (direction: cited-by → THIS paper), AND open each citing wiki page and add a back-link to the new page.

## S2 recommendations (candidates for future ingestion, not to be summarized here)

{rec_block}

## Extracted PDF section text (best-effort anchored)

These are raw PDF excerpts. Compress / summarize them when filling Key Contributions, Methodology, Results, and Limitations sections of the wiki page. Do NOT copy verbatim.

### First-page preamble (title block + intro, first 3500 chars)
```
{preamble}
```

{sections_rendered}

## References list (first 60 titles + DOIs)

{ref_block}

## Draft index.md entry (polish the short name; verify one-liner)

Target section: `## {category_hint or "TODO-category"}`.

```markdown
{index_line_draft}
```

The short name is a TODO — pick the most recognizable handle (e.g., "Evo 2", "MMseqs2", "Cas9-EDVs", "Bridge RNAs"). The one-line hook is seeded from the TLDR / abstract; compress to one sentence if needed.

---

## Instructions for the LLM

1. Create `wiki/{{category}}/{stem}.md` using the YAML block above (fill in `category` if marked TODO).
2. Fill the 6 standard sections per CLAUDE.md caps:
   - **Summary** (≤150 words): compress the S2 abstract + any extra context from PDF first-page preamble.
   - **Key Contributions** (≤10 bullets): extract from Methods + Results sections below.
   - **Methodology and Architecture** (≤200 words): distill Methods section.
   - **Results** (≤200 words, prefer table): distill Results section.
   - **Limitations** (≤100 words): Discussion section + obvious gaps.
   - **Related Papers** (≤6 entries): use both the **outgoing** and **incoming** cross-link candidates above. For each one, use `[[wikilink]]`; include the direction of citation in the one-liner.
3. **For every INCOMING cross-link**, open the citing wiki page and add a `[[wikilink]]` back to the new page in its Related Papers section. This closes the graph — both ends of the edge should reference each other.
4. Paste the draft `index.md` line into the right category section (polish the short name).
5. No Glossary section, no Document Information section (YAML has everything).
6. Delete this digest file when done: `rm .ingest/{stem}-digest.md`.
"""
    write_text_atomic(out, body)
    return out


# ---------- per-paper pipeline ----------

def process_one(src_pdf: Path, category_hint: str | None, no_move: bool,
                doi_override: str | None = None,
                supplementary: list[Path] | None = None) -> Path | None:
    """Full pipeline for a single PDF. Returns digest path on success, None on dedup skip."""
    if not src_pdf.exists():
        log(f"PDF not found: {src_pdf}")
        return None

    log(f"extracting text from {src_pdf.name}")
    pdf_text, pdf_meta = extract_pdf(src_pdf)

    if doi_override:
        doi = doi_override.strip().lower()
        log(f"DOI override (user-supplied): {doi}")
    else:
        doi = detect_doi(pdf_meta, pdf_text)
        log(f"DOI detected: {doi}")

    # Compute the wiki DOI index once and reuse it for both the dedup check
    # here and the cross-link intersection after the provider calls — nothing
    # writes a wiki page in between, so a second full rglob+YAML walk would
    # only re-derive the identical dict.
    wiki_dois = read_wiki_dois()

    # Dedup layer 1: DOI already in wiki? Skip before any provider calls.
    if doi:
        existing = wiki_dois.get(doi.lower())
        if existing:
            log(f"SKIP (duplicate): DOI {doi} is already in wiki as [[{existing}]]. "
                f"PDF left in inbox/ for manual deletion.")
            return None

    provider = get_default_provider()
    article: ScholarlyArticle | None = None
    if doi:
        article = provider.get_by_doi(doi)
    if article is None:
        fallback_title = (pdf_meta.get("/Title") or "").strip()
        if not fallback_title:
            fallback_title = pdf_text.split("\n", 1)[0][:200]
        log(f"provider fallback title search: {fallback_title[:80]}")
        article = provider.search_by_title(fallback_title)
        if article and not doi:
            doi = article.doi_lower or None
    if article is None:
        log("provider lookup failed; proceeding with minimal metadata from PDF")
        article = ScholarlyArticle(
            title=(pdf_meta.get("/Title") or "").strip() or "UNKNOWN",
            authors=[(pdf_meta.get("/Author") or "unknown").strip()],
        )

    stem = derive_stem(article.authors, article.year or "YYYY", article.title)
    log(f"derived stem: {stem}")

    # Dedup layer 2: stem already used. Branch on preprint-vs-journal verdict
    # so a journal version of an existing preprint can replace the PDF rather
    # than being silently dropped in inbox/.
    stem_hit = find_stem_collision(stem)
    if stem_hit is not None:
        verdict = classify_pdf_collision(doi, stem_hit)
        if verdict == "journal-upgrade":
            target = papers_dir() / f"{stem}.pdf"
            import shutil, uuid
            from ..wiki import _doi_from_existing_page
            from ..db.iterations import write_iteration
            old_doi = _doi_from_existing_page(stem_hit)
            shutil.move(str(src_pdf), str(target))
            log(f"PDF UPGRADE: replaced papers/{stem}.pdf with journal version "
                f"(existing page DOI is preprint; incoming DOI {doi} is not). "
                f"Wiki page YAML/body NOT touched — run `researchwiki preprint-check "
                f"--doi <existing-preprint-doi>` and update `doi:` / `venue:` manually.")
            try:
                write_iteration(
                    attempt_id=f"upgrade-{uuid.uuid4().hex[:8]}",
                    paper_stem=stem,
                    pdf_filename=src_pdf.name,
                    iteration=0,
                    role="pdf_upgrade",
                    decision="upgraded",
                    decision_reason=f"verdict=journal-upgrade; old_doi={old_doi}; new_doi={doi}",
                )
            except Exception as e:
                log(f"  (note: ingest_iterations row not written: {e})")
            return None
        elif verdict == "preprint-downgrade":
            log(f"SKIP (preprint-downgrade): wiki already has the journal version "
                f"of `{stem}` at {stem_hit}; incoming preprint DOI {doi} would be "
                f"a regression. PDF left in inbox/.")
        elif verdict == "duplicate":
            log(f"SKIP (duplicate): stem `{stem}` and DOI match {stem_hit} — "
                f"this PDF is already ingested. PDF left in inbox/ for deletion.")
        else:
            log(f"SKIP (collision-unclear): stem `{stem}` already exists at "
                f"{stem_hit} but DOIs don't fit a known preprint-journal pattern. "
                f"PDF left in inbox/ for manual investigation.")
        return None

    papers_hit = papers_dir() / f"{stem}.pdf"
    if papers_hit.exists() and src_pdf.resolve() != papers_hit.resolve():
        log(f"SKIP (duplicate): papers/{stem}.pdf already exists. "
            f"PDF left in inbox/ for manual investigation.")
        return None

    refs = provider.get_references(article)
    citations = provider.get_citations(article)
    recs = provider.get_recommendations(article)
    log(f"provider refs={len(refs)} cites={len(citations)} recs={len(recs)}")

    refs_raw = [r.raw for r in refs]

    # Fallback chain for when S2 `/references` is empty. Springer Nature and
    # bioRxiv now routinely return `data: null` ("fields elided by the
    # publisher") for most post-2024 papers, which would drop every new
    # ingest into the graph as an orphan. We try two cheaper fallbacks in
    # precision-first order:
    #
    #   1. Crossref /works — authoritative upstream where publishers deposit
    #      references. Clean structured DOIs, one cached HTTP call. Works
    #      for most journal papers; empty for bioRxiv preprints that don't
    #      deposit refs.
    #   2. PDF text — regex-harvest DOIs typeset in the paper's own
    #      References section. Last-resort because it reads the full PDF,
    #      has lower recall (misses Vancouver-style refs), and its regex
    #      occasionally picks up noise (`...doi:`, `...preprint`) that
    #      downstream tools then have to filter.
    #
    # Each fallback runs only if the previous returned empty — we do not
    # merge, since S2/Crossref refs when present are already complete.
    if not refs_raw and doi:
        from ..providers.crossref import fetch_crossref_refs
        cr_dois = fetch_crossref_refs(doi)
        if cr_dois:
            log(f"S2 /references empty; Crossref returned {len(cr_dois)} DOIs for this paper")
            refs_raw = dois_as_s2_refs(cr_dois)
    if not refs_raw and doi:
        pdf_dois = extract_ref_dois(src_pdf, own_doi=doi)
        if pdf_dois:
            log(f"S2 + Crossref both empty; parsed {len(pdf_dois)} DOIs from PDF references section")
            refs_raw = dois_as_s2_refs(pdf_dois)

    citations_raw = [c.raw for c in citations]
    crosslinks = intersect_crosslinks(refs_raw, wiki_dois)
    incoming = intersect_incoming(citations_raw, wiki_dois)
    log(f"wiki outgoing cross-links: {len(crosslinks)}; incoming citations: {len(incoming)}")

    sections = anchor_sections(pdf_text)
    log(f"anchored sections: {list(sections.keys())}")

    # Category auto-suggest: only when the user didn't pass --category.
    # Respects user input; falls back silently if the index isn't built.
    category_source = "user-supplied"
    if category_hint is None:
        from ..search import SearchBackendUnavailable, get_default_backend, suggest_category
        try:
            suggestion = suggest_category(
                get_default_backend(),
                article.title or "",
                article.abstract or "",
            )
            if suggestion is not None:
                category_hint = suggestion.category
                category_source = (
                    f"auto-suggested: {int(round(suggestion.confidence * 5))}/5 "
                    f"nearest neighbors agree (top: {suggestion.top_3})"
                )
                log(f"category auto-suggested: {category_hint} "
                    f"({suggestion.confidence:.0%}; top={suggestion.top_3})")
            else:
                log("category auto-suggest: neighbors split; falling back to `other`")
        except SearchBackendUnavailable as e:
            log(f"category auto-suggest skipped: {e}")

    pdf_dest = papers_dir() / f"{stem}.pdf"
    if no_move:
        log(f"--no-move: using existing PDF path {src_pdf}")
        pdf_dest = src_pdf
    # The actual move to papers/ happens AFTER the digest is written (below), so
    # a failure leaves the source PDF in inbox/ (re-ingestable) rather than moved
    # with no digest.

    staged_supp: list[dict] = []
    if supplementary:
        from .attach import stage_supplementary
        for sp in supplementary:
            try:
                staged = stage_supplementary(stem, sp)
                staged_supp.append(staged)
                log(f"supplementary staged: papers/{stem}.supp/{staged['filename']} "
                    f"({staged['kind']})")
            except (FileNotFoundError, FileExistsError, ValueError) as e:
                log(f"supplementary skipped ({sp}): {e}")

    digest_path = _write_digest(
        stem=stem,
        pdf_dest=pdf_dest,
        article=article,
        refs=refs,
        crosslinks=crosslinks,
        incoming=incoming,
        recommendations=recs,
        pdf_text=pdf_text,
        sections=sections,
        category_hint=category_hint,
        raw_source_pdf=src_pdf,
        category_source=category_source,
        staged_supp=staged_supp,
    )
    log(f"digest written: {digest_path}")

    # Move the PDF into papers/ now that the digest is durably written.
    # shutil.move (not Path.rename) handles a cross-filesystem inbox/ (EXDEV).
    if not no_move and src_pdf.resolve() != pdf_dest.resolve():
        papers_dir().mkdir(exist_ok=True)
        if pdf_dest.exists():
            log(f"destination exists; will overwrite: {pdf_dest}")
        shutil.move(str(src_pdf), str(pdf_dest))
        log(f"moved: {src_pdf} -> {pdf_dest}")

    # Append to log.md (chronological record).
    first_author = first_author_surname(article.authors)
    headline = f"{first_author} {article.year or '?'} — {article.title[:120]}"
    details = (
        f"Category: {category_hint or 'TODO'}. "
        f"DOI: {article.doi_lower or 'none'}. "
        f"Cross-links found: {len(crosslinks)} outgoing, {len(incoming)} incoming."
    )
    append_log_md("ingest", headline, details)

    return digest_path


# ---------- CLI ----------

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki ingest",
        description="Ingest one or more PDFs into wiki-ready digests.",
        epilog="Example: researchwiki ingest inbox/*.pdf --category cgt",
    )
    parser.add_argument("pdfs", nargs="*",
                        help="Paths to PDFs (typically in inbox/). Passing --workers >1 "
                             "or --resume activates crash-safe batch mode with a "
                             "checkpoint under .ingest/batch-<ts>/. Serial multi-PDF "
                             "invocation (default, no --workers) keeps historical "
                             "behavior with no batch dir.")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Concurrent subprocesses in batch mode. Setting this "
                             "activates the batch driver (checkpoint + --resume). "
                             "Omit for the current serial-loop behavior.")
    parser.add_argument("--resume", metavar="BATCH_DIR", default=None,
                        help="Continue an interrupted batch. Reads its plan.json + "
                             "checkpoint.json and re-runs only what's pending.")
    parser.add_argument("--no-retry", action="store_true",
                        help="With --resume: skip PDFs that previously failed "
                             "(default is to retry them).")
    parser.add_argument("--category", default=None,
                        help="Wiki content category applied to every paper in the batch. Must be an existing wiki/<category>/ directory (rejected otherwise — create it first). Omit to leave as TODO / auto-classify.")
    parser.add_argument("--no-move", action="store_true",
                        help="Skip moving the PDF from inbox/ to papers/ (useful for re-running on existing papers/ files).")
    parser.add_argument("--doi", dest="doi_override", default=None,
                        help="Override DOI detection with a user-supplied DOI. Use when a PDF's "
                             "first-page text contains multiple DOIs (e.g. correspondence bundles) "
                             "or the paper's metadata is wrong. Cannot combine with batch input.")
    parser.add_argument("--supplementary", dest="supplementary", action="append", default=None,
                        type=Path,
                        help="Path to a supplementary file to attach alongside the primary PDF. "
                             "Repeat for multiple files. Each file is copied into "
                             "`papers/{stem}.supp/` (filename normalized to lowercase + safe "
                             "chars) and listed under `supplementary:` in the digest's YAML "
                             "block. Defaults: PDF→kind=methods, xlsx/csv/tsv→kind=data, "
                             "other→kind=other. Edit the digest YAML block before pasting "
                             "into the wiki page if you need overrides. "
                             "Cannot combine with batch input.")
    args = parser.parse_args(argv)

    # --resume takes over completely; plan.json holds the subcommand + flags.
    if args.resume:
        from . import _ingest_batch
        return _ingest_batch.resume_batch(
            Path(args.resume).expanduser().resolve(),
            no_retry=args.no_retry,
            workers_override=args.workers,
        )

    if not args.pdfs:
        parser.error("need PDF path(s) (or --resume BATCH_DIR)")

    # Batch mode (parallel + checkpoint) is opt-in via --workers. Multi-PDF
    # without --workers keeps the historical serial-loop behavior below —
    # some callers rely on the ordered per-PDF logs and the final summary
    # line. `-w 1` is legal: gives you a checkpoint dir with one worker.
    if args.workers is not None:
        if args.doi_override:
            parser.error("--doi can only be used with a single PDF (drop --workers "
                         "for a targeted single-file invocation)")
        if args.supplementary:
            parser.error("--supplementary can only be used with a single PDF "
                         "(drop --workers for a targeted single-file invocation)")
        if args.category is not None:
            from ..categories import content_categories, is_valid
            if not is_valid(args.category):
                parser.error(
                    f"--category {args.category!r} does not exist. Create it first "
                    f"(`mkdir -p wiki/{args.category}`) or pick an existing one: "
                    f"{sorted(content_categories())}. Omit --category to auto-classify."
                )
        # Passthrough to per-worker `researchwiki ingest`. --category applies
        # uniformly across the batch; --no-move too. --doi and --supplementary
        # are per-PDF, already rejected above.
        extra: list[str] = []
        if args.category is not None:
            extra += ["--category", args.category]
        if args.no_move:
            extra.append("--no-move")
        from . import _ingest_batch
        return _ingest_batch.new_batch(
            args.pdfs, ["ingest"], extra, workers=args.workers,
        )

    if args.doi_override and len(args.pdfs) != 1:
        parser.error("--doi can only be used with a single PDF")
    if args.supplementary and len(args.pdfs) != 1:
        parser.error("--supplementary can only be used with a single PDF "
                     "(supp files attach to one paper at a time)")
    # Reject-unless-exists: a content category is valid only if its
    # wiki/<category>/ directory already exists. New categories must be created
    # explicitly first (e.g. `mkdir -p wiki/<category>` or `suggest-splits`) —
    # ingest never auto-creates one from a typo'd --category.
    if args.category is not None:
        from ..categories import content_categories, is_valid
        if not is_valid(args.category):
            parser.error(
                f"--category {args.category!r} does not exist. Create it first "
                f"(`mkdir -p wiki/{args.category}`) or pick an existing one: "
                f"{sorted(content_categories())}. Omit --category to auto-classify."
            )


    successes: list[Path] = []
    skipped: list[str] = []
    failures: list[str] = []
    for p in args.pdfs:
        src_pdf = Path(p).resolve()
        log("")
        log(f"━━━ ingesting: {src_pdf.name} ━━━")
        try:
            digest = process_one(src_pdf, args.category, args.no_move,
                                 doi_override=args.doi_override,
                                 supplementary=args.supplementary)
        except Exception as e:  # noqa: BLE001
            log(f"FAILED on {src_pdf.name}: {type(e).__name__}: {e}")
            failures.append(src_pdf.name)
            continue
        if digest:
            successes.append(digest)
        else:
            skipped.append(src_pdf.name)

    log("")
    log(f"━━━ batch complete: {len(successes)} ok, {len(skipped)} skipped, {len(failures)} failed ━━━")
    if skipped:
        log(f"skipped (duplicates): {', '.join(skipped)}")
    if failures:
        log(f"failures: {', '.join(failures)}")

    # End-of-batch saturation check: if this batch landed papers in
    # wiki/other/ that pushed the count over threshold, surface the
    # suggest-splits nudge once. The decay-stamp inside the helper
    # suppresses repeats for 7 days across status / ingest / agent.
    from ..categories import other_saturation_warning
    msg = other_saturation_warning()
    if msg:
        print()
        print(msg)
        print()

    for d in successes:
        print(d)
    return 1 if failures else 0
