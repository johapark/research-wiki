"""Deeper consistency checks over the local wiki (no network).

✅ Use when: you need a structured issue list (orphan pages, broken links,
   missing back-links, stale syntheses). Agents: pass `--json` for
   machine-parseable output. Pass `--fix` to auto-insert missing back-links.
❌ Don't use: for a one-screen dashboard (use `status`). Not a search tool.
   Concept-hub candidates — an *opportunity* signal, not a defect — live
   at `researchwiki candidates concepts` so `lint`'s output can stay a
   pure defect list.

This subpackage groups checks by what they read so each module is
focused and individually testable:

  walk            — page enumeration + link/yaml parsing primitives
  link_checks     — orphans, broken_wikilinks, missing_backlinks,
                    none_placeholders, --fix
  yaml_checks     — invalid_fm, type/category/doi/year/keywords drift,
                    venue_suspect
  staleness       — stale_synthesis, stale_by_content, audit_count, proposals
  audit_p2        — Priority-2 entries with audit anchor hits
  index_checks    — thin_index_text (what the embedder will see)
  db_checks       — ungraded_papers, zero_claim_papers,
                    stems_missing_claim_overlap, duplicate_claim_sets,
                    db_drift
  supplementary   — supp YAML ↔ disk consistency

The orchestrator below walks pages once, calls each check, then renders
either the prose report or the JSON object. Public CLI signature
(`researchwiki lint [--fix] [--json]`) is unchanged.

Pass `--fix` to apply the only deterministic fix we can make automatically:
inserting missing back-links into target pages' `## Related Papers` section.
Every inserted link is marked `(auto-added; refine)` so the LLM knows to
rewrite the one-liner on the next ingest/query pass.

For citation-graph gap-finding (cross-links that exist in S2 but not in the
wiki), run `researchwiki audit` instead — that one needs network.

Exit code: 0 always (lint reports findings; issues found are not failures).
"""

from __future__ import annotations

import argparse

from ...wiki import read_page, strip_non_prose
from .audit_p2 import find_p2_anchor_hits
from .claim_anchors import find_dangling_claim_anchors
from .concept_contract import find_concept_contract_violations
from .idea_contract import find_idea_contract_violations
from ...eval.pointers import broken as broken_prompt_pointers
from ...eval.pointers import orphans as orphan_prompt_files
from .db_checks import (
    db_drift_check_and_fix,
    find_duplicate_claim_sets,
    find_ungraded_papers,
    find_stems_missing_claim_overlap,
    find_zero_claim_papers,
)
from .index_checks import find_thin_index_text
from .report import _emit_json, _emit_prose
from .link_checks import (
    apply_backlink_fixes,
    find_none_placeholders,
    build_link_graph,
    find_missing_backlinks,
    find_orphans,
)
from .staleness import (
    find_stale_by_audit_count,
    find_stale_by_content,
    find_stale_evolution_proposals,
    find_stale_synthesis,
)
from .supplementary import find_supplementary_issues
from .walk import all_pages, page_key
from .yaml_checks import (
    find_category_drift,
    find_invalid_frontmatter,
    find_hook_too_long,
    find_missing_doi,
    find_missing_author_model,
    find_missing_hook,
    find_missing_keywords,
    find_missing_type,
    find_page_type_mismatches,
    find_stem_year_drift,
    find_unquoted_wikilink_lists,
    find_venue_suspect,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki lint",
        description="Local consistency checks over the wiki (no network).",
    )
    parser.add_argument("--fix", action="store_true",
                        help="Apply the only auto-fixable issue: insert missing back-links "
                             "into target pages' Related Papers section. Each inserted bullet "
                             "is marked `(auto-added; refine)` for later LLM polish.")
    parser.add_argument("--cross-paper", action="store_true",
                        help="Run the cross-paper contradiction lint (LLM-call-heavy, opt-in). "
                             "Pairs claims across papers at high embedding cosine and asks an "
                             "LLM judge to flag numeric/direction disagreements.")
    parser.add_argument("--cross-paper-threshold", type=float, default=0.85,
                        help="Cosine threshold for the cross-paper candidate pool (default: 0.85).")
    parser.add_argument("--cross-paper-max-pairs", type=int, default=50,
                        help="Cap on judged pairs to bound LLM-call cost (default: 50). "
                             "0 sizes the pool without judging anything, which is the "
                             "zero-cost way to see what a sweep would cost.")
    parser.add_argument("--cross-paper-rejudge", action="store_true",
                        help="Re-judge pairs already recorded in cross_paper_judgements. "
                             "By default a repeat run judges only pairs the previous one "
                             "never reached, so resuming an interrupted sweep is cheap.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit a structured JSON object instead of the prose report. "
                             "Keys: pages_scanned, orphans, broken_wikilinks, missing_backlinks, "
                             "missing_type, page_type_mismatches, category_yaml_drift, stale_synthesis, stale_by_content, "
                             "stale_by_audit_count, p2_entries_with_anchor_hits, "
                             "stale_evolution_proposals, missing_keywords, "
                             "missing_hook, hook_too_long, missing_author_model, "
                             "venue_suspect, none_placeholders, thin_index_text, "
                             "ungraded_papers, zero_claim_papers, "
                             "stems_missing_claim_overlap, "
                             "duplicate_claim_sets, "
                             "dangling_claim_anchors, "
                             "concept_contract_violations, "
                             "idea_contract_violations, orphan_prompts, "
                             "broken_prompt_pointers, db_drift, "
                             "cross_paper_contradictions, fix_applied.")
    args = parser.parse_args(argv)

    pages = all_pages()
    if not pages:
        print("No wiki pages found under `wiki/`. Nothing to lint.")
        return 0

    # --- strict YAML frontmatter check (runs before permissive read_page so
    #     breakage is reported even when read_page would silently recover)
    invalid_fm, fm_check_ran = find_invalid_frontmatter(pages)

    # --- walk every page once: cache fm, body, prose
    known = {page_key(p) for p in pages}
    pages_fm: dict = {}
    pages_body: dict = {}
    pages_prose: dict = {}
    # Files with no frontmatter fence at all. `read_page` returns None for them
    # and `read_pages` — what `build_index` walks — drops them, so they have no
    # index entry and index-side checks must not report on them.
    pages_unfenced: set = set()
    for md in pages:
        p = read_page(md)
        fm = p.fm if p else {}
        body = p.body if p else md.read_text(encoding="utf-8")
        if p is None:
            pages_unfenced.add(md)
        pages_fm[md] = fm
        pages_body[md] = body
        pages_prose[md] = strip_non_prose(body)

    # --- link graph + dependent checks (orphans, broken, missing_back)
    out_links, in_links, broken = build_link_graph(pages, pages_prose, known)
    orphans = find_orphans(pages, out_links, in_links)
    missing_back = find_missing_backlinks(out_links)
    none_placeholders = find_none_placeholders(pages_body)

    # --- frontmatter-shape checks
    missing_type = find_missing_type(pages, pages_fm)
    type_mismatches = find_page_type_mismatches(pages, pages_fm)
    category_drift = find_category_drift(pages, pages_fm)
    missing_doi = find_missing_doi(pages, pages_fm)
    stem_year_drift = find_stem_year_drift(pages, pages_fm)
    missing_keywords = find_missing_keywords(pages, pages_fm)
    missing_hook = find_missing_hook(pages, pages_fm)
    hook_too_long = find_hook_too_long(pages, pages_fm)
    missing_author_model = find_missing_author_model(pages, pages_fm)
    unquoted_wikilinks = find_unquoted_wikilink_lists(pages)
    venue_suspect = find_venue_suspect(pages, pages_fm)
    thin_index_text = find_thin_index_text(
        [p for p in pages if p not in pages_unfenced], pages_fm, pages_body
    )

    # --- staleness
    stale = find_stale_synthesis(pages, pages_fm, known)
    stale_by_content = find_stale_by_content(pages, pages_fm, known)
    stale_by_audit_count = find_stale_by_audit_count(pages, pages_fm)
    stale_proposals = find_stale_evolution_proposals()

    # --- audit + db + supp + claim anchors + concept contract
    p2_anchor_hits = find_p2_anchor_hits(pages)
    ungraded_papers = find_ungraded_papers()
    zero_claim_papers = find_zero_claim_papers()
    stems_missing_claim_overlap = find_stems_missing_claim_overlap()
    duplicate_claim_sets = find_duplicate_claim_sets()
    supp_yaml_missing, supp_orphans = find_supplementary_issues(pages, pages_fm)
    dangling_anchors = find_dangling_claim_anchors(pages_body)
    concept_contract = find_concept_contract_violations(pages, pages_body, pages_fm)
    idea_contract = find_idea_contract_violations(pages, pages_body, pages_fm)
    # Docs-layer reachability. Same class of check as broken_wikilinks, one
    # layer up: a prompt no CLAUDE.md pointer reaches is a procedure the agent
    # has no condition to read, and a pointer with no file sends it looking for
    # something that isn't there. Pure filesystem work, no provider call.
    orphan_prompts = orphan_prompt_files()
    broken_pointers = broken_prompt_pointers()

    # --- apply fixes BEFORE rendering so stats reflect post-fix state
    fix_written: dict[str, int] = {}
    if args.fix and missing_back:
        fix_written = apply_backlink_fixes(missing_back, pages_prose)
    # Provenance recovery reads this pipeline's own telemetry rather than
    # inferring anything, which is why it is a `--fix` repair and not a review
    # queue: `provenance.apply_provenance_fixes` fills a blank field with the
    # value the ingest run recorded, and fills nothing at all where no run did.
    if args.fix and missing_author_model:
        from .provenance import apply_provenance_fixes
        filled = set(apply_provenance_fixes()["author_model_keys"])
        missing_author_model = [k for k in missing_author_model if k not in filled]
    db_drift, db_drift_fixed = db_drift_check_and_fix(apply_fix=args.fix)

    # --- opt-in cross-paper contradiction lint (LLM-call-heavy)
    cross_paper: list[dict] = []
    # None (not {}) when the check didn't run, matching the `duplicate_claim_sets`
    # convention: null means skipped, a populated object means it ran.
    cross_paper_stats: dict | None = None
    if args.cross_paper:
        from .cross_paper import find_cross_paper_contradictions
        cross_paper_stats = {}
        cross_paper = find_cross_paper_contradictions(
            sim_threshold=args.cross_paper_threshold,
            max_pairs=args.cross_paper_max_pairs,
            rejudge=args.cross_paper_rejudge,
            stats=cross_paper_stats,
        )

    if args.as_json:
        return _emit_json(
            pages=pages, fm_check_ran=fm_check_ran, invalid_fm=invalid_fm,
            orphans=orphans, broken=broken, missing_back=missing_back,
            missing_type=missing_type,
        type_mismatches=type_mismatches, category_drift=category_drift,
            stale=stale, stale_by_content=stale_by_content,
            stale_by_audit_count=stale_by_audit_count,
            p2_anchor_hits=p2_anchor_hits, stale_proposals=stale_proposals,
            missing_keywords=missing_keywords, missing_doi=missing_doi,
            missing_hook=missing_hook, hook_too_long=hook_too_long,
            missing_author_model=missing_author_model,
            stem_year_drift=stem_year_drift, unquoted_wikilinks=unquoted_wikilinks,
            venue_suspect=venue_suspect, none_placeholders=none_placeholders,
            thin_index_text=thin_index_text,
            supp_yaml_missing=supp_yaml_missing, supp_orphans=supp_orphans,
            ungraded_papers=ungraded_papers,
            zero_claim_papers=zero_claim_papers,
            stems_missing_claim_overlap=stems_missing_claim_overlap,
            duplicate_claim_sets=duplicate_claim_sets,
            dangling_anchors=dangling_anchors,
            concept_contract=concept_contract,
            idea_contract=idea_contract,
            orphan_prompts=orphan_prompts,
            broken_prompt_pointers=broken_pointers,
            db_drift=db_drift, db_drift_fixed=db_drift_fixed,
            cross_paper=cross_paper,
            cross_paper_stats=cross_paper_stats,
            fix_applied=args.fix, fix_written=fix_written,
        )

    return _emit_prose(
        pages=pages, fm_check_ran=fm_check_ran, invalid_fm=invalid_fm,
        orphans=orphans, broken=broken, missing_back=missing_back,
        missing_type=missing_type,
        type_mismatches=type_mismatches, category_drift=category_drift,
        stale=stale, stale_by_content=stale_by_content,
        stale_by_audit_count=stale_by_audit_count,
        p2_anchor_hits=p2_anchor_hits, stale_proposals=stale_proposals,
        missing_keywords=missing_keywords, missing_doi=missing_doi,
        missing_hook=missing_hook, hook_too_long=hook_too_long,
        missing_author_model=missing_author_model,
        stem_year_drift=stem_year_drift, unquoted_wikilinks=unquoted_wikilinks,
        venue_suspect=venue_suspect, none_placeholders=none_placeholders,
        thin_index_text=thin_index_text,
        supp_yaml_missing=supp_yaml_missing, supp_orphans=supp_orphans,
        ungraded_papers=ungraded_papers,
        zero_claim_papers=zero_claim_papers,
        stems_missing_claim_overlap=stems_missing_claim_overlap,
        duplicate_claim_sets=duplicate_claim_sets,
        dangling_anchors=dangling_anchors,
        concept_contract=concept_contract,
        idea_contract=idea_contract,
        orphan_prompts=orphan_prompts,
        broken_prompt_pointers=broken_pointers,
        db_drift=db_drift, db_drift_fixed=db_drift_fixed,
        cross_paper=cross_paper,
        cross_paper_stats=cross_paper_stats,
        fix_applied=args.fix, fix_written=fix_written,
    )

