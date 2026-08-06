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
import json

from ...paths import s2_cache_dir
from ...wiki import read_page, strip_non_prose
from .audit_p2 import find_p2_anchor_hits
from .claim_anchors import find_dangling_claim_anchors
from .concept_contract import find_concept_contract_violations
from .db_checks import (
    db_drift_check_and_fix,
    find_duplicate_claim_sets,
    find_ungraded_papers,
    find_stems_missing_claim_overlap,
    find_zero_claim_papers,
)
from .index_checks import find_thin_index_text
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
    find_missing_hook,
    find_missing_keywords,
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
                        help="Cap on judged pairs to bound LLM-call cost (default: 50).")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit a structured JSON object instead of the prose report. "
                             "Keys: pages_scanned, orphans, broken_wikilinks, missing_backlinks, "
                             "page_type_mismatches, category_yaml_drift, stale_synthesis, stale_by_content, "
                             "stale_by_audit_count, p2_entries_with_anchor_hits, "
                             "stale_evolution_proposals, missing_keywords, "
                             "missing_hook, hook_too_long, "
                             "venue_suspect, none_placeholders, thin_index_text, "
                             "ungraded_papers, zero_claim_papers, "
                             "stems_missing_claim_overlap, "
                             "duplicate_claim_sets, "
                             "dangling_claim_anchors, "
                             "concept_contract_violations, db_drift, "
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
    for md in pages:
        p = read_page(md)
        fm = p.fm if p else {}
        body = p.body if p else md.read_text(encoding="utf-8")
        pages_fm[md] = fm
        pages_body[md] = body
        pages_prose[md] = strip_non_prose(body)

    # --- link graph + dependent checks (orphans, broken, missing_back)
    out_links, in_links, broken = build_link_graph(pages, pages_prose, known)
    orphans = find_orphans(pages, out_links, in_links)
    missing_back = find_missing_backlinks(out_links)
    none_placeholders = find_none_placeholders(pages_body)

    # --- frontmatter-shape checks
    type_mismatches = find_page_type_mismatches(pages, pages_fm)
    category_drift = find_category_drift(pages, pages_fm)
    missing_doi = find_missing_doi(pages, pages_fm)
    stem_year_drift = find_stem_year_drift(pages, pages_fm)
    missing_keywords = find_missing_keywords(pages, pages_fm)
    missing_hook = find_missing_hook(pages, pages_fm)
    hook_too_long = find_hook_too_long(pages, pages_fm)
    unquoted_wikilinks = find_unquoted_wikilink_lists(pages)
    venue_suspect = find_venue_suspect(pages, pages_fm)
    thin_index_text = find_thin_index_text(pages, pages_fm, pages_body)

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

    # --- apply fixes BEFORE rendering so stats reflect post-fix state
    fix_written: dict[str, int] = {}
    if args.fix and missing_back:
        fix_written = apply_backlink_fixes(missing_back, pages_prose)
    db_drift, db_drift_fixed = db_drift_check_and_fix(apply_fix=args.fix)

    # --- opt-in cross-paper contradiction lint (LLM-call-heavy)
    cross_paper: list[dict] = []
    if args.cross_paper:
        from .cross_paper import find_cross_paper_contradictions
        cross_paper = find_cross_paper_contradictions(
            sim_threshold=args.cross_paper_threshold,
            max_pairs=args.cross_paper_max_pairs,
        )

    if args.as_json:
        return _emit_json(
            pages=pages, fm_check_ran=fm_check_ran, invalid_fm=invalid_fm,
            orphans=orphans, broken=broken, missing_back=missing_back,
            type_mismatches=type_mismatches, category_drift=category_drift,
            stale=stale, stale_by_content=stale_by_content,
            stale_by_audit_count=stale_by_audit_count,
            p2_anchor_hits=p2_anchor_hits, stale_proposals=stale_proposals,
            missing_keywords=missing_keywords, missing_doi=missing_doi,
            missing_hook=missing_hook, hook_too_long=hook_too_long,
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
            db_drift=db_drift, db_drift_fixed=db_drift_fixed,
            cross_paper=cross_paper,
            fix_applied=args.fix, fix_written=fix_written,
        )

    return _emit_prose(
        pages=pages, fm_check_ran=fm_check_ran, invalid_fm=invalid_fm,
        orphans=orphans, broken=broken, missing_back=missing_back,
        type_mismatches=type_mismatches, category_drift=category_drift,
        stale=stale, stale_by_content=stale_by_content,
        stale_by_audit_count=stale_by_audit_count,
        p2_anchor_hits=p2_anchor_hits, stale_proposals=stale_proposals,
        missing_keywords=missing_keywords, missing_doi=missing_doi,
        missing_hook=missing_hook, hook_too_long=hook_too_long,
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
        db_drift=db_drift, db_drift_fixed=db_drift_fixed,
        cross_paper=cross_paper,
        fix_applied=args.fix, fix_written=fix_written,
    )


def _emit_json(**kw) -> int:
    out = {
        "pages_scanned": len(kw["pages"]),
        "invalid_frontmatter": (
            [
                {"page": page_key(md), "error": msg, "line": line_no}
                for md, msg, line_no in kw["invalid_fm"]
            ]
            if kw["fm_check_ran"]
            else None  # null = check skipped (PyYAML missing)
        ),
        "orphans": kw["orphans"],
        "broken_wikilinks": [
            {"page": src, "targets": sorted(set(bad))}
            for src, bad in kw["broken"]
        ],
        "missing_backlinks": [
            {"src": src, "tgt": t} for src, t in kw["missing_back"]
        ],
        "page_type_mismatches": [
            {"page": key, "reason": reason} for key, reason in kw["type_mismatches"]
        ],
        "category_yaml_drift": [
            {"page": key, "yaml_category": yc, "dir_category": dc}
            for key, yc, dc in kw["category_drift"]
        ],
        "stale_synthesis": [
            {"page": page_key(md), "newer_references": newer}
            for md, newer in kw["stale"]
        ],
        "stale_by_content": [
            {"page": page_key(md), "unreferenced_top_hits": hits}
            for md, hits in kw["stale_by_content"]
        ],
        "stale_by_audit_count": [
            {"page": page_key(md), "cached_count": cached, "current_count": current}
            for md, cached, current in kw["stale_by_audit_count"]
        ],
        "p2_entries_with_anchor_hits": kw["p2_anchor_hits"],
        "stale_evolution_proposals": [
            {"source_stem": stem, "n_proposal_files": nf, "age_days": age}
            for stem, nf, age in kw["stale_proposals"]
        ],
        "missing_keywords": [
            {"page": key, "n_keywords": n} for key, n in kw["missing_keywords"]
        ],
        "missing_hook": kw["missing_hook"],
        "hook_too_long": [
            {"page": key, "chars": n, "ceiling": cap}
            for key, n, cap in kw["hook_too_long"]
        ],
        "missing_doi": kw["missing_doi"],
        "stem_year_drift": kw["stem_year_drift"],
        "unquoted_wikilink_lists": [
            {"page": key, "field": field} for key, field in kw["unquoted_wikilinks"]
        ],
        "supplementary_missing_on_disk": kw["supp_yaml_missing"],
        "supplementary_orphaned_files": kw["supp_orphans"],
        "dangling_claim_anchors": [
            {"page": page_key(d["page"]), "stem": d["stem"], "slug": d["slug"]}
            for d in kw["dangling_anchors"]
        ],
        "concept_contract_violations": [
            {"page": page_key(v["page"]), "kind": v["kind"], "detail": v["detail"]}
            for v in kw["concept_contract"]
        ],
        "ungraded_papers": kw["ungraded_papers"],
        "venue_suspect": kw["venue_suspect"],
        "none_placeholders": kw["none_placeholders"],
        "thin_index_text": kw["thin_index_text"],
        "zero_claim_papers": kw["zero_claim_papers"],
        "stems_missing_claim_overlap": kw["stems_missing_claim_overlap"],
        # null = check skipped (claim-embedding cache cold or <50% of claims
        # cached); [] = ran and found nothing. Same convention as
        # invalid_frontmatter above.
        "duplicate_claim_sets": kw["duplicate_claim_sets"],
        "db_drift": kw["db_drift"],
        "cross_paper_contradictions": kw.get("cross_paper", []),
        "fix_applied": {
            "backlinks_added": sum(kw["fix_written"].values()),
            "pages_updated": kw["fix_written"],
            "db_upserted": kw["db_drift_fixed"].get("upserted", 0),
            "db_deleted": kw["db_drift_fixed"].get("deleted", 0),
        } if kw["fix_applied"] else None,
    }
    print(json.dumps(out, indent=2))
    return 0


def _emit_prose(**kw) -> int:
    pages = kw["pages"]
    print("# researchwiki lint report")
    print()
    print(f"Pages scanned: {len(pages)}")
    print()

    if kw["fm_check_ran"]:
        print(f"## Invalid frontmatter ({len(kw['invalid_fm'])})")
        if kw["invalid_fm"]:
            for md, msg, line_no in kw["invalid_fm"]:
                loc = f"{page_key(md)}:{line_no}" if line_no is not None else page_key(md)
                print(f"- **{loc}** — {msg}")
            print()
            print("_Obsidian's Properties panel falls back to raw `---` display for any page "
                  "with non-strict YAML. Common cause: unquoted `': '` inside a value "
                  "(e.g. `authors: X, Y (senior: Z)` — the parenthetical parses as a nested "
                  "mapping). Fix by splitting into a separate YAML field or quoting the value._")
        else:
            print("_none — every page parses as strict YAML._")
    else:
        print("## Invalid frontmatter (skipped)")
        print("_PyYAML not installed; install it (`pip install pyyaml` or `pip install -e .`) "
              "to enable strict-YAML checks for Obsidian-compatible frontmatter._")
    print()

    print(f"## Orphans ({len(kw['orphans'])})")
    if kw["orphans"]:
        for o in kw["orphans"]:
            print(f"- {o}")
    else:
        print("_none — every paper page is linked._")
    print()

    broken = kw["broken"]
    print(f"## Broken wikilinks ({sum(len(b) for _, b in broken)} across {len(broken)} pages)")
    if broken:
        for src, bad in broken:
            print(f"- **{src}** → {', '.join(sorted(set(bad)))}")
    else:
        print("_none._")
    print()

    missing_back = kw["missing_back"]
    print(f"## Missing back-links ({len(missing_back)})")
    if missing_back:
        shown = missing_back[:40]
        for src, t in shown:
            print(f"- {src} → {t} (but {t} does not link back)")
        if len(missing_back) > 40:
            print(f"_... ({len(missing_back) - 40} more)_")
    else:
        print("_none — every paper→paper wikilink is symmetric._")
    print()

    type_mismatches = kw["type_mismatches"]
    print(f"## Page-type mismatches ({len(type_mismatches)})")
    if type_mismatches:
        for key, reason in type_mismatches:
            print(f"- {key}: {reason}")
    else:
        print("_none._")
    print()

    category_drift = kw["category_drift"]
    print(f"## Category YAML ↔ directory drift ({len(category_drift)})")
    if category_drift:
        for key, yc, dc in category_drift:
            print(f"- {key}: YAML category=[{yc}] but lives in {dc}/ "
                  f"(directory wins — update the frontmatter to [{dc}])")
    else:
        print("_none._")
    print()

    stale = kw["stale"]
    print(f"## Stale synthesis pages ({len(stale)})")
    if stale:
        for md, newer in stale:
            print(f"- {page_key(md)} — newer referenced pages: {', '.join(newer[:5])}"
                  + (f" (+{len(newer) - 5} more)" if len(newer) > 5 else ""))
    else:
        print("_none — all synthesis pages are fresh relative to their referenced papers._")
    print()

    stale_by_content = kw["stale_by_content"]
    print(f"## Stale by content — pages with `topic_seed` whose top search hits aren't cited ({len(stale_by_content)})")
    if stale_by_content:
        for md, hits in stale_by_content:
            print(f"- **{page_key(md)}** — unreferenced top hits for its topic_seed:")
            for h in hits[:5]:
                print(f"    - [[{h['key']}]]  (score {h['score']})  — {h['title'][:70]}")
            if len(hits) > 5:
                print(f"    - _... +{len(hits) - 5} more_")
    else:
        print("_none — pages with `topic_seed` cite every top-ranked paper in the index for that seed._")
    print()

    stale_by_audit_count = kw["stale_by_audit_count"]
    print(f"## Stale by audit count — pages with cached paper count drift ≥ threshold ({len(stale_by_audit_count)})")
    if stale_by_audit_count:
        for md, cached, current in stale_by_audit_count:
            print(f"- **{page_key(md)}** — `wiki_papers_at_audit: {cached}` → current is {current} "
                  f"(+{current - cached}). Re-run `researchwiki audit` and merge.")
    else:
        print("_none — all pages with `wiki_papers_at_audit` are within drift tolerance._")
    print()

    p2_anchor_hits = kw["p2_anchor_hits"]
    print(f"## Priority-2 entries surfaced in audit anchors ({len(p2_anchor_hits)})")
    if p2_anchor_hits:
        print("_DOIs listed under suggested-additions.md §Priority 2 that now appear in the latest "
              "cached audit's `shared_citation_anchors`. Informational — the LLM decides whether "
              "to move, re-annotate, or leave these entries._")
        print()
        for hit in p2_anchor_hits:
            cats = ", ".join(hit["categories"]) if hit["categories"] else "(no category)"
            title = (hit["title"] or "(no title)")[:80]
            print(f"- `{hit['doi']}` — {hit['current_count']}× [{cats}] — {title}")
    elif not list((s2_cache_dir() / ".").glob("audit-*.json")):
        print("_no cached audit snapshot found — run `researchwiki audit --json` first to populate `.s2-cache/audit-{date}.json`._")
    else:
        print("_none — no P2 DOIs intersect the latest audit's anchor list._")
    print()

    stale_proposals = kw["stale_proposals"]
    print(f"## Stale evolution proposals ({len(stale_proposals)})")
    if stale_proposals:
        print("Directories under `.ingest/*-evolution-proposals/` that have "
              "sat unactioned for ≥7 days. Apply them, or `rm -rf` the directory.")
        for stem, nf, age in stale_proposals[:10]:
            print(f"- `{stem}` — {nf} proposal file(s), {age}d old")
        if len(stale_proposals) > 10:
            print(f"- ... and {len(stale_proposals) - 10} more")
    else:
        print("_none._")
    print()

    missing_doi = kw["missing_doi"]
    print(f"## Paper pages missing DOI ({len(missing_doi)})")
    if missing_doi:
        print("Pages with `type: paper` but no `doi:` value (or DOI is "
              "`TODO`/`none`). Without a DOI, audit / preprint-check / "
              "retraction-check can't reach S2 or PubMed for this page, "
              "and the provenance audit terminates here.")
        for k in missing_doi[:20]:
            print(f"- {k}")
        if len(missing_doi) > 20:
            print(f"- ... and {len(missing_doi) - 20} more")
    else:
        print("_all paper pages carry a DOI._")
    print()

    stem_year_drift = kw["stem_year_drift"]
    print(f"## Stem ↔ YAML year drift ({len(stem_year_drift)})")
    if stem_year_drift:
        print("Pages where the stem-encoded year differs from YAML `year:`. "
              "Note: this check can only surface a *residual* mismatch — "
              "if a buggy reconcile stamped both stem and YAML with the "
              "same wrong year, lint can't tell. Drift entries here "
              "appear after the YAML was patched but the stem hasn't "
              "been (yet). Two legitimate causes:\n"
              "  • preprint→journal version updates that shift year by "
              "+1 (CLAUDE.md keeps the preprint-era stem to preserve "
              "back-links — informational, no action needed); or\n"
              "  • a buggy reconcile at ingest time that you've patched "
              "in YAML but should now propagate to the stem (mv the PDF + "
              "wiki .md, repoint inbound `[[wikilinks]]`, then `db rebuild` "
              "+ `reindex`).\n"
              "Verify which case applies for each entry below.")
        for entry in stem_year_drift[:20]:
            print(f"- **{entry['page']}** — stem={entry['stem_year']}, "
                  f"yaml={entry['yaml_year']}")
        if len(stem_year_drift) > 20:
            print(f"- ... and {len(stem_year_drift) - 20} more")
    else:
        print("_no drift — every paper page's stem year matches its YAML year._")
    print()

    missing_keywords = kw["missing_keywords"]
    print(f"## Paper pages missing keywords ({len(missing_keywords)})")
    if missing_keywords:
        print("Pages with `keywords:` empty or fewer than 3 items. The "
              "`keywords:` field is the retrieval-token list indexed by both "
              "BM25 and the semantic page index — sparse keywords degrade "
              "search recall on terms the Summary doesn't mention.")
        for key, n in missing_keywords[:20]:
            print(f"- {key} — {n} keyword(s)")
        if len(missing_keywords) > 20:
            print(f"- ... and {len(missing_keywords) - 20} more")
    else:
        print("_all paper pages carry ≥3 keywords._")
    print()

    missing_hook = kw["missing_hook"]
    print(f"## Catalog pages missing a hook ({len(missing_hook)})")
    if missing_hook:
        print("Pages with no `hook:` — the one-line gloss `index.md` renders "
              "after the citation. Agent ingest writes it from the author's "
              "HOOK trailer, so a page here either predates the field, came "
              "from another framework, or had a malformed trailer (the field "
              "is left unset rather than salvaged from a Summary slice, which "
              "would state the paper's question instead of its finding). "
              "Write a result-first gloss — method + scale + distinguishing "
              "finding — to clear the entry.")
        for key in missing_hook[:20]:
            print(f"- {key}")
        if len(missing_hook) > 20:
            print(f"- ... and {len(missing_hook) - 20} more")
    else:
        print("_every catalog page carries a hook._")
    print()

    hook_too_long = kw["hook_too_long"]
    print(f"## Hooks over the advisory ceiling ({len(hook_too_long)})")
    if hook_too_long:
        print("Hooks longer than their page type's ceiling (paper 400, "
              "concept / synthesis / reference 1000, idea 2000 chars). "
              "Advisory: a long hook is index bloat, not a defect, and nothing "
              "truncates it — trimming is the author's call.")
        for key, n, cap in hook_too_long[:20]:
            print(f"- {key} — {n} chars (ceiling {cap})")
        if len(hook_too_long) > 20:
            print(f"- ... and {len(hook_too_long) - 20} more")
    else:
        print("_every hook is within its ceiling._")
    print()

    unquoted_wikilinks = kw["unquoted_wikilinks"]
    print(f"## Unquoted wikilink-list frontmatter ({len(unquoted_wikilinks)})")
    if unquoted_wikilinks:
        print("Frontmatter list fields whose items are unquoted `[[wikilink]]`s. "
              "PyYAML parses `- [[cat/stem]]` as a nested list, which Obsidian's "
              "Properties panel renders as \"?\". Quote each item "
              "(`- \"[[cat/stem]]\"`) so it renders as a link.")
        for key, field in unquoted_wikilinks[:20]:
            print(f"- {key} — `{field}:`")
        if len(unquoted_wikilinks) > 20:
            print(f"- ... and {len(unquoted_wikilinks) - 20} more")
    else:
        print("_no unquoted wikilink lists — all render cleanly in Obsidian._")
    print()

    supp_yaml_missing = kw["supp_yaml_missing"]
    supp_orphans = kw["supp_orphans"]
    if supp_yaml_missing or supp_orphans:
        n_missing_files = sum(len(e["missing"]) for e in supp_yaml_missing)
        n_orphan_files = sum(len(e["files"]) for e in supp_orphans)
        print(f"## Supplementary file inconsistencies "
              f"({n_missing_files} listed-but-missing across {len(supp_yaml_missing)} page(s); "
              f"{n_orphan_files} orphan-on-disk across {len(supp_orphans)} stem(s))")
        if supp_yaml_missing:
            print("YAML lists supplementary files not found in `papers/{stem}.supp/`:")
            for entry in supp_yaml_missing[:20]:
                print(f"- **{entry['page']}** → missing: {', '.join(entry['missing'])}")
            if len(supp_yaml_missing) > 20:
                print(f"_... +{len(supp_yaml_missing) - 20} more_")
        if supp_orphans:
            if supp_yaml_missing:
                print()
            print("Files present in `papers/{stem}.supp/` not listed in any page's YAML:")
            for entry in supp_orphans[:20]:
                print(f"- `{entry['stem']}` → orphan: {', '.join(entry['files'])}")
            if len(supp_orphans) > 20:
                print(f"_... +{len(supp_orphans) - 20} more_")
        print()

    concept_contract = kw["concept_contract"]
    if concept_contract:
        print(f"## Concept-hub contract violations ({len(concept_contract)}, advisory)")
        print("Concept hubs whose authored prose isn't pulling its weight. Warn-only — "
              "these don't fail lint, but each one is worth a look. Definition should "
              "be ≥40 content words, bridge hubs need a populated `## Cross-domain "
              "connections` section, and the Definition shouldn't paraphrase a single "
              "spoke's claim (that means you copied one member paper's text instead "
              "of synthesizing across the corpus).")
        by_page: dict[str, list[tuple[str, str]]] = {}
        for v in concept_contract:
            by_page.setdefault(page_key(v["page"]), []).append((v["kind"], v["detail"]))
        for pk, entries in sorted(by_page.items())[:20]:
            print(f"- **{pk}**")
            for kind, detail in entries:
                print(f"    - `{kind}` — {detail}")
        if len(by_page) > 20:
            print(f"_... +{len(by_page) - 20} more hubs_")
        print()

    dangling_anchors = kw["dangling_anchors"]
    if dangling_anchors:
        print(f"## Dangling claim anchors ({len(dangling_anchors)})")
        print("`[[stem#slug]]` anchors whose (stem, slug) pair no longer resolves in state.db. "
              "Either the target paper was removed, or the target claim's text has changed — "
              "regenerate the slug via `researchwiki claims --by-stem <stem>` and update the citation.")
        # Group by page for readability.
        by_page: dict[str, list[tuple[str, str]]] = {}
        for d in dangling_anchors:
            by_page.setdefault(page_key(d["page"]), []).append((d["stem"], d["slug"]))
        for pk, pairs in sorted(by_page.items())[:20]:
            print(f"- **{pk}**")
            for stem, slug in pairs[:10]:
                print(f"    - `[[{stem}#{slug}]]`")
            if len(pairs) > 10:
                print(f"    - _... +{len(pairs) - 10} more on this page_")
        if len(by_page) > 20:
            print(f"_... +{len(by_page) - 20} more pages_")
        print()

    ungraded_papers = kw["ungraded_papers"]
    if ungraded_papers:
        n_total = sum(p["n_ungraded"] for p in ungraded_papers)
        print(f"## Paper pages with ungraded claims ({len(ungraded_papers)} pages, "
              f"{n_total} claims)")
        for p in ungraded_papers[:20]:
            tail = (
                f"{p['n_ungraded']}/{p['n_claims']} ungraded"
            )
            print(f"- {p['stem']:<60} {tail}")
        if len(ungraded_papers) > 20:
            print(f"- ... ({len(ungraded_papers) - 20} more)")
        print()
        print("_Backfill with `researchwiki grade regression`. New ingests "
              "grade automatically; this only matters for pre-Phase-A pages._")
        print()

    venue_suspect = kw["venue_suspect"]
    if venue_suspect:
        print(f"## Pages whose venue looks like page furniture ({len(venue_suspect)})")
        print("   Fix via `backfill doi` → provider venue, or edit the field directly.")
        for e in venue_suspect[:20]:
            print(f"- {e['page']} — venue={e['venue']!r}")
        print()

    thin_index_text = kw["thin_index_text"]
    if thin_index_text:
        print(f"## Pages too thin for semantic retrieval ({len(thin_index_text)})")
        print("   Invisible to see-also, ingest cross-linking, evolve's KNN and")
        print("   candidates — check the page's H2 names against its page type.")
        for e in thin_index_text[:20]:
            print(f"- {e['page']} ({e['page_type']}) — {e['reason']}")
        print()

    none_placeholders = kw["none_placeholders"]
    if none_placeholders:
        print(f"## Related Papers with a stale `(none)` placeholder above real bullets "
              f"({len(none_placeholders)})")
        print("   Cosmetic. Delete the placeholder line; nothing downstream reads it.")
        for p_ in none_placeholders[:20]:
            print(f"- {p_}")
        if len(none_placeholders) > 20:
            print(f"- ... ({len(none_placeholders) - 20} more)")
        print()

    zero_claim_papers = kw["zero_claim_papers"]
    if zero_claim_papers:
        print(f"## Paper pages with NO claims ({len(zero_claim_papers)})")
        for p in zero_claim_papers[:20]:
            print(f"- {p['category']}/{p['stem']}")
        if len(zero_claim_papers) > 20:
            print(f"- ... ({len(zero_claim_papers) - 20} more)")
        print()
        print("_These are inert as evidence: `claims` returns nothing for them, "
              "no `[[stem#slug]]` anchor exists, and `grade synthesis` can't "
              "verify a citation to them — while every other check stays quiet "
              "(the ungraded-claims list above JOINs `claims`, so it can't see "
              "a page with none). Almost always non-canonical H2 headings: "
              "extraction reads only `## Key Contributions` / `## Results` / "
              "`## Limitations` / `## Methodology and Architecture`. Rename the "
              "heading, then `db rebuild`._")
        print()

    pending_overlap = kw["stems_missing_claim_overlap"]
    if pending_overlap:
        print(f"## Paper pages not yet covered by claim-overlap ({len(pending_overlap)})")
        for s in pending_overlap[:20]:
            print(f"- {s}")
        if len(pending_overlap) > 20:
            print(f"- ... ({len(pending_overlap) - 20} more)")
        print()
        print("_Advisory, not a defect — nothing on disk is wrong. Claim-overlap is "
              "opt-in at ingest (`--claim-overlap`) because it spends an LLM judge "
              "call per candidate pair to confirm a link on roughly one paper in "
              "ten, so it's batched. A stem appears here when it has never been "
              "examined, or when its claims changed since it was (regrade / "
              "re-ingest), which invalidates the earlier comparison. Drain with "
              "`researchwiki claim-overlap --backlog`._")
        print()

    dup_claims = kw["duplicate_claim_sets"]
    if dup_claims is None:
        print("## Near-duplicate claim sets (skipped)")
        print("_The claim-embedding cache (`.semantic-cache/claims.npy`) is cold or covers "
              "less than half the corpus's claims. This check reads it directly rather than "
              "loading the bi-encoder, so `lint` stays instant; warm it with any "
              "`researchwiki claim-overlap` run and the check starts reporting._")
        print()
    elif dup_claims:
        print(f"## Near-duplicate claim sets ({len(dup_claims)}, advisory)")
        print("Page pairs where each page's claims mostly point at the other's as their "
              "nearest neighbour (reciprocal top-1 concentration ≥ 0.25). The failure this "
              "catches is a *commentary* on a paper — a research highlight, News & Views, "
              "editorial — ingested as `type: paper`, whose extracted claims then credit the "
              "original authors' contributions to the commentary. Both fidelity gates pass on "
              "such a page (the claims really are in its PDF), so this structural signal is "
              "the only one available.")
        print()
        print("Advisory, not a defect: legitimate near-duplication is common — a paper and "
              "its own preprint, two trials of one therapy, two reviews of one disease, two "
              "papers from one group. Compare the pair's venues and page types; the shorter "
              "page is the usual suspect, and the fix is `type:`, not the prose.")
        for d in dup_claims[:20]:
            a, b = d["pages"]
            print(f"- **{d['score']:.2f}** — {a['stem']} ↔ {b['stem']}")
            print(f"    - {a['stem']}: {a['n_top1_in_other']}/{a['n_claims']} claims "
                  f"(share {a['top1_share']:.2f}) nearest-match into the other page")
            print(f"    - {b['stem']}: {b['n_top1_in_other']}/{b['n_claims']} claims "
                  f"(share {b['top1_share']:.2f}) nearest-match into the other page")
        if len(dup_claims) > 20:
            print(f"- ... ({len(dup_claims) - 20} more)")
        print()

    db_drift = kw["db_drift"]
    db_drift_fixed = kw["db_drift_fixed"]
    if db_drift:
        n_drift = sum(len(v) for v in db_drift.values())
        if n_drift:
            print("## Structured-DB drift")
            for kind, items in db_drift.items():
                if not items:
                    continue
                print(f"- {kind}: {len(items)}")
                for item in items[:5]:
                    if isinstance(item, dict):
                        print(f"    - {item.get('stem', item)}")
                    else:
                        print(f"    - {item}")
                if len(items) > 5:
                    print(f"    ... ({len(items) - 5} more)")
            print()
            if not kw["fix_applied"]:
                print("_Run with `--fix` to upsert/delete these against the DB, "
                      "or `researchwiki db rebuild` for a full walk._")
                print()

    cross_paper = kw.get("cross_paper") or []
    if cross_paper:
        print(f"## Cross-paper contradictions ({len(cross_paper)})")
        for c in cross_paper[:10]:
            a, b = c["pair"]
            print(f"- **{c['verdict']}** (sim={c['similarity']:.2f})")
            print(f"    A: [[{a['paper_stem']}]] ({a['section']}#{a['position']}) — {a['text']}")
            print(f"    B: [[{b['paper_stem']}]] ({b['section']}#{b['position']}) — {b['text']}")
            if c.get("rationale"):
                print(f"    Judge: {c['rationale']}")
        if len(cross_paper) > 10:
            print(f"- ... ({len(cross_paper) - 10} more)")
        print()
        print("_Cross-paper lint is opt-in (`--cross-paper`); the LLM judge classified each pair_")
        print("_as `disagree_numeric` or `disagree_direction`. Verify against the source PDFs before correcting._")
        print()

    fix_written = kw["fix_written"]
    if kw["fix_applied"] and fix_written:
        total_links = sum(fix_written.values())
        print(f"## Applied `--fix`: {total_links} back-link(s) across {len(fix_written)} page(s)")
        for tgt, n in fix_written.items():
            print(f"- {tgt}: +{n} bullet(s)")
        print()
        print("_Each inserted bullet is marked `(auto-added; refine)`. "
              "On the next ingest/query pass, the LLM should rewrite those one-liners "
              "with the real citation relationship._")
        print()
    if kw["fix_applied"] and db_drift_fixed:
        up = db_drift_fixed.get("upserted", 0)
        de = db_drift_fixed.get("deleted", 0)
        print(f"## Applied `--fix` (DB): {up} upsert(s), {de} delete(s)")
        print()

    return 0
