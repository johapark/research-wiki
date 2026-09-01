"""Rendering for `researchwiki lint` — the JSON contract and the prose report.

Split out of the package `__init__` because presentation was 58% of that file and
none of it participates in *deciding* anything: both emitters take the finished
check results as keyword arguments and only choose how to print them. Keeping
them here means the dispatcher reads as a list of checks, which is what a reader
looking for "what does lint actually verify" comes for.

`_emit_json` is a **contract**: agents parse those keys, so removing or renaming
one is a breaking change (see CONTRIBUTING.md § Releasing). Adding a key is not.
"""

from __future__ import annotations

import json

from ...paths import s2_cache_dir
from .contracts import LINT_JSON_KEYS
from .walk import page_key


def _contract_json(violations: list[dict]) -> list[dict]:
    """Serialize the shared page/kind/detail shape of advisory contracts."""
    return [{"page": page_key(v["page"]), "kind": v["kind"], "detail": v["detail"]}
            for v in violations]


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
        # Two keys for what a hand-deleted page strands. `broken_index_bullets`
        # is `index.md`-only by design (root meta pages are excluded from
        # `broken_wikilinks`, for `log.md`'s sake); `orphan_pdfs` is a list of
        # stems, since the path is always `papers/{stem}.pdf`.
        "broken_index_bullets": kw["broken_index_bullets"],
        "orphan_pdfs": kw["orphan_pdfs"],
        "missing_backlinks": [
            {"src": src, "tgt": t} for src, t in kw["missing_back"]
        ],
        "missing_type": kw["missing_type"],
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
        "missing_author_model": kw["missing_author_model"],
        "acknowledged_legacy_provenance": kw["acknowledged_legacy_provenance"],
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
        "orphan_prompts": kw["orphan_prompts"],
        "broken_prompt_pointers": kw["broken_prompt_pointers"],
        "concept_contract_violations": _contract_json(kw["concept_contract"]),
        "idea_contract_violations": _contract_json(kw["idea_contract"]),
        "dashboard_contract_violations": _contract_json(kw["dashboard_contract"]),
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
        # null = the opt-in check didn't run; an object means it did. Carries the
        # pool size even when max_pairs=0 judged nothing, so sizing a sweep costs
        # no LLM calls. Same null-means-skipped convention as duplicate_claim_sets.
        "cross_paper_stats": kw.get("cross_paper_stats"),
        "fix_applied": {
            "backlinks_added": sum(kw["fix_written"].values()),
            "pages_updated": kw["fix_written"],
            "db_upserted": kw["db_drift_fixed"].get("upserted", 0),
            "db_deleted": kw["db_drift_fixed"].get("deleted", 0),
        } if kw["fix_applied"] else None,
    }
    assert set(out) == LINT_JSON_KEYS
    print(json.dumps(out, indent=2))
    return 0


def _emit_structural_sections(kw: dict) -> None:
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

    from .report_deletions import print_broken_index_bullets, print_orphan_pdfs
    print_broken_index_bullets(kw["broken_index_bullets"])
    print_orphan_pdfs(kw["orphan_pdfs"])

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

    missing_type = kw["missing_type"]
    print(f"## Pages with no `type:` ({len(missing_type)})")
    if missing_type:
        print("Every consumer reads `type` as `fm.get(\"type\", \"paper\")`, so these "
              "behave as papers and no other check sees them. A non-paper page that "
              "lost the field would have its claims extracted and misattributed.")
        for key in missing_type[:20]:
            print(f"- {key}")
        if len(missing_type) > 20:
            print(f"- ... and {len(missing_type) - 20} more")
    else:
        print("_none._")
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
    print(f"## Stale by citation-scout count — pages with cached paper count drift ≥ threshold ({len(stale_by_audit_count)})")
    if stale_by_audit_count:
        for md, cached, current in stale_by_audit_count:
            print(f"- **{page_key(md)}** — `wiki_papers_at_audit: {cached}` → current is {current} "
                  f"(+{current - cached}). Re-run `researchwiki scout` and merge.")
    else:
        print("_none — all pages with `wiki_papers_at_audit` are within drift tolerance._")
    print()

    p2_anchor_hits = kw["p2_anchor_hits"]
    print(f"## Priority-2 entries surfaced in audit anchors ({len(p2_anchor_hits)})")
    if p2_anchor_hits:
        print("_DOIs listed under suggested-additions.md §Priority 2 that now appear in the latest "
              "cached citation scout's `shared_citation_anchors`. Informational — the LLM decides whether "
              "to move, re-annotate, or leave these entries._")
        print()
        for hit in p2_anchor_hits:
            cats = ", ".join(hit["categories"]) if hit["categories"] else "(no category)"
            title = (hit["title"] or "(no title)")[:80]
            print(f"- `{hit['doi']}` — {hit['current_count']}× [{cats}] — {title}")
    elif not list((s2_cache_dir() / ".").glob("audit-*.json")):
        print("_no cached citation-scout snapshot found — run `researchwiki scout --json` first to populate the legacy `.s2-cache/audit-{date}.json` contract._")
    else:
        print("_none — no P2 DOIs intersect the latest citation scout's anchor list._")
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
              "`TODO`/`none`). Without a DOI, scout / preprint-check / "
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


def _emit_metadata_sections(kw: dict) -> None:
    """Render page metadata and supplementary-file findings."""
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

    from .report_provenance import print_author_provenance_sections
    print_author_provenance_sections(
        kw["missing_author_model"], kw.get("acknowledged_legacy_provenance", [])
    )

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


def _emit_page_contract_sections(kw: dict) -> None:
    """Render page-shape and durable-anchor contract findings."""
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

    idea_contract = kw["idea_contract"]
    if idea_contract:
        print(f"## Idea-page contract violations ({len(idea_contract)}, advisory)")
        print("Idea pages whose headings or verdict mirror drift from CLAUDE.md §4. "
              "Warn-only — these don't fail lint. Neither page gate reads headings "
              "(both parse paragraphs), so this is the only check that sees them. "
              "Required order is Verdict → Background → Opportunities → Plans → "
              "Caveats, because sourcing policy differs per section; `verdict:` in "
              "YAML must mirror the label written in the Verdict section.")
        by_page: dict[str, list[tuple[str, str]]] = {}
        for v in idea_contract:
            by_page.setdefault(page_key(v["page"]), []).append((v["kind"], v["detail"]))
        for pk, entries in sorted(by_page.items())[:20]:
            print(f"- **{pk}**")
            for kind, detail in entries:
                print(f"    - `{kind}` — {detail}")
        if len(by_page) > 20:
            print(f"_... +{len(by_page) - 20} more pages_")
        print()

    dashboard_contract = kw["dashboard_contract"]
    if dashboard_contract:
        print(f"## Dashboard contract violations ({len(dashboard_contract)}, advisory)")
        print("`wiki/views.md` has drifted from the dashboard's semantic contract. "
              "Warn-only — custom prose and extra columns are allowed; these findings "
              "cover table order, date sources, limits, and membership semantics.")
        for violation in dashboard_contract:
            print(f"- `{violation['kind']}` — {violation['detail']}")
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


def _emit_corpus_evidence_sections(kw: dict) -> None:
    """Render grading, retrieval, and claim-coverage findings."""
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


def _emit_similarity_and_db_sections(kw: dict) -> None:
    """Render duplicate-claim, DB-drift, cross-paper, and fix summaries."""
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

    from .report_cross_paper import print_cross_paper_section
    print_cross_paper_section(kw.get("cross_paper") or [], kw.get("cross_paper_stats"))

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


def _emit_prose(**kw) -> int:
    """Render the human report from focused, independently testable sections."""
    _emit_structural_sections(kw)
    _emit_metadata_sections(kw)
    _emit_page_contract_sections(kw)
    _emit_corpus_evidence_sections(kw)
    _emit_similarity_and_db_sections(kw)
    return 0
