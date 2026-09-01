"""Machine-readable compatibility constants for lint output."""

LINT_JSON_KEYS = frozenset({
    "pages_scanned", "invalid_frontmatter", "orphans", "broken_wikilinks",
    "broken_index_bullets", "orphan_pdfs", "missing_backlinks", "missing_type",
    "page_type_mismatches", "category_yaml_drift", "stale_synthesis",
    "stale_by_content", "stale_by_audit_count", "p2_entries_with_anchor_hits",
    "stale_evolution_proposals", "missing_keywords", "missing_hook",
    "missing_author_model", "acknowledged_legacy_provenance", "hook_too_long",
    "missing_doi", "stem_year_drift", "unquoted_wikilink_lists",
    "supplementary_missing_on_disk", "supplementary_orphaned_files",
    "dangling_claim_anchors", "orphan_prompts", "broken_prompt_pointers",
    "concept_contract_violations", "idea_contract_violations",
    "dashboard_contract_violations", "ungraded_papers", "venue_suspect",
    "none_placeholders", "thin_index_text", "zero_claim_papers",
    "stems_missing_claim_overlap", "duplicate_claim_sets", "db_drift",
    "cross_paper_contradictions", "cross_paper_stats", "fix_applied",
})
