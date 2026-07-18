"""CLI subcommands. Each module must expose `main(argv: list[str]) -> int`.

Subcommands are auto-discovered by `researchwiki/__main__.py`; this list is
informational. Run `researchwiki --help` for the canonical set.

  - ingest     — PDF → wiki-ready digest pipeline
  - agent      — full agentic ingest (LLM author + grader gate + promote)
  - attach     — attach a supplementary file to an existing wiki page
  - audit      — citation-graph audit across all current wiki papers (needs S2)
  - status     — one-screen local health check (no network)
  - synthesize — scaffold entity / concept / synthesis / comparison pages
  - lint       — deeper local consistency checks (orphans, broken links, stale syntheses)
  - reindex    — rebuild the Tantivy + semantic page indexes
  - search     — hybrid (BM25 + semantic) search over the wiki
  - evolve     — propose memory-evolution edits for neighboring synthesis pages
  - … see `researchwiki --help` for the full set.
"""
