-- researchwiki state DB schema, v1.
--
-- This DB is a *derived index* over the markdown wiki + papers + caches.
-- Markdown is canonical. The DB is rebuildable from `wiki/` + `papers/` +
-- `.s2-cache/` + `.grade-cache/` via `researchwiki db rebuild`.
--
-- Invariants:
--   - No LLM-authored content lives here. The DB never contains prose that
--     should be edited by hand. If a column starts looking like prose, it
--     belongs in a markdown file instead.
--   - Every column is either parsed deterministically from a markdown file
--     or computed deterministically by the grader. Reproducibility is the
--     property we trade everything for.
--   - On drift between DB and markdown, markdown wins. `db verify` reports
--     drift; `db rebuild` reconciles.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- papers: one row per wiki markdown page (any type — paper / entity / concept /
-- synthesis / comparison / guidance / etc). Mirrors YAML frontmatter so joins
-- and structured queries become trivial without re-parsing markdown.
CREATE TABLE IF NOT EXISTS papers (
    stem TEXT PRIMARY KEY,
    category TEXT NOT NULL,            -- subdir under wiki/ (compbio / cgt / genomics / overviews / references / etc)
    page_type TEXT NOT NULL,           -- frontmatter `type:` (paper / entity / concept / synthesis / comparison / guidance / ...)
    title TEXT NOT NULL,
    year INTEGER,
    doi TEXT,
    venue TEXT,
    publication_status TEXT,           -- 'accelerated-article-preview' / null / etc
    authors TEXT,                      -- raw author list string
    senior_authors TEXT,
    tags TEXT,                         -- JSON array
    pdf_path TEXT,                     -- frontmatter `pdf_path:`, may be null for entity / concept / synthesis / overview pages
    page_path TEXT NOT NULL,           -- absolute path to the wiki .md file
    page_mtime INTEGER NOT NULL,       -- file mtime; used for staleness checks
    pdf_mtime INTEGER,                 -- file mtime of the source PDF, when present
    raw_frontmatter TEXT NOT NULL,     -- JSON of full YAML frontmatter for fidelity
    indexed_at INTEGER NOT NULL        -- when this row was written by `db rebuild`
);

CREATE INDEX IF NOT EXISTS idx_papers_category   ON papers(category);
CREATE INDEX IF NOT EXISTS idx_papers_page_type  ON papers(page_type);
CREATE INDEX IF NOT EXISTS idx_papers_year       ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_doi        ON papers(doi);

-- claims: every gradable bullet from every paper page (Key Contributions +
-- Results, today). Drives the grader, the future synthesis layer, and the
-- structured-similarity surface for cross-paper linking.
--
-- Grader columns are nullable until the grader has run on the claim. They
-- get UPDATEd in place by `grade` (see researchwiki.grade.fidelity.paper).
--
-- `claim_slug` is a deterministic content-addressed identifier (see
-- researchwiki.claim_graph.slug.compute_claim_slug). Invariant-legal: it's a
-- pure function of (section, text), computed at upsert time — no LLM in the
-- loop. The claim-graph edge cache under .claim-graph/edges.db keys on
-- (paper_stem, claim_slug) so edges survive rebuild.
CREATE TABLE IF NOT EXISTS claims (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_stem      TEXT NOT NULL REFERENCES papers(stem) ON DELETE CASCADE,
    section         TEXT NOT NULL,     -- 'key_contributions' | 'results' | 'limitations' | 'methodology' (post-L6)
    position        INTEGER NOT NULL,  -- 0-indexed within section
    text            TEXT NOT NULL,
    claim_slug      TEXT,              -- {section-prefix}-{blake2s-8-hex}, disambiguated by -{position} on collision. See researchwiki/claim_graph/slug.py.
    is_cross_ref    INTEGER NOT NULL DEFAULT 0,   -- 0/1; cross-ref claims are skipped by the grader
    -- Grader output (NULL until graded):
    bm25_top1            REAL,
    bm25_top3_mean       REAL,
    bm25_top1_chunk_id   INTEGER,
    supporting_text      TEXT,         -- verbatim text of the bm25_top1 chunk (≤500 chars). Cached source context for the claim — surfaces in `claims --by-stem` and is read by the cross-paper contradiction judge so it sees the experimental setting alongside the bare claim.
    semantic_score       REAL,         -- max cosine similarity over top-K chunks (bi-encoder)
    embed_model          TEXT,         -- bi-encoder identifier (e.g. 'BAAI/bge-small-en-v1.5'); NULL if semantic skipped
    negation_mismatch    INTEGER,      -- 0/1; deterministic negation parity check
    numeric_tokens       TEXT,         -- JSON array of strings
    numeric_unmatched    TEXT,         -- JSON array; subset of numeric_tokens not found in PDF
    last_graded_at       INTEGER,
    UNIQUE(paper_stem, section, position),
    UNIQUE(paper_stem, claim_slug)
);

CREATE INDEX IF NOT EXISTS idx_claims_stem      ON claims(paper_stem);
CREATE INDEX IF NOT EXISTS idx_claims_section   ON claims(section);
CREATE INDEX IF NOT EXISTS idx_claims_semantic  ON claims(semantic_score);
CREATE INDEX IF NOT EXISTS idx_claims_xref      ON claims(is_cross_ref);
-- idx_claims_slug is created by connection._migrate after the ADD COLUMN;
-- keeping it out of schema.sql means executescript() doesn't fail on
-- existing DBs where the column hasn't been added yet.

-- ingest_iterations: append-only event log of every step the ingest agent
-- takes. The framework writes a row after each phase / each LLM call;
-- the LLM never inserts here directly. See researchwiki/agents/runner.py
-- for the state machine that drives writes.
--
-- Roles:
--   reconcile  : metadata reconciler resolved {title, year, doi, ...}
--   extract    : section extractor produced sections + claims
--   author     : an LLM (or stub) generated a draft for `section`
--   grade      : grader scored a prior author draft (parent_iteration_id)
--   tournament : framework picked a winner across sibling author drafts
--   critic     : critic flagged issues on a prior draft
--   commit     : final draft chosen, markdown written to wiki/
CREATE TABLE IF NOT EXISTS ingest_iterations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id          TEXT NOT NULL,                      -- groups iterations of one ingest run
    paper_stem          TEXT,                               -- nullable: filled once reconcile picks a stem
    pdf_filename        TEXT NOT NULL,                      -- inbox source for traceability
    iteration           INTEGER NOT NULL,                   -- monotonic within attempt
    role                TEXT NOT NULL,                      -- reconcile / extract / author / grade / tournament / critic / commit
    section             TEXT,                               -- summary / key_contributions / results / NULL
    draft_text          TEXT,                               -- author drafts only
    parent_iteration_id INTEGER,                            -- evolution / grade / tournament / critic links back to authored draft
    grader_scores       TEXT,                               -- JSON: {mean_top1, semantic_score, n_drift, n_negation_mismatches, ...}
    critic_notes        TEXT,
    decision            TEXT,                               -- kept / discarded / committed / observed / rejected
    decision_reason     TEXT,
    model_used          TEXT,                               -- 'claude-sonnet-4-6' / 'claude-opus-4-7' / 'stub'
    temperature         REAL,
    cost_input_tokens   INTEGER,
    cost_output_tokens  INTEGER,
    created_at          INTEGER NOT NULL,
    FOREIGN KEY (parent_iteration_id) REFERENCES ingest_iterations(id)
);

CREATE INDEX IF NOT EXISTS idx_iter_attempt  ON ingest_iterations(attempt_id);
CREATE INDEX IF NOT EXISTS idx_iter_stem     ON ingest_iterations(paper_stem);
CREATE INDEX IF NOT EXISTS idx_iter_role     ON ingest_iterations(role);
CREATE INDEX IF NOT EXISTS idx_iter_created  ON ingest_iterations(created_at);
