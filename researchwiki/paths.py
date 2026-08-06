"""Canonical wiki repo paths, resolved relative to the current working directory.

The package is invoked from the wiki root (where CLAUDE.md and wiki/ live).
Paths are resolved lazily so callers can `cd` into the repo before running.
"""

from pathlib import Path


def wiki_root() -> Path:
    """Current wiki root — the directory containing `wiki/`, `papers/`, etc.

    We assume the CLI is invoked from the wiki root (matching `scripts/*`
    behaviour). If a caller wants a different root, they can set CWD first.
    """
    return Path.cwd()


def wiki_dir() -> Path:
    return wiki_root() / "wiki"


def papers_dir() -> Path:
    return wiki_root() / "papers"


def supp_dir(stem: str) -> Path:
    """Sibling directory holding supplementary files for `papers/{stem}.pdf`.

    Layout: `papers/{stem}.supp/{filename}`. Sibling — not a containing
    directory — so every existing `papers/{stem}.pdf` path keeps working
    untouched. Returned path may not exist.
    """
    return papers_dir() / f"{stem}.supp"


def benchmark_pdfs_dir() -> Path:
    """Bundled OA-corpus PDFs shipped with the benchmark harness.

    `papers/` is gitignored (per-user), so fresh clones can't score benchmark
    fixtures against the maintainer's personal corpus. Committing CC-BY OA PDFs
    here makes `benchmark-fixtures/` runnable end-to-end after `git clone`.
    """
    return wiki_root() / "benchmark-fixtures" / "pdfs"


def resolve_pdf(stem: str) -> Path:
    """Locate a paper PDF, falling back from personal corpus to bundled OA.

    Precedence:
      1. `papers/{stem}.pdf` — user's canonical wiki-integrated copy.
      2. `benchmark-fixtures/pdfs/{stem}.pdf` — OA corpus shipped with the
         benchmark harness, so fresh clones can score without an ingest.

    Personal corpus wins because when a user has ingested a stem into their
    wiki, that copy is the source of truth for their grading; the bundled
    version is only there so benchmarks work on a fresh checkout.
    """
    primary = papers_dir() / f"{stem}.pdf"
    if primary.exists():
        return primary
    bundled = benchmark_pdfs_dir() / f"{stem}.pdf"
    if bundled.exists():
        return bundled
    raise FileNotFoundError(
        f"PDF not found for stem={stem!r} in {papers_dir()} or {benchmark_pdfs_dir()}"
    )


def inbox_dir() -> Path:
    return wiki_root() / "inbox"


def ensure_scaffold() -> list[Path]:
    """Create the content dirs a working wiki needs; return the ones created.

    `wiki/`, `papers/`, `inbox/`, and the page-type scaffold under `wiki/`
    (`categories.DEFAULT_DIRS`). These are gitignored in full — no `.gitkeep`,
    nothing committed — so a fresh clone has none of them and this is what puts
    them there. Idempotent.

    Deliberately NOT called on every CLI invocation: `__main__` treats a missing
    `wiki/` as "you are in the wrong directory" and exits 2, a guard worth
    keeping. Auto-creating would turn a typo'd `cd` into a phantom empty wiki.

    A path that exists as a *dangling symlink* (a synced folder that has not
    mounted yet) is reported rather than created — `mkdir` on one raises
    `FileExistsError`, and replacing the link would strand the real content.
    """
    from .categories import DEFAULT_DIRS

    root = wiki_root()
    top = [root / "wiki", root / "papers", root / "inbox"]

    # Top level first, and bail before touching the subdirs: `wiki/` is
    # routinely a symlink, and if it dangles then `wiki/<scaffold>/.mkdir()`
    # fails on the *parent* with a bare FileExistsError that names the link and
    # explains nothing.
    dangling = [p for p in top if p.is_symlink() and not p.exists()]
    if dangling:
        names = ", ".join(str(p.relative_to(root)) for p in dangling)
        raise FileExistsError(
            f"dangling symlink(s): {names} — what they point at is missing (an "
            f"unmounted synced folder?). Fix the link or the mount; refusing to "
            f"replace it with an empty directory."
        )

    created = []
    for t in top + [root / "wiki" / d for d in sorted(DEFAULT_DIRS)]:
        if t.is_dir():
            continue
        t.mkdir(parents=True, exist_ok=True)
        created.append(t)
    return created


# The bookkeeping markdown files live INSIDE wiki/ (not at the repo root) so an
# Obsidian vault opened on `wiki/` sees them. Wikilinks from these files into
# wiki/<category>/<stem> resolve cleanly inside the vault.
def index_path() -> Path:
    """Catalog page — `wiki/index.md`. LLM-maintained; gitignored."""
    return wiki_dir() / "index.md"


def log_path() -> Path:
    """Per-user operation history — `wiki/log.md`. Gitignored."""
    return wiki_dir() / "log.md"


# `pdfs_failed_parsing_path()` was removed with the `wiki/pdfs-failed-parsing.md`
# ledger. Extraction failures are recorded per page in YAML `pdf_extraction_note:`
# and surfaced by `researchwiki status`, so there is no separate file to keep in
# sync — the old one had drifted to nonexistent while pages still carried the
# marker, which made the status count read a silent zero. `db rebuild` still
# skips the filename (see `_META_FILENAMES`) for wikis that already have one.


def ingest_dir() -> Path:
    return wiki_root() / ".ingest"


def evolve_cache_dir() -> Path:
    """Derived cache for memory_evolve's judged-pair ledger (`.evolve-cache/`).

    Separate from `state.db` because it holds LLM verdicts, which violate that
    DB's deterministic/rebuildable invariant — same rationale as
    `.claim-graph/`. Gitignored; safe to delete (pairs are re-judged on the
    next run).
    """
    return wiki_root() / ".evolve-cache"


def s2_cache_dir() -> Path:
    return wiki_root() / ".s2-cache"


def crossref_cache_dir() -> Path:
    return wiki_root() / ".crossref-cache"


def web_cache_dir() -> Path:
    """Shared cache for structured-API responses beyond S2/Crossref
    (PubMed, bioRxiv, ORCID, Retraction Watch). One dir so the provenance
    audit can grep a single place."""
    return wiki_root() / ".web-cache"


def search_index_dir() -> Path:
    return wiki_root() / ".tantivy-index"


def grade_cache_dir() -> Path:
    """Per-PDF Tantivy chunk indexes for the coverage grader.
    Each paper gets its own subdirectory: .grade-cache/{stem}/."""
    return wiki_root() / ".grade-cache"


def semantic_cache_dir() -> Path:
    """Page-level semantic embedding store.

    Holds `pages.npy` (L2-normalized float32 row per page) and `pages_meta.json`
    (row-aligned list of {key, stem, category, page_type, title, content_hash}).
    Rebuilt by `researchwiki reindex` alongside the Tantivy index.
    """
    return wiki_root() / ".semantic-cache"


def claim_graph_dir() -> Path:
    """Derived cache for the claim-edge graph.

    Holds `edges.db` — a separate SQLite file so LLM-judged edges never
    contaminate the invariant-pure `state.db`. Gitignored; rebuildable from
    the source judges.
    """
    return wiki_root() / ".claim-graph"
