"""Helpers for reading the local wiki state (YAML frontmatter, stems, DOIs).

`read_page(md)` is the canonical parser — it parses the frontmatter with PyYAML,
so values keep their real types (lists, ints, dates). Use the type-tolerant
`Page.str_field` / `Page.list_field` / `Page.year_int` accessors to read fields
that may be non-scalar. Higher-level helpers (`read_wiki_dois`,
`read_wiki_papers`) are thin wrappers over it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import papers_dir, wiki_dir

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]*`")
MD_HEADING_RE = re.compile(r"^#{1,6}\s.*$", re.MULTILINE)


@dataclass
class Page:
    """Parsed wiki page: frontmatter dict + body text."""
    path: Path
    stem: str
    category: str            # directory name: "compbio", "cgt", "synthesis", ...
    fm: dict[str, Any]       # YAML frontmatter; values keep their parsed types
    body: str                # post-frontmatter text

    @property
    def key(self) -> str:
        """Relative stem path for [[wikilink]] resolution, e.g. 'compbio/abramson-2024-...'"""
        return f"{self.category}/{self.stem}"

    @property
    def page_type(self) -> str:
        return self.fm.get("type", "paper")

    # --- type-tolerant frontmatter accessors -----------------------------
    # These read the same field correctly whether `fm` came from the legacy
    # line parser (every value a str, lists as the literal "[a, b]") or from
    # real YAML (native list / int / date). Call sites use these so the parser
    # can switch underneath without touching them.

    def str_field(self, key: str, default: str = "") -> str:
        """Frontmatter value as a string. Lists join with ', '; None → default."""
        v = self.fm.get(key)
        if v is None:
            return default
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return str(v)

    def list_field(self, key: str) -> list[str]:
        """Frontmatter value as a list of trimmed strings.

        Handles a native YAML list, the line parser's literal "[a, b]" string,
        a bare "a, b" string, and None/missing (→ []).
        """
        v = self.fm.get(key)
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        return [t.strip() for t in s.split(",") if t.strip()]

    def year_int(self) -> int | None:
        """`year` as an int (YAML int or numeric string), else None."""
        v = self.fm.get("year")
        if isinstance(v, bool):       # bool is an int subclass; never a year
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        return None


def read_page(md: Path) -> Page | None:
    """Parse a wiki markdown file into frontmatter + body.

    Returns None only when the file has no frontmatter block at all (no leading
    `---` fence) — i.e. it isn't a wiki page. A present-but-malformed YAML block
    yields a Page with `fm={}` rather than None, so a single typo never silently
    drops the page from search / audit / lint (which flag the bad YAML
    separately via `invalid_frontmatter`).
    """
    text = md.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    try:
        fm = yaml.safe_load(text[4:end])
    except yaml.YAMLError:
        fm = None
    if not isinstance(fm, dict):
        fm = {}
    body = text[end + 5:]
    return Page(
        path=md,
        stem=md.stem,
        category=md.parent.name,
        fm=fm,
        body=body,
    )


def read_pages(exclude_synthesis: bool = False) -> list[Page]:
    """Return every parseable wiki page. Sorted by path for stability."""
    out: list[Page] = []
    root = wiki_dir()
    if not root.exists():
        return out
    for md in sorted(root.rglob("*.md")):
        if exclude_synthesis and md.parent.name == "synthesis":
            continue
        p = read_page(md)
        if p is not None:
            out.append(p)
    return out


def find_orphan_pdfs() -> list[str]:
    """PDF stems in `papers/` with no `wiki/**/{stem}.md`. Sorted.

    `papers/` holds one canonically-named PDF per page, and the stem is the join
    key — `db rebuild` derives `papers/{stem}.pdf` from it. So a PDF whose stem
    matches no page is a file nothing in the wiki can reach: not searchable, not
    citable, not in `index.md`, and invisible to every other check, all of which
    start from the page corpus and walk outwards. `researchwiki remove` deletes
    page and PDF together, but a page deleted by hand strands the PDF silently.

    **Not always a defect.** `remove --keep-pdf` produces this state on purpose,
    so the paper can be re-ingested clean. Two exits: `agent ingest
    papers/{stem}.pdf`, or delete the file.

    Takes no arguments and walks `wiki_dir()` itself, rather than accepting the
    caller's page list. Its two consumers had disagreed: `lint` walks every
    `*.md`, while `status` holds `read_pages()`, which drops any file with no
    `---` frontmatter fence — so a hand-written page missing its fence would
    have had its PDF reported as orphaned by one and not the other. Walking
    here makes them the same answer by construction. It also keeps `status` off
    the `lint` package, whose `__init__` costs ~20 ms to import for one
    predicate.

    Non-recursive glob on purpose: `papers/{stem}.supp/*.pdf` is supplementary
    material, which belongs to its parent page and is `find_supplementary_issues`'
    business. A recursive walk would report every one of those here as well.
    """
    pdir = papers_dir()
    if not pdir.is_dir():
        return []
    root = wiki_dir()
    page_stems = {md.stem for md in root.rglob("*.md")} if root.is_dir() else set()
    return sorted(
        pdf.stem for pdf in pdir.glob("*.pdf")
        if pdf.is_file() and pdf.stem not in page_stems
    )


def strip_non_prose(text: str) -> str:
    """Remove HTML comments, fenced code, inline code, and markdown headers.

    These contain template examples, acronyms, and section names that produce
    false positives in wikilink / concept-candidate extraction.
    """
    text = HTML_COMMENT_RE.sub("", text)
    text = CODE_FENCE_RE.sub("", text)
    text = INLINE_CODE_RE.sub("", text)
    text = MD_HEADING_RE.sub("", text)
    return text


def extract_section(body: str, heading: str) -> str:
    """Return the text of a `## {heading}` section (or empty string if absent).

    The section ends at the next `##` heading or EOF.
    """
    heading_re = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.IGNORECASE | re.MULTILINE)
    m = heading_re.search(body)
    if not m:
        return ""
    start = m.end()
    next_m = re.search(r"^##?\s+", body[start:], re.MULTILINE)
    end = start + next_m.start() if next_m else len(body)
    return body[start:end].strip()


# ---------- higher-level convenience functions ----------

def read_wiki_dois() -> dict[str, str]:
    """Return {doi_lower: category/stem} for each paper page with a `doi:` line."""
    out: dict[str, str] = {}
    for p in read_pages(exclude_synthesis=True):
        # `doi:` with no value parses as None (the .get default doesn't apply
        # when the key exists), so coerce before lowering.
        doi = str(p.fm.get("doi") or "").lower()
        if doi:
            out[doi] = p.key
    return out


def read_wiki_stems() -> dict[str, Path]:
    """`{stem: path}` for every page in the wiki. One directory walk.

    The batched sibling of `find_stem_collision`, for callers testing many
    stems at once. That function re-walks `wiki/` on every call, which is right
    for the single-shot uses (ingest, promote, synthesize) but turns a loop over
    N records into O(N x pages): measured at 0.5 ms per call over 117 pages, so
    ~0.3 s for a 500-record import today and worse as the corpus grows, against
    one walk here.

    Includes *every* `.md` — synthesis, ideas, concepts, bookkeeping — because a
    stem must be unique across the whole wiki, not just among paper pages. That
    is the one way this differs from `read_wiki_dois`, which excludes synthesis
    deliberately.

    Not cached: callers that write pages and re-check (`import apply` between
    waves) need to see their own writes.
    """
    root = wiki_dir()
    if not root.exists():
        return {}
    return {md.stem: md for md in root.rglob("*.md")}


def read_wiki_papers() -> list[dict[str, str]]:
    """Return list of {stem, category, doi, title, year} dicts for each paper page."""
    papers: list[dict[str, str]] = []
    for p in read_pages(exclude_synthesis=True):
        doi = p.fm.get("doi")
        if not doi:
            continue
        papers.append({
            "stem": p.stem,
            "category": p.category,
            "doi": str(doi),
            "title": p.str_field("title"),
            "year": p.str_field("year"),
        })
    return papers


def find_stem_collision(stem: str) -> Path | None:
    """Return the path of an existing wiki page with this stem, else None."""
    root = wiki_dir()
    if not root.exists():
        return None
    for md in root.rglob("*.md"):
        if md.stem == stem:
            return md
    return None


# Preprint server DOI prefixes (same set used by `researchwiki preprint-check`).
PREPRINT_DOI_PREFIXES = ("10.1101/", "10.64898/", "10.31219/", "10.20944/", "10.48550/")


def is_preprint_doi(doi: str | None) -> bool:
    """True if the DOI's prefix is a known preprint server (bioRxiv, medRxiv,
    OSF, Preprints.org, arXiv)."""
    d = (doi or "").lower().strip()
    return any(d.startswith(p) for p in PREPRINT_DOI_PREFIXES)


def _doi_from_existing_page(page_path: Path) -> str | None:
    """Cheap YAML scrape — read the `doi:` field without invoking pyyaml."""
    try:
        text = page_path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    for line in text[4:end].splitlines():
        if line.startswith("doi:"):
            return line[4:].strip().strip('"').strip("'") or None
    return None


def classify_pdf_collision(new_doi: str | None, existing_page_path: Path) -> str:
    """Classify a stem collision into one of:

    - `duplicate`: same DOI on both sides — incoming PDF is a re-ingest.
    - `journal-upgrade`: existing page is preprint, incoming is journal version.
    - `preprint-downgrade`: existing page is journal, incoming is preprint.
    - `unclear`: missing DOIs or both/neither look like preprints.
    """
    existing_doi = _doi_from_existing_page(existing_page_path)
    if not new_doi or not existing_doi:
        return "unclear"
    if new_doi.lower().strip() == existing_doi.lower().strip():
        return "duplicate"
    new_pre = is_preprint_doi(new_doi)
    old_pre = is_preprint_doi(existing_doi)
    if old_pre and not new_pre:
        return "journal-upgrade"
    if new_pre and not old_pre:
        return "preprint-downgrade"
    return "unclear"


def _normalize_external_doi(item: dict | None) -> str | None:
    if not item:
        return None
    ex = item.get("externalIds") or {}
    doi = (ex.get("DOI") or "").lower().strip()
    return doi or None


def intersect_crosslinks(refs: list[dict], wiki_dois: dict[str, str]) -> list[dict]:
    """For each reference whose DOI matches a wiki paper, return a hit dict."""
    hits: list[dict] = []
    for r in refs:
        doi = _normalize_external_doi(r)
        if doi and doi in wiki_dois:
            hits.append({
                "doi": doi,
                "title": (r or {}).get("title") or "",
                "year": (r or {}).get("year"),
                "wikilink": f"[[{wiki_dois[doi]}]]",
            })
    return hits


def commit_page(md: Path) -> None:
    """Persist a just-written wiki page into the structured index.

    Call this immediately after writing/editing a markdown page under `wiki/`.
    Single-page DB upsert is cheap (one connection, one transaction); doing it
    at every write keeps the DB from drifting behind markdown so downstream
    surfaces (`status`, `claims`, `db query`) stay trustworthy without anyone
    having to remember `db rebuild`.

    Failures are swallowed by design — markdown is canonical and the DB is a
    derived index. A failed commit logs and moves on; the next `db rebuild` or
    `lint --fix` will reconcile.
    """
    try:
        from .db import upsert_page as _upsert
        _upsert(md)
    except Exception as e:
        from .log import log
        log(f"WARN: db commit_page failed for {md}: {e}", tag="wiki")


def intersect_incoming(citations: list[dict], wiki_dois: dict[str, str]) -> list[dict]:
    """For each citing paper whose DOI matches a wiki paper, return a hit dict."""
    hits: list[dict] = []
    for c in citations:
        doi = _normalize_external_doi(c)
        if doi and doi in wiki_dois:
            hits.append({
                "doi": doi,
                "title": (c or {}).get("title") or "",
                "year": (c or {}).get("year"),
                "wikilink": f"[[{wiki_dois[doi]}]]",
            })
    return hits
