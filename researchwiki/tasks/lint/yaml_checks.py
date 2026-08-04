"""Frontmatter-shape checks: every check here reads a page's YAML and
flags a structural mismatch.

  - invalid_frontmatter: strict YAML parse failure (Obsidian compat)
  - page_type_mismatches: type=X but lives under wiki/Y/
  - category_drift: YAML category disagrees with parent dir
  - missing_doi: paper-type page without a DOI / no_doi_reason
  - stem_year_drift: stem-encoded year ≠ YAML year
  - missing_keywords: paper page with fewer than MIN_KEYWORDS keywords
  - missing_hook: catalog page with no `hook:` gloss
  - hook_too_long: `hook:` past its page type's advisory ceiling
"""

from __future__ import annotations

import re
from pathlib import Path

from ...categories import PAGE_TYPE_DIRS, content_categories
from .walk import count_keywords, first_category, page_key

try:
    import yaml
    _HAS_PYYAML = True
except ImportError:
    _HAS_PYYAML = False


REFERENCE_TYPES = ("guidance", "protocol", "whitepaper", "book")

#: Minimum acceptable `keywords:` count. Mirrors
#: `agents.phases.commit.MIN_KEYWORDS`, which is the canonical owner because the
#: writer is what enforces it (`render_keywords_yaml` returns None below it).
#:
#: Deliberately duplicated rather than imported: `tasks.lint` pulls in no
#: `agents` module today, and importing one adds ~107 ms to a command that
#: otherwise runs in ~1 s. `tests/test_keywords_threshold.py` asserts the two
#: agree, so the copy can't drift — a test is the cheaper enforcement here.
#:
#: This used to be 3 while the writer refused below 5, which left a dead zone:
#: a page with 3-4 keywords passed lint, so nothing flagged it, while
#: `backfill keywords` refused to write a replacement. Nothing could move it.
MIN_KEYWORDS = 5
_STEM_YEAR_RE = re.compile(r"^[a-z0-9-]+?-(\d{4})[a-z]?-")

# Page types that carry no `index.md` bullet and so need no `hook:`. Everything
# else under a category dir is catalogued and does. Exempting by explicit type
# (rather than requiring by type) keeps the 23 pages that predate the `type:`
# requirement in scope instead of silently excusing them.
HOOK_EXEMPT_TYPES = ("meta", "dashboard")

# Advisory `hook:` ceilings in characters, by page type — mirrors the spec table
# in CLAUDE.md Step 3, itself derived from observed practice. Reported, never
# enforced: silently shortening a hand-written gloss is the failure mode the
# `hook:` field exists to remove, so lint warns and leaves the text alone.
HOOK_MAX_CHARS = {
    "paper": 400,
    "synthesis": 1000,
    "concept": 1000,
    "guidance": 1000,
    "protocol": 1000,
    "whitepaper": 1000,
    "book": 1000,
    "idea": 2000,
}
HOOK_MAX_DEFAULT = 1000


def find_invalid_frontmatter(
    pages: list[Path],
) -> tuple[list[tuple[Path, str, int | None]], bool]:
    """Flag pages whose frontmatter fails strict YAML parsing.

    Returns ``(findings, check_ran)``. ``findings`` is
    ``[(path, error, line_in_file)]`` where line_in_file is 1-indexed
    (or None if the parser didn't mark a position). ``check_ran`` is
    False when PyYAML is missing; callers should surface "skipped"
    rather than pretend everything is clean.

    Obsidian's Properties panel uses a strict YAML parser (js-yaml) and
    silently falls back to raw `---` display when parsing fails. The
    wiki's own ``read_page`` is permissive (line-by-line `partition(":")`)
    so broken frontmatter still round-trips through researchwiki tooling
    — but shows up as a visible inconsistency in Obsidian. This check
    catches that before it accumulates across pages.
    """
    if not _HAS_PYYAML:
        return [], False
    findings: list[tuple[Path, str, int | None]] = []
    for md in pages:
        text = md.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            continue
        end = text.find("\n---\n", 4)
        if end < 0:
            continue
        fm_text = text[4:end]
        try:
            yaml.safe_load(fm_text)
        except yaml.YAMLError as e:
            msg = str(e).split("\n")[0].strip()
            line_no: int | None = None
            mark = getattr(e, "problem_mark", None)
            if mark is not None:
                # PyYAML line numbers are 0-indexed within the block we
                # passed in; +2 converts to 1-indexed file line (skipping
                # the opening `---\n`).
                line_no = mark.line + 2
            findings.append((md, msg, line_no))
    return findings, True


def find_page_type_mismatches(
    pages: list[Path], pages_fm: dict[Path, dict],
) -> list[tuple[str, str]]:
    """Flag pages whose `type:` disagrees with their parent directory.

    `wiki/synthesis/X.md` with `type: paper` (or vice versa).
    `wiki/references/X.md` must declare a reference-document type.
    """
    out: list[tuple[str, str]] = []
    for md in pages:
        key = page_key(md)
        fm = pages_fm.get(md, {})
        ptype = fm.get("type", "paper")
        in_synthesis = md.parent.name == "synthesis"
        in_references = md.parent.name == "references"
        in_concepts = md.parent.name == "concepts"
        if in_synthesis and ptype == "paper":
            out.append((key, f"in synthesis/ but type={ptype}"))
        if not in_synthesis and ptype == "synthesis":
            out.append((key, f"type={ptype} but not in synthesis/"))
        if in_concepts and ptype == "paper":
            out.append((key, f"in concepts/ but type={ptype}"))
        if not in_concepts and ptype == "concept":
            out.append((key, f"type={ptype} but not in concepts/"))
        if in_references and ptype not in REFERENCE_TYPES:
            out.append((key, f"in references/ but type={ptype} (expected one of {REFERENCE_TYPES})"))
        if not in_references and ptype in REFERENCE_TYPES:
            out.append((key, f"type={ptype} but not in references/"))
    return out


def find_category_drift(
    pages: list[Path], pages_fm: dict[Path, dict],
) -> list[tuple[str, str, str]]:
    """Pages whose YAML `category:` disagrees with their parent directory.

    For content-category dirs the directory is canonical: `db rebuild`
    derives a page's category from its parent dir and ignores YAML
    `category:` (denormalized bookkeeping). A page moved between category
    dirs without its YAML updated is functionally correct but its
    frontmatter lies — which silently misleads anyone reading the page
    source or an Obsidian property view, and `broken_wikilinks` won't
    catch it (the page still resolves).

    Page-type dirs (`synthesis`/`ideas`/`references`) are the exception:
    the directory names the page *type*, not a content category, so their
    YAML `category:` legitimately carries the page's *content* category
    instead (e.g. an idea about `ai` lives in `ideas/` but classifies as
    `ai`). There, directory-equality doesn't apply — accept the page-type
    name itself (legacy/decorative) or any valid content category, and
    flag only a value that is neither (a typo or a stale/invalid cat).
    """
    out: list[tuple[str, str, str]] = []
    valid_content: frozenset[str] | None = None
    for md in pages:
        raw_cat = pages_fm.get(md, {}).get("category")
        if not raw_cat:
            continue
        yaml_cat = first_category(raw_cat)
        if not yaml_cat:
            continue
        dir_cat = md.parent.name
        if dir_cat in PAGE_TYPE_DIRS:
            if yaml_cat == dir_cat:
                continue
            if valid_content is None:
                valid_content = content_categories()
            if yaml_cat not in valid_content:
                out.append((page_key(md), yaml_cat,
                            f"{dir_cat}/ (page-type dir — expected a content category)"))
            continue
        if yaml_cat != dir_cat:
            out.append((page_key(md), yaml_cat, dir_cat))
    return out


def find_missing_doi(pages: list[Path], pages_fm: dict[Path, dict]) -> list[str]:
    """Paper-type pages without a DOI value (or DOI is `TODO`/`none`).

    Without a DOI, audit/preprint-check/retraction-check can't query
    S2 or PubMed for this page, and any provenance trail terminates.
    Skip synthesis/reference pages (no DOI by design), explicit
    `doi: TODO` placeholders (the digest path emits these for offline
    ingests), AND pages that declare `no_doi_reason:` (the explicit
    escape hatch for papers without DOIs by design — NeurIPS posters,
    workshop papers, internal tech reports).
    """
    out: list[str] = []
    for md in pages:
        if md.parent.name in ("synthesis", "references", "concepts"):
            continue
        fm = pages_fm.get(md, {})
        if fm.get("type", "paper") != "paper":
            continue
        if (fm.get("no_doi_reason") or "").strip():
            continue
        doi_raw = (fm.get("doi") or "").strip().strip('"').strip("'").lower()
        if not doi_raw or doi_raw == "todo" or doi_raw == "none":
            out.append(page_key(md))
    out.sort()
    return out


def find_stem_year_drift(
    pages: list[Path], pages_fm: dict[Path, dict],
) -> list[dict]:
    """Pages where the stem-encoded year differs from YAML `year:`.

    The stem encodes year at ingest time; YAML can be patched later.
    When they diverge, surface so the human can decide which is
    authoritative (usually YAML, since the stem-stability rule blocks
    re-stemming). Lettered years (smith-2024b-...) match the bare year
    (2024).

    Two legitimate causes:
      • preprint→journal version updates that shift year by +1
        (CLAUDE.md keeps the preprint-era stem to preserve back-links)
      • a buggy reconcile at ingest that's been patched in YAML but
        not yet propagated to the stem
    """
    out: list[dict] = []
    for md in pages:
        if md.parent.name in ("synthesis", "references", "concepts"):
            continue
        fm = pages_fm.get(md, {})
        if fm.get("type", "paper") != "paper":
            continue
        m = _STEM_YEAR_RE.match(md.stem)
        if not m:
            continue
        stem_year = int(m.group(1))
        yaml_year_raw = str(fm.get("year") or "").strip()
        try:
            yaml_year = int(yaml_year_raw)
        except (ValueError, TypeError):
            continue
        if stem_year != yaml_year:
            out.append({
                "page": page_key(md),
                "stem_year": stem_year,
                "yaml_year": yaml_year,
            })
    return out


_FM_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:")
_UNQUOTED_WIKILINK_ITEM_RE = re.compile(r"^\s*-\s*\[\[")


def find_unquoted_wikilink_lists(pages: list[Path]) -> list[tuple[str, str]]:
    """Frontmatter list fields whose items are unquoted `[[wikilink]]`s.

    PyYAML parses an unquoted `- [[cat/stem]]` item as a *nested* list
    (`[['cat/stem']]`), which Obsidian's Properties panel can't type and renders
    as "?". Quoting the item (`- "[[cat/stem]]"`) makes it a plain string that
    renders as a clickable link. Applies to any wikilink-list field
    (`companion_synthesis`, concept `referenced_papers`, …). Returns
    `(page_key, field)`. Scans the raw frontmatter block so it's independent of
    how the frontmatter parser recovers.
    """
    out: list[tuple[str, str]] = []
    for md in pages:
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        m = re.match(r"(?s)^---\n(.*?)\n---\n", text)
        if not m:
            continue
        field = ""
        seen: set[str] = set()
        for line in m.group(1).split("\n"):
            km = _FM_KEY_RE.match(line)
            if km:
                field = km.group(1)
            elif field and field not in seen and _UNQUOTED_WIKILINK_ITEM_RE.match(line):
                out.append((page_key(md), field))
                seen.add(field)
    out.sort()
    return out


def find_missing_keywords(
    pages: list[Path], pages_fm: dict[Path, dict],
) -> list[tuple[str, int]]:
    """Paper- and reference-type pages with fewer than `MIN_KEYWORDS`
    `keywords:` entries.

    The field is indexed by both BM25 and the semantic page index;
    sparse keywords degrade search recall on terms the Summary doesn't
    mention. `keywords:` is required (not optional) on `type: paper`
    pages and on the four reference-document types (guidance, protocol,
    whitepaper, book) — see CLAUDE.md Page Types §1 and §3. Synthesis /
    idea / concept pages are exempt (their citation-oriented body carries
    the search substrate instead).
    """
    out: list[tuple[str, int]] = []
    for md in pages:
        if md.parent.name in ("synthesis", "concepts", "ideas"):
            continue
        fm = pages_fm.get(md, {})
        ptype = fm.get("type", "paper")
        if ptype != "paper" and ptype not in REFERENCE_TYPES:
            continue
        n_kw = count_keywords(fm.get("keywords", ""))
        if n_kw < MIN_KEYWORDS:
            out.append((page_key(md), n_kw))
    out.sort()
    return out


def _catalog_pages(
    pages: list[Path], pages_fm: dict[Path, dict],
) -> list[tuple[Path, str, str]]:
    """Pages that get an `index.md` bullet → (path, page type, hook text).

    Skips root-level bookkeeping (`index.md`, `log.md`, `views.md`,
    `pdfs-failed-parsing.md`), which produce slashless page keys, and any page
    whose declared type is explicitly exempt.
    """
    out: list[tuple[Path, str, str]] = []
    for md in pages:
        key = page_key(md)
        if "/" not in key:
            continue
        fm = pages_fm.get(md, {})
        ptype = str(fm.get("type") or "paper").strip().strip("\"'")
        if ptype in HOOK_EXEMPT_TYPES:
            continue
        hook = str(fm.get("hook") or "").strip()
        out.append((md, ptype, hook))
    return out


def find_missing_hook(
    pages: list[Path], pages_fm: dict[Path, dict],
) -> list[str]:
    """Catalog pages with no `hook:` — the one-line gloss `index.md` renders
    after the citation (CLAUDE.md Step 3).

    Agent ingest writes the field automatically from the author's HANDLE/HOOK
    trailer, so a page appearing here means one of three things: it predates the
    field, it came from another framework, or the author's trailer was missing or
    malformed and the field was deliberately left unset rather than salvaged from
    a Summary slice (which yields the paper's question, not its finding).

    Backfill route for all three: `phases.commit.propose_short_name` derives the
    handle and hook from an existing page body in one lightweight call.
    """
    out = [page_key(md) for md, _, hook in _catalog_pages(pages, pages_fm) if not hook]
    out.sort()
    return out


def find_hook_too_long(
    pages: list[Path], pages_fm: dict[Path, dict],
) -> list[tuple[str, int, int]]:
    """Pages whose `hook:` exceeds the advisory ceiling for their page type.

    Returns (page key, actual chars, ceiling), longest first. Advisory only —
    a long hook is index bloat, not a defect, and trimming is a judgement call
    left to the author.
    """
    out: list[tuple[str, int, int]] = []
    for md, ptype, hook in _catalog_pages(pages, pages_fm):
        cap = HOOK_MAX_CHARS.get(ptype, HOOK_MAX_DEFAULT)
        if hook and len(hook) > cap:
            out.append((page_key(md), len(hook), cap))
    out.sort(key=lambda row: (-row[1], row[0]))
    return out
