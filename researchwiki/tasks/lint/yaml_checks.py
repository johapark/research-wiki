"""Frontmatter-shape checks: every check here reads a page's YAML and
flags a structural mismatch.

  - invalid_frontmatter: strict YAML parse failure (Obsidian compat)
  - page_type_mismatches: type=X but lives under wiki/Y/
  - category_drift: YAML category disagrees with parent dir
  - missing_doi: paper-type page without a DOI / no_doi_reason
  - stem_year_drift: stem-encoded year ≠ YAML year
  - missing_keywords: paper page with <3 keywords
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
_STEM_YEAR_RE = re.compile(r"^[a-z0-9-]+?-(\d{4})[a-z]?-")


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
    """Paper- and reference-type pages with empty or fewer-than-3
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
        if n_kw < 3:
            out.append((page_key(md), n_kw))
    out.sort()
    return out
