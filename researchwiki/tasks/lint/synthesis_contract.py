"""Synthesis-page heading contract lint.

Synthesis pages share one fixed H2 structure (CLAUDE.md §2, procedure in
`prompts/synthesis-page-author.md`):

  Question → Short answer → Background → Organizing framework → Findings →
  Cross-cutting insights → Tensions / open questions → Outlook → References

The structure is fixed so every page reads the same way, and two of its
sections carry rules the gates depend on: `Outlook` is the one section where
`*(inference)*` and `*(model prior)*` labels count (`grounding.py`'s
`_LABELLED_SECTION_RES["synthesis"]`), and `Short answer` must come first in
the semantic index. Neither page gate reads headings, so this is the only check
that sees a missing or misplaced section.

  synthesis_legacy_spine
    The page predates the fixed structure: it has no `## Organizing framework`
    or still carries `## What would update this page`. One finding per page,
    instead of a missing-section finding for each new heading, because the fix
    is a single upgrade (procedure Mode C), not nine edits.

  synthesis_missing_section
    One of the required H2s is absent. `References` is required only when the
    page uses `[^id]` footnotes.

  synthesis_section_order
    All present required sections are there but out of order.

  synthesis_duplicate_section
    A required H2 appears more than once. Split `## References` blocks scatter
    the footnote definitions, and `wiki.extract_section` — which `impact_review`
    and the semantic index both use — reads only the first occurrence, so the
    second copy is invisible to every consumer that works by section name.
    (Grounding's labelled ranges do cover both copies; the risk is the reader's
    and the tooling's, not a lost label.)

  synthesis_unexpected_h2
    An H2 outside the structure. Extra material belongs in an H3 under
    `Findings`.

  synthesis_findings_without_themes
    `## Findings` has no H3. Themes are what keep the page organized by idea
    rather than by paper.

  synthesis_label_outside_outlook
    A `*(inference)*` or `*(model prior)*` label outside `## Outlook`. It does
    nothing there — the unit is graded, or flagged, as ordinary prose — so the
    author's intent is silently lost.

  synthesis_footnotes_undefined
    `[^id]` references with no definition anywhere on the page.

Checks are **warn-only** — reported by `lint --json` under
`synthesis_contract_violations` but never flip the exit code, the same staging
as `idea_contract`.
"""

from __future__ import annotations

import re
from pathlib import Path


# The structure, in order. CLAUDE.md §2 is canonical; keep these identical.
REQUIRED_SECTIONS = (
    "question",
    "short answer",
    "background",
    "organizing framework",
    "findings",
    "cross-cutting insights",
    "tensions / open questions",
    "outlook",
    "references",
)

# Headings that mark the pre-structure shape. `What would update this page`
# is the strongest signal: the fixed structure replaced it with `Outlook`.
_LEGACY_MARKERS = ("what would update this page", "evidence from the wiki")

_H2_RE = re.compile(r"^##[ \t]+(.+?)\s*$", re.MULTILINE)
_H3_RE = re.compile(r"^###[ \t]+", re.MULTILINE)
_LABEL_RE = re.compile(r"\*\(\s*(?:inference|model\s+prior)\s*\)\*", re.IGNORECASE)
_FOOTNOTE_REF_RE = re.compile(r"\[\^([^\]\s]+)\](?!:)")
# Leading indent allowed, matching `grounding._FOOTNOTE_DEF_RE` and
# `claim_anchors`: an indented definition is valid CommonMark (and is how
# the authoring prompt's own fenced example renders), so rejecting it here
# reported a false `synthesis_footnotes_undefined` on a page the other two
# readers were happy with.
_FOOTNOTE_DEF_RE = re.compile(r"^[ \t]*\[\^([^\]\s]+)\]:", re.MULTILINE)
_FENCED_CODE_RE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _clean(body: str) -> str:
    """Blank fenced code and HTML comments, preserving line count."""
    def _blank(m: re.Match) -> str:
        return "\n" * m.group(0).count("\n")
    return _HTML_COMMENT_RE.sub(_blank, _FENCED_CODE_RE.sub(_blank, body))


def _canonical(name: str) -> str | None:
    """Map an H2 title to its section key. Exact match only: the names are
    part of the contract (`Outlook` is matched by the grounding gate and
    `Short answer` by the semantic index), so a variant is a defect."""
    return name if name in REQUIRED_SECTIONS else None


def _sections(cleaned: str) -> list[tuple[str, str, str]]:
    """(title, lowercased name, body) for each H2, in document order."""
    headers = list(_H2_RE.finditer(cleaned))
    out = []
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(cleaned)
        title = m.group(1).strip()
        out.append((title, title.lower(), cleaned[m.end():end]))
    return out


def is_legacy(body: str) -> bool:
    """True when the page has the pre-structure shape (see module docstring)."""
    names = {name for _, name, _ in _sections(_clean(body))}
    return "organizing framework" not in names or any(m in names for m in _LEGACY_MARKERS)


def check_page(path: Path, body: str) -> list[dict]:
    """Run every contract check on one synthesis page. Returns violation dicts:
    {page, kind, detail}. Empty when the page passes."""
    cleaned = _clean(body)
    if is_legacy(body):
        return [{
            "page": path,
            "kind": "synthesis_legacy_spine",
            "detail": "predates the fixed structure; upgrade with "
                      "prompts/synthesis-page-author.md (Mode C)",
        }]

    violations: list[dict] = []
    sections = _sections(cleaned)
    refs = {m.group(1) for m in _FOOTNOTE_REF_RE.finditer(cleaned)}
    defs = {m.group(1) for m in _FOOTNOTE_DEF_RE.finditer(cleaned)}

    required = [k for k in REQUIRED_SECTIONS if k != "references" or refs or defs]
    seen: list[str] = []
    counts: dict[str, int] = {}
    for _, name, _ in sections:
        key = _canonical(name)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        if key not in seen:
            seen.append(key)

    # A repeated section is not a cosmetic slip, though not for the reason it
    # might seem: `_labelled_line_ranges` iterates *every* matching header, so
    # labels in a second `## Outlook` are live, not inert. What breaks is the
    # reader's model of the page — two `## References` split the footnote
    # definitions, and a second `## Outlook` puts labelled prose where nobody
    # looks for it. De-duplicating into `seen` is what let this pass.
    for key, n in counts.items():
        if n > 1:
            violations.append({
                "page": path,
                "kind": "synthesis_duplicate_section",
                "detail": f"`## {key[0].upper() + key[1:]}` appears {n} times; "
                          "merge them — a reader (and `extract_section`) finds "
                          "only the first",
            })

    for key in required:
        if key not in seen:
            violations.append({
                "page": path,
                "kind": "synthesis_missing_section",
                "detail": f"no `## {key[0].upper() + key[1:]}` H2 found",
            })

    expected_order = [k for k in REQUIRED_SECTIONS if k in seen]
    if seen != expected_order:
        violations.append({
            "page": path,
            "kind": "synthesis_section_order",
            "detail": "expected " + " → ".join(expected_order)
                      + "; found " + " → ".join(seen),
        })

    for title, name, _ in sections:
        if _canonical(name) is None:
            violations.append({
                "page": path,
                "kind": "synthesis_unexpected_h2",
                "detail": f"`## {title}` is not part of the synthesis structure; "
                          "use an H3 under `## Findings`",
            })

    for _, name, text in sections:
        if name == "findings" and not _H3_RE.search(text):
            violations.append({
                "page": path,
                "kind": "synthesis_findings_without_themes",
                "detail": "`## Findings` has no H3 themes",
            })
        if name != "outlook" and _LABEL_RE.search(text):
            violations.append({
                "page": path,
                "kind": "synthesis_label_outside_outlook",
                "detail": f"source label under `## {name}` has no effect outside "
                          "`## Outlook`",
            })

    undefined = sorted(refs - defs)
    if undefined:
        shown = ", ".join(f"`[^{r}]`" for r in undefined[:5])
        more = f" (+{len(undefined) - 5} more)" if len(undefined) > 5 else ""
        violations.append({
            "page": path,
            "kind": "synthesis_footnotes_undefined",
            "detail": f"{len(undefined)} footnote ref(s) with no definition: "
                      f"{shown}{more}",
        })

    return violations


def find_synthesis_contract_violations(
    pages: list[Path], pages_body: dict[Path, str], pages_fm: dict[Path, dict],
) -> list[dict]:
    """Run contract checks on every wiki/synthesis/*.md typed `synthesis`.
    `type: meta` pages there (suggested-additions.md) are skipped."""
    out: list[dict] = []
    for md in pages:
        if md.parent.name != "synthesis":
            continue
        fm = pages_fm.get(md, {}) or {}
        # `.lower()` to agree with `grounding._page_type`, which lowercases.
        # A page typed `Synthesis` otherwise got the Outlook label privileges
        # from both gates while being invisible to the only check that reads
        # its headings — so a renamed or duplicated `## Outlook` went
        # unreported on exactly the pages holding the gate exemption.
        if str(fm.get("type", "")).strip("\"'").lower() != "synthesis":
            continue
        out.extend(check_page(md, pages_body.get(md, "")))
    return out
