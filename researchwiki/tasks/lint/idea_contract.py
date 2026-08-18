"""Idea-page heading contract lint.

CLAUDE.md §4 fixes five H2 sections in one order — Verdict → Background →
Opportunities → Plans → Caveats — because the *sourcing policy differs per
section*: Verdict/Background/Caveats are strict-grounded while
Opportunities/Plans admit `*(model prior)*` clauses. A section in the wrong
place, or missing, therefore isn't cosmetic: it puts prose under a grounding
policy that wasn't written for it.

Nothing else in the toolchain checks this. Both mandatory page gates
(`check-grounding`, `grade synthesis`) parse *units* — paragraphs and bullets —
so they never read a heading. `grounding.py`'s `_PERMISSIVE_IDEA_SECTION_RE`
matches only `^(opportunities|plans)\\b`, and it exists to locate the
model-prior-eligible ranges, not to validate the contract. An idea page whose
Verdict prose sits above the first H2 with no `## Verdict` heading passes both
gates green — which is the case this module was written for.

  idea_missing_section
    One of the five required H2s is absent. Reported once per missing section.

  idea_section_order
    All five are present but not in canonical order. Names the observed order.

  idea_unexpected_h2
    An H2 outside the allowed set (the five, plus `## References` and
    `## What would update this page`). Catches `## Related Papers`, which is a
    *paper*-page section that several idea pages carry as an empty trailing
    stub.

  idea_missing_verdict_field
    No YAML `verdict:`. The prompt's step 6 requires mirroring the section's
    label into frontmatter so `index.md` / Dataview can triage without parsing
    prose.

  idea_verdict_label_mismatch
    YAML `verdict:` disagrees with the label written in the Verdict section.
    Whichever is stale, a reader gets two different answers.

  idea_verdict_label_unparseable
    A `## Verdict` section exists but carries no recognizable strength label,
    so the mirror can't be checked at all.

  idea_footnotes_without_references
    The body uses `[^id]` footnote references but has no `## References` H2 to
    define them, leaving the citations dangling in Obsidian.

Checks are **warn-only** — reported by `lint --json` under
`idea_contract_violations` but never flip the exit code. Same staging as
`concept_contract`: promote to defect after two calibration rounds on real
pages.
"""

from __future__ import annotations

import re
from pathlib import Path


# The contract, in order. CLAUDE.md §4 is canonical; keep these identical.
REQUIRED_SECTIONS = ("verdict", "background", "opportunities", "plans", "caveats")

# H2s that may appear alongside the required five without being a defect.
#   references — where `[^id]` footnotes are defined; near-universal in practice
#   what would update this page — gate-exempt by name in grounding.py, and
#     CLAUDE.md §4 folds it into Caveats, so authors sometimes promote it to H2
_ALLOWED_EXTRA_SECTIONS = ("references", "what would update this page")

_H2_RE = re.compile(r"^##[ \t]+(.+?)\s*$", re.MULTILINE)

# Strength label inside the Verdict section. Anchored on the opening `**` so a
# parenthetical qualifier can't hijack the match: in
# `**Strength: incremental (with strong upside contingent on Phase 5)**` the
# first match is `incremental`, not the later `strong`. The `Strength:` prefix
# is optional because some pages open with a bare `**incremental** — …`.
_VERDICT_LABEL_RE = re.compile(
    r"\*\*\s*(?:Strength\s*:\s*)?(strong|incremental|weak)\b",
    re.IGNORECASE,
)

VALID_LABELS = frozenset({"strong", "incremental", "weak"})

# A footnote *reference* (`[^ashr]`) vs a *definition* (`[^ashr]: …`). The
# reference pattern deliberately excludes the definition form so a page that
# only defines footnotes it never cites isn't counted as using them. The two
# are checked separately because the failure modes differ in severity: a ref
# with no definition anywhere is a broken citation, while a definition block
# that simply isn't under a `## References` H2 still renders correctly in
# Obsidian and is only convention drift.
_FOOTNOTE_REF_RE = re.compile(r"\[\^([^\]\s]+)\](?!:)")
_FOOTNOTE_DEF_RE = re.compile(r"^\[\^([^\]\s]+)\]:", re.MULTILINE)

# Structural markup neutralized before scanning, so an H2-shaped line or a
# label inside a fenced code block can't leak into the checks.
_FENCED_CODE_RE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _clean(body: str) -> str:
    """Blank fenced code and HTML comments, preserving line count so any future
    line-number reporting stays accurate."""
    def _blank(m: re.Match) -> str:
        return "\n" * m.group(0).count("\n")
    return _HTML_COMMENT_RE.sub(_blank, _FENCED_CODE_RE.sub(_blank, body))


def _h2_titles(cleaned: str) -> list[str]:
    """H2 titles in document order, original case preserved (the checks
    lowercase for matching; the reports quote the author's own text)."""
    return [m.group(1).strip() for m in _H2_RE.finditer(cleaned)]


def _canonical(name: str) -> str | None:
    """Map an H2 title to its required-section key, tolerating a descriptive
    suffix (`## Plans — how to actually build this`). Returns None when the
    title isn't one of the five."""
    for key in REQUIRED_SECTIONS:
        if name == key or name.startswith(key):
            return key
    return None


def _is_allowed_extra(name: str) -> bool:
    return any(name == a or name.startswith(a) for a in _ALLOWED_EXTRA_SECTIONS)


def _verdict_section(cleaned: str) -> str | None:
    """Body text of the `## Verdict` section, or None when absent."""
    headers = list(_H2_RE.finditer(cleaned))
    for i, m in enumerate(headers):
        if _canonical(m.group(1).strip().lower()) == "verdict":
            start = m.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(cleaned)
            return cleaned[start:end]
    return None


def _yaml_verdict(fm: dict) -> str | None:
    """Frontmatter `verdict:`, unquoted and lowercased. None when absent/blank."""
    raw = fm.get("verdict")
    if raw is None:
        return None
    v = str(raw).strip().strip("\"'").strip().lower()
    return v or None


def check_page(path: Path, body: str, fm: dict) -> list[dict]:
    """Run every contract check on one idea page. Returns violation dicts:
    {page, kind, detail}. Empty when the page passes."""
    violations: list[dict] = []
    cleaned = _clean(body)
    titles = _h2_titles(cleaned)
    names = [t.lower() for t in titles]

    # Observed order of the required sections, de-duplicated on first sight.
    seen: list[str] = []
    for name in names:
        key = _canonical(name)
        if key and key not in seen:
            seen.append(key)

    # (a) missing sections — one violation each, so the report names them all
    for key in REQUIRED_SECTIONS:
        if key not in seen:
            violations.append({
                "page": path,
                "kind": "idea_missing_section",
                "detail": f"no `## {key.capitalize()}` H2 found",
            })

    # (b) order — only meaningful once all five are present; otherwise the
    #     missing-section violations above are the actionable report.
    if len(seen) == len(REQUIRED_SECTIONS) and tuple(seen) != REQUIRED_SECTIONS:
        violations.append({
            "page": path,
            "kind": "idea_section_order",
            "detail": "expected " + " → ".join(REQUIRED_SECTIONS)
                      + "; found " + " → ".join(seen),
        })

    # (c) unexpected H2s
    for title, name in zip(titles, names):
        if _canonical(name) is None and not _is_allowed_extra(name):
            violations.append({
                "page": path,
                "kind": "idea_unexpected_h2",
                "detail": f"`## {title}` is not part of the idea-page contract",
            })

    # (d)/(e)/(f) verdict label mirror
    yaml_verdict = _yaml_verdict(fm)
    section = _verdict_section(cleaned)
    section_label: str | None = None
    if section is not None:
        m = _VERDICT_LABEL_RE.search(section)
        if m:
            section_label = m.group(1).lower()
        else:
            violations.append({
                "page": path,
                "kind": "idea_verdict_label_unparseable",
                "detail": "`## Verdict` carries no **strong**/**incremental**/"
                          "**weak** label",
            })

    if yaml_verdict is None:
        violations.append({
            "page": path,
            "kind": "idea_missing_verdict_field",
            "detail": "no YAML `verdict:`"
                      + (f" (section says `{section_label}`)" if section_label else ""),
        })
    elif yaml_verdict not in VALID_LABELS:
        violations.append({
            "page": path,
            "kind": "idea_verdict_label_mismatch",
            "detail": f"YAML `verdict: {yaml_verdict}` is not one of "
                      + "/".join(sorted(VALID_LABELS)),
        })
    elif section_label and yaml_verdict != section_label:
        violations.append({
            "page": path,
            "kind": "idea_verdict_label_mismatch",
            "detail": f"YAML `verdict: {yaml_verdict}` vs section label "
                      f"`{section_label}`",
        })

    # (g) footnote refs with nowhere to resolve — a broken citation
    refs = {m.group(1) for m in _FOOTNOTE_REF_RE.finditer(cleaned)}
    defs = {m.group(1) for m in _FOOTNOTE_DEF_RE.finditer(cleaned)}
    undefined = sorted(refs - defs)
    if undefined:
        shown = ", ".join(f"`[^{r}]`" for r in undefined[:5])
        more = f" (+{len(undefined) - 5} more)" if len(undefined) > 5 else ""
        violations.append({
            "page": path,
            "kind": "idea_footnotes_undefined",
            "detail": f"{len(undefined)} footnote ref(s) with no definition: "
                      f"{shown}{more}",
        })

    # (h) definitions exist but aren't declared under a `## References` H2.
    #     Renders fine — this is convention drift, not a broken citation.
    has_references = any(n == "references" or n.startswith("references")
                         for n in names)
    if defs and not has_references:
        violations.append({
            "page": path,
            "kind": "idea_references_section_missing",
            "detail": f"{len(defs)} footnote definition(s) but no "
                      f"`## References` H2 declaring them",
        })

    return violations


def find_idea_contract_violations(
    pages: list[Path], pages_body: dict[Path, str], pages_fm: dict[Path, dict],
) -> list[dict]:
    """Run contract checks on every wiki/ideas/*.md. Non-idea pages are skipped
    silently. Idempotent, no DB reads — every input is already on disk by the
    time lint gets here."""
    out: list[dict] = []
    for md in pages:
        if md.parent.name != "ideas":
            continue
        fm = pages_fm.get(md, {}) or {}
        if str(fm.get("type", "")).strip("\"'") != "idea":
            continue
        out.extend(check_page(md, pages_body.get(md, ""), fm))
    return out
