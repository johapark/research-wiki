"""Per-page assessment of an incoming corpus. Read-only.

The corpus is mixed — some pages come from an older release of this framework and
already satisfy the contract, others from a simpler generator with its own
heading names. So every page is classified individually and nothing is written.

The headline number is **claims_before → claims_after**: parse the page as-is,
then parse the rewritten body in memory, and report both. A page reading `0 → 0`
is the one that would otherwise look migrated — `backfill hook` succeeds on it,
`lint` stays quiet, and it is inert as evidence. Quantifying that before anyone
commits is the point of this module.

Compliance is judged by the **claim parser's** exact-match rule, never by
`coherence`'s prefix match. `## Results (main text)` counts as present to
coherence (`coherence.py:155`) while yielding zero claims from the parser, so
classifying with coherence would mark exactly the broken pages as already fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..stems import derive_stem
from ..wiki import Page, classify_pdf_collision, find_stem_collision
from .frontmatter import FrontmatterPlan, map_keys
from .sections import HeadingPlan, plan_headings, rewrite_headings

#: Verdicts, in order of severity. `apply` acts on `compliant` and `fixable`.
VERDICTS = ("compliant", "fixable", "needs-human", "blocked", "duplicate")

#: A page needs at least this many *graded* canonical sections after aliasing to
#: be one-paper-shaped. Below it we're looking at a note, not a paper page, and
#: importing it produces an unciteable stub — see the "not for" list in
#: `prompts/migration-backfill.md`.
_MIN_GRADED_SECTIONS = 1


@dataclass
class PageAssessment:
    src_page: Path
    src_pdf: Path | None = None
    verdict: str = "fixable"
    reasons: list[str] = field(default_factory=list)
    headings: HeadingPlan | None = None
    frontmatter: FrontmatterPlan | None = None
    derived_stem: str | None = None
    target_category: str = "other"
    page_type: str = "paper"
    claims_before: int = 0
    claims_after: int = 0
    collision: dict | None = None

    @property
    def actionable(self) -> bool:
        return self.verdict in ("compliant", "fixable")

    def as_dict(self) -> dict:
        fm = self.frontmatter
        hp = self.headings
        return {
            "src_page": str(self.src_page),
            "src_pdf": str(self.src_pdf) if self.src_pdf else None,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "derived_stem": self.derived_stem,
            "target_category": self.target_category,
            "page_type": self.page_type,
            "claims_before": self.claims_before,
            "claims_after": self.claims_after,
            "heading_renames": [
                {"from": c.original, "to": c.canonical, "graded": c.graded,
                 "merged": c.merged_into_earlier}
                for c in (hp.changes if hp else [])
            ],
            "heading_ambiguous": [
                {"heading": h, "suggestion": s, "why": w}
                for h, s, w in (hp.ambiguous if hp else [])
            ],
            "heading_unmapped": hp.unmapped if hp else [],
            "fm_renames": [{"from": a, "to": b} for a, b in (fm.renames if fm else [])],
            "fm_conflicts": [
                {"field": k, "values": [{"key": sk, "value": str(v)} for sk, v in pairs]}
                for k, pairs in (fm.conflicts if fm else [])
            ],
            "fm_missing_required": fm.missing_required if fm else [],
            "fm_lookup_needed": fm.lookup_needed if fm else [],
            "fm_notes": fm.notes if fm else [],
            "collision": self.collision,
        }


def _split_authors(raw) -> list[str]:
    """Split a byline into per-author strings, `;` taking precedence over `,`.

    Order matters: `"Doe, Alice; Roe, Bob"` is two authors in `Last, First` form,
    so splitting on both separators at once yields four bogus entries and
    `derive_stem` reads the wrong surname. Splitting on `;` when present keeps
    each `Last, First` pair intact, and `stems.first_author_surname` (called
    inside `derive_stem`) already understands that form.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(a).strip() for a in raw if str(a).strip()]
    s = str(raw).strip()
    if not s:
        return []
    sep = ";" if ";" in s else ","
    return [p.strip() for p in s.split(sep) if p.strip()]


def _count_claims(body: str, fm: dict, *, stem: str, category: str) -> int:
    """Claims a body would yield, without writing anything."""
    from ..grade.parser import parse_claims
    page = Page(path=Path(f"{category}/{stem}.md"), stem=stem, category=category,
                fm=fm or {"type": "paper"}, body=body)
    try:
        return len(parse_claims(page))
    except Exception:
        return 0


def _split_page(text: str) -> tuple[dict, str]:
    """-> (frontmatter dict, body). ({}, whole text) when there's no block."""
    import yaml
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    raw = text[3:end]
    body = text[end + 4:].lstrip("\n")
    try:
        fm = yaml.safe_load(raw)
    except Exception:
        fm = None
    return (fm if isinstance(fm, dict) else {}), body


def assess(
    src_page: Path,
    *,
    pdf_dir: Path,
    category: str = "other",
    require_pdf: bool = True,
) -> PageAssessment:
    """Classify one incoming page. Touches nothing on disk but the source read."""
    a = PageAssessment(src_page=src_page)
    try:
        text = src_page.read_text(encoding="utf-8")
    except Exception as e:
        a.verdict, a.reasons = "blocked", [f"unreadable: {e}"]
        return a

    fm, body = _split_page(text)
    if not fm:
        a.verdict = "blocked"
        a.reasons.append("no YAML frontmatter — add one with at least `title:`")
        return a

    a.frontmatter = fmp = map_keys(fm)
    a.headings = hp = plan_headings(body)
    a.target_category = category
    declared = str(fm.get("type") or fm.get("page_type") or "").strip().strip("\"'")
    a.page_type = declared or "paper"

    # PDF, by matching filename stem.
    candidate = pdf_dir / f"{src_page.stem}.pdf"
    a.src_pdf = candidate if candidate.exists() else None

    # Claims, before and after the rewrite. This is the number that matters.
    rewritten, _ = rewrite_headings(body)
    stem_guess = src_page.stem
    a.claims_before = _count_claims(body, fm, stem=stem_guess, category=category)
    a.claims_after = _count_claims(rewritten, fm, stem=stem_guess, category=category)

    # Stem from frontmatter only — never opens the PDF.
    if not fmp.blocked:
        author_list = _split_authors(fmp.mapped.get("authors"))
        try:
            a.derived_stem = derive_stem(author_list, fmp.mapped["year"],
                                         str(fmp.mapped["title"]))
        except Exception as e:
            a.reasons.append(f"stem derivation failed: {e}")

    # --- verdict, most severe wins ---
    if fmp.blocked:
        a.verdict = "blocked"
        a.reasons.append(
            f"missing required frontmatter: {', '.join(fmp.missing_required)} "
            "(needed to derive a stem; never auto-filled)"
        )
        return a
    if a.derived_stem is None:
        a.verdict = "blocked"
        a.reasons.append("could not derive a stem from frontmatter")
        return a
    if require_pdf and a.src_pdf is None:
        a.verdict = "blocked"
        a.reasons.append(
            f"no {src_page.stem}.pdf beside the page — without it claims can't be "
            "graded, so the page could never ground a citation"
        )
        return a

    # One-paper-shaped check, using post-rewrite canonical sections.
    graded_present = len([c for c in hp.changes if c.graded]) + len(
        [h for h in hp.already_canonical if h in ("Key Contributions", "Results",
                                                  "Limitations",
                                                  "Methodology and Architecture")]
    )
    if a.page_type == "paper" and graded_present < _MIN_GRADED_SECTIONS:
        a.verdict = "blocked"
        a.reasons.append(
            "no gradable section found even after heading aliasing — this doesn't "
            "look like a one-paper page; file it as a references/ doc or a "
            "synthesis page by hand"
        )
        return a

    # Collisions against the existing corpus.
    existing = find_stem_collision(a.derived_stem)
    if existing is not None:
        verdict = classify_pdf_collision(str(fmp.mapped.get("doi") or "") or None, existing)
        a.collision = {"existing_page": str(existing), "classification": verdict}
        if verdict == "duplicate":
            a.verdict = "duplicate"
            a.reasons.append(f"same DOI already at {existing}")
            return a
        a.verdict = "needs-human"
        a.reasons.append(
            f"stem collides with {existing} ({verdict}); a journal upgrade is a "
            "re-ingest decision, see prompts/recovery.md"
        )
        return a

    if fmp.needs_human:
        a.verdict = "needs-human"
        a.reasons.append(
            "frontmatter disagrees with itself: "
            + "; ".join(f"{k} from {[sk for sk, _ in pairs]}" for k, pairs in fmp.conflicts)
        )
        return a
    if hp.needs_human:
        a.verdict = "needs-human"
        for heading, suggestion, why in hp.ambiguous:
            a.reasons.append(f"`## {heading}` → {suggestion}? {why}")
        return a

    if not hp.changes and not fmp.renames:
        a.verdict = "compliant"
    else:
        a.verdict = "fixable"
    return a


def assess_all(
    src_dir: Path, *, pdf_dir: Path | None = None, category: str = "other",
    require_pdf: bool = True,
) -> list[PageAssessment]:
    """Assess every `*.md` directly under `src_dir`, sorted by filename."""
    pdf_dir = pdf_dir or src_dir
    out = [
        assess(md, pdf_dir=pdf_dir, category=category, require_pdf=require_pdf)
        for md in sorted(src_dir.glob("*.md"))
    ]
    resolve_stem_collisions(out)
    return out


def resolve_stem_collisions(assessments: list[PageAssessment]) -> None:
    """Letter-suffix stems that collide *within the incoming batch*.

    CLAUDE.md: a second same-author-same-year paper takes a BibTeX letter and the
    first keeps the bare year. `derive_stem` doesn't do this, so it's done here.
    Deterministic in source-path order, so re-running `inspect` yields the same
    letters.
    """
    seen: dict[str, int] = {}
    for a in sorted(assessments, key=lambda x: str(x.src_page)):
        if not a.derived_stem or not a.actionable:
            continue
        base = a.derived_stem
        n = seen.get(base, 0)
        if n:
            # smith-2024-title -> smith-2024b-title  (b for the 2nd, c for 3rd)
            letter = chr(ord("a") + n)
            parts = base.split("-")
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) == 4:
                    parts[i] = f"{part}{letter}"
                    break
            a.derived_stem = "-".join(parts)
            a.reasons.append(f"stem lettered to {a.derived_stem} (collides with {base} in this batch)")
        seen[base] = n + 1
