"""Grounding post-processor — flag claim-bearing units without a citation.

A wiki answer is "grounded" when every factual claim carries either:

  - a `[[category/stem]]` (or `[[stem]]`) wikilink pointing into wiki/, OR
  - a `[[stem#claim_slug]]` **claim anchor** — the durable, content-addressed
    citation form. The anchor grounds a unit only when the (stem, slug) pair
    resolves against `claims(paper_stem, claim_slug)` in state.db; a bogus
    slug (or an anchor into a claim whose text has drifted) does NOT count
    as a citation, and the unit is reported ungrounded, OR
  - a `[^id]` footnote reference whose definition line (`[^id]: … [[wikilink]]`)
    resolves to a wikilink — academic citation style, where the link lives in
    the reference list at the bottom rather than inline in the prose, OR
  - a `claim_id:NNN` reference to a row in the claims table (legacy; pages
    should cite with `[[wikilink]]` / footnotes — claim ids are volatile).

This module splits a markdown answer into *units* (paragraphs and bullets)
and checks each one. Units are the right granularity — splitting into
sentences would force every clause in a bullet to repeat the citation,
which inflates ungrounded counts on perfectly fine answers.

Public API:

  - parse_units(text) → list[Unit]
  - check(text) → GroundingReport
  - annotate(text) → str    (markdown with `⚠ ungrounded` markers)
  - strip(text) → str       (replaces ungrounded units with a placeholder)
  - extract_claim_anchors(text) → list[(stem, slug)]  (for lint's dangling check)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# A wikilink: `[[anything]]`. We don't validate the *target* here (that's the
# cross-link verifier in researchwiki.agents.phases), but we DO validate a
# claim anchor (`[[stem#slug]]`) — the slug is content-addressed, so a bogus
# one is a specific mistake that must fail grounding. Validation happens
# through the `valid_anchors` param threaded to _make_unit.
_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
# Extract (stem, slug) from `[[stem#slug]]` or `[[stem#slug|display]]`. The
# stem may or may not include a `category/` prefix; both forms are used in
# the wiki, so we accept either. Whitespace inside the anchor is disallowed.
_CLAIM_ANCHOR_RE = re.compile(
    r"\[\[([^\]\|#\s]+)#([^\]\|\s]+)(?:\|[^\]]+)?\]\]"
)
_CLAIM_ID_RE = re.compile(r"\bclaim_id\s*[:=]\s*\d+\b", re.IGNORECASE)

# Academic footnote citations. A *reference* `[^id]` appears inline in a claim
# unit; the citation itself lives in a *definition* line `[^id]: … [[link]]`.
# A unit carrying `[^id]` is grounded only when that definition resolves to a
# wikilink (or claim_id) — see `_grounded_footnotes`. Ids are non-whitespace.
_FOOTNOTE_REF_RE = re.compile(r"\[\^([^\]\s]+)\]")
_FOOTNOTE_DEF_RE = re.compile(r"^[ \t]*\[\^([^\]\s]+)\]:[ \t]*(.*)$", re.MULTILINE)
_FOOTNOTE_DEF_LINE_RE = re.compile(r"^[ \t]*\[\^[^\]\s]+\]:")

# A few sentence shapes that legitimately carry no citation.
# These are Rule-4 disclaimers and meta statements — flagging them as
# ungrounded would be a false positive.
_DISCLAIMER_PATTERNS = [
    re.compile(r"\bno (paper|wiki page) (covers|in this wiki)", re.IGNORECASE),
    re.compile(r"\bthe wiki (has no|does not|doesn't)", re.IGNORECASE),
    re.compile(r"\bi (don't have|do not have|haven't|need to check)", re.IGNORECASE),
    re.compile(r"\bi'?d need to check the pdf", re.IGNORECASE),
    re.compile(r"\bi (would need|cannot answer|can't answer)", re.IGNORECASE),
]

# Strip markdown decorations when measuring "real" word count.
_MD_STRIP_RE = re.compile(r"[*_`>~]+")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_BULLET_RE = re.compile(r"^(\s*)(?:[-*+]|\d+\.)\s+")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
# Caller-controlled "skip this region" marker: HTML comments are invisible
# in rendered markdown, so this is a no-op for the reader. Used by
# `evaluate` to wrap verbatim-idea and run-details sections.
_SKIP_REGION_RE = re.compile(
    r"<!--\s*skip-grounding-start\s*-->.*?<!--\s*skip-grounding-end\s*-->",
    re.DOTALL,
)

# Any HTML comment. These are invisible to the rendered output (and to a
# reader looking at the page in Obsidian), so the grounding gate should not
# treat them as claim-bearing prose. The motivating case: `researchwiki
# synthesize` emits template comments (`<!-- claim_lookup(...) -->`,
# `<!-- no claims indexed for this paper -->`) that often survive into the
# committed page; these would otherwise show up as ungrounded "claims".
# Matched non-greedily so adjacent comments don't get merged.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Idea pages: in permissive mode, units inside Opportunities/Plans H2 sections
# may use the `*(model prior)*` marker as an explicit acknowledgment that the
# claim comes from training knowledge rather than a wiki paper (CLAUDE.md §4).
# `\b` lets the title carry a descriptive suffix like "## Plans — how to ...".
_PERMISSIVE_IDEA_SECTION_RE = re.compile(r"^(opportunities|plans)\b", re.IGNORECASE)
_FRONTMATTER_TYPE_RE = re.compile(r"^type:\s*([^\s#]+)", re.MULTILINE)
_H2_RE = re.compile(r"^##[ \t]+(.+?)\s*$", re.MULTILINE)
_MODEL_PRIOR_RE = re.compile(r"\*\(\s*model\s+prior\s*\)\*", re.IGNORECASE)

# Sections whose content is *meta-commentary* about wiki coverage rather than
# factual claims about source PDFs. Synthesis and idea pages declare their gaps
# under `## What would update this page` — forward-looking statements about
# papers we *don't yet have*, which by construction can't be cited. Skipping
# these sections at parse time means the grounding gate stops treating
# legitimate coverage-declaration prose as ungrounded.
_GATE_SKIP_SECTION_RE = re.compile(r"^what would update this page\b", re.IGNORECASE)

# Minimum word count to treat a unit as a "claim". Below this, it's
# probably a label/lead-in/punctuation noise, not a factual claim.
MIN_CLAIM_WORDS = 8


@dataclass
class Unit:
    """One claim-candidate block: a paragraph or list item."""
    index: int                # 0-based unit index in the document
    line_start: int           # 1-based source line
    text: str                 # raw markdown of the unit (may span multiple lines)
    kind: str                 # 'paragraph' | 'bullet' | 'heading' | 'other'
    is_claim: bool            # passed the claim-shape heuristic
    has_citation: bool        # wiki-cited OR (marker-cited in eligible context)
    is_model_prior: bool = False  # grounded only by the *(model prior)* marker
    citations: list[str] = field(default_factory=list)
    flag_reason: str | None = None  # set when ungrounded

    @property
    def grounded(self) -> bool:
        return (not self.is_claim) or self.has_citation


class ClaimDBUnavailable(RuntimeError):
    """Raised when state.db can't be reached to resolve `[[stem#slug]]` anchors.

    Distinguishes an environment failure (DB locked/missing) from a genuine
    empty resolution, so callers don't misread "DB down" as "every anchor is
    dangling" and spuriously fail a grounded page.
    """


@dataclass
class GroundingReport:
    units: list[Unit]
    # True when anchor resolution was skipped because state.db was unreachable
    # (anchors were then counted permissively rather than failed).
    anchor_db_unavailable: bool = False

    @property
    def total_claims(self) -> int:
        return sum(1 for u in self.units if u.is_claim)

    @property
    def grounded_claims(self) -> int:
        """Wiki-grounded claims (excludes model-prior). The strict-mode count."""
        return sum(1 for u in self.units
                   if u.is_claim and u.has_citation and not u.is_model_prior)

    @property
    def model_prior_claims(self) -> int:
        """Claims grounded only by `*(model prior)*` marker. Default-mode-only."""
        return sum(1 for u in self.units if u.is_claim and u.is_model_prior)

    @property
    def model_prior_units(self) -> list[Unit]:
        return [u for u in self.units if u.is_claim and u.is_model_prior]

    @property
    def ungrounded_units(self) -> list[Unit]:
        return [u for u in self.units if u.is_claim and not u.has_citation]

    @property
    def coverage(self) -> float:
        """Fraction of claim-units acknowledged (wiki-cited OR model-prior).
        1.0 if no claims. Matches pass/fail: any unit not in this fraction is
        an ungrounded failure. In strict mode, model-prior units don't carry
        has_citation, so they fall into the ungrounded slice naturally."""
        if self.total_claims == 0:
            return 1.0
        acknowledged = sum(1 for u in self.units if u.is_claim and u.has_citation)
        return acknowledged / self.total_claims


# ---------- core split ----------

def _blank_region(match: re.Match) -> str:
    """Replace a matched region with the same number of newlines it spanned, so
    downstream line numbers stay aligned with the *original* text."""
    return "\n" * match.group(0).count("\n")


def _is_idea_page(text: str) -> bool:
    """True if frontmatter declares `type: idea`. Tolerates `"idea"` / `'idea'`
    in case an author quotes the value defensively (YAML accepts both)."""
    fm = _FRONTMATTER_RE.match(text)
    if not fm:
        return False
    m = _FRONTMATTER_TYPE_RE.search(fm.group(0))
    return bool(m and m.group(1).strip("\"'").lower() == "idea")


def _o_p_line_ranges(cleaned: str) -> list[tuple[int, int]]:
    """Line ranges (1-based, half-open) covered by Opportunities/Plans H2
    sections. Each range runs from the H2 header line to the next H2 (or EOF).

    Operates on already-cleaned text (frontmatter / code / skip regions blanked)
    so that an Opportunities-shaped header inside a code block can't leak in."""
    headers = list(_H2_RE.finditer(cleaned))
    ranges: list[tuple[int, int]] = []
    for i, m in enumerate(headers):
        if not _PERMISSIVE_IDEA_SECTION_RE.match(m.group(1).strip()):
            continue
        start_line = cleaned.count("\n", 0, m.start()) + 1
        end_pos = headers[i + 1].start() if i + 1 < len(headers) else len(cleaned)
        end_line = cleaned.count("\n", 0, end_pos) + 1
        ranges.append((start_line, end_line))
    return ranges


def _strip_for_processing(text: str) -> str:
    """Neutralize frontmatter, fenced code blocks, caller-marked skip regions,
    and gate-exempt H2 sections (`## What would update this page`) before unit
    splitting.

    These regions are blanked (replaced with an equal number of newlines), not
    deleted: `parse_units` reports each unit's `line_start`, and `annotate` /
    `strip` index those numbers back into the *original* text. Deleting lines
    here would shift every subsequent line number by the removed-line count and
    land the markers on the wrong lines (e.g. on frontmatter)."""
    text = _FRONTMATTER_RE.sub(_blank_region, text, count=1)
    text = _FENCED_CODE_RE.sub(_blank_region, text)
    text = _SKIP_REGION_RE.sub(_blank_region, text)
    # `_SKIP_REGION_RE` matches paired markers; this catches every other
    # HTML comment (synthesize-template leftovers, inline `<!-- TODO -->`
    # markers). Runs after the paired-marker pass so a region wrapped in
    # skip-grounding-start/end is consumed there first.
    text = _HTML_COMMENT_RE.sub(_blank_region, text)
    text = _blank_gate_skip_sections(text)
    return text


def _blank_gate_skip_sections(text: str) -> str:
    """Blank every line of every H2 section whose title matches
    `_GATE_SKIP_SECTION_RE` (currently `## What would update this page`).

    Runs after frontmatter/code/skip-region blanking so a matching heading
    inside a code block can't leak in. Line-count-preserving by construction:
    each blanked line keeps its newline.
    """
    headers = list(_H2_RE.finditer(text))
    if not headers:
        return text
    out_chars = list(text)
    for i, m in enumerate(headers):
        if not _GATE_SKIP_SECTION_RE.match(m.group(1).strip()):
            continue
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        # Preserve newlines so line numbers in annotations still line up.
        for j in range(start, end):
            if out_chars[j] != "\n":
                out_chars[j] = " "
    return "".join(out_chars)


def _grounded_footnotes(text: str) -> set[str]:
    """Footnote ids whose definition line carries a [[wikilink]] (or claim_id).

    A claim unit citing `[^id]` is grounded only if `id` is in this set — i.e.
    the reference points at a real citation, not an empty/prose-only footnote.
    """
    out: set[str] = set()
    for m in _FOOTNOTE_DEF_RE.finditer(text):
        fid, body = m.group(1).strip(), m.group(2)
        if _WIKILINK_RE.search(body) or _CLAIM_ID_RE.search(body):
            out.add(fid)
    return out


def extract_claim_anchors(text: str) -> list[tuple[str, str]]:
    """Return every `(stem, slug)` pair present in `[[stem#slug]]` anchors.

    Deduplicated per (stem, slug) but order-preserving on first occurrence.
    The stem is emitted as written — `stem` or `category/stem`. Callers
    resolving against state.db should normalise both forms.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for m in _CLAIM_ANCHOR_RE.finditer(text):
        stem, slug = m.group(1).strip(), m.group(2).strip()
        key = (stem, slug)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _resolve_claim_anchors(
    pairs: list[tuple[str, str]] | set[tuple[str, str]],
) -> set[tuple[str, str]]:
    """Query state.db for the resolving subset of `(stem, slug)` pairs.

    Returns the set of anchors that resolve. Both `stem` and `category/stem`
    input forms are accepted — the state.db lookup uses only the bare stem.
    Raises `ClaimDBUnavailable` on any DB failure so callers can distinguish it
    from a genuine empty resolution (see `check()`).
    """
    pair_set = {p for p in pairs}
    if not pair_set:
        return set()
    # Bare stem is what claims.paper_stem holds. Callers may pass either
    # `stem` or `category/stem`; strip any leading `category/`.
    bare_pairs = {(s.rsplit("/", 1)[-1], slug) for s, slug in pair_set}
    try:
        from ..db.connection import get_connection
        conn = get_connection()
    except Exception as e:
        raise ClaimDBUnavailable(f"cannot open state.db: {e}") from e
    try:
        placeholders = ",".join(["(?, ?)"] * len(bare_pairs))
        params: list[str] = []
        for s, slug in bare_pairs:
            params.extend([s, slug])
        rows = conn.execute(
            f"SELECT paper_stem, claim_slug FROM claims "
            f" WHERE (paper_stem, claim_slug) IN (VALUES {placeholders})",
            params,
        ).fetchall()
    except Exception as e:
        raise ClaimDBUnavailable(f"claims query failed: {e}") from e
    finally:
        conn.close()
    resolved: set[tuple[str, str]] = set()
    for r in rows:
        # An anchor written as `category/stem#slug` resolves the same way
        # `stem#slug` does — re-emit both forms as valid.
        for orig_stem, orig_slug in pair_set:
            if r["paper_stem"] == orig_stem.rsplit("/", 1)[-1] and r["claim_slug"] == orig_slug:
                resolved.add((orig_stem, orig_slug))
    return resolved


def parse_units(
    text: str,
    permissive: bool = False,
    valid_anchors: set[tuple[str, str]] | None = None,
) -> list[Unit]:
    """Split markdown into units (paragraph or bullet) and tag each.

    A bullet unit absorbs nested bullets at strictly greater indent —
    "Hypothesis: ..." / "Approach: ..." / "Decision rule: ..." sub-bullets
    inherit the parent bullet's citation rather than being graded
    independently. A new top-level unit starts only when a bullet at indent
    ≤ the current parent's indent appears.

    When `permissive=True` AND the document is an idea page, units inside
    Opportunities/Plans H2 sections may use the `*(model prior)*` marker as
    a citation token (per CLAUDE.md §4). Those units are reported as
    `model_prior` rather than `grounded`.

    `valid_anchors`: set of `(stem, slug)` pairs that resolve against
    state.db. When None, claim anchors are NOT validated (any `[[stem#slug]]`
    counts as a citation — legacy behavior). When an empty set is passed
    explicitly, every anchor is treated as dangling. `check()` populates
    this from state.db by default; tests inject their own set.
    """
    cleaned = _strip_for_processing(text)
    # Resolve footnote definitions from the RAW text, not `cleaned`: a
    # definition line (`[^id]: [[link]]`) is a citation *target*, and its
    # ability to ground a reference must not depend on which section it sits
    # under. `_strip_for_processing` blanks the gate-exempt `## What would
    # update this page` section to end-of-document, and footnote defs are
    # conventionally placed at the very bottom — computing on `cleaned` silently
    # dropped them, marking every footnote-only claim ungrounded.
    grounded_fn = _grounded_footnotes(text)
    op_ranges = _o_p_line_ranges(cleaned) if (permissive and _is_idea_page(text)) else []
    lines = cleaned.splitlines()

    units: list[Unit] = []
    buf: list[str] = []
    buf_start = 1
    buf_kind: str | None = None         # 'paragraph' | 'bullet'
    parent_bullet_indent: int | None = None

    def _eligible(line: int) -> bool:
        return any(s <= line < e for s, e in op_ranges)

    def _flush():
        nonlocal buf, buf_start, buf_kind, parent_bullet_indent
        if buf and any(line.strip() for line in buf):
            unit_text = "\n".join(buf).rstrip()
            units.append(_make_unit(len(units), buf_start, unit_text,
                                    buf_kind or "paragraph", grounded_fn,
                                    _eligible(buf_start),
                                    valid_anchors=valid_anchors))
        buf = []
        buf_kind = None
        parent_bullet_indent = None

    for i, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped:
            _flush()
            continue
        if _HEADING_RE.match(raw):
            _flush()
            units.append(_make_unit(len(units), i, raw.rstrip(), "heading",
                                    grounded_fn, _eligible(i),
                                    valid_anchors=valid_anchors))
            continue
        bm = _BULLET_RE.match(raw)
        if bm:
            indent = len(bm.group(1).expandtabs())
            if buf_kind == "bullet" and parent_bullet_indent is not None and indent > parent_bullet_indent:
                # Nested bullet — keep collecting under the current parent.
                buf.append(raw)
            else:
                _flush()
                buf = [raw]
                buf_start = i
                buf_kind = "bullet"
                parent_bullet_indent = indent
            continue
        # Continuation line. Belongs to the in-progress unit if there is one.
        if buf:
            buf.append(raw)
        else:
            buf = [raw]
            buf_start = i
            buf_kind = "paragraph"

    _flush()
    return units


def _make_unit(index: int, line_start: int, text: str, kind: str,
               grounded_fn: frozenset[str] | set[str] = frozenset(),
               model_prior_eligible: bool = False,
               valid_anchors: set[tuple[str, str]] | None = None) -> Unit:
    # Wikilinks split into two classes: (a) plain wikilinks (no anchor) —
    # always count as citations at grounding time; (b) claim anchors
    # `[[stem#slug]]` — count only when the pair resolves via valid_anchors
    # (when validation is enabled).
    wikilinks = _WIKILINK_RE.findall(text)
    counted: list[str] = []
    dangling: list[str] = []
    for wl in wikilinks:
        m = _CLAIM_ANCHOR_RE.match(wl)
        if m is None:
            counted.append(wl)
            continue
        # Claim anchor: validate against the resolver if one was provided.
        if valid_anchors is None:
            counted.append(wl)  # legacy no-validate mode
            continue
        stem, slug = m.group(1).strip(), m.group(2).strip()
        if (stem, slug) in valid_anchors:
            counted.append(wl)
        else:
            dangling.append(wl)

    citations = counted + _CLAIM_ID_RE.findall(text)
    citations += [f"[^{fid}]" for fid in _FOOTNOTE_REF_RE.findall(text) if fid in grounded_fn]
    has_wiki = bool(citations)
    has_marker = model_prior_eligible and bool(_MODEL_PRIOR_RE.search(text))
    if has_marker:
        citations.append("*(model prior)*")
    has_citation = has_wiki or has_marker
    is_model_prior = has_marker and not has_wiki
    is_claim, reason = _classify_claim(text, kind)
    # Dangling-anchor-only units aren't grounded — report the specific fault
    # rather than the generic "no citation" so authors know it's a slug
    # mistake, not a missing citation.
    flag_reason: str | None = None
    if is_claim and not has_citation:
        flag_reason = "claim with dangling [[stem#slug]] anchor" if (dangling and not counted) else reason
    return Unit(
        index=index,
        line_start=line_start,
        text=text,
        kind=kind,
        is_claim=is_claim,
        has_citation=has_citation,
        is_model_prior=is_model_prior,
        citations=citations,
        flag_reason=flag_reason,
    )


def _classify_claim(text: str, kind: str) -> tuple[bool, str]:
    """Return (is_claim, reason_if_ungrounded)."""
    if kind == "heading":
        return False, ""
    # Footnote definition lines (`[^id]: …`) are reference apparatus, not prose
    # claims — even a one-line-per-def block. Exempt them so the bottom-of-page
    # reference list never counts as ungrounded claims.
    nonblank = [ln for ln in text.splitlines() if ln.strip()]
    if nonblank and all(_FOOTNOTE_DEF_LINE_RE.match(ln) for ln in nonblank):
        return False, ""
    cleaned = _MD_STRIP_RE.sub("", text)
    cleaned = re.sub(r"\[\[[^\]]+\]\]", "", cleaned)        # don't count link text
    cleaned = re.sub(r"\bclaim_id\s*[:=]\s*\d+", "", cleaned)
    cleaned = _FOOTNOTE_REF_RE.sub("", cleaned)            # don't count footnote-ref ids
    body = cleaned.strip()
    if not body:
        return False, ""
    # Questions don't make claims.
    if body.endswith("?"):
        return False, ""
    # Rule-4 disclaimers are legitimate non-citations.
    for pat in _DISCLAIMER_PATTERNS:
        if pat.search(body):
            return False, ""
    words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]+", body)
    if len(words) < MIN_CLAIM_WORDS:
        return False, ""
    return True, "claim with no [[wikilink]] or claim_id:NNN"


# ---------- public-facing helpers ----------

def check(
    text: str,
    permissive: bool = False,
    valid_anchors: set[tuple[str, str]] | None = None,
    resolve_anchors: bool = True,
) -> GroundingReport:
    """Grade `text` for grounding.

    `valid_anchors`: explicit set of resolving `(stem, slug)` pairs. When
    provided, no DB access. Useful for tests + callers that already have
    the resolved set.

    `resolve_anchors`: when True (default) and `valid_anchors` is None,
    query state.db to determine which claim anchors resolve. When False,
    skip validation entirely and treat every `[[stem#slug]]` as a citation
    (legacy behavior — matches pre-anchor grounding).
    """
    db_unavailable = False
    if valid_anchors is None and resolve_anchors:
        anchors_in_text = extract_claim_anchors(text)
        if anchors_in_text:
            try:
                valid_anchors = _resolve_claim_anchors(anchors_in_text)
            except ClaimDBUnavailable:
                # DB unreachable ≠ "every anchor dangling". Fall back to the
                # permissive legacy mode (valid_anchors=None → anchors count),
                # and flag it so the CLI can report an environment error rather
                # than failing an otherwise-grounded page.
                valid_anchors = None
                db_unavailable = True
        else:
            valid_anchors = set()
    return GroundingReport(
        units=parse_units(text, permissive=permissive, valid_anchors=valid_anchors),
        anchor_db_unavailable=db_unavailable,
    )


_ANNOTATION = " ⚠ ungrounded"
_MODEL_PRIOR_ANNOTATION = " ⚠ model prior"
_STRIP_PLACEHOLDER = "*[ungrounded — I'd need to check the PDF for this.]*"


def annotate(
    text: str,
    permissive: bool = False,
    valid_anchors: set[tuple[str, str]] | None = None,
) -> str:
    """Return `text` with `⚠ ungrounded` appended to each ungrounded unit's
    last line, and `⚠ model prior` on each marker-grounded unit (default mode
    only). Frontmatter / code blocks are preserved as-is."""
    report = check(text, permissive=permissive, valid_anchors=valid_anchors)
    if not report.ungrounded_units and not report.model_prior_units:
        return text

    lines = text.splitlines()
    # Map each flagged unit to the index of its LAST non-empty source line,
    # so we attach the marker to the actual prose line rather than a blank.
    out_lines = list(lines)
    for unit, marker in [(u, _ANNOTATION) for u in report.ungrounded_units] + \
                        [(u, _MODEL_PRIOR_ANNOTATION) for u in report.model_prior_units]:
        end = unit.line_start - 1 + unit.text.count("\n")
        # Walk backward from end to find a non-blank source line.
        while end >= 0 and not out_lines[end].strip():
            end -= 1
        if 0 <= end < len(out_lines) and marker not in out_lines[end]:
            out_lines[end] = out_lines[end].rstrip() + marker
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


def strip(
    text: str,
    permissive: bool = False,
    valid_anchors: set[tuple[str, str]] | None = None,
) -> str:
    """Return `text` with each ungrounded unit replaced by a placeholder.
    Headings, code, and frontmatter are preserved."""
    report = check(text, permissive=permissive, valid_anchors=valid_anchors)
    if not report.ungrounded_units:
        return text

    lines = text.splitlines()
    redacted: set[int] = set()
    out_lines: list[str | None] = list(lines)
    for unit in report.ungrounded_units:
        start = unit.line_start - 1
        end = start + unit.text.count("\n")
        # Preserve bullet prefix so list structure survives.
        replacement = _STRIP_PLACEHOLDER
        if unit.kind == "bullet":
            m = _BULLET_RE.match(lines[start])
            if m:
                replacement = m.group(0) + _STRIP_PLACEHOLDER
        out_lines[start] = replacement
        for j in range(start + 1, end + 1):
            redacted.add(j)
    final = [ln for i, ln in enumerate(out_lines) if i not in redacted]
    return "\n".join(final) + ("\n" if text.endswith("\n") else "")
