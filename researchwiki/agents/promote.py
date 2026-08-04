"""Auto-promote a winning agent draft from sandbox to the canonical wiki.

Phase 2.7 closes the agent-output → wiki/ loop, but only when grader gates
pass. The default is conservative: if any signal is weak, the page stays in
.agent-output/ and the user reviews + promotes manually. The user can force
either side with --auto-promote / --force-sandbox CLI flags.

What promotion does, in order:
  1. Decide a category via search.suggest_category (same heuristic the legacy
     ingest uses) and bake it into the YAML frontmatter.
  2. mv {pdf_path} → papers/{stem}.pdf when the source is in inbox/. If the
     PDF is already in papers/ (re-ingest) leave it.
  3. Write wiki/{category}/{stem}.md (the agent's draft, frontmatter rebuilt
     to match CLAUDE.md conventions).
  4. Append back-links — for each verified outgoing wikilink, add a bullet
     to the target page's Related Papers section. Idempotent.
  5. Append a single-line entry to index.md under the category heading.
  6. Append a multi-line ingest entry to log.md (parseable-prefix format).

Failure modes are surfaced via PromotionResult.warnings — a missing index.md
or log.md, an unwritable wiki/ directory, etc. The runner persists the
result to ingest_iterations.decision_reason so any partial state is auditable.
"""

from __future__ import annotations

import datetime as _dt
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .. import backlinks as _bl
from ..fsatomic import write_text_atomic
from ..paths import inbox_dir, papers_dir, wiki_dir


# Promotion gates — all must pass.
# Tentative bi-encoder thresholds; recalibrate after first eval run on the
# 5-paper fixture. BGE-small cosine for true-supported claim/passage pairs
# typically clusters 0.55-0.80; setting the gate at 0.55 should accept most
# faithful pages while flagging serious paraphrase drift.
SEMANTIC_MEAN_THRESHOLD = 0.55
SEMANTIC_MEAN_THRESHOLD_REVIEW = 0.45
MIN_GRADED_CLAIMS = 5
MIN_KEY_CONTRIBUTIONS = 4           # Summary alone doesn't earn auto-promote


@dataclass
class GateResult:
    promoted: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # non-blocking signals (e.g. drafter hallucinated wikilinks that verify already stripped)


def should_auto_promote(
    scores: dict,
    verification,
    n_key_contributions: int = 0,
    paper_type: str = "research",
) -> GateResult:
    """Evaluate the promotion gates. All must pass for auto-promote.

    `scores` is the aggregate dict from phases.grade_draft(). `verification` is
    the VerificationReport from verify_crosslinks. `n_key_contributions` is
    a structural check — pages with too few KCs likely missed the prompt.
    `paper_type` is 'research' (default) or 'review'; reviews use a relaxed
    semantic-similarity threshold because their claims paraphrase other
    papers and only loosely match the source PDF.
    """
    fails: list[str] = []
    warnings: list[str] = []
    sem_threshold = (
        SEMANTIC_MEAN_THRESHOLD_REVIEW if paper_type == "review" else SEMANTIC_MEAN_THRESHOLD
    )

    if not scores.get("semantic_available"):
        fails.append("semantic scorer was disabled — auto-promote requires semantic signal")
    else:
        sem = scores.get("semantic_score") or 0.0
        if sem < sem_threshold:
            fails.append(
                f"mean semantic {sem:.2f} < {sem_threshold:.2f} threshold "
                f"(paper_type={paper_type})"
            )

    drift = scores.get("n_drift") or 0
    if drift > 0:
        fails.append(f"{drift} numeric drift claim(s)")

    # Qualitative-support veto — the entailment analogue of the numeric-drift
    # veto (grade.support). Absent/0 when the support check didn't run, so this
    # is inert until the check is enabled and calibrated; once populated, a
    # claim the cited passage does not support blocks auto-promote the same way
    # a drifted number does.
    unsupported = scores.get("n_unsupported") or 0
    if unsupported > 0:
        fails.append(f"{unsupported} unsupported claim(s)")

    # Broken wikilinks are NOT a hard gate-fail. By construction they are
    # already stripped from the cleaned text by verify_crosslinks — the list
    # records targets the drafter wrote that didn't exist (typically
    # external baselines hallucinated as wiki pages, e.g. [[GEARS]] for a
    # benchmark not in the wiki). The post-strip text is shippable; the
    # drafter's intent is preserved as a warning so iteration logs and
    # PromotionResult.warnings still surface the signal.
    if verification is not None and verification.broken:
        warnings.append(
            f"{len(verification.broken)} broken wikilink(s) stripped at verify: "
            f"{verification.broken}"
        )

    n_graded = scores.get("n_graded") or 0
    if n_graded < MIN_GRADED_CLAIMS:
        fails.append(f"only {n_graded} graded claims (need {MIN_GRADED_CLAIMS})")

    if n_key_contributions < MIN_KEY_CONTRIBUTIONS:
        fails.append(
            f"only {n_key_contributions} Key Contribution bullets "
            f"(need {MIN_KEY_CONTRIBUTIONS} — page may be incomplete)"
        )

    return GateResult(promoted=not fails, reasons=fails, warnings=warnings)


# ---------- promotion ----------

@dataclass
class PromotionResult:
    promoted: bool
    wiki_path: Path | None = None
    pdf_path: Path | None = None
    backlinks_added: list[str] = field(default_factory=list)
    backlinks_skipped: list[str] = field(default_factory=list)
    index_updated: bool = False
    log_appended: bool = False
    category: str | None = None
    warnings: list[str] = field(default_factory=list)
    pdf_upgrade: dict | None = None  # set when _move_pdf swapped a preprint PDF for a journal version


_HEADING_RE = re.compile(r"^## (.+?)\s*$", re.MULTILINE)


def _suggest_category(title: str, summary: str) -> tuple[str | None, str]:
    """Use the existing search-index helper.

    Returns a (category, strength) tuple:
      - (cat, 'strong') when ≥ 3 of top-5 neighbors agree.
      - (cat, 'weak') when there's at least one neighbor but no consensus —
        first-of-kind topics. Caller flags via frontmatter.
      - (None, 'none') when the index is empty or seed text is blank.
    """
    try:
        from ..search import SearchBackendUnavailable, get_default_backend, suggest_category
    except ImportError:
        return None, "none"
    try:
        backend = get_default_backend()
    except SearchBackendUnavailable:
        return None, "none"
    try:
        suggestion = suggest_category(backend, title, summary)
    except Exception:
        return None, "none"
    if suggestion is None:
        return None, "none"
    return suggestion.category, suggestion.strength


def _extract_section(body: str, heading_re_lower: str) -> str:
    """Return the body of a `## {heading}` section, or ''."""
    pat = re.compile(rf"^##\s+{heading_re_lower}\s*$", re.IGNORECASE | re.MULTILINE)
    m = pat.search(body)
    if not m:
        return ""
    start = m.end()
    next_m = re.search(r"^##\s+", body[start:], re.MULTILINE)
    end = start + next_m.start() if next_m else len(body)
    return body[start:end].strip()


def _count_key_contributions(body: str) -> int:
    section = _extract_section(body, "key contributions")
    return sum(
        1 for line in section.splitlines()
        if line.strip().startswith(("- ", "* "))
    )


# `_extract_summary_first_sentence` used to build the index gloss from sentence 1
# of `## Summary`, sliced to 200 chars. It was removed with the `hook:` field:
# a Summary opener states the paper's *question*, not its finding, and the slice
# landed after the sentence match, so 97 of 244 generated entries ended mid-word.
# The gloss now comes from the author's HOOK trailer (see
# `phases.draft.split_gloss_trailer`) and is never derived from page prose — if
# no hook is available the field is left unset for `lint` to flag.


def detect_publication_status(
    pdf_text: str | None,
    doi: str | None,
    pdf_meta: dict | None = None,
) -> str | None:
    """Cheap rules-based publication-status detection.

    Returns one of: 'arxiv-preprint', 'biorxiv-preprint', 'medrxiv-preprint',
    'accelerated-article-preview', 'proof-pdf', or None for a regular
    published paper.

    Only two text fingerprints mean "not yet the peer-reviewed final version":
    Nature's AAP banner and the accepted-manuscript disclaimer. Two earlier
    fingerprints were dropped/retargeted because neither indicates a preview —
    together they mislabeled 80 of 83 flagged pages:

    - `"advance access"` was removed. It is OUP's online-first program, which
      publishes the *final* copyedited version ahead of pagination — and the
      string also appears in OUP's unfilled LaTeX template
      (`Advance Access Publication Date: Day Month Year`), so it flagged an
      arXiv preprint typeset in the Bioinformatics class as a journal preview.
    - the `Published online: xx xx xxxx` placeholder now returns 'proof-pdf'.
      It marks a Springer *proof* of the final version, downloaded before the
      online date was stamped — the article is or will be published normally,
      so calling it a preview overstated the case. 'proof-pdf' says only what
      the file is; re-download to clear it.
    """
    if doi:
        d = doi.lower()
        if d.startswith("10.48550/arxiv."):
            return "arxiv-preprint"
        if d.startswith("10.1101/"):
            # bioRxiv and medRxiv share the 10.1101/ prefix; differentiate
            # via the year-formatted ID. Both treated as preprint.
            return "biorxiv-preprint"

    head = (pdf_text or "")[:6000].lower()
    if "accelerated article preview" in head:
        return "accelerated-article-preview"
    if "this is a pdf file of a peer-reviewed paper that has been accepted" in head:
        return "accelerated-article-preview"
    # Typeset proof of the final version: Springer leaves the publication-online
    # field as a literal placeholder until the online date is assigned. Not a
    # preview — see the docstring.
    if re.search(r"published\s+online:\s*x{1,2}\s+x{1,3}\s+x{2,4}", head):
        return "proof-pdf"
    return None


def _detect_senior_authors(authors_str: str | None) -> str | None:
    """Heuristic: senior authors are typically the LAST author plus an optional
    second-to-last. Returns 'X' or 'X, Y' when there are at least 4 authors —
    on shorter lists the senior/first conflation makes the field misleading."""
    if not authors_str:
        return None
    parts = [a.strip() for a in authors_str.split(",") if a.strip()]
    if len(parts) < 4:
        return None
    return parts[-1] if len(parts) < 6 else f"{parts[-2]}, {parts[-1]}"


def _yaml_dq(value: str) -> str:
    """Double-quoted YAML scalar, collapsed to one line.

    Used for `hook:`, where quoting is load-bearing rather than cosmetic: hooks
    routinely contain `[[wikilinks]]`, which PyYAML parses as a nested flow
    sequence when unquoted (the failure `lint` reports as
    `unquoted_wikilink_lists`), and `:` / `#`, which break a bare scalar.
    """
    flat = " ".join(str(value).split())
    escaped = flat.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _build_frontmatter(
    metadata: dict,
    stem: str,
    category: str,
    body: str,
    short_name: str | None = None,
    hook: str | None = None,
    category_strength: str = "strong",
    keywords: list[str] | None = None,
    author_model: str | None = None,
) -> str:
    """Produce a CLAUDE.md-shaped YAML frontmatter for the agent's draft.

    `category_strength` is 'strong' (≥ 3-of-5 neighbor agreement),
    'weak' (top-1 fallback for first-of-kind topics), or 'none' (no
    category suggested — typically means category is 'TODO'). Only
    'weak' is surfaced in the frontmatter so the user knows the auto-
    suggested category needs review.

    `author_model` (when set) is the LLM model that produced the page
    body — emitted as `author_model:` in the YAML so a reader can
    tell which model authored the page (Sonnet 4.6 vs Opus 4.7 vs a
    local model vs stub). Per-phase model tracking lives in the
    `ingest_iterations` DB table; this field is the at-a-glance
    surface for the page itself. Distinct from the paper's `authors:`
    field (which is the paper's actual authors).
    """
    title = (metadata.get("title") or "").replace('"', '\\"')
    authors = metadata.get("authors") or "unknown"
    year = metadata.get("year") or "unknown"
    doi = metadata.get("doi") or ""
    venue = metadata.get("venue") or ""
    pub_status = metadata.get("publication_status")
    # Skip senior-author detection when authors came from --authors override
    # AND the user-supplied list is short — they likely truncated for
    # convenience, and inferring "senior = last entry" would silently mislabel.
    # When the list is long enough (≥10) it's plausibly complete.
    authors_origin = metadata.get("authors_origin")
    n_authors = len(authors.split(",")) if isinstance(authors, str) else 0
    if authors_origin == "override" and n_authors < 10:
        senior = None
    else:
        senior = _detect_senior_authors(authors if isinstance(authors, str) else None)

    fm_lines = [
        "---",
        f'title: "{title}"',
        f"authors: {authors}",
    ]
    if senior:
        fm_lines.append(f"senior_authors: {senior}")
    fm_lines.append(f"year: {year}")
    if doi:
        fm_lines.append(f"doi: {doi}")
    if doi and doi.lower().startswith("10.48550/arxiv."):
        # arXiv ID for traceability: 10.48550/arXiv.NNNN.NNNNN → arxiv_id: NNNN.NNNNN
        arxiv_id = doi.split(".", 1)[1].split(".", 1)[1] if "arXiv." in doi else None
        if arxiv_id:
            # Quote: an unquoted arXiv id like 2501.01230 parses as a float
            # under any real YAML reader (Obsidian, yaml.safe_load), losing
            # the trailing zero / precision. Keep it a string.
            fm_lines.append(f'arxiv_id: "{arxiv_id}"')
    if venue:
        fm_lines.append(f"venue: {venue}")
    fm_lines.extend([
        "type: paper",
        f"category: [{category}]",
        # `pdf_path` is an Obsidian wikilink to the source PDF so the property
        # renders as a click-to-open link (the vault root holds wiki/ + papers/
        # side by side, so `[[stem.pdf]]` resolves and opens in Obsidian's PDF
        # viewer). Must be quoted — an unquoted leading `[[` parses as a YAML
        # flow sequence. `db rebuild` derives the real filesystem path from the
        # stem; the basename lives in this wikilink, so no separate
        # `pdf_filename` field is needed.
        f'pdf_path: "[[{stem}.pdf]]"',
        "source_collection: external",
    ])
    if category_strength == "weak":
        fm_lines.append("category_suggestion_strength: weak  # first-of-kind — review")
    if pub_status:
        fm_lines.append(f"publication_status: {pub_status}")
    if short_name and short_name != "TODO":
        fm_lines.append(f"short_name: {short_name}")
    # `hook:` is always double-quoted — hooks routinely contain `[[wikilinks]]`
    # (which YAML reads as a nested flow sequence when bare) and `:`. Omitted
    # entirely when the author gave us nothing, so `lint`'s `missing_hook` picks
    # the page up rather than us committing an empty field.
    if hook:
        fm_lines.append(f"hook: {_yaml_dq(hook)}")
    # Keywords — render only when we got enough quality items;
    # `render_keywords_yaml` returns None below the floor and we'd rather
    # write no field than a half-list lint would flag immediately.
    if keywords:
        from .phases import render_keywords_yaml
        kw_line = render_keywords_yaml(keywords)
        if kw_line:
            fm_lines.append(kw_line)
    if author_model:
        # Quote the model id to keep YAML parsers from interpreting hyphens
        # / dots as numerics. e.g. `claude-sonnet-4-6` is a string, not a
        # subtraction; `claude-opus-4.7` would otherwise round-trip wrong.
        fm_lines.append(f'author_model: "{author_model}"')
    # `ingested_at` is the ingest event timestamp. Stamped at every commit —
    # first ingest AND re-ingest both update it, so a re-ingest properly
    # bubbles the page back to the top of "Recent additions" (re-ingest is a
    # meaningful edit; sed sweeps and other in-place rewrites are not).
    # Carrying the timestamp in YAML rather than relying on `file.cday`
    # makes it durable across `sed -i ''` rewrites that reset macOS btime.
    # Format: ISO 8601 local time, second precision.
    from datetime import datetime as _dt_for_ingest_stamp
    fm_lines.append(
        f"ingested_at: {_dt_for_ingest_stamp.now().isoformat(timespec='seconds')}"
    )
    fm_lines.extend([
        "tags: [ingested-via-agent]",
        "---",
        "",
    ])
    return "\n".join(fm_lines) + body


def _move_pdf(
    pdf_path: Path,
    stem: str,
    *,
    new_doi: str | None = None,
) -> tuple[Path, dict | None]:
    """If the PDF is in inbox/, mv it to papers/{stem}.pdf. If it's already in
    papers/ at any name, rename it to the stem name. Otherwise (external path)
    copy it into papers/.

    On collision (target exists with a different file body), classify as
    preprint→journal upgrade vs duplicate via DOI comparison and act:

    - `journal-upgrade` — proceed with the swap (the explicit user intent
      when the journal version arrives after the preprint).
    - anything else — refuse, raise, surface a clear error so the agent
      run aborts before the wiki page is overwritten.

    Returns (target_path, upgrade_info). `upgrade_info` is a small dict
    `{old_doi, new_doi}` when a preprint→journal swap happened, else None.
    Caller is responsible for emitting an `ingest_iterations` row.
    """
    from ..wiki import _doi_from_existing_page, classify_pdf_collision, find_stem_collision

    target = papers_dir() / f"{stem}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)

    if pdf_path.resolve() == target.resolve():
        return target, None

    upgrade_info: dict | None = None
    if target.exists():
        existing_page = find_stem_collision(stem)
        if existing_page is not None:
            verdict = classify_pdf_collision(new_doi, existing_page)
            if verdict != "journal-upgrade":
                raise RuntimeError(
                    f"papers/{stem}.pdf already exists and incoming PDF is not a "
                    f"journal-version upgrade (verdict={verdict}). Refusing to "
                    f"overwrite. Existing page: {existing_page}. To force, delete "
                    f"the existing PDF + page first."
                )
            upgrade_info = {
                "old_doi": _doi_from_existing_page(existing_page),
                "new_doi": new_doi,
            }

    if pdf_path.parent.resolve() == inbox_dir().resolve():
        shutil.move(str(pdf_path), str(target))
    elif pdf_path.parent.resolve() == papers_dir().resolve():
        # Already in papers/ but at a different filename — rename in place.
        shutil.move(str(pdf_path), str(target))
    else:
        # External — never mv someone's file out of place; copy.
        shutil.copy2(str(pdf_path), str(target))
    return target, upgrade_info


# The back-link bullet states a relationship, so it has to match how the
# candidate was found. Keyed by `CrosslinkCandidate.kind`; the bullet is written
# on the TARGET page and points back at the newly-ingested source paper.
#
#   cited_by_source — source cites the target → on the target: "cites this paper"
#   cites_source    — target cites the source → the same wording would be
#                     BACKWARDS, so it reverses
#   topical         — semantic-KNN + LLM judgement, no citation evidence either
#                     way. Claiming a citation here is a fabrication: CLAUDE.md
#                     requires a wikilink be source-supported, and "cites this
#                     paper" asserts exactly the support that was never checked.
# Sourced from `backlinks` rather than spelled out here: `lint --fix` reads
# these phrasings back off the page to recover an edge's direction, so a
# second copy that drifted would break mirroring silently.
_BACKLINK_NOTES = {
    "cited_by_source": _bl.CITES_NOTE,
    "cites_source": _bl.CITED_BY_NOTE,
    "topical": _bl.TOPICAL_NOTE,
}
_BACKLINK_NOTE_DEFAULT = _BACKLINK_NOTES["topical"]


def _append_backlinks(
    candidates: list,
    source_category: str,
    source_stem: str,
) -> tuple[list[str], list[str]]:
    """Append a back-link to each verified candidate's Related Papers section.

    Uses the shared `backlinks.append_related_paper` helper for consistent
    on-disk convention with `lint --fix`. The bullet's note comes from the
    candidate's `kind` (see `_BACKLINK_NOTES`) — an unknown kind falls back to
    the topical phrasing, which asserts the least.

    Returns (added, skipped) — `skipped` covers both already-linked targets
    and missing target files.
    """
    from ..backlinks import append_related_paper

    source_key = f"{source_category}/{source_stem}"
    added: list[str] = []
    skipped: list[str] = []
    for cand in candidates:
        target_path = wiki_dir() / f"{cand.wikilink}.md"
        note = _BACKLINK_NOTES.get(
            getattr(cand, "kind", ""), _BACKLINK_NOTE_DEFAULT
        )
        if append_related_paper(target_path, source_key, note=note):
            added.append(cand.wikilink)
        else:
            skipped.append(cand.wikilink)
    return added, skipped


def _append_index_entry(
    stem: str,
    category: str,
    title: str,
    year: int | str | None,
    venue: str | None,
    hook: str,
    short_name: str | None = None,
) -> bool:
    """Append `- [[category/stem]] — **{short_name}** (...): {hook}` under the
    `## {category}` section in wiki/index.md. Returns True if updated, False
    if the catalog doesn't exist or the section is missing."""
    from ..paths import index_path as _index_path_fn
    from ..fsatomic import update_locked
    index_path = _index_path_fn()
    if not index_path.exists():
        return False

    venue_clean = venue.lstrip('*').rstrip('*') if venue else ""
    year_str = f"{year}" if year else ""
    parts = [year_str, venue_clean]
    citation_paren = ", ".join(p for p in parts if p)
    citation_paren = f" ({citation_paren})" if citation_paren else ""
    handle = short_name if short_name and short_name != "TODO" else "**TODO short-name**"
    handle_md = f"**{handle}**" if not handle.startswith("**") else handle
    title_str = title or stem
    # The gloss is the page's own `hook:` — the same string, verbatim, so the
    # bullet and the YAML can't drift. The old path sliced sentence 1 of the
    # Summary to 200 chars, which stated the paper's question and cut mid-word on
    # 97 of 244 entries. When the author gave us no hook we say so plainly rather
    # than substituting a Summary slice, and `lint`'s `missing_hook` queues it.
    if hook:
        gloss = hook
    else:
        gloss = "_(no hook — set `hook:` on the page)_"
    line = (
        f"- [[{category}/{stem}]] — {handle_md} — *{title_str}*{citation_paren}: "
        f"{gloss}"
    )

    def _splice(text: str) -> str:
        # index.md is written by every concurrent ingest subprocess; the flock
        # in update_locked serializes this read-modify-write so entries can't
        # clobber each other, and the atomic replace prevents truncation.
        # Idempotent: a re-ingest of the same stem must not append a second
        # bullet — return the text unchanged (update_locked treats that as a
        # no-op) when this paper already has its own entry. Anchored to the
        # leading-bullet form so an incidental mention in another entry's hook
        # doesn't falsely suppress this paper's bullet.
        if re.search(rf"^- \[\[{re.escape(category)}/{re.escape(stem)}\]\]",
                     text, re.MULTILINE):
            return text
        pat = re.compile(rf"^## {re.escape(category)}\s*$", re.MULTILINE)
        m = pat.search(text)
        if not m:
            # Fall back to appending under a new section at end.
            return text.rstrip() + f"\n\n## {category}\n\n{line}\n"
        # Find next ## heading or EOF.
        start = m.end()
        next_m = re.search(r"^##\s+", text[start:], re.MULTILINE)
        end = start + next_m.start() if next_m else len(text)
        # Insert at end of section, before any trailing blank lines.
        section = text[start:end]
        section_trimmed = section.rstrip()
        new_section = section_trimmed + "\n" + line + "\n"
        if next_m:
            new_section += "\n"
        return text[:start] + new_section + text[end:]

    return update_locked(index_path, _splice, missing_ok=False)


def _append_log_entry(
    stem: str,
    metadata: dict,
    category: str,
    n_outgoing: int,
    n_backlinks: int,
    attempt_id: str,
    gates_passed: bool,
    gate_reasons: list[str] | None = None,
) -> bool:
    """Append a parseable-prefix log entry. Returns True if appended."""
    from ..paths import log_path as _log_path_fn
    log_path = _log_path_fn()
    today = _dt.date.today().isoformat()
    short_title = (metadata.get("title") or stem)
    if len(short_title) > 90:
        short_title = short_title[:90] + "…"
    short_id = attempt_id[:8]

    promote_status = "auto-promoted" if gates_passed else "sandbox-only"
    lines = [
        "",
        f"## [{today}] ingest | {short_title}",
        (
            f"Category: {category}. DOI: {metadata.get('doi') or 'none'}. "
            f"Cross-links: {n_outgoing} outgoing verified, {n_backlinks} back-links added. "
            f"Agent attempt {short_id} ({promote_status})."
        ),
    ]
    if not gates_passed and gate_reasons:
        lines.append(f"Gates failed: {'; '.join(gate_reasons)}.")
    body = "\n".join(lines) + "\n"

    try:
        with log_path.open("a") as f:
            f.write(body)
        return True
    except Exception:
        return False


def promote_to_wiki(
    *,
    stem: str,
    draft_text: str,
    metadata: dict,
    candidates: list,
    source_pdf_path: Path,
    attempt_id: str,
    short_name: str | None = None,
    hook: str | None = None,
    keywords: list[str] | None = None,
    author_model: str | None = None,
) -> PromotionResult:
    """Move PDF + write wiki page + add back-links + update index/log."""
    res = PromotionResult(promoted=True)

    # Category resolution.
    summary_text = _extract_section(draft_text, "summary")
    cat_suggestion, cat_strength = _suggest_category(
        metadata.get("title") or "", summary_text
    )
    # Abstention fallback. `other` is the structured "uncategorized backlog"
    # bucket; `suggest-splits` proposes promotions when it grows.
    category = cat_suggestion or "other"
    res.category = category

    # Write the wiki page BEFORE moving the PDF. The frontmatter's pdf_path is
    # computed from the stem (papers/{stem}.pdf), not from the move, so the page
    # can be written first — and doing so keeps the source PDF in inbox/ until
    # the page is durably committed. A failure here therefore leaves the paper
    # fully re-ingestable (batch --resume re-runs the still-present inbox PDF)
    # instead of stranding a moved PDF with no page.
    target_dir = wiki_dir() / category
    target_dir.mkdir(parents=True, exist_ok=True)
    page_path = target_dir / f"{stem}.md"
    full_page = _build_frontmatter(
        metadata, stem, category, draft_text,
        short_name=short_name,
        hook=hook,
        category_strength=cat_strength,
        keywords=keywords,
        author_model=author_model,
    )
    write_text_atomic(page_path, full_page)
    from ..wiki import commit_page as _commit_page
    _commit_page(page_path)
    res.wiki_path = page_path

    # Move/copy PDF only after the page is committed.
    try:
        res.pdf_path, res.pdf_upgrade = _move_pdf(
            source_pdf_path, stem, new_doi=metadata.get("doi"),
        )
    except Exception as e:
        res.warnings.append(f"PDF move/copy failed: {e}")
        res.promoted = False
        return res

    # Back-links.
    added, skipped = _append_backlinks(
        candidates=candidates,
        source_category=category,
        source_stem=stem,
    )
    res.backlinks_added = added
    res.backlinks_skipped = skipped

    # index.md — same `hook` string that went into the page's YAML, so the
    # bullet and the frontmatter cannot drift.
    if _append_index_entry(
        stem=stem,
        category=category,
        title=metadata.get("title") or stem,
        year=metadata.get("year"),
        venue=metadata.get("venue"),
        hook=hook or "",
        short_name=short_name,
    ):
        res.index_updated = True
    else:
        res.warnings.append("index.md not updated (file missing or category section absent)")

    # log.md (always — needs no preexisting content).
    res.log_appended = _append_log_entry(
        stem=stem,
        metadata=metadata,
        category=category,
        n_outgoing=len(candidates),
        n_backlinks=len(added),
        attempt_id=attempt_id,
        gates_passed=True,
    )

    return res
