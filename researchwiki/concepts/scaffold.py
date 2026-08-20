"""Concept-hub scaffolding + post-ingest attachment.

Two public entry points:

  - `run(term, ...)`             — scaffold a new `wiki/concepts/{slug}.md`
                                    from a recurring term, gather member
                                    papers, add reciprocal `[[concepts/…]]`
                                    back-links.
  - `attach_after_ingest(stem)`  — post-ingest hook: for every existing hub
                                    whose term appears in the new paper's
                                    contribution claims, add a spoke bullet
                                    and reciprocal back-link.

Both share the term↔claim substrate in `term_claims`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from pathlib import Path

from ..backlinks import append_related_paper
from ..log import append_log_md, log
from ..paths import wiki_dir
from ..search import format_claim_ref
from ..fsatomic import update_locked, write_text_atomic
from ..wiki import commit_page, find_stem_collision, read_page, read_pages, strip_non_prose
# Reuse the synthesis scaffolder's slug + dominant-category helpers.
from ..tasks.synthesize import _dominant_category, _slugify
from .semantic_members import semantic_member_candidates, suggested_alias_set
from .term_claims import (
    _matching_claims,
    _page_mentions,
    _papers_where_keywords_match,
    _promote_instantiates_edges_for,
    _term_claim_hint,
    _top_kc_claim_slug,
)

_NOTE = "instantiates this concept (auto-added; concept-link)"

def find_members(
    term: str, aliases: list[str] | None = None,
) -> list[tuple[str, str, str | None, str | None]]:
    """Return [(key, category, best_slug, matched_term)] of paper pages that
    instantiate `term` as a contribution.

    `matched_term` is which of `term`/`aliases` actually found the paper, or
    None for a keyword-only member. Aliases widen membership without saying
    what they cost — five of them once took a hub from 5 members to 17 — so
    the caller aggregates this into `alias_hits` for the author to read. It is
    not part of the `--json` contract: `result["members"]` stays a list of keys.

    Three signals feed the member set:

      1. **LLM keywords / tags on the paper page** (primary). The ingest LLM
         already normalized the vocabulary — a paper that lists "DMS" or
         "saturation mutagenesis" among its keywords instantiates the same
         concept as one that spells out "deep mutational scanning" in a
         claim. Using the LLM's normalization for matching closes the
         acronym/synonym gap that pure substring matching misses.
      2. **Caller-supplied `aliases`**. For concepts whose vocabulary
         diverges across papers (DMS ↔ "saturation mutagenesis" ↔ "MAVE"),
         the hub author supplies the alias list and every check runs against
         `term + aliases`. Persist these to the hub YAML as
         `topic_seed_aliases:` so downstream hooks (attach, refresh) see them.
      3. **Claim-substrate substring match** (recall fallback). Papers
         without the term (or an alias) in their keywords but with a
         contribution claim mentioning any of them still count — an author
         may have added the concept to a paper page before the keywords
         caught up.

    `best_slug` is the highest-priority contribution-claim slug for the
    paper, in this order:
      - a claim whose text contains `term` (or any alias) verbatim;
      - a claim whose text contains one of the paper's own matching
        keywords (LLM-normalized alias);
      - `None` when a keyword-only member has no claim matching any of them.
        `format_claim_ref` renders that as a bare `[[stem]]`. There is
        deliberately no last-resort anchor: falling back to the paper's first
        key_contributions claim produced spokes citing a claim about something
        else (seqLens cited as "Introduced seqLens, a DeBERTa-v2 based gLM
        family…" on a parameter-efficient-fine-tuning hub), and an unrelated
        anchor is invisible where a bare one asks the author to fix it.
        `attach_after_ingest` still has that fallback, because there the anchor
        doubles as the membership test — separating them is a change to what
        auto-attaches at ingest, not to how a spoke is cited.

    Papers that mention the term only in body prose (intro / limitations /
    discussion) do NOT become members — a mention isn't an instantiation.
    Sorted by category then key. Only `type: paper` pages are spoke-eligible.
    """
    alias_list = list(aliases or [])
    search_terms: list[str] = [term, *alias_list]
    keyword_hits = _papers_where_keywords_match(term, aliases=alias_list)

    members: list[tuple[str, str, str | None, str | None]] = []
    for p in read_pages():
        if p.page_type != "paper":
            continue

        # Step 1: direct claim-substring match on the term OR any alias.
        best_slug: str | None = None
        matched_term: str | None = None
        for t in search_terms:
            hits = _matching_claims(p.stem, t)
            if hits:
                best_slug = hits[0]["claim_slug"]
                matched_term = t
                break

        # Step 2: keyword-hit paper without a direct claim match — widen
        # the claim search to the paper's own matching keywords (each is a
        # further alias the LLM already normalized).
        paper_aliases = keyword_hits.get(p.stem)
        if best_slug is None and paper_aliases:
            for alias in paper_aliases:
                alias_hits = _matching_claims(p.stem, alias)
                if alias_hits:
                    best_slug = alias_hits[0]["claim_slug"]
                    matched_term = alias
                    break

        # Not a member if no signal fired.
        if best_slug is None and p.stem not in keyword_hits:
            continue

        # A keyword-only member keeps `best_slug is None` on purpose, which
        # `format_claim_ref` renders as a bare `[[stem]]`. Anchoring to the
        # paper's top key_contribution instead used to produce spokes citing a
        # claim about something else entirely — seqLens cited as "Introduced
        # seqLens, a DeBERTa-v2 based gLM family…" on a hub about
        # parameter-efficient fine-tuning. Bare is the form CLAUDE.md prescribes
        # for citing a paper as a whole, it announces itself to the author, and
        # `concepts --upgrade-spokes` fills it in once a matching claim exists.
        members.append((p.key, p.category, best_slug, matched_term))
    members.sort(key=lambda kc: (kc[1], kc[0]))
    return members

def _template(
    title: str, slug: str, term: str,
    members: list[tuple[str, str, str | None, str | None]], span: int,
    thesis: str, aliases: list[str] | None = None,
) -> str:
    today = date.today().isoformat()
    category = _dominant_category([k for k, _, _, _ in members])
    cat_line = (
        "category: [TODO]  # dominant content field of the member papers; set "
        "to a valid content category (type is carried by type:/the concepts/ dir)"
        if category == "TODO" else f"category: [{category}]"
    )
    # Quote each wikilink so Obsidian types the list as links (unquoted
    # `- [[..]]` parses as a nested list → "?" in the Properties panel).
    ref_lines = "\n".join(f'  - "[[{k}]]"' for k, _, _, _ in members)

    # concept_thesis: the one-sentence discriminator between concept vs
    # glossary/synthesis, collected at scaffold time. Rendered as a blockquote
    # under the H1 so a reviewer sees the intent before the Definition.
    #
    # Use YAML block-scalar form so multi-line theses keep intact — the "|"
    # style preserves newlines, and the indented body is standard YAML.
    thesis_stripped = thesis.strip().replace("\r\n", "\n")
    thesis_indented = "\n".join("  " + line for line in thesis_stripped.split("\n"))

    yaml = [
        f'title: "{title}"',
        "type: concept",
        cat_line,
        "referenced_papers:",
        ref_lines,
        "concept_thesis: |",
        thesis_indented,
        f"concept_span: {span}",
        f"generated_at: {today}",
        f'topic_seed: "{term.replace(chr(34), chr(39))}"',
    ]
    # topic_seed_aliases: only emitted when non-empty. Downstream hooks
    # (find_members on refresh, attach_after_ingest) read this list to
    # expand the vocabulary they search across paper claims and keywords.
    alias_list = [a for a in (aliases or []) if a and a.strip()]
    if alias_list:
        yaml.append("topic_seed_aliases:")
        for a in alias_list:
            yaml.append(f'  - "{a.replace(chr(34), chr(39))}"')
    yaml.append(f"tags: [concept, {slug}]")

    # Spoke list: group under H3-per-category when the term bridges domains
    # (span ≥ 2) so the cross-domain reach is visible; flat otherwise.
    #
    # Each spoke cites the specific claim that instantiates the concept via
    # `[[stem#slug]]` when a matching contribution claim was found; falls
    # back to bare `[[stem]]` when best_slug is None. `referenced_papers:` in
    # the frontmatter stays bare — that field enumerates whole-paper members,
    # not per-claim anchors.
    by_cat: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for k, c, best_slug, _matched in members:
        by_cat[c].append((k, best_slug))
    spoke_lines: list[str] = []
    for cat in sorted(by_cat):
        if span >= 2:
            spoke_lines.append(f"### {cat}")
        for k, best_slug in by_cat[cat]:
            stem = k.split("/", 1)[1] if "/" in k else k
            hint = _term_claim_hint(stem, term, aliases)
            hint_c = f' — <!-- how this paper uses {term}. hint: "{hint}" -->' if hint \
                else f" — <!-- how this paper uses {term} -->"
            cite = format_claim_ref({"paper_stem": k, "claim_slug": best_slug})
            spoke_lines.append(f"- {cite}{hint_c}")
        if span >= 2:
            spoke_lines.append("")
    spokes = "\n".join(spoke_lines).rstrip()

    crossdomain = (
        "## Cross-domain connections\n"
        f"<!-- {term} spans {span} categories. How does it manifest DIFFERENTLY "
        "across them? Strict-grounded — cite the contrast, or delete this section. -->\n\n"
        if span >= 2 else ""
    )

    # The thesis is authoritative in YAML `concept_thesis:` — Dataview and
    # `views.md` surface it from there. We used to also render it as a visible
    # blockquote under the H1 wrapped in `<!-- skip-grounding-start/end -->`
    # markers, but Obsidian's Live Preview + Source modes render HTML comments
    # as literal text, defeating the point. Keep the thesis YAML-only.

    body = (
        "## Definition\n"
        f"<!-- 1–3 sentences: what {term} is, as the member papers use it. "
        "Cite each sentence to a member via footnote [^id]. Strict-grounded. -->\n\n"
        "## How it appears across the corpus\n"
        f"{spokes}\n\n"
        f"{crossdomain}"
        "## What would update this page\n"
        f"<!-- ≤3 bullets: paper classes whose ingestion would extend/reframe {term}. -->\n"
    )
    return "---\n" + "\n".join(yaml) + "\n---\n\n" + body

def run(
    term: str, *, thesis: str, aliases: list[str] | None = None,
    title: str | None = None, slug: str | None = None,
    min_members: int = 3, force: bool = False, dry_run: bool = False,
) -> dict:
    """Core: discover members → write stub → apply reciprocal spoke links.

    `thesis` is the one-sentence "why is this a concept, not a glossary term
    or synthesis topic?" answer. Required — a hub without a thesis is what
    produced the retracted PAM/RNP/LNP/DSB glossary hubs. Empty thesis
    raises ValueError so `main()` can surface the failure. `--dry-run` still
    requires a thesis: the discipline is that if you can't write one, you
    shouldn't scaffold.

    Returns a decisions dict. Raises ValueError on user-input errors (empty
    slug, empty thesis, too few members, target exists without force) so
    `main` can map them to exit 1 with a message.
    """
    if not dry_run and (not thesis or not thesis.strip()):
        # Dry runs are exempt: the thesis test asks *why is this a concept and
        # not glossary*, which you can only answer once you have seen the member
        # list, so demanding it in order to look was circular. Nothing is
        # written and `_template` never runs, so there is no field to fill.
        raise ValueError(
            f"a `concept_thesis` is required to scaffold a hub. Provide one "
            f"sentence answering *why {term!r} is a concept* (an idea the corpus "
            f"disagrees about or elaborates on) rather than a *glossary* term or "
            f"a *synthesis* topic. Pass via `--thesis` or answer the interactive "
            f"prompt."
        )
    slug = slug or _slugify(term)
    if not slug:
        raise ValueError(f"could not derive slug from term `{term}`")
    title = title or term
    aliases_clean = [a.strip() for a in (aliases or []) if a and a.strip()]

    members = find_members(term, aliases=aliases_clean)
    span = len({c for _, c, _, _ in members})
    # Which search term found each member. Keyword-only members are attributed
    # to "(keywords)" so the counts always sum to len(members).
    alias_hits: dict[str, int] = {}
    for _, _, _, matched in members:
        key = matched or "(keywords)"
        alias_hits[key] = alias_hits.get(key, 0) + 1
    result = {
        "term": term, "slug": slug, "title": title,
        "members": [k for k, _, _, _ in members], "span": span,
        "alias_hits": alias_hits,
        "thesis": thesis.strip(), "aliases": aliases_clean,
        "dry_run": dry_run, "linked": [], "path": None,
    }

    # Semantic recall tier: report the members lexical matching missed. Never
    # added to `members` — see semantic_members.__doc__ for why cosine cannot
    # carry membership. The author converts a candidate by re-running with the
    # suggested alias, which routes it through `find_members` unchanged.
    member_stems = {k.split("/")[-1] for k, _, _, _ in members}
    candidates = semantic_member_candidates(
        term, aliases=aliases_clean, exclude_stems=member_stems,
    )
    result["semantic_candidates"] = [
        {
            "stem": c.stem, "category": c.category, "score": round(c.score, 3),
            "claim_slug": c.claim_slug, "section": c.section,
            "text": c.text, "suggested_alias": c.suggested_alias,
        }
        for c in candidates
    ]
    result["suggested_aliases"] = suggested_alias_set(candidates)
    result["semantic_span_gain"] = len(
        {c.category for c in candidates} - {c for _, c, _, _ in members}
    )

    if len(members) < min_members:
        hint = ""
        suggested = result.get("suggested_aliases") or []
        n_cand = len(result.get("semantic_candidates") or [])
        if suggested:
            # The common cause of a thin member list is vocabulary, not
            # absence: the corpus names the concept differently elsewhere.
            hint = (
                f" {n_cand} paper(s) match semantically but not lexically; "
                f"retry with --aliases \"{','.join(suggested)}\" if those are "
                f"the same concept."
            )
        elif n_cand:
            hint = f" {n_cand} paper(s) match semantically but not lexically."
        raise ValueError(
            f"`{term}` appears in {len(members)} paper(s) (<{min_members}). "
            f"Not concept-worthy — lower --min-members to override.{hint}"
        )

    out = wiki_dir() / "concepts" / f"{slug}.md"
    if out.exists() and not force:
        raise ValueError(f"page already exists: {out} (use --force to overwrite)")
    if (hit := find_stem_collision(slug)) is not None and hit != out:
        log(f"WARN: stem `{slug}` already used at {hit} (different dir)", tag="concepts")

    if dry_run:
        return result

    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out, _template(title, slug, term, members, span, thesis.strip(),
                             aliases=aliases_clean))
    commit_page(out)
    result["path"] = str(out)

    # Reciprocal half: link each member back to the hub. Idempotent — a member
    # already linking the concept is a no-op.
    concept_key = f"concepts/{slug}"
    pages = {p.key: p for p in read_pages()}
    for k, _, _, _ in members:
        page = pages.get(k)
        if page and append_related_paper(page.path, concept_key, note=_NOTE):
            result["linked"].append(k)

    # Promote any pre-scaffold `instantiates` edges from the detector to
    # `confirmed`. The scaffold is the review signal — the human decided this
    # term is hub-worthy, so every candidate edge pointing at it is endorsed
    # en bloc (this is the special case for `instantiates`: no per-edge
    # LLM review, unlike `contradicts` which goes through claim-graph review).
    promoted = _promote_instantiates_edges_for(term)
    result["instantiates_promoted"] = promoted

    append_log_md(
        "concept",
        f"concept: {title} (span {span}) → wiki/concepts/{slug}.md",
        f"Members: {', '.join(k.split('/', 1)[-1] for k, _, _, _ in members)}. "
        f"Reciprocal links added: {len(result['linked'])}/{len(members)}. "
        + (f"instantiates edges promoted: {promoted}." if promoted else ""),
    )
    return result

def _attach_to_concept(
    concept_path: Path, paper_key: str, term: str,
    best_slug: str | None = None,
) -> bool:
    """Add `paper_key` as a spoke to an existing concept page.

    Inserts a `referenced_papers` entry (bare stem — that field enumerates
    whole-paper members) and a spoke bullet in *How it appears across the
    corpus* citing `[[paper_key#best_slug]]` when a matching claim slug is
    known (falls back to bare `[[paper_key]]`). Recomputes `concept_span`
    from the member category prefixes and refreshes `generated_at`. The
    spoke carries an empty `how this paper uses …` comment for the author
    to fill on the next refresh. Idempotent (no-op if the paper is already
    linked). Returns True iff written.
    """
    def _splice(text: str) -> str:
        # attach_after_ingest runs per-paper inside each batch subprocess, so
        # two papers attaching to the same hub race on this page — the flock in
        # update_locked serializes them. Returning `text` unchanged signals a
        # no-op (already linked, or structurally not a concept page).
        if f"[[{paper_key}]]" in text or f"[[{paper_key}#" in text:
            return text
        m = re.match(r"(?s)^(---\n.*?\n---\n)(.*)$", text)
        if not m or "referenced_papers:" not in m.group(1):
            return text
        fm, body = m.group(1), m.group(2)
        today = date.today().isoformat()

        fm = re.sub(r"(referenced_papers:\n)", rf'\1  - "[[{paper_key}]]"\n', fm, count=1)
        span = len(set(re.findall(r'-\s*"?\[\[([^/\]]+)/', fm))) or 1
        fm = re.sub(r"concept_span:\s*\d+", f"concept_span: {span}", fm, count=1)
        fm = re.sub(r"generated_at:\s*\S+", f"generated_at: {today}", fm, count=1)

        cite = format_claim_ref({"paper_stem": paper_key, "claim_slug": best_slug})
        spoke = f"- {cite} — <!-- how this paper uses {term} (auto-added; concept-link) -->"
        sec = re.compile(r"(## How it appears across the corpus\n)(.*?)(?=\n## |\Z)", re.S)
        new_body, n = sec.subn(
            lambda mm: mm.group(1) + mm.group(2).rstrip() + "\n" + spoke + "\n", body, count=1
        )
        if n == 0:  # section missing (hand-edited page) — append one
            new_body = body.rstrip() + f"\n\n## How it appears across the corpus\n{spoke}\n"
        return fm + new_body

    changed = update_locked(concept_path, _splice, missing_ok=False)
    if changed:
        commit_page(concept_path)
    return changed

def attach_after_ingest(stem: str, committed_path) -> dict | None:
    """Post-ingest hook: attach a just-promoted paper to every existing concept
    hub whose term the paper *instantiates* (i.e., a claim in kc / results /
    methodology mentions the term). Body-prose-only mentions log at INFO but
    do NOT trigger attachment — a passing reference in an intro isn't the
    same as instantiating the concept.

    Uses the same three-signal member-detection logic as `find_members` (the
    scaffold-time builder), so a paper the scaffold would have picked up as a
    member is also picked up here after ingest:

      1. Direct claim-substring match on the hub's `topic_seed` OR any entry
         in `topic_seed_aliases:` (medical abbreviations, spelling variants —
         `FH` / `HeFH` / `familial hypercholesterolaemia` for the FH hub).
      2. Keyword-hit fallback via `_papers_where_keywords_match`: if the
         paper's LLM-authored keywords match the hub's vocabulary, use those
         keywords as further aliases to widen the claim search.
      3. Last-resort anchor: `_top_kc_claim_slug(paper.stem)` when the paper
         is a keyword-hit member but no alias appears in claim text either.

    Near-miss detection (`_page_mentions`) also runs across term + aliases so
    a British-spelling paper still logs a near-miss instead of silent skip.

    Mirrors `claim_overlap.run_after_ingest` — reads the committed page, never
    raises (a hiccup must not fail an otherwise-good ingest), and no-ops when
    there are no concept pages / the page wasn't promoted to wiki/.
    """
    try:
        paper = read_page(Path(committed_path))
        if paper is None or paper.page_type != "paper":
            return None
        cdir = wiki_dir() / "concepts"
        if not cdir.exists():
            return None
        prose = strip_non_prose(paper.body)
        attached: list[str] = []
        near_missed: list[str] = []
        for cp in sorted(cdir.glob("*.md")):
            cpage = read_page(cp)
            if cpage is None:
                continue
            term = str(cpage.fm.get("topic_seed") or cpage.fm.get("title") or "").strip().strip('"').strip("'")
            if not term:
                continue

            # Hub-supplied vocabulary variants (`FH`, `HeFH`, British
            # spelling). Same list `find_members` uses at scaffold time.
            raw_aliases = cpage.fm.get("topic_seed_aliases") or []
            aliases = [a.strip() for a in raw_aliases
                       if isinstance(a, str) and a.strip()]
            search_terms = [term, *aliases]

            # Step 1: direct claim-substring match on the term or any alias.
            best_slug: str | None = None
            for t in search_terms:
                hits = _matching_claims(paper.stem, t)
                if hits:
                    best_slug = hits[0]["claim_slug"]
                    break

            # Step 2: keyword-hit fallback. Widen the claim search to the
            # paper's own matching keywords (LLM-normalized synonyms). If
            # any of those hit a claim, anchor to that; else anchor to the
            # paper's top kc claim slug so we still get a `[[stem#slug]]`
            # citation.
            if best_slug is None:
                keyword_hits = _papers_where_keywords_match(term, aliases=aliases)
                paper_aliases = keyword_hits.get(paper.stem)
                if paper_aliases:
                    for alias in paper_aliases:
                        alias_hits = _matching_claims(paper.stem, alias)
                        if alias_hits:
                            best_slug = alias_hits[0]["claim_slug"]
                            break
                    if best_slug is None:
                        best_slug = _top_kc_claim_slug(paper.stem)

            if best_slug is None:
                # No claim / keyword signal. Near-miss: log if the term or
                # any alias appears in body prose so a British-spelling
                # paper still surfaces as reviewable.
                if any(_page_mentions(t, prose) for t in search_terms):
                    log(f"concept-attach: skipped {paper.stem}→{cp.stem}, "
                        f"term only in body prose (not in kc/results/methodology)",
                        tag="concepts")
                    near_missed.append(cp.stem)
                continue

            if _attach_to_concept(cp, paper.key, term, best_slug=best_slug):
                append_related_paper(paper.path, f"concepts/{cp.stem}", note=_NOTE)
                attached.append(cp.stem)
        if attached:
            print()
            print(f"concept-attach → {stem} joined {len(attached)} hub(s): {', '.join(attached)}")
            log(f"concept-attach {stem}: joined {', '.join(attached)}", tag="concepts")
        return {"stem": stem, "attached": attached, "near_missed": near_missed}
    except Exception as e:  # never let attachment break an otherwise-good ingest
        log(f"concept-attach hook skipped: {type(e).__name__}: {e}", tag="concepts")
        return None
