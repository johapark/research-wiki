"""Per-item verdicts: is this record importable, and if not, why not.

Three verdicts, `ready | review | skip`, with the detail in `reasons`. Three
rather than five because every extra state is another branch to get wrong, and
the reason strings carry more information than a finer verdict vocabulary would.

`apply` acts on `ready` only. `review` means a human should look — never a
silent best guess. `skip` means there is nothing to do with this record.

The gates and their frequencies come from a real 532-item ReadCube library; the
counts in the docstrings are what that library actually produced, not estimates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..stems import derive_stem, strip_diacritics
from .pair import (
    TITLE_ACCEPT,
    TITLE_MARGIN,
    Pairing,
    PdfFacts,
    find_duplicate_doi_losers,
)
from .parse import ExportItem

READY, REVIEW, SKIP = "ready", "review", "skip"

#: Below this many characters of extractable text per page, a PDF has no usable
#: text layer. Real papers land at 2000-4000; a scan with no OCR lands near 0.
#: 200 sits in the empty middle, so the threshold is not a close call.
MIN_CHARS_PER_PAGE = 200

#: Item types that are not papers. Authoritative only where the exporter
#: populates the field — Zotero does; ReadCube typed 531 of 532 records
#: `JOUR`, two actual books included.
NON_PAPER_TYPES = frozenset({"book", "chapter", "thesis", "report", "webpage"})

#: Nature's news, views and comment DOI prefix. A free, precise commentary
#: signal: no network call, no heuristic on the PDF.
_COMMENTARY_DOI_PREFIXES = ("10.1038/d41586-",)

_PUNCT = re.compile(r"[^a-z0-9]+")


def _norm_title(title: str | None) -> str:
    """Title reduced to a comparison key for `find_superseded`.

    Folded through `strip_diacritics` before lowering, matching `pair._tokens`.
    Without the fold, `[^a-z0-9]+` *deletes* every non-ASCII letter: `Grünewald`
    becomes `gr newald` while the same paper spelled `Grunewald` becomes
    `grunewald`, so the preprint and its published version land in different
    buckets and the supersede gate never sees the pair — the one gate that finds
    them, since such pairs carry two different DOIs.

    Fold *before* lower, because `_TRANSLITERATE` maps uppercase forms. The fold
    also turns Unicode dashes into ASCII `-`, which `[^a-z0-9]+` then maps to a
    space, preserving the word boundary instead of welding two words together.
    """
    return _PUNCT.sub(" ", strip_diacritics(title or "").lower()).strip()


@dataclass
class ItemAssessment:
    item: ExportItem
    pairing: Pairing
    verdict: str = READY
    reasons: list[str] = field(default_factory=list)
    derived_stem: str | None = None
    chars_per_page: float | None = None
    page_count: int | None = None
    collision: dict | None = None
    ingest_args: list[str] = field(default_factory=list)

    def _flag(self, verdict: str, reason: str) -> None:
        """Record a reason and keep the most severe verdict seen.

        Severity ordering matters: an item that is both `already-present` and
        `weak-pairing` is a skip, not a review, and the order gates run in
        should not decide that.
        """
        order = {READY: 0, REVIEW: 1, SKIP: 2}
        if reason not in self.reasons:
            self.reasons.append(reason)
        if order[verdict] > order[self.verdict]:
            self.verdict = verdict

    def as_dict(self) -> dict:
        return {
            "key": self.item.key,
            "title": self.item.title,
            "doi": self.item.doi,
            "year": self.item.year,
            "authors": self.item.authors,
            "item_type": self.item.item_type,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "derived_stem": self.derived_stem,
            "primary_pdf": str(self.pairing.primary) if self.pairing.primary else None,
            "supplementary": [str(p) for p in self.pairing.supplementary],
            "pair_rung": self.pairing.rung,
            "pair_confidence": self.pairing.confidence,
            "pair_rival": self.pairing.rival,
            "pair_margin": self.pairing.margin,
            "pair_candidates": [[str(p), s] for p, s in self.pairing.candidates],
            "chars_per_page": (round(self.chars_per_page, 1)
                               if self.chars_per_page is not None else None),
            "page_count": self.page_count,
            "collision": self.collision,
            "ingest_args": self.ingest_args,
        }


def build_ingest_args(item: ExportItem, pairing: Pairing) -> list[str]:
    """The `agent ingest` overrides this record contributes.

    Venue is deliberately absent: there is no override flag for it, and
    reconcile derives it from the DOI lookup, which is more trustworthy than a
    reference manager's often-abbreviated journal field.
    """
    args: list[str] = []
    if item.has_usable_doi:
        args += ["--doi", item.doi]
    if item.title:
        args += ["--title", item.title]
    if item.authors:
        args += ["--authors", "; ".join(item.authors)]
    if item.year:
        args += ["--year", str(item.year)]
    for supp in pairing.supplementary:
        args += ["--supplementary", str(supp)]
    return args


def find_superseded(items: list[ExportItem]) -> set[int]:
    """ids() of records superseded by a published version of the same title.

    The highest-value gate in the whole feature, and invisible to DOI-level
    dedupe: the real library had **10 such pairs and zero duplicate DOIs**, so
    nothing but title comparison finds them. Every pair was a bioRxiv record
    alongside its journal publication.

    The survivor is chosen deliberately rather than by file order — the two
    exports of the same library listed the pairs in different orders, so
    "last one wins" would have produced different imports from the same data.
    """
    by_title: dict[str, list[ExportItem]] = {}
    for item in items:
        key = _norm_title(item.title)
        if key:
            by_title.setdefault(key, []).append(item)

    superseded: set[int] = set()
    for group in by_title.values():
        if len(group) < 2:
            continue
        published = [i for i in group if not i.is_preprint_doi and i.has_usable_doi]
        if not published:
            continue
        for i in group:
            if i.is_preprint_doi or (i is not published[0] and not i.has_usable_doi):
                superseded.add(id(i))
    return superseded


def assess_all(items: list[ExportItem], pairings: list[Pairing],
               facts_by_path: dict[Path, PdfFacts], *,
               known_dois: dict[str, str] | None = None,
               stem_exists=None,
               superseded: set[int] | None = None) -> list[ItemAssessment]:
    """Run every gate over every record.

    `known_dois` maps a lowercase DOI to the wiki stem carrying it, and
    `stem_exists(stem)` reports whether a page already uses that stem. Both are
    injected rather than read here so this module stays free of wiki state and
    testable without a repo on disk.
    """
    known_dois = {k.lower(): v for k, v in (known_dois or {}).items()}
    superseded = find_superseded(items) if superseded is None else superseded
    duplicate_losers = find_duplicate_doi_losers(
        [item for item in items if id(item) not in superseded]
    )
    by_item = {id(p.item): p for p in pairings}
    out: list[ItemAssessment] = []

    for item in items:
        pairing = by_item.get(id(item)) or Pairing(item=item)
        a = ItemAssessment(item=item, pairing=pairing)

        if item.title and item.year and item.authors:
            a.derived_stem = derive_stem(item.authors, item.year, item.title)

        facts = facts_by_path.get(pairing.primary) if pairing.primary else None
        if facts is not None:
            a.page_count = facts.page_count
            a.chars_per_page = facts.chars_per_page

        # --- identity and metadata ---
        if id(item) in superseded:
            a._flag(SKIP, "superseded-by-journal")

        duplicate_survivor = duplicate_losers.get(id(item))
        if duplicate_survivor is not None:
            a.collision = {"kind": "duplicate-doi", "key": duplicate_survivor.key}
            a._flag(SKIP, "duplicate-doi")

        if item.item_type in NON_PAPER_TYPES:
            a._flag(SKIP, "not-a-paper")

        if not item.has_usable_doi and not item.authors and not item.year:
            # The empirical fallback for exporters whose type field is useless.
            # 5 of 532 real records, including a Django book and a Rust book
            # that the exporter typed as journal articles.
            a._flag(REVIEW, "unresolvable")
        elif not item.has_usable_doi and (not item.title or not item.year):
            a._flag(REVIEW, "thin-metadata")

        if item.doi and any(item.doi.startswith(p) for p in _COMMENTARY_DOI_PREFIXES):
            a._flag(REVIEW, "maybe-commentary")

        # --- already in the wiki ---
        if item.has_usable_doi and item.doi in known_dois:
            a.collision = {"kind": "doi", "stem": known_dois[item.doi]}
            a._flag(SKIP, "already-present")
        elif a.derived_stem and stem_exists and stem_exists(a.derived_stem):
            a.collision = {"kind": "stem", "stem": a.derived_stem}
            a._flag(SKIP, "already-present")

        # --- the PDF ---
        # A superseded record is not being imported, so whether it has a file
        # is not a finding. Reporting `no-pdf` for it would also inflate the
        # fetch-list count with versions the user deliberately isn't importing.
        if id(item) in superseded or duplicate_survivor is not None:
            pass
        elif pairing.primary is None:
            a._flag(SKIP, "no-pdf")
        else:
            if facts is None or facts.page_count is None:
                a._flag(SKIP, "pdf-unreadable")
            else:
                if (a.chars_per_page or 0) < MIN_CHARS_PER_PAGE:
                    a._flag(SKIP, "no-text-layer")
                if facts.page_count == 1 and not item.has_usable_doi:
                    a._flag(REVIEW, "maybe-commentary")
            if pairing.rung == "title":
                if pairing.confidence < TITLE_ACCEPT:
                    a._flag(REVIEW, "weak-pairing")
                elif pairing.margin < TITLE_MARGIN:
                    # Confident-looking but not distinctive: another record
                    # scored nearly as well against this same PDF, so the score
                    # came from shared vocabulary rather than from identity.
                    # On 313 DOI-confirmed pairs this gate is the difference
                    # between 6 silently wrong pairings and none.
                    a._flag(REVIEW, "ambiguous-pairing")

        a.ingest_args = build_ingest_args(item, pairing)
        out.append(a)

    _flag_stem_collisions(out)
    return out


def _flag_stem_collisions(assessments: list[ItemAssessment]) -> None:
    """Two importable records deriving one stem.

    Left for a human: CLAUDE.md's BibTeX-letter rule turns on which paper keeps
    the bare year, and that is a judgement about the corpus, not something to
    settle by iteration order.
    """
    seen: dict[str, list[ItemAssessment]] = {}
    for a in assessments:
        if a.verdict == READY and a.derived_stem:
            seen.setdefault(a.derived_stem, []).append(a)
    for group in seen.values():
        if len(group) > 1:
            for a in group:
                a._flag(REVIEW, "stem-collision")


def summarize(assessments: list[ItemAssessment]) -> dict:
    """Counts for the report and for `--json`."""
    verdicts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for a in assessments:
        verdicts[a.verdict] = verdicts.get(a.verdict, 0) + 1
        for r in a.reasons:
            reasons[r] = reasons.get(r, 0) + 1
    return {
        "total": len(assessments),
        "verdicts": verdicts,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def reference_doc_candidates(assessments: list[ItemAssessment]) -> list[dict]:
    """Records skipped as non-papers, which are `wiki/references/` material.

    Books, guidance documents, theses and reports are legitimate wiki pages —
    they are just hand-written ones (CLAUDE.md → Page Types §3), not ingested.
    Without this the run reports `not-a-paper: 20` and discards which twenty,
    leaving the user a count and a dead end. Same reasoning as the missing-PDF
    fetch list: don't throw away information the reader needs to act.

    Only fires where the exporter populates `type`. Zotero does; ReadCube typed
    531 of 532 records as journal articles, two actual books included — those
    surface under `unresolvable` instead, which is a review item.
    """
    out = []
    for a in assessments:
        if "not-a-paper" not in a.reasons:
            continue
        out.append({
            "title": a.item.title, "item_type": a.item.item_type,
            "doi": a.item.doi, "year": a.item.year,
            "authors": a.item.authors, "key": a.item.key,
            "primary_pdf": str(a.pairing.primary) if a.pairing.primary else None,
        })
    return out


def missing_pdf_fetch_list(assessments: list[ItemAssessment]) -> list[dict]:
    """Records that clear every gate except having a file.

    On a cloud-hosted library this is the most useful artifact the whole command
    produces: without it, a metadata-only run reports "483 skipped" and nothing
    actionable. Emitted as plain DOIs so the list can be piped.

    Deduplicated by DOI as a final guard so a metadata-only export never asks
    the user to fetch the same paper twice.
    """
    out = []
    seen: set[str] = set()
    for a in assessments:
        if a.reasons == ["no-pdf"] and a.item.has_usable_doi:
            if a.item.doi in seen:
                continue
            seen.add(a.item.doi)
            out.append({"doi": a.item.doi, "title": a.item.title,
                        "year": a.item.year, "key": a.item.key})
    return out
