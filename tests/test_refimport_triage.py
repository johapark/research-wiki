"""Triage gates (`refimport.triage`).

One test per gate, asserting the **verdict and the reason string**, because the
reason is what a user reads and what `--json` consumers key on. Frequencies in
the docstrings are from a real 532-item ReadCube library.
"""

from pathlib import Path

from researchwiki.refimport.pair import Pairing, PdfFacts
from researchwiki.refimport.parse import ExportItem
from researchwiki.refimport.triage import (
    READY,
    REVIEW,
    SKIP,
    assess_all,
    build_ingest_args,
    missing_pdf_fetch_list,
    summarize,
)


def mk_item(**kw) -> ExportItem:
    base = dict(key="k1", item_type="article", title="A fine test paper about things",
                authors=["Ada Fixture"], year=2024, doi="10.1234/test.1")
    base.update(kw)
    return ExportItem(**base)


def mk_facts(path="/pdfs/a.pdf", pages=12, chars_per_page=3000, doi=None) -> PdfFacts:
    f = PdfFacts(path=Path(path), page_count=pages, doi=doi)
    f.text = "x" * int(chars_per_page * min(pages, 3))
    return f


def run(items, pairings=None, facts=None, **kw):
    pairings = pairings if pairings is not None else [
        Pairing(item=i, primary=Path("/pdfs/a.pdf"), rung="doi", confidence=0.9)
        for i in items
    ]
    facts = facts if facts is not None else {Path("/pdfs/a.pdf"): mk_facts()}
    return assess_all(items, pairings, facts, **kw)


def only(assessments):
    assert len(assessments) == 1
    return assessments[0]


# ---------- the happy path ----------

def test_complete_record_with_a_good_pdf_is_ready():
    a = only(run([mk_item()]))
    assert a.verdict == READY and a.reasons == []
    # Five words, and `about` is a content word, so the window closes there.
    assert a.derived_stem == "fixture-2024-a-fine-test-paper-about"


# ---------- PDF gates ----------

def test_no_text_layer_is_skipped():
    """The silent one: a scan extracts to nothing, ingest logs a warning nobody
    reads, and the page passes every later gate on grounding that isn't there."""
    facts = {Path("/pdfs/a.pdf"): mk_facts(chars_per_page=12)}
    a = only(run([mk_item()], facts=facts))
    assert a.verdict == SKIP and "no-text-layer" in a.reasons


def test_a_real_paper_clears_the_text_layer_threshold():
    """2000-4000 chars/page is normal; the 200 threshold sits in the empty
    middle so it is never a close call."""
    facts = {Path("/pdfs/a.pdf"): mk_facts(chars_per_page=2500)}
    assert only(run([mk_item()], facts=facts)).verdict == READY


def test_unreadable_pdf_is_skipped_not_fatal():
    facts = {Path("/pdfs/a.pdf"): PdfFacts(path=Path("/pdfs/a.pdf"), page_count=None)}
    a = only(run([mk_item()], facts=facts))
    assert a.verdict == SKIP and "pdf-unreadable" in a.reasons


def test_no_pdf_is_a_skip_not_a_failure():
    """Metadata-only rows are the majority of a big library."""
    a = only(run([mk_item()], pairings=[Pairing(item=mk_item())], facts={}))
    assert a.verdict == SKIP and a.reasons == ["no-pdf"]


# ---------- identity gates ----------

def test_typed_non_paper_is_skipped():
    for kind in ("book", "webpage", "thesis", "chapter", "report"):
        a = only(run([mk_item(item_type=kind)]))
        assert a.verdict == SKIP and "not-a-paper" in a.reasons, kind


def test_untyped_book_is_caught_by_the_metadata_fallback():
    """531 of 532 real records were typed `article`, two actual books included.
    The type field cannot catch this; absent DOI+author+year can."""
    a = only(run([mk_item(item_type="article", title="Packt.Django.5.By.Example.pdf",
                          authors=[], year=None, doi=None)]))
    assert a.verdict == REVIEW and "unresolvable" in a.reasons


def test_no_author_or_year_but_a_doi_is_still_ready():
    """11 of 532 real records look like this. A DOI alone is a sufficient
    override — reconcile resolves the rest."""
    a = only(run([mk_item(authors=[], year=None, doi="10.1234/nmeth.6")]))
    assert a.verdict == READY
    assert "--doi" in a.ingest_args and "--year" not in a.ingest_args


def test_thin_metadata_without_a_doi_goes_to_review():
    a = only(run([mk_item(doi=None, year=None)]))
    assert a.verdict == REVIEW and "thin-metadata" in a.reasons


def test_nature_comment_doi_prefix_flags_commentary():
    """`10.1038/d41586-` is Nature news/comment. Free and precise: no network
    call, no heuristic on the PDF."""
    a = only(run([mk_item(doi="10.1038/d41586-026-00008-1")]))
    assert a.verdict == REVIEW and "maybe-commentary" in a.reasons


def test_single_page_pdf_without_a_doi_flags_commentary():
    facts = {Path("/pdfs/a.pdf"): mk_facts(pages=1)}
    a = only(run([mk_item(doi=None)], facts=facts))
    assert "maybe-commentary" in a.reasons


def test_single_page_pdf_with_a_doi_is_not_flagged():
    """A one-page paper with a registered DOI is a paper. The page-count arm is
    only a fallback for when there's no DOI to judge by."""
    facts = {Path("/pdfs/a.pdf"): mk_facts(pages=1)}
    assert "maybe-commentary" not in only(run([mk_item()], facts=facts)).reasons


# ---------- preprint / published pairs ----------

def test_preprint_is_superseded_by_its_published_version():
    """The highest-value gate, and invisible to DOI dedupe: 10 such pairs in
    532 real records, with zero duplicate DOIs."""
    title = "Sequence modeling and design from molecular to genome scale"
    pre = mk_item(key="pre", title=title, doi="10.1101/2024.01.01.500001")
    pub = mk_item(key="pub", title=title, doi="10.1234/science.5")
    out = run([pre, pub])
    by_key = {a.item.key: a for a in out}
    assert by_key["pre"].verdict == SKIP
    assert "superseded-by-journal" in by_key["pre"].reasons
    assert by_key["pub"].verdict == READY


def test_survivor_does_not_depend_on_record_order():
    """The two real exports of one library listed the pairs in different
    orders, so 'last one wins' would import differently from the same data."""
    title = "Same paper twice"
    pre = mk_item(key="pre", title=title, doi="10.1101/2024.01.01.1")
    pub = mk_item(key="pub", title=title, doi="10.1234/j.1")
    for order in ([pre, pub], [pub, pre]):
        by_key = {a.item.key: a.verdict for a in run(list(order))}
        assert by_key["pre"] == SKIP and by_key["pub"] == READY


def test_titles_differing_only_in_diacritics_still_pair():
    """`[^a-z0-9]+` *deletes* a non-ASCII letter rather than folding it, so
    `Grünewald` reduced to `gr newald` while `Grunewald` reduced to `grunewald`
    — different buckets, and the pair escaped the one gate that finds it. A
    preprint and its published version are commonly typeset differently."""
    pre = mk_item(key="pre", title="Grünewald transformer editing",
                  doi="10.1101/2024.1.9")
    pub = mk_item(key="pub", title="Grunewald transformer editing",
                  doi="10.1234/j.9")
    by_key = {a.item.key: a for a in run([pre, pub])}
    assert by_key["pre"].verdict == SKIP
    assert "superseded-by-journal" in by_key["pre"].reasons
    assert by_key["pub"].verdict == READY


def test_titles_differing_only_in_punctuation_still_pair():
    pre = mk_item(key="pre", title="Deep-learning: a review", doi="10.1101/2024.1.1")
    pub = mk_item(key="pub", title="Deep learning  a review!", doi="10.1234/j.2")
    assert {a.item.key: a.verdict for a in run([pre, pub])}["pre"] == SKIP


def test_two_preprints_with_no_published_version_are_left_alone():
    """Nothing supersedes them, so neither should be dropped."""
    pre1 = mk_item(key="a", title="Only a preprint", doi="10.1101/2024.1.1")
    pre2 = mk_item(key="b", title="Only a preprint", doi="10.1101/2024.1.2")
    assert all("superseded-by-journal" not in a.reasons for a in run([pre1, pre2]))


def test_distinct_titles_are_not_deduped():
    a = mk_item(key="a", title="First distinct paper", doi="10.1101/2024.1.1")
    b = mk_item(key="b", title="Second distinct paper", doi="10.1234/j.3")
    assert all(x.verdict == READY for x in run([a, b]))


# ---------- already in the wiki ----------

def test_doi_already_in_the_wiki_is_skipped():
    """Makes top-up imports idempotent."""
    a = only(run([mk_item()], known_dois={"10.1234/test.1": "cgt/existing-2024-paper"}))
    assert a.verdict == SKIP and "already-present" in a.reasons
    assert a.collision == {"kind": "doi", "stem": "cgt/existing-2024-paper"}


def test_stem_already_in_the_wiki_is_skipped():
    a = only(run([mk_item(doi=None)], stem_exists=lambda s: True))
    assert a.verdict == SKIP and "already-present" in a.reasons
    assert a.collision["kind"] == "stem"


def test_doi_matching_is_case_insensitive():
    a = only(run([mk_item(doi="10.1234/test.1")],
                 known_dois={"10.1234/TEST.1": "cgt/x"}))
    assert "already-present" in a.reasons


# ---------- pairing quality ----------

def test_weak_title_pairing_goes_to_review():
    """Never guess an identity: a wrong pairing writes the wrong paper under
    the right stem."""
    p = Pairing(item=mk_item(), primary=Path("/pdfs/a.pdf"), rung="title", confidence=0.6)
    a = only(assess_all([p.item], [p], {Path("/pdfs/a.pdf"): mk_facts()}))
    assert a.verdict == REVIEW and "weak-pairing" in a.reasons


def test_strong_title_pairing_is_accepted():
    p = Pairing(item=mk_item(), primary=Path("/pdfs/a.pdf"), rung="title", confidence=0.92)
    a = only(assess_all([p.item], [p], {Path("/pdfs/a.pdf"): mk_facts()}))
    assert a.verdict == READY


def test_in_batch_stem_collision_goes_to_review():
    """Which paper keeps the bare year is a judgement about the corpus, not
    something to settle by iteration order."""
    one = mk_item(key="a", doi="10.1234/a")
    two = mk_item(key="b", doi="10.1234/b")
    out = run([one, two])
    assert all(a.verdict == REVIEW and "stem-collision" in a.reasons for a in out)


# ---------- the same DOI twice in one export ----------
#
# `plan_wave` re-checks the wiki, not the manifest, so without this gate both
# records dispatch in one wave and the paper is ingested twice.


def test_duplicate_doi_keeps_one_and_skips_the_rest():
    a = mk_item(key="a", title="Same paper", doi="10.1234/dup")
    b = mk_item(key="b", title="Same paper", doi="10.1234/dup")
    verdicts = [x.verdict for x in run([a, b])]
    assert sorted(verdicts) == [READY, SKIP]
    loser = next(x for x in run([a, b]) if x.verdict == SKIP)
    assert "duplicate-doi" in loser.reasons
    assert loser.collision["kind"] == "duplicate-doi"


def test_duplicate_doi_is_caught_without_a_year():
    """The hole this gate exists for. `derived_stem` is only set when title *and*
    year *and* authors are present, so a DOI-bearing record with no year has no
    stem and `_flag_stem_collisions` never sees it."""
    a = mk_item(key="a", title="Same paper", doi="10.1234/dup", year=None)
    b = mk_item(key="b", title="Same paper", doi="10.1234/dup", year=None)
    out = run([a, b])
    assert all(x.derived_stem is None for x in out), "fixture no longer has the hole"
    assert sorted(x.verdict for x in out) == [READY, SKIP]


def test_duplicate_doi_survivor_does_not_depend_on_record_order():
    """Same reasoning as the supersede gate: two exports of one library listed
    their records in different orders."""
    a = mk_item(key="a", title="Same paper", doi="10.1234/dup")
    b = mk_item(key="b", title="Same paper", doi="10.1234/dup")
    for order in ([a, b], [b, a]):
        survivors = [x.item.key for x in run(list(order)) if x.verdict == READY]
        assert survivors == ["a"], f"order {[i.key for i in order]} chose {survivors}"


def test_the_duplicate_doi_survivor_is_not_also_flagged_stem_collision():
    """Gate ordering. Both records derive the same stem, so running the stem gate
    first would send a clean import to review over a record about to be
    skipped."""
    a = mk_item(key="a", title="Same paper", doi="10.1234/dup")
    b = mk_item(key="b", title="Same paper", doi="10.1234/dup")
    survivor = next(x for x in run([a, b]) if x.verdict == READY)
    assert "stem-collision" not in survivor.reasons


def test_distinct_dois_are_untouched_by_the_dedup_gate():
    a = mk_item(key="a", title="First paper here", doi="10.1234/a")
    b = mk_item(key="b", title="Second paper here", doi="10.1234/b")
    assert all("duplicate-doi" not in x.reasons for x in run([a, b]))


def test_the_fetch_list_asks_for_each_doi_once():
    """A metadata-only export should ask for each duplicated DOI once."""
    a = mk_item(key="a", title="Same paper", doi="10.1234/dup")
    b = mk_item(key="b", title="Same paper", doi="10.1234/dup")
    out = assess_all([a, b], [Pairing(item=a), Pairing(item=b)], {})
    assert [r["doi"] for r in missing_pdf_fetch_list(out)] == ["10.1234/dup"]


# ---------- verdict severity ----------

def test_the_most_severe_verdict_wins_regardless_of_gate_order():
    """An item that is both weakly paired and already present is a skip."""
    p = Pairing(item=mk_item(), primary=Path("/pdfs/a.pdf"), rung="title", confidence=0.6)
    a = only(assess_all([p.item], [p], {Path("/pdfs/a.pdf"): mk_facts()},
                        known_dois={"10.1234/test.1": "cgt/x"}))
    assert a.verdict == SKIP
    assert {"weak-pairing", "already-present"} <= set(a.reasons)


def test_reasons_are_not_duplicated():
    facts = {Path("/pdfs/a.pdf"): mk_facts(pages=1)}
    a = only(run([mk_item(doi="10.1038/d41586-026-1")], facts=facts))
    assert a.reasons.count("maybe-commentary") == 1


# ---------- ingest argv ----------

def test_ingest_args_carry_every_available_override():
    args = build_ingest_args(mk_item(), Pairing(item=mk_item()))
    assert args == ["--doi", "10.1234/test.1",
                    "--title", "A fine test paper about things",
                    "--authors", "Ada Fixture",
                    "--year", "2024"]


def test_ingest_args_omit_a_malformed_doi():
    """`--doi` with a broken value costs a failed lookup; omitting it lets
    reconcile do its normal job."""
    item = mk_item(doi=None)
    assert "--doi" not in build_ingest_args(item, Pairing(item=item))


def test_ingest_args_include_supplementary_paths():
    item = mk_item()
    p = Pairing(item=item, primary=Path("/pdfs/a.pdf"),
                supplementary=[Path("/pdfs/a-supp.pdf")])
    assert build_ingest_args(item, p)[-2:] == ["--supplementary", "/pdfs/a-supp.pdf"]


def test_ingest_args_never_pass_venue():
    """No override flag exists for it, and reconcile's DOI lookup is more
    trustworthy than a manager's abbreviated journal field."""
    item = mk_item(venue="Journal of Test Genomics")
    assert "Journal of Test Genomics" not in build_ingest_args(item, Pairing(item=item))


# ---------- reporting ----------

def test_summarize_counts_verdicts_and_reasons():
    out = run([mk_item(key="a", doi="10.1234/a"),
               mk_item(key="b", title="Another different paper", doi=None, year=None)])
    s = summarize(out)
    assert s["total"] == 2
    assert s["verdicts"][REVIEW] >= 1
    assert "thin-metadata" in s["reasons"]


def test_fetch_list_holds_exactly_the_items_blocked_only_by_a_missing_pdf():
    """On a cloud-hosted library this is the most useful artifact the command
    produces — without it a metadata-only run reports a count and nothing
    actionable."""
    good = mk_item(key="want", doi="10.1234/want.1")
    typed_out = mk_item(key="book", item_type="book", doi="10.1234/book.1")
    out = assess_all([good, typed_out],
                     [Pairing(item=good), Pairing(item=typed_out)], {})
    fetch = missing_pdf_fetch_list(out)
    assert [f["key"] for f in fetch] == ["want"]
    assert fetch[0]["doi"] == "10.1234/want.1"


def test_fetch_list_excludes_items_with_other_problems_too():
    """An item that is also already in the wiki does not belong on a to-fetch
    list — the reason set must be exactly `no-pdf`."""
    item = mk_item()
    out = assess_all([item], [Pairing(item=item)], {},
                     known_dois={"10.1234/test.1": "cgt/x"})
    assert missing_pdf_fetch_list(out) == []


def test_as_dict_is_json_safe():
    import json
    a = only(run([mk_item()]))
    json.dumps(a.as_dict())


# ---------- pairing distinctiveness ----------
#
# Measured against 313 DOI-confirmed pairs from a real library: without this
# gate, 6 records were silently paired to the wrong PDF. With it, none were, at
# a cost of 4 confident pairings demoted to review.

def test_a_confident_but_undistinctive_title_match_goes_to_review():
    """Another record scored nearly as well against this same PDF, so the score
    came from shared vocabulary rather than from identity."""
    p = Pairing(item=mk_item(), primary=Path("/pdfs/a.pdf"), rung="title",
                confidence=0.90, rival=0.88)
    a = only(assess_all([p.item], [p], {Path("/pdfs/a.pdf"): mk_facts()}))
    assert a.verdict == REVIEW and "ambiguous-pairing" in a.reasons


def test_a_distinctive_title_match_is_accepted():
    p = Pairing(item=mk_item(), primary=Path("/pdfs/a.pdf"), rung="title",
                confidence=0.90, rival=0.40)
    a = only(assess_all([p.item], [p], {Path("/pdfs/a.pdf"): mk_facts()}))
    assert a.verdict == READY and "ambiguous-pairing" not in a.reasons


def test_a_sole_candidate_has_no_rival_and_is_accepted():
    p = Pairing(item=mk_item(), primary=Path("/pdfs/a.pdf"), rung="title",
                confidence=0.80, rival=0.0)
    assert only(assess_all([p.item], [p],
                           {Path("/pdfs/a.pdf"): mk_facts()})).verdict == READY


def test_the_margin_gate_does_not_apply_to_doi_pairings():
    """A DOI match is an identity, not a similarity — a rival score against the
    same file is irrelevant to it."""
    p = Pairing(item=mk_item(), primary=Path("/pdfs/a.pdf"), rung="doi",
                confidence=0.9, rival=0.9)
    assert only(assess_all([p.item], [p],
                           {Path("/pdfs/a.pdf"): mk_facts()})).verdict == READY


def test_weak_pairing_takes_precedence_over_ambiguity():
    """Below the confidence bar is the more informative complaint; reporting
    both would be noise."""
    p = Pairing(item=mk_item(), primary=Path("/pdfs/a.pdf"), rung="title",
                confidence=0.60, rival=0.59)
    a = only(assess_all([p.item], [p], {Path("/pdfs/a.pdf"): mk_facts()}))
    assert "weak-pairing" in a.reasons and "ambiguous-pairing" not in a.reasons


# ---------- reference material ----------

def test_typed_non_papers_are_listed_not_just_counted():
    """A bare `not-a-paper: 20` discards which twenty and leaves the user a
    dead end. Books and guidance are legitimate wiki/references/ pages — just
    hand-written ones."""
    from researchwiki.refimport.triage import reference_doc_candidates

    book = mk_item(key="b", item_type="book", title="The Test Framework Manual")
    paper = mk_item(key="p", title="A normal paper", doi="10.1234/p")
    out = run([book, paper])
    refs = reference_doc_candidates(out)
    assert [r["key"] for r in refs] == ["b"]
    assert refs[0]["item_type"] == "book"
    assert refs[0]["title"] == "The Test Framework Manual"


def test_an_untyped_book_does_not_reach_the_reference_list():
    """ReadCube types books as journal articles, so they land in `unresolvable`
    — a review item — rather than here. The list only claims what the exporter
    actually asserted."""
    from researchwiki.refimport.triage import reference_doc_candidates

    a = only(run([mk_item(item_type="article", title="Packt.Django.5.By.Example.pdf",
                          authors=[], year=None, doi=None)]))
    assert "unresolvable" in a.reasons
    assert reference_doc_candidates([a]) == []
