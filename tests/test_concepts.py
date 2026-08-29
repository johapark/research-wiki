"""Concept hub scaffolder (researchwiki.tasks.concepts).

Key invariants under test:

  1. `_page_mentions` — membership uses a boundary-anchored token match, NOT
     a substring match: "CAR" must not match "car", "RAG" must not match
     "storage".
  2. `find_members` — a paper is a member iff it has a graded claim in a
     contribution section (kc / results / methodology) mentioning the term.
     Body-prose-only mentions do NOT make a paper a member. Idea / synthesis
     pages never become members regardless.
  3. `_template` — spoke bullets carry the specific claim's `#slug` when a
     matching claim is known; falls back to bare `[[stem]]` when best_slug
     is None. `referenced_papers:` in frontmatter stays bare.
  4. `attach_after_ingest` — matches against the claim substrate (not body
     prose); a body-prose-only mention is logged as a near-miss but does
     not attach the paper.
"""

from pathlib import Path

import pytest

from researchwiki.concepts import scaffold as concepts_mod
from researchwiki.concepts import term_claims as term_claims_mod
from researchwiki.concepts.term_claims import _page_mentions
from researchwiki.concepts.scaffold import _template, find_members


# ---------- _page_mentions (pure) ----------

def test_page_mentions_acronym_token_not_substring():
    assert _page_mentions("RAG", "We use RAG for retrieval.")
    assert not _page_mentions("RAG", "dragging storage around")  # substring, no token
    assert not _page_mentions("CAR", "the car drove off")        # lowercase


def test_page_mentions_titlecase_phrase():
    assert _page_mentions("Virtual Cell", "The Virtual Cell paradigm is emerging.")
    assert not _page_mentions("Virtual Cell", "a virtual cell model")  # lowercased


# ---------- find_members (with a stubbed claim resolver) ----------

@pytest.fixture
def tmp_wiki(tmp_path, monkeypatch):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    monkeypatch.setattr("researchwiki.wiki.wiki_dir", lambda: wiki)
    # Patching wiki_dir only redirects markdown I/O. `commit_page` (fired by
    # `append_related_paper` / concept-attach writes) resolves the DB path
    # independently via `wiki_root()` (= cwd), which this fixture does NOT
    # touch — without this, a real write here silently lands in the actual
    # per-repo state.db instead of a throwaway one.
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    return wiki


def _page(wiki: Path, key: str, body: str, ptype: str = "paper") -> None:
    p = wiki / f"{key}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f'---\ntitle: "x"\ntype: {ptype}\n---\n\n{body}\n')


def _stub_matching_claims(monkeypatch, per_stem_hits):
    """Replace concepts._matching_claims with a stub reading from a dict.

    per_stem_hits: {stem: [{section, position, claim_slug, text}, ...]}
      OR
      {(stem, term_lc): [...]}  (term-specific hits — mimics the real matcher
                                 which returns different rows per search term).

    Stems / (stem, term) not in the dict return [] (no matching claim).
    """
    def fake(stem, term):
        # Try term-specific first, then fall back to stem-only.
        specific = per_stem_hits.get((stem, term.lower()))
        if specific is not None:
            return specific
        return per_stem_hits.get(stem, [])
    monkeypatch.setattr(concepts_mod, "_matching_claims", fake)
    # The spoke hint resolves the same function inside term_claims (that is the
    # point of routing both through one query), so stub both call sites.
    monkeypatch.setattr(term_claims_mod, "_matching_claims", fake)


def _stub_keyword_hits(monkeypatch, per_stem_kws: dict[str, list[str]]):
    """Replace `_papers_where_keywords_match` with a fixed mapping.

    per_stem_kws: {stem: [matched_keyword, ...]}. `_load_paper_metadata` isn't
    called at all in the stubbed path, so tests don't need a live state.db.
    """
    def fake(term, aliases=None):
        return dict(per_stem_kws)
    monkeypatch.setattr(concepts_mod, "_papers_where_keywords_match", fake)


def test_find_members_uses_claim_substrate_not_body_prose(tmp_wiki, monkeypatch):
    # a-2024-x mentions RAPTOR in a claim → member.
    # b-2024-y mentions RAPTOR in body prose but has no claim → NOT a member.
    # c-2024-z doesn't mention it at all.
    # d-idea is an idea page (not spoke-eligible even if it had a claim).
    _page(tmp_wiki, "ai/a-2024-x", "RAPTOR builds a tree of summaries.")
    _page(tmp_wiki, "single-cell/b-2024-y", "We compare against RAPTOR here.")
    _page(tmp_wiki, "single-cell/c-2024-z", "No mention of the term at all.")
    _page(tmp_wiki, "ideas/d-idea", "RAPTOR everywhere.", ptype="idea")

    _stub_matching_claims(monkeypatch, {
        "a-2024-x": [{"section": "key_contributions", "position": 0,
                       "claim_slug": "kc-aaaa1111",
                       "text": "RAPTOR builds a tree of summaries.",
                       "semantic_score": 0.9}],
        # b-2024-y intentionally absent — body-prose mention shouldn't matter.
    })

    members = find_members("RAPTOR")
    keys = [k for k, _, _, _ in members]
    slugs = [s for _, _, s, _ in members]

    assert keys == ["ai/a-2024-x"]
    assert slugs == ["kc-aaaa1111"]


def test_find_members_promotes_keyword_hit_paper_without_claim_match(
    tmp_wiki, monkeypatch,
):
    """The DMS-shaped failure: a paper whose LLM keywords include a synonym
    of the term (e.g., `saturation mutagenesis` for a `DMS` hub) but whose
    claim text uses only that synonym. Under the fixed matcher, the paper's
    own keyword becomes an alias that widens the claim search.
    """
    _page(tmp_wiki, "cgt/zhou-2025-ldlr", "body")

    # Paper's keyword is "saturation mutagenesis"; its claim text uses that
    # exact phrase. Term is "DMS" — direct search finds nothing, but the
    # paper's own keyword bridges the vocabulary gap.
    _stub_keyword_hits(monkeypatch, {"zhou-2025-ldlr": ["saturation mutagenesis"]})
    _stub_matching_claims(monkeypatch, {
        # No hit on the term "DMS".
        # Hit on the aliasing keyword.
        ("zhou-2025-ldlr", "saturation mutagenesis"): [{
            "section": "key_contributions", "position": 0,
            "claim_slug": "kc-abcd1234",
            "text": "Saturation mutagenesis of LDLR residues 137–219.",
            "semantic_score": 0.9,
        }],
    })

    members = find_members("DMS")
    assert len(members) == 1
    assert members[0][0] == "cgt/zhou-2025-ldlr"
    assert members[0][2] == "kc-abcd1234"


def test_find_members_expands_via_caller_aliases(tmp_wiki, monkeypatch):
    """`find_members(term, aliases=[...])` widens the claim-substring search
    to every alias — closes the vocabulary-divergence gap across papers."""
    _page(tmp_wiki, "cgt/one", "body")
    _page(tmp_wiki, "cgt/two", "body")

    _stub_keyword_hits(monkeypatch, {})  # no keyword hits — force alias path
    _stub_matching_claims(monkeypatch, {
        # Paper `one` claims match "DMS"; paper `two` claims match "MAVE".
        ("one", "dms"): [{"section": "key_contributions", "position": 0,
                          "claim_slug": "kc-oneone11",
                          "text": "DMS results across the library.",
                          "semantic_score": 0.9}],
        ("two", "mave"): [{"section": "key_contributions", "position": 0,
                           "claim_slug": "kc-twotwo22",
                           "text": "MAVE-derived functional annotation.",
                           "semantic_score": 0.9}],
    })

    # Without aliases: neither paper matches "deep mutational scanning".
    assert find_members("deep mutational scanning") == []

    # With aliases: both papers become members.
    members = find_members("deep mutational scanning",
                            aliases=["DMS", "MAVE"])
    stems = {k for k, _, _, _ in members}
    assert stems == {"cgt/one", "cgt/two"}
    # The alias that found each one is recorded, which is what `alias_hits`
    # aggregates for the author.
    assert {k: m for k, _, _, m in members} == {"cgt/one": "DMS", "cgt/two": "MAVE"}


def test_matching_claims_respects_word_boundaries(monkeypatch, tmp_path):
    """Short acronyms like `DMS` must not substring-match inside `Dmse` or
    `DMSO`. Fires against a live state.db (in tmp) so the SQL path + regex
    filter both run."""
    from researchwiki.concepts import scaffold as _concepts
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))

    from researchwiki.db.connection import get_connection
    conn = get_connection()
    conn.execute(
        "INSERT INTO papers (stem, category, page_type, title, page_path, page_mtime, raw_frontmatter, indexed_at) "
        "VALUES ('p1', 'ai', 'paper', 'x', '/tmp/x.md', 0, '{}', 0)"
    )
    conn.execute(
        "INSERT INTO claims (paper_stem, section, position, text, claim_slug, is_cross_ref) "
        "VALUES ('p1', 'results', 0, 'Dmse distortion parameter for quantization.', 'res-aaaa1111', 0)"
    )
    conn.execute(
        "INSERT INTO claims (paper_stem, section, position, text, claim_slug, is_cross_ref) "
        "VALUES ('p1', 'results', 1, 'DMS scores across all variants.', 'res-bbbb2222', 0)"
    )
    conn.commit()
    conn.close()

    got = _concepts._matching_claims("p1", "DMS")
    slugs = [r["claim_slug"] for r in got]
    # "Dmse" MUST NOT match; "DMS scores" MUST.
    assert slugs == ["res-bbbb2222"]


def test_find_members_returns_best_slug_per_paper(tmp_wiki, monkeypatch):
    _page(tmp_wiki, "ai/a-2024-x", "body")
    _stub_matching_claims(monkeypatch, {
        "a-2024-x": [
            {"section": "key_contributions", "position": 0,
             "claim_slug": "kc-first0000", "text": "top pick", "semantic_score": 0.9},
            {"section": "results", "position": 2,
             "claim_slug": "res-later123", "text": "secondary", "semantic_score": 0.8},
        ],
    })
    members = find_members("X")
    assert members[0][2] == "kc-first0000"  # first row = top pick


def test_alias_hits_attribute_each_member_to_the_term_that_found_it(
    tmp_wiki, monkeypatch,
):
    """Aliases widen membership silently; `alias_hits` makes the cost visible.

    Five plausible aliases took the parameter-efficient-fine-tuning hub from 5
    members to 17 across 4 categories, admitting a Bayesian-optimization paper
    and Feynman's restaurant problem via substring hits on "low-rank" and
    "adapter". Nothing said which alias did it. This is a diagnostic, not a cap:
    the governing decision for this tier is propose-never-decide, and a cap
    would block legitimately broad hubs.
    """
    _page(tmp_wiki, "ai/a-2024-x", "body")
    _page(tmp_wiki, "compbio/b-2024-y", "body")
    _page(tmp_wiki, "single-cell/c-2024-z", "body")
    _stub_matching_claims(monkeypatch, {
        ("a-2024-x", "peft"): [{"claim_slug": "kc-1", "text": "uses PEFT"}],
        ("b-2024-y", "lora"): [{"claim_slug": "kc-2", "text": "uses LoRA"}],
        ("c-2024-z", "lora"): [{"claim_slug": "kc-3", "text": "uses LoRA"}],
    })
    _stub_keyword_hits(monkeypatch, {})

    res = concepts_mod.run("PEFT", aliases=["LoRA"], thesis="", dry_run=True)
    assert res["alias_hits"] == {"PEFT": 1, "LoRA": 2}


def test_find_members_leaves_the_anchor_bare_when_no_claim_matches(
    tmp_wiki, monkeypatch,
):
    """A keyword-only member gets no anchor rather than an unrelated one.

    The seqLens case: the paper joined the parameter-efficient-fine-tuning hub
    on keywords, no claim matched the term, and the spoke was anchored to the
    paper's *first* key_contributions claim — "Introduced seqLens, a DeBERTa-v2
    based gLM family…", which says nothing about the concept. Nucleotide
    Transformer got its model list the same way. A bare `[[stem]]` is the
    citation form CLAUDE.md prescribes for referring to a paper as a whole, and
    `concepts --upgrade-spokes` backfills it once a matching claim exists.
    """
    _page(tmp_wiki, "compbio/seqlens-2025-x", "body")
    # Keyword signal fires; neither the term nor the paper's own keyword hits
    # any claim text.
    _stub_keyword_hits(monkeypatch, {"seqlens-2025-x": ["genomic language model"]})
    _stub_matching_claims(monkeypatch, {})
    # Make the discarded fallback loudly available: if find_members still calls
    # it, the assertion below fails with this slug rather than passing because a
    # tmp DB happened to be empty.
    monkeypatch.setattr(concepts_mod, "_top_kc_claim_slug",
                        lambda stem: "kc-unrelated0")

    members = find_members("parameter-efficient fine-tuning")
    assert len(members) == 1                       # still a member
    assert members[0][0] == "compbio/seqlens-2025-x"
    assert members[0][2] is None                   # but NOT a fabricated anchor


# ---------- _term_claim_hint (shares one query with the anchor) ----------

def test_term_claim_hint_is_case_insensitive_like_the_anchor(tmp_wiki, monkeypatch):
    """Hint and anchor must come from one query or they disagree.

    The hint used its own scan — case-*sensitive*, no section filter, no
    word-boundary check — so a spoke could carry a valid `#claim_slug` from
    `find_members` and an empty hint, or a hint drawn from a `limitations`
    claim the anchor would never cite. Sentence-initial capitals alone were
    enough to break it.
    """
    from researchwiki.concepts.term_claims import _term_claim_hint

    _stub_matching_claims(monkeypatch, {
        "a-2024-x": [{"section": "key_contributions", "position": 0,
                      "claim_slug": "kc-aaaa1111",
                      "text": "Prime editing installs substitutions up to 50 bp.",
                      "semantic_score": 0.9}],
    })
    # Term is lowercase; the claim starts the sentence with a capital.
    assert "Prime editing installs" in _term_claim_hint("a-2024-x", "prime editing")


# ---------- _template (pure; passes 3-tuple members) ----------

def test_template_bridge_shape():
    members = [
        ("ai/a-2024-x", "ai", "kc-abcd1234", "RAPTOR"),
        ("single-cell/b-2024-y", "single-cell", None, None),
    ]
    out = _template("RAPTOR", "raptor", "RAPTOR", members, span=2,
                    thesis="Same tool, different epistemic role across ai and single-cell.")
    assert "type: concept" in out
    assert "concept_span: 2" in out
    # Spoke with slug uses [[stem#slug]]; spoke without slug falls back to bare.
    assert "[[ai/a-2024-x#kc-abcd1234]]" in out
    assert "[[single-cell/b-2024-y]]" in out
    assert "[[single-cell/b-2024-y#" not in out  # no anchor when best_slug is None
    # referenced_papers cites the paper (quoted so Obsidian renders the link).
    assert '  - "[[ai/a-2024-x]]"' in out
    assert "## Definition" in out
    assert "## Cross-domain connections" in out
    assert "### ai" in out and "### single-cell" in out
    # Thesis lands as a YAML field only — the visible `> **Thesis.**`
    # blockquote was dropped (Obsidian Live Preview renders the
    # skip-grounding HTML comments as literal text).
    assert "concept_thesis: |" in out
    assert 'author_model: "TODO"' in out
    assert "> **Thesis.**" not in out


def test_template_single_category_is_flat():
    members = [("ai/a", "ai", "kc-11111111", "X"), ("ai/b", "ai", None, None),
               ("ai/c", "ai", "res-22222222", "X")]
    out = _template("X", "x", "X", members, span=1,
                    thesis="One direction of dependency in ai — X as design substrate.")
    assert "## Cross-domain connections" not in out
    assert "### ai" not in out
    # Bare + slug forms both present.
    assert "[[ai/a#kc-11111111]]" in out
    assert "[[ai/b]]" in out
    assert "[[ai/c#res-22222222]]" in out
    assert "concept_thesis: |" in out


# ---------- attach_after_ingest (post-ingest hook, claim substrate) ----------

def _concept_wiki(tmp_path, monkeypatch):
    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    monkeypatch.setattr("researchwiki.wiki.wiki_dir", lambda: wiki)
    monkeypatch.setattr("researchwiki.concepts.scaffold.wiki_dir", lambda: wiki)
    # See tmp_wiki's comment above — attach_after_ingest writes trigger
    # commit_page, which needs its own DB isolation independent of wiki_dir.
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    (wiki / "concepts" / "raptor.md").write_text(
        '---\ntitle: "RAPTOR"\ntype: concept\ncategory: [ai]\n'
        "referenced_papers:\n  - [[ai/sarthi-2024-x]]\n"
        'concept_span: 1\ngenerated_at: 2020-01-01\ntopic_seed: "RAPTOR"\ntags: [concept, raptor]\n---\n\n'
        "## Definition\nRAPTOR builds trees.[^s]\n\n"
        "## How it appears across the corpus\n- [[ai/sarthi-2024-x]] — builds the tree.\n\n"
        "[^s]: [[ai/sarthi-2024-x]]\n"
    )
    return wiki


def test_attach_after_ingest_adds_spoke_span_and_reciprocal(tmp_path, monkeypatch):
    wiki = _concept_wiki(tmp_path, monkeypatch)
    (wiki / "single-cell").mkdir()
    newp = wiki / "single-cell" / "b-2026-y.md"
    newp.write_text('---\ntitle: "y"\ntype: paper\n---\n\nWe adapt RAPTOR to single cells.\n')

    _stub_matching_claims(monkeypatch, {
        "b-2026-y": [{"section": "key_contributions", "position": 0,
                       "claim_slug": "kc-newpaper1", "text": "adapts RAPTOR",
                       "semantic_score": 0.95}],
    })

    res = concepts_mod.attach_after_ingest("b-2026-y", newp)
    assert res["attached"] == ["raptor"]

    ctext = (wiki / "concepts" / "raptor.md").read_text()
    # Spoke cites the specific claim slug now.
    assert "[[single-cell/b-2026-y#kc-newpaper1]]" in ctext
    # referenced_papers entry (quoted so Obsidian renders the link).
    assert '  - "[[single-cell/b-2026-y]]"' in ctext
    assert "concept_span: 2" in ctext                    # ai + single-cell
    assert "generated_at: 2020-01-01" not in ctext       # refreshed
    assert "[[concepts/raptor]]" in newp.read_text()      # reciprocal back-link

    # idempotent: a second run attaches nothing
    assert concepts_mod.attach_after_ingest("b-2026-y", newp)["attached"] == []


def test_attach_after_ingest_body_prose_only_is_near_miss(tmp_path, monkeypatch, caplog):
    """The new rule: body-prose mention without a matching claim triggers an
    INFO log entry (near-miss) but does NOT attach the paper."""
    wiki = _concept_wiki(tmp_path, monkeypatch)
    (wiki / "single-cell").mkdir()
    newp = wiki / "single-cell" / "b-2026-y.md"
    newp.write_text('---\ntitle: "y"\ntype: paper\n---\n\nWe adapt RAPTOR to single cells.\n')

    # No matching claim → no attach; body prose mentions the term → near-miss.
    _stub_matching_claims(monkeypatch, {})

    res = concepts_mod.attach_after_ingest("b-2026-y", newp)
    assert res["attached"] == []
    assert "raptor" in res["near_missed"]
    # Hub file must not carry the paper.
    assert "[[single-cell/b-2026-y]]" not in (wiki / "concepts" / "raptor.md").read_text()


def test_run_refuses_without_thesis(tmp_wiki, monkeypatch):
    """Scaffold gate: empty `thesis` → ValueError with an actionable message.
    This is the discipline that would have caught the retracted PAM/RNP/LNP/DSB
    glossary hubs — you can't type a real thesis for a glossary term."""
    from researchwiki.concepts import scaffold as concepts

    _stub_matching_claims(monkeypatch, {
        "a-2024-x": [{"section": "key_contributions", "position": 0,
                       "claim_slug": "kc-aaaa1111", "text": "...",
                       "semantic_score": 0.9}],
        "b-2024-y": [{"section": "key_contributions", "position": 0,
                       "claim_slug": "kc-bbbb2222", "text": "...",
                       "semantic_score": 0.9}],
        "c-2024-z": [{"section": "key_contributions", "position": 0,
                       "claim_slug": "kc-cccc3333", "text": "...",
                       "semantic_score": 0.9}],
    })
    _page(tmp_wiki, "cgt/a-2024-x", "body")
    _page(tmp_wiki, "cgt/b-2024-y", "body")
    _page(tmp_wiki, "cgt/c-2024-z", "body")

    with pytest.raises(ValueError, match="concept_thesis"):
        concepts.run("some-term", thesis="")
    with pytest.raises(ValueError, match="concept_thesis"):
        concepts.run("some-term", thesis="   ")


def test_dry_run_inspects_without_a_thesis(tmp_wiki, monkeypatch):
    """A dry run must not demand the sentence it exists to help you write.

    The thesis test asks *why is this a concept and not glossary* — a judgement
    you can only make once you have seen the member list, so requiring it to
    look was circular. Building the parameter-efficient-fine-tuning hub needed
    three inspections, each passing `--thesis "provisional"` to get past a gate
    that then discarded the value. The write path keeps the gate.
    """
    _page(tmp_wiki, "ai/a-2024-x", "body")
    _page(tmp_wiki, "compbio/b-2024-y", "body")
    _page(tmp_wiki, "single-cell/c-2024-z", "body")
    _stub_matching_claims(monkeypatch, {
        "a-2024-x": [{"claim_slug": "kc-1", "text": "uses X"}],
        "b-2024-y": [{"claim_slug": "kc-2", "text": "uses X"}],
        "c-2024-z": [{"claim_slug": "kc-3", "text": "uses X"}],
    })
    _stub_keyword_hits(monkeypatch, {})

    res = concepts_mod.run("X", thesis="", dry_run=True)
    assert res["dry_run"] is True
    assert len(res["members"]) == 3
    assert res["path"] is None            # nothing written


def _concept_wiki_with_aliases(tmp_path, monkeypatch):
    """Same shape as `_concept_wiki` but the hub declares `topic_seed_aliases`.

    Mirrors the FH-shaped failure: hub topic_seed is the full phrase, but the
    claim extractor uses medical abbreviations (`FH`, `HeFH`, `HoFH`) and
    papers may use British spelling (`hypercholesterolaemia`).
    """
    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    monkeypatch.setattr("researchwiki.wiki.wiki_dir", lambda: wiki)
    monkeypatch.setattr("researchwiki.concepts.scaffold.wiki_dir", lambda: wiki)
    # See tmp_wiki's comment above — attach_after_ingest writes trigger
    # commit_page, which needs its own DB isolation independent of wiki_dir.
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    (wiki / "concepts" / "familial-hypercholesterolemia.md").write_text(
        '---\ntitle: "familial hypercholesterolemia"\ntype: concept\ncategory: [genetics]\n'
        "referenced_papers:\n  - [[genetics/koyama-2026-x]]\n"
        'concept_span: 1\ngenerated_at: 2020-01-01\n'
        'topic_seed: "familial hypercholesterolemia"\n'
        'topic_seed_aliases: ["FH", "HeFH", "familial hypercholesterolaemia"]\n'
        "tags: [concept, familial-hypercholesterolemia]\n---\n\n"
        "## Definition\nFH is a monogenic disorder.[^s]\n\n"
        "## How it appears across the corpus\n- [[genetics/koyama-2026-x]] — anchors it.\n\n"
        "[^s]: [[genetics/koyama-2026-x]]\n"
    )
    return wiki


def test_attach_after_ingest_uses_alias_when_claim_uses_abbreviation(
    tmp_path, monkeypatch,
):
    """FH-shaped failure: hub topic_seed is `familial hypercholesterolemia`
    but the ingested paper's claims say `FH` / `HeFH`. Without alias support
    the attach hook silently misses. With aliases, direct claim-substring
    match on `HeFH` picks it up."""
    wiki = _concept_wiki_with_aliases(tmp_path, monkeypatch)
    (wiki / "genetics").mkdir()
    newp = wiki / "genetics" / "ahmad-2026-y.md"
    newp.write_text('---\ntitle: "y"\ntype: paper\n---\n\nUpdate on familial hypercholesterolemia.\n')

    # Claims use only the abbreviation — the exact term "familial
    # hypercholesterolemia" is absent from every contribution claim.
    _stub_matching_claims(monkeypatch, {
        ("ahmad-2026-y", "hefh"): [{
            "section": "key_contributions", "position": 0,
            "claim_slug": "kc-fh1", "text": "HeFH prevalence ~1 in 311.",
            "semantic_score": 0.95,
        }],
    })

    res = concepts_mod.attach_after_ingest("ahmad-2026-y", newp)
    assert res["attached"] == ["familial-hypercholesterolemia"]

    ctext = (wiki / "concepts" / "familial-hypercholesterolemia.md").read_text()
    assert "[[genetics/ahmad-2026-y#kc-fh1]]" in ctext
    assert "[[concepts/familial-hypercholesterolemia]]" in newp.read_text()


def test_attach_after_ingest_falls_back_to_keyword_hits(tmp_path, monkeypatch):
    """Second signal: a paper whose LLM-authored keywords match the hub
    vocabulary but whose claim text uses a further synonym. The keyword-hit
    fallback widens the claim search using the paper's own keywords, then
    anchors to _top_kc_claim_slug when even that misses."""
    wiki = _concept_wiki_with_aliases(tmp_path, monkeypatch)
    (wiki / "genetics").mkdir()
    newp = wiki / "genetics" / "santos-2025-z.md"
    newp.write_text('---\ntitle: "z"\ntype: paper\n---\n\nFH review.\n')

    # No direct-term claim hits.
    _stub_matching_claims(monkeypatch, {})
    # But the paper's keywords match "familial hypercholesterolaemia" alias.
    _stub_keyword_hits(monkeypatch,
                       {"santos-2025-z": ["familial hypercholesterolaemia"]})
    # And the last-resort anchor returns the paper's top kc claim.
    monkeypatch.setattr(concepts_mod, "_top_kc_claim_slug",
                        lambda stem: "kc-topclaim")

    res = concepts_mod.attach_after_ingest("santos-2025-z", newp)
    assert res["attached"] == ["familial-hypercholesterolemia"]

    ctext = (wiki / "concepts" / "familial-hypercholesterolemia.md").read_text()
    assert "[[genetics/santos-2025-z#kc-topclaim]]" in ctext


def test_attach_after_ingest_near_miss_uses_aliases_for_body_prose(
    tmp_path, monkeypatch,
):
    """British-spelling near-miss: the hub topic_seed is `familial
    hypercholesterolemia` (US), the paper's body has only `familial
    hypercholesterolaemia` (UK), and no claim matches. Old behavior:
    silent skip. New behavior: alias-aware near-miss log so the paper
    surfaces for manual review instead of vanishing."""
    wiki = _concept_wiki_with_aliases(tmp_path, monkeypatch)
    (wiki / "genetics").mkdir()
    newp = wiki / "genetics" / "wiegman-2026-uk.md"
    newp.write_text(
        '---\ntitle: "uk"\ntype: paper\n---\n\n'
        "This is a pediatric familial hypercholesterolaemia consensus.\n"
    )

    _stub_matching_claims(monkeypatch, {})
    _stub_keyword_hits(monkeypatch, {})

    res = concepts_mod.attach_after_ingest("wiegman-2026-uk", newp)
    assert res["attached"] == []
    assert "familial-hypercholesterolemia" in res["near_missed"]


def test_attach_after_ingest_skips_when_term_absent(tmp_path, monkeypatch):
    wiki = _concept_wiki(tmp_path, monkeypatch)
    (wiki / "single-cell").mkdir()
    newp = wiki / "single-cell" / "c-2026-z.md"
    newp.write_text('---\ntitle: "z"\ntype: paper\n---\n\nNo mention of that method here.\n')

    _stub_matching_claims(monkeypatch, {})

    res = concepts_mod.attach_after_ingest("c-2026-z", newp)
    assert res["attached"] == []
    assert res["near_missed"] == []  # term not even in body prose
    assert "[[single-cell/c-2026-z]]" not in (wiki / "concepts" / "raptor.md").read_text()
