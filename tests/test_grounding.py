"""Grounding checks (researchwiki.grade.grounding).

Two behaviours that are easy to break silently:

  1. Academic footnote citations — a claim that cites `[^id]` inline is grounded
     iff the matching `[^id]: … [[wikilink]]` definition resolves to a wikilink.
     The link lives in the reference list, not in the prose unit.
  2. Line-number fidelity — frontmatter / fenced code / skip regions are blanked
     (not deleted) before unit splitting, so `annotate` paints the `⚠ ungrounded`
     marker on the *original* source line, never on shifted-up frontmatter.
"""

from researchwiki.grade import grounding


def _unit_starting(report, prefix):
    return next(u for u in report.units if u.text.lstrip().startswith(prefix))


# ---------- footnote-citation grounding ----------

FOOTNOTE_DOC = """---
type: idea
---

scGPT emits both cell and gene embeddings from one transformer that scales widely.[^scgpt]

Geneformer compares gene embeddings by cosine similarity across the whole network.[^geneformer]

This claim points at a footnote whose definition has no link at all so it must flag.[^nolink]

This claim references a footnote that was never defined anywhere in the body here.[^missing]

[^scgpt]: [[single-cell/cui-2024-scgpt-toward-building-a-foundation]] — scGPT
[^geneformer]: [[single-cell/theodoris-2023-transfer-learning-enables-predictions-in-network]]
[^nolink]: Just some prose with no wikilink in the definition body.
"""


def test_footnote_with_wikilink_definition_grounds_the_claim():
    rep = grounding.check(FOOTNOTE_DOC)
    assert _unit_starting(rep, "scGPT").has_citation
    assert _unit_starting(rep, "Geneformer").has_citation


def test_footnote_definition_without_wikilink_does_not_ground():
    rep = grounding.check(FOOTNOTE_DOC)
    assert not _unit_starting(rep, "This claim points").has_citation


def test_undefined_footnote_reference_does_not_ground():
    rep = grounding.check(FOOTNOTE_DOC)
    assert not _unit_starting(rep, "This claim references").has_citation


def test_footnote_definition_lines_are_not_claims():
    rep = grounding.check(FOOTNOTE_DOC)
    defs = [u for u in rep.units if u.text.lstrip().startswith("[^")]
    assert defs, "expected at least one footnote-definition unit"
    assert all(not u.is_claim for u in defs)


def test_footnote_coverage_counts_only_real_claims():
    rep = grounding.check(FOOTNOTE_DOC)
    # 4 prose claims; 2 grounded by resolving footnotes, 2 not.
    assert rep.total_claims == 4
    assert rep.grounded_claims == 2


# `## What would update this page` is gate-exempt and blanked to end-of-file
# before unit splitting. Footnote definitions are conventionally placed at the
# very bottom of the page — i.e. *below* that heading — so their resolution must
# read the raw text, not the blanked copy, or every footnote-only claim silently
# flags ungrounded.
FOOTNOTE_BELOW_EXEMPT_DOC = """---
type: concept
---

## Definition

Retrieval-augmented generation grounds output in a retrieved corpus.[^rag]

## What would update this page

A non-text retrieval paper would extend the concept.

[^rag]: [[ai/gutierrez-2024-hipporag-neurobiologically-inspired-long-term-memory]]
"""


def test_footnote_definition_below_exempt_section_still_grounds():
    rep = grounding.check(FOOTNOTE_BELOW_EXEMPT_DOC)
    assert _unit_starting(rep, "Retrieval-augmented").has_citation


# ---------- plain wikilink still grounds (no regression) ----------

def test_inline_wikilink_grounds():
    doc = "A factual paragraph long enough to be a real claim here [[ai/foo-2025-bar]].\n"
    rep = grounding.check(doc)
    assert _unit_starting(rep, "A factual").has_citation


# ---------- line-offset fidelity (regression) ----------

def test_annotate_marks_original_line_not_frontmatter():
    doc = (
        "---\n"
        "title: T\n"
        "type: idea\n"
        "tags: [a, b]\n"
        "---\n"
        "\n"
        "This paragraph makes a factual claim about scaling laws with no citation at all.\n"
        "\n"
        "This one is grounded by a link [[ai/foo-2025-bar]] and should never be flagged.\n"
    )
    annotated = grounding.annotate(doc)
    marked = [i for i, ln in enumerate(annotated.splitlines(), start=1)
              if "ungrounded" in ln]
    # The only ungrounded claim is the original line 7; frontmatter must be untouched.
    assert marked == [7]
    assert "title: T" not in annotated.splitlines()[marked[0] - 1]


def test_fenced_code_does_not_shift_line_numbers():
    doc = (
        "Intro paragraph that is itself a grounded claim here [[ai/x-2025-y]] for sure.\n"
        "\n"
        "```\n"
        "code block line one\n"
        "code block line two\n"
        "```\n"
        "\n"
        "Ungrounded claim paragraph that comes after the fenced code block and is long.\n"
    )
    annotated = grounding.annotate(doc)
    marked = [i for i, ln in enumerate(annotated.splitlines(), start=1)
              if "ungrounded" in ln]
    assert marked == [8]


def test_what_would_update_section_skipped():
    """Forward-looking gap statements under `## What would update this page`
    describe papers the wiki *doesn't yet have*, so they can't carry a citation.
    The section is exempt from the grounding gate — its bullets must not
    appear as ungrounded units (or as units at all) in the report.
    """
    doc = (
        "## Background\n"
        "\n"
        "A clear factual claim with a wikilink citation that supports it here in this longer paragraph [[compbio/foo-2024-bar]] for verification by the reader.\n"
        "\n"
        "## What would update this page\n"
        "\n"
        "- A trans-ancestry GWAS that enumerates lead variants beyond X, Y, and Z.\n"
        "- Polygenic-score papers translating these variants into clinical risk stratification.\n"
        "- Functional validation of the novel target nominated by paper foo-2024.\n"
    )
    report = grounding.check(doc)
    # Only the Background claim is a unit; the three "would update" bullets
    # are blanked out of processing entirely.
    assert report.total_claims == 1
    assert report.grounded_claims == 1
    assert not report.ungrounded_units, (
        f"unexpected ungrounded: {[u.text[:60] for u in report.ungrounded_units]}"
    )


def test_what_would_update_only_matches_the_exact_heading():
    """The exemption is narrow: only `## What would update this page` triggers
    it. A differently-worded heading (e.g. `## What's next`) does not."""
    doc = (
        "## What's next\n"
        "\n"
        "An uncited forward-looking claim that should still be flagged as ungrounded.\n"
    )
    report = grounding.check(doc)
    assert len(report.ungrounded_units) == 1


def test_html_comments_are_not_claim_units():
    """HTML comments are invisible in the rendered output and must not appear
    as ungrounded claim-units. Motivating case: `researchwiki synthesize`
    emits template comments (`<!-- claim_lookup(...) -->`, `<!-- no claims
    indexed -->`) that can survive into a committed page."""
    doc = (
        "<!-- claim_lookup('foo bar baz', k=10). Curate this section. -->\n"
        "\n"
        "<!-- no claims indexed for this paper -->\n"
        "\n"
        "A real grounded claim about something specific [[compbio/foo-2024-bar]] with enough words to count as a claim under the heuristic.\n"
    )
    report = grounding.check(doc)
    assert report.total_claims == 1
    assert not report.ungrounded_units


def test_inline_html_comment_in_paragraph_is_ignored():
    """A `<!-- TODO -->` marker inside an otherwise-grounded paragraph
    shouldn't change the grounding verdict."""
    doc = (
        "The paper makes a clear factual claim with a citation [[compbio/foo-2024-bar]] supporting the assertion. <!-- TODO: cross-check this number -->\n"
    )
    report = grounding.check(doc)
    assert report.total_claims == 1
    assert report.grounded_claims == 1


# ---------- three-category grounding for idea pages ----------

# Same fixture across tests: every section has one claim, one of which (in
# Opportunities) carries a `*(model prior)*` marker. H2 suffixes ("— why")
# exercise the `\b` match in `_PERMISSIVE_IDEA_SECTION_RE`.
IDEA_DOC = """---
type: idea
---

## Background — why

Background claim with enough words to count as a real claim with no citation.

## Opportunities — design

Opportunities claim with enough words to count as a real claim *(model prior)*.

## Plans — phases

Plans claim with enough words to count as a real claim with no citation.

## Caveats — risks

Caveats claim with enough words to count as a real claim with no citation.
"""


def test_marker_grounds_only_in_opportunities_plans():
    """In default mode on an idea page, `*(model prior)*` grounds the claim
    iff it's inside Opportunities or Plans. Background and Caveats stay strict."""
    rep = grounding.check(IDEA_DOC, permissive=True)
    assert rep.total_claims == 4
    # Opportunities marker takes effect → 1 model_prior
    assert rep.model_prior_claims == 1
    # Background, Plans, Caveats all unmarked → 3 ungrounded
    assert len(rep.ungrounded_units) == 3
    assert rep.grounded_claims == 0


def test_strict_collapses_marker_into_ungrounded():
    """Strict mode ignores the marker — every claim needs a real citation."""
    rep = grounding.check(IDEA_DOC, permissive=False)
    assert rep.total_claims == 4
    assert rep.model_prior_claims == 0
    assert len(rep.ungrounded_units) == 4   # marker no longer exempts


def test_marker_in_background_does_not_ground():
    """Caveats/Background shouldn't accept the marker even in default mode."""
    bad = IDEA_DOC.replace(
        "Background claim with enough words to count as a real claim with no citation.",
        "Background claim with enough words to count as a real claim *(model prior)*.",
    )
    rep = grounding.check(bad, permissive=True)
    # Background's marker is ignored; only Opportunities' marker counts.
    assert rep.model_prior_claims == 1
    assert len(rep.ungrounded_units) == 3


def test_marker_no_effect_on_non_idea_pages():
    """The marker only acts as a citation on idea pages."""
    synth = IDEA_DOC.replace("type: idea", "type: synthesis")
    rep_perm = grounding.check(synth, permissive=True)
    rep_strict = grounding.check(synth, permissive=False)
    assert rep_perm.model_prior_claims == 0
    assert rep_strict.model_prior_claims == 0
    # Marker just becomes prose noise; that unit is still ungrounded.
    assert len(rep_perm.ungrounded_units) == 4
    assert len(rep_strict.ungrounded_units) == 4


def test_wiki_citation_takes_precedence_over_marker():
    """A unit with both a wiki citation AND the marker is `grounded`,
    not `model_prior` — wiki always wins."""
    doc = """---
type: idea
---

## Opportunities — design

Mixed claim with enough words and both [[ai/foo-2025-bar]] and *(model prior)*.
"""
    rep = grounding.check(doc, permissive=True)
    u = next(u for u in rep.units if u.is_claim)
    assert u.has_citation
    assert not u.is_model_prior
    assert rep.grounded_claims == 1
    assert rep.model_prior_claims == 0


def test_quoted_type_idea_still_detected():
    """`type: "idea"` and `type: 'idea'` must still trigger permissive mode —
    YAML allows quoted scalars, so a defensive author shouldn't lose the gate."""
    for variant in ('type: "idea"', "type: 'idea'"):
        doc = IDEA_DOC.replace("type: idea", variant)
        rep = grounding.check(doc, permissive=True)
        assert rep.model_prior_claims == 1, variant


# ---------- labelled Outlook on synthesis pages ----------

# One claim per section. Background stays strict; Outlook holds a cited
# inference, a model prior, and an uncited inference (which must fail).
SYNTH_DOC = """---
type: synthesis
---

## Background

Background claim with enough words to count as a real claim [[compbio/foo-2024-bar]].

## Outlook

Combining both assays implies a shared detection floor across the field *(inference)*.[^foo][^baz]

Long-read sequencing will probably be the next route around that floor *(model prior)*.

Another conclusion with enough words to count but no cited papers at all *(inference)*.

[^foo]: [[compbio/foo-2024-bar]]
[^baz]: [[compbio/baz-2025-qux]]
"""


def _claims(rep):
    return [u for u in rep.units if u.is_claim]


def test_synthesis_outlook_accepts_both_labels():
    """Outlook is the synthesis page's labelled section: a cited inference is
    grounded and flagged, a model prior is reported as a warning, and an
    inference with nothing cited stays ungrounded."""
    rep = grounding.check(SYNTH_DOC, permissive=True, valid_anchors=set())
    assert rep.total_claims == 4
    assert rep.inference_claims == 1
    assert rep.model_prior_claims == 1
    assert rep.grounded_claims == 2          # Background + the cited inference
    (bad,) = rep.ungrounded_units
    assert "no cited papers" in bad.text
    assert bad.flag_reason == "*(inference)* without the papers it is drawn from"


def test_inference_label_does_not_ground_without_a_citation():
    """The label relabels a grounded unit; it is never a citation by itself.
    Otherwise an author could exempt any Outlook sentence from both gates."""
    rep = grounding.check(SYNTH_DOC, permissive=True, valid_anchors=set())
    uncited = [u for u in _claims(rep) if "no cited papers" in u.text]
    assert uncited and not uncited[0].has_citation
    assert not uncited[0].is_inference


def test_labels_outside_outlook_have_no_effect_on_synthesis():
    """Background stays strictly grounded: a label there neither grounds the
    unit nor marks it as an inference."""
    doc = SYNTH_DOC.replace(
        "Background claim with enough words to count as a real claim [[compbio/foo-2024-bar]].",
        "Background claim with enough words to count as a real claim *(model prior)*.",
    )
    rep = grounding.check(doc, permissive=True, valid_anchors=set())
    background = next(u for u in _claims(rep) if u.text.startswith("Background"))
    assert not background.has_citation
    assert rep.model_prior_claims == 1      # only the Outlook one


def test_cited_inference_outside_outlook_is_an_ordinary_claim():
    """Outside a labelled section the label is prose: the unit is graded as a
    normal cited claim, so the fidelity gate still checks it."""
    doc = SYNTH_DOC.replace("## Outlook", "## Findings")
    rep = grounding.check(doc, permissive=True, valid_anchors=set())
    assert rep.inference_claims == 0
    assert rep.model_prior_claims == 0


def test_strict_ignores_both_labels_on_synthesis():
    """`--strict`: the model prior and the uncited inference both fail, and
    the cited inference is an ordinary cited claim."""
    rep = grounding.check(SYNTH_DOC, permissive=False, valid_anchors=set())
    assert rep.inference_claims == 0
    assert rep.model_prior_claims == 0
    assert len(rep.ungrounded_units) == 2
    assert rep.grounded_claims == 2


def test_outlook_is_not_labelled_on_idea_pages():
    """Each page type labels only its own sections: an idea page's Outlook
    heading (not part of the idea contract) stays strict."""
    doc = SYNTH_DOC.replace("type: synthesis", "type: idea")
    rep = grounding.check(doc, permissive=True, valid_anchors=set())
    assert rep.model_prior_claims == 0
    assert rep.inference_claims == 0


def test_idea_pages_accept_the_inference_label():
    """Opportunities/Plans take `*(inference)*` too, so the two labels mean
    the same thing on both page types."""
    doc = IDEA_DOC.replace(
        "Plans claim with enough words to count as a real claim with no citation.",
        "Plans claim with enough words to count as a real claim [[ai/foo-2025-bar]] *(inference)*.",
    )
    rep = grounding.check(doc, permissive=True, valid_anchors=set())
    assert rep.inference_claims == 1
    assert rep.model_prior_claims == 1


def test_model_prior_carrying_a_number_stays_ungrounded():
    """Review finding: `*(model prior)*` grounded a unit whatever it said, so an
    uncited "98% efficacy" passed check-grounding — and the fidelity gate skips
    an uncited unit, so a fabricated figure cleared both. CLAUDE.md §4 is
    explicit that numbers always need a paper citation, so the marker grounds a
    qualitative forecast only."""
    doc = SYNTH_DOC.replace(
        "Long-read sequencing will probably be the next route around that floor *(model prior)*.",
        "Long-read assays will likely reach 98% efficacy at detecting these sites *(model prior)*.",
    )
    rep = grounding.check(doc, permissive=True, valid_anchors=set())
    numeric = next(u for u in _claims(rep) if "98%" in u.text)
    assert not numeric.has_citation
    assert not numeric.is_model_prior
    assert "carrying a number" in (numeric.flag_reason or "")
    assert numeric in rep.ungrounded_units


def test_qualitative_model_prior_still_grounds():
    """The counterpart: a forecast with no quantity is exactly what the label
    is for, and must not be collateral damage."""
    doc = SYNTH_DOC.replace(
        "Long-read sequencing will probably be the next route around that floor *(model prior)*.",
        "Long-read assays are likely to become the standard route around that floor *(model prior)*.",
    )
    rep = grounding.check(doc, permissive=True, valid_anchors=set())
    assert rep.model_prior_claims == 1


def test_digits_inside_a_name_are_not_a_quantity():
    """`Cas9`, `ABE8e`, `p53` must not count as numbers, or the rule would
    forbid naming an editor in a model prior. Shares the fidelity grader's
    tokenizer, whose lookbehind already excludes word-internal digits."""
    doc = SYNTH_DOC.replace(
        "Long-read sequencing will probably be the next route around that floor *(model prior)*.",
        "A Cas9 variant engineered on ABE8e scaffolds will probably solve this *(model prior)*.",
    )
    rep = grounding.check(doc, permissive=True, valid_anchors=set())
    assert rep.model_prior_claims == 1


def test_idea_design_parameters_may_carry_numbers_under_the_marker():
    """The numeric rule is synthesis-only, and the asymmetry is deliberate. An
    idea page's Opportunities numbers are parameters the author is *choosing*
    ("enumeration cap of 64", "be-window 4-8"), not results being asserted —
    proposing them is what the section is for. Applying the synthesis rule here
    flagged 12 units across 5 existing idea pages, every one a parameter."""
    doc = IDEA_DOC.replace(
        "Opportunities claim with enough words to count as a real claim *(model prior)*.",
        "Use a tied-path enumeration cap of 64 with a deterministic tiebreaker *(model prior)*.",
    )
    rep = grounding.check(doc, permissive=True, valid_anchors=set())
    assert rep.model_prior_claims == 1
    # IDEA_DOC's Background/Plans/Caveats claims are deliberately uncited, so
    # assert about the parameter unit itself rather than the whole report.
    param = next(u for u in _claims(rep) if "cap of 64" in u.text)
    assert param.is_model_prior and param.flag_reason is None
