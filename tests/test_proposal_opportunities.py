"""Proposal opportunities: surfacing proposals to users who never ask for them.

Proposals only run on request, so a user who has not heard of the command never
gets one. These pin the local detector that finds proposal-shaped signals in the
corpus and the places it speaks up (`status`, ingest, the CLI), and — the part
that matters most — that finding them never costs a model call.

Hermetic: every source is faked; no claim graph, index, DB or model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from researchwiki import proposal_opportunities as po


def _edge(src, tgt, relation="builds_on"):
    return SimpleNamespace(src_stem=src, tgt_stem=tgt, relation=relation)


def _cluster(members, keywords, verdict="new"):
    return SimpleNamespace(members=[f"cat/{m}" for m in members],
                           common_keywords=keywords, verdict=verdict)


def _pair(a, b, terms, cat_a="genomics", cat_b="cgt"):
    return SimpleNamespace(stem_a=a, stem_b=b, category_a=cat_a, category_b=cat_b,
                           shared_terms=terms)


@pytest.fixture
def sources(monkeypatch):
    """Fake the three local sources; tests fill them in."""
    state = {"edges": [], "clusters": [], "pairs": [], "explored": []}
    monkeypatch.setattr(po, "_live_edges", lambda: state["edges"])
    monkeypatch.setattr(po, "_explored_paper_sets", lambda: state["explored"])

    import researchwiki.synthesis_candidates.detect as detect
    import researchwiki.tasks.claim_discover as discover
    monkeypatch.setattr(detect, "find_candidates", lambda: (state["clusters"], {}))
    monkeypatch.setattr(discover, "discover_pairs", lambda **kw: state["pairs"])
    return state


def test_no_model_call_is_ever_made(sources, monkeypatch):
    """The whole point of the detector: it is free. Every surface that runs it
    (`status` on every invocation, ingest, the chat agent) depends on that."""
    from researchwiki.agents import llm
    monkeypatch.setattr(llm, "call", lambda **kw: pytest.fail("detector called a model"))
    sources["edges"] = [_edge("a-2020-x", "b-2021-y", "contradicts")]
    sources["clusters"] = [_cluster(["c1", "c2", "c3"], ["false discovery rate"])]
    sources["pairs"] = [_pair("p-2020-a", "q-2021-b", ["atac-seq", "chip-seq"])]
    assert po.find_opportunities()


def test_contradictions_rank_first(sources):
    sources["pairs"] = [_pair("p-2020-a", "q-2021-b", ["atac-seq"])]
    sources["clusters"] = [_cluster(["c1", "c2", "c3"], ["false discovery rate"])]
    sources["edges"] = [_edge("x-2020-a", "y-2021-b"),
                        _edge("a-2020-x", "b-2021-y", "contradicts")]
    kinds = [o.kind for o in po.find_opportunities()]
    assert kinds == ["tension", "build-on", "uncovered-cluster", "bridge"]


def test_edge_chains_collapse_into_one_opportunity(sources):
    """Five papers that refine one another are one line of work, not four
    near-duplicate questions."""
    sources["edges"] = [
        _edge("jiang-2026-a", "lucas-2026-b", "refines"),
        _edge("jiang-2026-a", "ashraf-2026-c", "refines"),
        _edge("zheng-2026-d", "lucas-2026-b", "refines"),
        _edge("zheng-2026-d", "yi-2026-e", "refines"),
    ]
    opps = po.find_opportunities()
    assert len(opps) == 1
    assert sorted(opps[0].papers) == sorted(
        ["jiang-2026-a", "lucas-2026-b", "ashraf-2026-c", "zheng-2026-d", "yi-2026-e"])
    assert opps[0].papers[0] in {"jiang-2026-a", "lucas-2026-b", "zheng-2026-d"}


def test_a_group_with_a_contradiction_is_a_tension(sources):
    sources["edges"] = [_edge("a-2020-x", "b-2021-y"),
                        _edge("b-2021-y", "c-2022-z", "contradicts")]
    (opp,) = po.find_opportunities()
    assert opp.kind == "tension"
    assert "B 2021" in opp.why and "C 2022" in opp.why


@pytest.mark.parametrize("size, kept", [(2, False), (3, True), (40, True), (41, False)])
def test_clusters_are_bounded_to_a_single_page_question(sources, size, kept):
    """A 78-paper cluster is a field survey the eight-paper packet cannot hold."""
    sources["clusters"] = [_cluster([f"m{i}" for i in range(size)], ["base editing"])]
    assert bool(po.find_opportunities()) is kept


def test_extend_clusters_and_keywordless_clusters_are_skipped(sources):
    sources["clusters"] = [_cluster(["a", "b", "c"], ["base editing"], verdict="extend"),
                           _cluster(["d", "e", "f"], [])]
    assert po.find_opportunities() == []


def test_bridges_need_a_named_method_not_shared_register(sources):
    """`lifestyle`, `translating`, `guidance` are shared vocabulary; a transfer
    needs something to transfer, like `needleman-wunsch` or `atac-seq`."""
    sources["pairs"] = [
        _pair("a-2020-x", "b-2021-y", ["lifestyle", "statins", "incorporating"]),
        _pair("c-2020-x", "d-2021-y", ["needleman-wunsch", "smith-waterman"]),
    ]
    (opp,) = po.find_opportunities()
    assert "needleman-wunsch" in opp.why


def test_corpus_wide_jargon_does_not_make_a_bridge(sources):
    many = [_pair(f"a{i}-2020-x", f"b{i}-2021-y", ["fine-tuning"]) for i in range(30)]
    sources["pairs"] = many + [_pair("c-2020-x", "d-2021-y", ["fine-tuning", "atac-seq"])]
    opps = po.find_opportunities()
    assert [o.papers for o in opps] == [["d-2021-y"]]


def test_bridge_commands_pin_only_target_category_papers(sources):
    """Cross-category generation rejects any pinned paper outside the target
    category, so pinning both sides made every bridge command fail."""
    sources["pairs"] = [_pair("src-2020-x", "tgt-2021-y", ["atac-seq"],
                              cat_a="genomics", cat_b="cgt")]
    (opp,) = po.find_opportunities()
    assert opp.papers == ["tgt-2021-y"] and opp.related == ["src-2020-x"]
    assert "--target-category cgt --cross-category" in opp.command()
    assert "src-2020-x" not in opp.command()


def test_an_existing_proposal_suppresses_its_opportunity(sources):
    sources["edges"] = [_edge("a-2020-x", "b-2021-y")]
    sources["explored"] = [frozenset({"a-2020-x", "b-2021-y", "z-2019-q"})]
    assert po.find_opportunities() == []


def test_one_signal_cannot_crowd_out_the_others(sources):
    sources["edges"] = [_edge(f"a{i}-2020-x", f"b{i}-2021-y") for i in range(12)]
    sources["clusters"] = [_cluster(["c1", "c2", "c3"], ["false discovery rate"])]
    kinds = [o.kind for o in po.find_opportunities(limit=20)]
    assert kinds.count("build-on") == po.PER_KIND_CAP
    assert "uncovered-cluster" in kinds


def test_commands_quote_the_topic_for_the_shell(sources):
    opp = po.Opportunity(kind="tension", topic='Is "X" real?', why="w",
                         papers=["a-2020-x", "b-2021-y"])
    assert opp.command() == (
        'researchwiki proposals generate "Is \\"X\\" real?" --papers a-2020-x b-2021-y')


def test_a_failing_source_yields_fewer_opportunities_not_an_error(sources, monkeypatch):
    import researchwiki.synthesis_candidates.detect as detect
    monkeypatch.setattr(detect, "find_candidates",
                        lambda: (_ for _ in ()).throw(RuntimeError("index missing")))
    sources["edges"] = [_edge("a-2020-x", "b-2021-y")]
    assert [o.kind for o in po.find_opportunities()] == ["build-on"]


def test_paper_hint_matches_involved_papers_including_bridge_sources(sources):
    sources["edges"] = [_edge("a-2020-x", "b-2021-y")]
    assert [o.kind for o in po.opportunities_for_paper("b-2021-y")] == ["build-on"]
    assert po.opportunities_for_paper("unrelated-2020-z") == []


# ---------- the status nudge ----------


@pytest.fixture
def stamp(tmp_path, monkeypatch):
    path = tmp_path / ".proposal-opportunity-stamp"
    monkeypatch.setattr(po, "_stamp_path", lambda: path)
    return path


def test_status_line_names_counts_and_the_top_command(sources, stamp):
    sources["edges"] = [_edge("a-2020-x", "b-2021-y", "contradicts"),
                        _edge("c-2020-x", "d-2021-y")]
    msg = po.opportunity_warning()
    assert msg.startswith("Proposal opportunities: 1 tension, 1 build-on")
    assert "researchwiki proposals generate" in msg
    assert "researchwiki proposals opportunities" in msg
    assert stamp.exists()


@pytest.mark.parametrize("surface", ["status", "ingest"])
def test_latency_bound_surfaces_never_run_the_slow_scans(sources, stamp, monkeypatch, surface):
    """The cluster scan (~0.7 s) and the claim-pair scan (~2 s) took `status`
    from ~2.5 s to ~6 s. Only the dedicated command may pay for them."""
    import researchwiki.synthesis_candidates.detect as detect
    import researchwiki.tasks.claim_discover as discover
    monkeypatch.setattr(detect, "find_candidates",
                        lambda: pytest.fail("cluster scan on a latency-bound surface"))
    monkeypatch.setattr(discover, "discover_pairs",
                        lambda **kw: pytest.fail("pair scan on a latency-bound surface"))
    sources["edges"] = [_edge("a-2020-x", "b-2021-y")]
    if surface == "status":
        assert po.opportunity_warning() is not None
    else:
        assert po.opportunities_for_paper("a-2020-x")


def test_status_stays_silent_on_cluster_and_bridge_signals_alone(sources, stamp):
    """Those are listed by `proposals opportunities`; the status line does not
    pay to discover them."""
    sources["clusters"] = [_cluster(["c1", "c2", "c3"], ["false discovery rate"])]
    sources["pairs"] = [_pair("p-2020-a", "q-2021-b", ["atac-seq"])]
    assert po.opportunity_warning() is None
    assert not stamp.exists()


def test_status_line_goes_quiet_for_the_decay_window(sources, stamp):
    sources["edges"] = [_edge("a-2020-x", "b-2021-y")]
    assert po.opportunity_warning() is not None
    assert po.opportunity_warning() is None


def test_status_line_checks_the_stamp_before_scanning(sources, stamp, monkeypatch):
    """The scan is the slowest thing `status` would run; a quiet period must not
    pay for it on every invocation."""
    stamp.write_text("9999999999")
    monkeypatch.setattr(po, "find_opportunities",
                        lambda **kw: pytest.fail("scanned during the quiet period"))
    assert po.opportunity_warning() is None


def test_peek_does_not_start_the_quiet_period(sources, stamp):
    sources["edges"] = [_edge("a-2020-x", "b-2021-y")]
    assert po.opportunity_warning(touch=False) is not None
    assert not stamp.exists()


def test_silent_when_the_corpus_has_nothing(sources, stamp):
    assert po.opportunity_warning() is None
    assert not stamp.exists()


# ---------- the CLI ----------


def test_opportunities_cli_json_is_parseable_and_carries_commands(sources, capsys):
    from researchwiki.tasks import proposals as cli
    sources["edges"] = [_edge("a-2020-x", "b-2021-y", "contradicts")]
    assert cli.main(["opportunities", "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["kind"] == "tension" and row["rank"] == 0
    assert row["command"].startswith("researchwiki proposals generate")
    assert set(row) >= {"kind", "topic", "why", "papers", "command"}


def test_opportunities_cli_explains_an_empty_result(sources, capsys):
    from researchwiki.tasks import proposals as cli
    assert cli.main(["opportunities"]) == 0
    assert "No proposal opportunities found" in capsys.readouterr().out


def test_ingest_hint_prints_one_opportunity_and_never_raises(sources, capsys, monkeypatch):
    from researchwiki.tasks import agent
    sources["edges"] = [_edge("new-2026-x", "old-2020-y")]
    agent._print_proposal_hint("new-2026-x")
    out = capsys.readouterr().out
    assert out.count("Proposal opportunity:") == 1
    assert "researchwiki proposals generate" in out

    monkeypatch.setattr(po, "opportunities_for_paper",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    agent._print_proposal_hint("new-2026-x")          # must not raise
    agent._print_proposal_hint("unrelated-2020-z")
