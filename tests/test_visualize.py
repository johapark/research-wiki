"""The `visualize` graph builder.

What is worth pinning here is the set of things that make the picture *lie* if
they regress, because a wrong graph looks exactly as convincing as a right one:

  - `index.md` must never become a node. It links to every page, so including it
    makes one node adjacent to all others and the layout collapses to a star.
  - `instantiates` edges carry the literal `concepts` in `tgt_stem`, not a paper
    stem. Resolving that as a stem silently drops every concept spoke.
  - `stale` claim edges are excluded by default, because a relation nobody
    currently stands behind must not be drawn like a live one. (Not a volume
    argument: parallel pairs collapse per page pair, so 13,535 stale edges on one
    real corpus amount to only ~20 more visible strands.)
  - An edge whose endpoint has no page is *counted*, not silently skipped —
    otherwise a partial graph reports itself as complete.

Hermetic: tmp wiki, tmp claim DB, no network, no LLM.
"""

from __future__ import annotations

import json

import pytest

from researchwiki import paths, visualize
from researchwiki.claim_graph.edges import Edge, open_edges_db, upsert_edge


def _page(root, cat, stem, body, **fm):
    d = root / "wiki" / cat
    d.mkdir(parents=True, exist_ok=True)
    front = "".join(f"{k}: {v}\n" for k, v in fm.items())
    (d / f"{stem}.md").write_text(f"---\n{front}---\n\n{body}\n", encoding="utf-8")


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    _page(tmp_path, "compbio", "smith-2024-a-paper-about-things",
          "## Summary\nLinks to [[compbio/jones-2025-another-paper]] and [[concepts/pangenome]].",
          type="paper", title="A paper about things", year=2024, short_name="Smith 2024")
    _page(tmp_path, "compbio", "jones-2025-another-paper",
          # bare-stem link: Obsidian resolves it, so the graph must too
          "## Summary\nSee [[smith-2024-a-paper-about-things]] and itself [[compbio/jones-2025-another-paper]].",
          type="paper", title="Another paper", year=2025)
    _page(tmp_path, "concepts", "pangenome", "## Definition\nA thing.", type="concept",
          title="Pangenome")
    _page(tmp_path, "synthesis", "field-map", "## Question\nWhat?", type="synthesis",
          title="A field map")
    # Bookkeeping page WITH frontmatter — the guard must hold even when
    # `read_pages` would happily parse it.
    (tmp_path / "wiki" / "index.md").write_text(
        "---\ntype: index\n---\n\n- [[compbio/smith-2024-a-paper-about-things]]\n"
        "- [[compbio/jones-2025-another-paper]]\n- [[concepts/pangenome]]\n",
        encoding="utf-8")
    return tmp_path


def _edge(**kw):
    kw.setdefault("slug_scheme_version", 1)
    return Edge(**kw)


def _seed_claims(root, edges):
    (root / ".claim-graph").mkdir(parents=True, exist_ok=True)
    conn = open_edges_db(root / ".claim-graph" / "edges.db")
    try:
        for e in edges:
            upsert_edge(conn, e)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------- nodes

def test_index_and_log_are_not_nodes(wiki):
    g = visualize.build_graph()
    ids = {n["id"] for n in g.nodes}
    assert "wiki/index" not in ids and "index" not in ids
    assert ids == {
        "compbio/smith-2024-a-paper-about-things",
        "compbio/jones-2025-another-paper",
        "concepts/pangenome",
        "synthesis/field-map",
    }


def test_short_name_preferred_as_label_then_author_year(wiki):
    by = {n["id"]: n for n in visualize.build_graph().nodes}
    assert by["compbio/smith-2024-a-paper-about-things"]["label"] == "Smith 2024"
    # no short_name -> derived from the stem, not the whole 40-char stem
    assert by["compbio/jones-2025-another-paper"]["label"] == "Jones 2025"


def test_node_carries_type_category_and_degree(wiki):
    by = {n["id"]: n for n in visualize.build_graph().nodes}
    n = by["compbio/smith-2024-a-paper-about-things"]
    assert n["type"] == "paper" and n["category"] == "compbio" and n["year"] == 2024
    assert n["deg"] >= 2


# ---------------------------------------------------------------- wikilinks

def test_bare_stem_links_resolve_and_self_links_are_dropped(wiki):
    g = visualize.build_graph()
    wl = {(e["source"], e["target"]) for e in g.edges if e["kind"] == "wikilink"}
    # bare `[[smith-...]]` from jones resolved to the category-qualified key
    assert ("compbio/jones-2025-another-paper",
            "compbio/smith-2024-a-paper-about-things") in wl
    # jones links to itself; that must not become an edge
    assert not any(s == t for s, t in wl)


def test_wikilink_edges_are_deduped(wiki, tmp_path):
    _page(tmp_path, "compbio", "dup-2020-a-paper",
          "## Summary\n[[concepts/pangenome]] again [[concepts/pangenome]] and [[concepts/pangenome]].",
          type="paper", title="Dup")
    g = visualize.build_graph()
    pairs = [(e["source"], e["target"]) for e in g.edges if e["kind"] == "wikilink"]
    assert pairs.count(("compbio/dup-2020-a-paper", "concepts/pangenome")) == 1


# ---------------------------------------------------------------- claim edges

def test_instantiates_resolves_the_concepts_sentinel(wiki):
    _seed_claims(wiki, [_edge(
        src_stem="smith-2024-a-paper-about-things", src_slug="kc-1111",
        tgt_stem="concepts", tgt_slug="pangenome",
        relation="instantiates", status="confirmed", directed=True)])
    g = visualize.build_graph()
    claims = [e for e in g.edges if e["kind"] == "claim"]
    assert len(claims) == 1
    assert claims[0]["source"] == "compbio/smith-2024-a-paper-about-things"
    # resolved to the concept PAGE, not a paper stem named "concepts"
    assert claims[0]["target"] == "concepts/pangenome"


def test_stale_edges_excluded_by_default_and_includable_on_request(wiki):
    _seed_claims(wiki, [
        _edge(src_stem="smith-2024-a-paper-about-things", src_slug="kc-1",
              tgt_stem="jones-2025-another-paper", tgt_slug="kc-2",
              relation="builds_on", status="stale"),
        _edge(src_stem="smith-2024-a-paper-about-things", src_slug="kc-3",
              tgt_stem="jones-2025-another-paper", tgt_slug="kc-4",
              relation="corroborates", status="candidate"),
    ])
    default = visualize.build_graph()
    assert {e["relation"] for e in default.edges if e["kind"] == "claim"} == {"corroborates"}

    widened = visualize.build_graph(claim_statuses=("candidate", "stale"))
    assert {e["relation"] for e in widened.edges if e["kind"] == "claim"} == {
        "builds_on", "corroborates"}


def test_edges_with_no_page_are_counted_not_silently_dropped(wiki):
    _seed_claims(wiki, [_edge(
        src_stem="smith-2024-a-paper-about-things", src_slug="kc-1",
        tgt_stem="ghost-2019-removed-from-the-wiki", tgt_slug="kc-2",
        relation="builds_on", status="candidate")])
    g = visualize.build_graph()
    assert [e for e in g.edges if e["kind"] == "claim"] == []
    assert g.meta["claim_edges_dropped"] == 1


def test_parallel_claim_pairs_collapse_to_one_edge_with_a_count(wiki):
    _seed_claims(wiki, [
        _edge(src_stem="smith-2024-a-paper-about-things", src_slug=f"kc-{i}",
              tgt_stem="jones-2025-another-paper", tgt_slug=f"kc-t{i}",
              relation="measures_same", status="candidate", confidence=0.5 + i / 10)
        for i in range(3)
    ])
    g = visualize.build_graph()
    claims = [e for e in g.edges if e["kind"] == "claim"]
    assert len(claims) == 1
    assert claims[0]["n"] == 3
    assert claims[0]["confidence"] == pytest.approx(0.7)   # strongest wins


def test_the_strongest_live_status_wins_when_edges_collapse(wiki):
    """`promoted` outranks `confirmed` outranks `candidate`. The first version
    only special-cased `confirmed`, so a promoted edge collapsed with a
    candidate displayed as candidate."""
    _seed_claims(wiki, [
        _edge(src_stem="smith-2024-a-paper-about-things", src_slug="kc-1",
              tgt_stem="jones-2025-another-paper", tgt_slug="kc-t1",
              relation="builds_on", status="promoted"),
        _edge(src_stem="smith-2024-a-paper-about-things", src_slug="kc-2",
              tgt_stem="jones-2025-another-paper", tgt_slug="kc-t2",
              relation="builds_on", status="candidate"),
        _edge(src_stem="smith-2024-a-paper-about-things", src_slug="kc-3",
              tgt_stem="jones-2025-another-paper", tgt_slug="kc-t3",
              relation="builds_on", status="confirmed"),
    ])
    g = visualize.build_graph()
    claims = [e for e in g.edges if e["kind"] == "claim"]
    assert len(claims) == 1
    assert claims[0]["status"] == "promoted"
    assert claims[0]["n"] == 3


def test_no_claim_db_is_not_an_error_and_creates_nothing(wiki):
    g = visualize.build_graph()
    assert g.meta["n_claim_edges"] == 0
    # A *view* command must not bring a database into being as a side effect.
    assert not (wiki / ".claim-graph").exists()


def test_include_claims_false_skips_the_db_entirely(wiki):
    _seed_claims(wiki, [_edge(
        src_stem="smith-2024-a-paper-about-things", src_slug="kc-1",
        tgt_stem="jones-2025-another-paper", tgt_slug="kc-2",
        relation="builds_on", status="candidate")])
    g = visualize.build_graph(include_claims=False)
    assert g.meta["n_claim_edges"] == 0
    assert g.meta["claim_statuses"] == []


# ---------------------------------------------------------------- rendering

def test_render_substitutes_the_placeholder(wiki):
    html = visualize.render_html(visualize.build_graph())
    assert "__GRAPH_DATA__" not in html
    assert "const DATA = {" in html


def test_render_escapes_script_close_in_page_text(wiki, tmp_path):
    # A title containing `</script>` would otherwise terminate the block and
    # blank the whole page.
    _page(tmp_path, "compbio", "evil-2020-a-paper",
          "## Summary\nnothing", type="paper", title='"a </script> title"')
    html = visualize.render_html(visualize.build_graph())
    payload = next(l for l in html.splitlines() if l.startswith("const DATA = "))
    assert "</script>" not in payload
    assert "<\\/script>" in payload
    # exactly one real closing tag, the template's own
    assert html.count("</script>") == 1


def test_empty_wiki_renders_a_valid_page(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    (tmp_path / "wiki").mkdir()
    g = visualize.build_graph()
    assert g.nodes == [] and g.edges == []
    html = visualize.render_html(g)
    assert "__GRAPH_DATA__" not in html and "<canvas" in html


# ---------------------------------------------------------------- CLI

def test_cli_writes_html_and_exits_zero(wiki, monkeypatch, capsys):
    from researchwiki.tasks import visualize as cli
    monkeypatch.chdir(wiki)
    out = wiki / "out" / "g.html"
    assert cli.main(["-o", str(out)]) == 0
    assert out.exists() and "<canvas" in out.read_text(encoding="utf-8")
    assert "4 pages" in capsys.readouterr().out


def test_cli_json_mode_emits_the_graph(wiki, monkeypatch, capsys):
    from researchwiki.tasks import visualize as cli
    monkeypatch.chdir(wiki)
    assert cli.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"nodes", "edges", "meta"}
    assert payload["meta"]["n_nodes"] == 4


def test_cli_missing_wiki_is_an_environment_error(tmp_path, monkeypatch):
    from researchwiki.tasks import visualize as cli
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cli.main([]) == 2


# ---------------------------------------------------------------- template invariants
#
# The renderer is JavaScript in an HTML template, so pytest cannot drive it. These
# are source-level assertions, which is weak — but they pin the two specific
# regressions that were shipped once, and a source check that names the bug beats
# no check at all.

def _template() -> str:
    from importlib import resources
    return (resources.files("researchwiki")
            .joinpath("templates/graph.html").read_text(encoding="utf-8"))


def _mousemove_handler(src: str) -> str:
    start = src.index("cv.addEventListener('mousemove'")
    return src[start:src.index("cv.addEventListener('mousedown'", start)]


def _code_only(js: str) -> str:
    """Drop `//` comments, so an assertion about what the code *calls* isn't
    satisfied (or broken) by a comment explaining what it deliberately doesn't."""
    import re
    return "\n".join(re.sub(r"//.*$", "", line) for line in js.splitlines())


def test_dragging_does_not_reheat_the_whole_layout():
    """The shake. A press used to become a drag instantly, so one pixel of hand
    jitter called `kick()` (alpha 0.55) and re-annealed a settled 477-node layout
    — clicking a node to read its panel visibly threw the graph around. The drag
    path must use the gentle `nudge()` instead, and only one node moved so there is
    nothing a full re-heat could correctly fix."""
    handler = _code_only(_mousemove_handler(_template()))
    assert "nudge()" in handler
    assert "kick()" not in handler, "the drag path must not re-heat the full layout"


def test_a_press_is_not_a_drag_until_it_moves():
    """Same bug, other half: a bare click must not pin the node it landed on.
    Pinning is what a *drag* means, and a click that nails a node down leaves the
    layout re-settling around it for reasons the user never asked for."""
    src = _template()
    assert "DRAG_SLOP" in src
    handler = _mousemove_handler(src)
    # The pin happens on the resolved-to-drag transition, inside mousemove, not on
    # mousedown where a click is indistinguishable from a press.
    assert "pinned = true" in handler
    mousedown = src[src.index("cv.addEventListener('mousedown'"):
                    src.index("window.addEventListener('mouseup'")]
    assert "pinned = true" not in mousedown


def _alpha_floor(src: str, fn: str) -> float:
    import re
    m = re.search(rf"function {fn}\(\)\{{ alpha = Math\.max\(alpha, ([\d.]+)\); \}}", src)
    assert m, f"could not find {fn}()'s alpha floor"
    return float(m.group(1))


def test_the_two_reheat_levels_stay_distinct_and_ordered():
    """`kick()` is for a change that invalidates every position; `nudge()` is for
    one node moving. Collapsing them re-introduces the shake.

    Asserts the *relationship*, not the constants — an earlier version of this test
    pinned `0.55` literally and broke the moment the layout was tuned calmer, which
    told us nothing about whether the distinction still held.
    """
    src = _template()
    kick, nudge = _alpha_floor(src, "kick"), _alpha_floor(src, "nudge")
    assert 0 < nudge < kick <= 1.0
    assert kick / nudge >= 2, "the levels should differ by enough to be worth having"


def test_the_hot_phase_runs_before_the_first_paint():
    """The layout's opening frames are its most violent. They are burned in
    `warmup()` off-screen, so a viewer never sees them — measured at 27px of
    per-tick node movement before this, under 1px after."""
    src = _template()
    assert "function warmup()" in src and "WARMUP_TICKS" in src
    boot = src[src.index("resize(); applyFilters();"):]
    assert boot.index("warmup();") < boot.index("frame();"), (
        "warmup must run before the render loop starts, or the point is lost"
    )


def test_reset_view_does_not_restart_the_layout():
    """It is a *camera* action. It used to set `alpha = 1`, which made it the most
    violent thing in the UI — worse than first load."""
    src = _template()
    handler = _code_only(src[src.index("getElementById('reset').onclick"):][:600])
    assert "fitView()" in handler
    assert "alpha = 1" not in handler
    assert "kick()" not in handler
