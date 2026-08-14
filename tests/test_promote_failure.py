"""A promote that doesn't complete must not report success.

`promote_to_wiki` is five multi-file steps with no transaction: the page and
its DB row land first, then the PDF move, back-links, `index.md` and `log.md`.
When a later step fails it returns `promoted=False` rather than raising — and
until WI-1 nothing read that flag. The commit phase fell straight through to
`decision = "committed-to-wiki"`, so the reachable case (a duplicate PDF, where
`_move_pdf` refuses a stem collision that isn't a journal upgrade) left a page
on disk with no PDF, no back-links, no index bullet and no log entry, while the
process exited 0 and the batch checkpoint recorded `completed`.

These pin the exit code, because the exit code is the part that was wrong.

Hermetic: `_phase_commit` is driven directly with stubbed collaborators — no
LLM, no PDF, no state.db.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.agents import promote as promote_mod
from researchwiki.agents import runner
from researchwiki.agents.context import Context, PromoteFailed


class _Draft:
    """Minimal stand-in for the tournament winner."""
    iteration_id = 7
    text = "## Summary\nbody\n"
    scores: dict = {}
    model = "test-model"
    handle = "Smith 2024"
    hook = "Does a thing, 2x faster."


@pytest.fixture
def ctx(tmp_path):
    c = Context(attempt_id="a1", pdf_path=tmp_path / "in.pdf", pdf_filename="in.pdf")
    c.paper_stem = "smith-2024-a-paper-about-things"
    c.metadata = {"title": "A paper about things", "paper_type": "research"}
    c.winner = _Draft()
    c.crosslink_candidates = []
    c.promote_mode = "always"
    c.sandbox_dir = tmp_path / "sandbox"
    return c


@pytest.fixture
def commit_env(monkeypatch, tmp_path):
    """Stub everything `_phase_commit` touches except the code under test."""
    rows: list[dict] = []
    monkeypatch.setattr(runner, "write_iteration",
                        lambda **kw: rows.append(kw) or 1)
    monkeypatch.setattr(runner.phases, "verify_crosslinks",
                        lambda text, cands: (text, _Verification()))
    monkeypatch.setattr(promote_mod, "_count_key_contributions", lambda t: 5)
    monkeypatch.setattr(promote_mod, "should_auto_promote",
                        lambda **kw: promote_mod.GateResult(promoted=True))
    monkeypatch.setattr(runner.phases, "propose_keywords",
                        lambda *a, **k: _Keywords())
    return rows


class _Keywords:
    keywords = ["alpha", "beta", "gamma", "delta", "epsilon"]
    model = "test-model"
    input_tokens = 0
    output_tokens = 0


class _Verification:
    verified: list = []
    unverified: list = []
    broken: list = []


def _failing_promote(page_path: Path):
    """Stand in for the real promote: page written, PDF move failed."""
    def promote(**kwargs):
        return promote_mod.PromotionResult(
            promoted=False,
            wiki_path=page_path,
            pdf_path=None,
            category="compbio",
            warnings=["PDF move/copy failed: papers/smith-2024-a-paper-about-things.pdf "
                      "already exists and incoming PDF is not a journal-version upgrade"],
        )
    return promote


def test_failed_promote_raises_rather_than_reporting_success(ctx, commit_env, monkeypatch,
                                                             tmp_path):
    page = tmp_path / "page.md"
    page.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(promote_mod, "promote_to_wiki", _failing_promote(page))

    with pytest.raises(PromoteFailed) as exc:
        runner._phase_commit(ctx, conn=None)

    assert exc.value.stem == ctx.paper_stem
    assert exc.value.page_path == page
    assert "already exists" in exc.value.warnings[0]


def test_failed_promote_records_the_iteration_before_unwinding(ctx, commit_env, monkeypatch,
                                                               tmp_path):
    """`run_ingest`'s `finally` closes the connection, so the row has to be
    written before the exception propagates or the trace loses the event."""
    page = tmp_path / "page.md"
    monkeypatch.setattr(promote_mod, "promote_to_wiki", _failing_promote(page))

    with pytest.raises(PromoteFailed):
        runner._phase_commit(ctx, conn=None)

    commits = [r for r in commit_env if r.get("role") == "commit"]
    assert len(commits) == 1
    assert commits[0]["decision"] == "promote-failed"
    assert "already exists" in commits[0]["decision_reason"]
    assert commits[0]["paper_stem"] == ctx.paper_stem


def test_successful_promote_is_unaffected(ctx, commit_env, monkeypatch, tmp_path):
    """Guard the happy path against the new early-exit."""
    page = tmp_path / "page.md"
    monkeypatch.setattr(promote_mod, "promote_to_wiki", lambda **kw:
                        promote_mod.PromotionResult(
                            promoted=True, wiki_path=page, pdf_path=tmp_path / "p.pdf",
                            category="compbio", index_updated=True, log_appended=True))
    monkeypatch.setattr(runner.phases, "evolve_memory", lambda *a, **k: None)
    monkeypatch.setattr(runner.phases, "persist_grades", lambda *a, **k: None)

    out = runner._phase_commit(ctx, conn=None)

    assert out == page
    commits = [r for r in commit_env if r.get("role") == "commit"]
    assert commits[0]["decision"] == "committed-to-wiki"


def test_cli_maps_promote_failure_to_exit_2(monkeypatch, capsys, tmp_path):
    """Exit 2 = environment error, which `_should_retry` treats as retryable:
    the user deletes the duplicate, the retry then works."""
    from researchwiki.tasks import agent as agent_cli

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")

    def boom(*a, **kw):
        raise PromoteFailed(
            stem="smith-2024-a-paper-about-things",
            page_path=tmp_path / "wiki" / "compbio" / "smith-2024-a-paper.md",
            warnings=["PDF move/copy failed: papers/x.pdf already exists"],
        )
    monkeypatch.setattr(agent_cli, "run_ingest", boom)

    rc = agent_cli.main(["ingest", str(pdf)])

    assert rc == 2
    err = capsys.readouterr().err
    assert "PARTIALLY landed" in err
    assert "smith-2024-a-paper-about-things" in err
    assert "already exists" in err
    assert "recovery.md" in err
    assert "Traceback" not in err, "known-failure mode must not print a stack trace"


# ---------- the real promote, rolled back (WI-4) ----------
#
# The tests above stub `promote_to_wiki`. These drive the real one, so what is
# pinned is that a failure mid-promote leaves the tree as it found it — which
# is what WI-1 could only report and WI-4 actually fixes.

@pytest.fixture
def wiki_root(tmp_path, monkeypatch):
    from researchwiki import mutation as mut
    from researchwiki import paths, wiki as wiki_mod
    from researchwiki.agents import promote as pm

    (tmp_path / "wiki" / "compbio").mkdir(parents=True)
    (tmp_path / "papers").mkdir()
    (tmp_path / "inbox").mkdir()
    (tmp_path / "wiki" / "index.md").write_text(
        "# index.md\n\n## compbio\n\n", encoding="utf-8")
    (tmp_path / "wiki" / "log.md").write_text("# log\n\n", encoding="utf-8")

    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.setattr(mut, "mutation_dir", lambda: tmp_path / ".mutation")
    monkeypatch.setattr(pm, "wiki_dir", lambda: tmp_path / "wiki")
    monkeypatch.setattr(pm, "papers_dir", lambda: tmp_path / "papers")
    monkeypatch.setattr(pm, "_suggest_category", lambda *a, **k: ("compbio", "strong"))
    monkeypatch.setattr(wiki_mod, "commit_page", lambda md: None)
    monkeypatch.delenv("RW_MUTATION_JOURNAL", raising=False)
    return tmp_path


def _promote(tmp_path, **over):
    from researchwiki.agents import promote as pm
    pdf = tmp_path / "inbox" / "in.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    kwargs = dict(
        stem="smith-2024-a-paper",
        draft_text="## Summary\nbody\n\n## Key Contributions\n- one\n",
        metadata={"title": "A paper", "year": 2024},
        candidates=[],
        source_pdf_path=pdf,
        attempt_id="a1",
        short_name="Smith 2024",
        hook="Does a thing.",
        keywords=["a", "b", "c", "d", "e"],
    )
    kwargs.update(over)
    return pm.promote_to_wiki(**kwargs)


def test_real_promote_rolls_back_a_failed_pdf_move(wiki_root, monkeypatch):
    from researchwiki.agents import promote as pm
    index_before = (wiki_root / "wiki" / "index.md").read_text(encoding="utf-8")
    log_before = (wiki_root / "wiki" / "log.md").read_text(encoding="utf-8")

    monkeypatch.setattr(pm, "_move_pdf", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("papers/smith-2024-a-paper.pdf already exists")))

    res = _promote(wiki_root)

    assert res.promoted is False
    assert not (wiki_root / "wiki" / "compbio" / "smith-2024-a-paper.md").exists(), \
        "the page must be un-written, not left as an orphan"
    assert (wiki_root / "inbox" / "in.pdf").exists(), "the input stays re-ingestable"
    assert (wiki_root / "wiki" / "index.md").read_text(encoding="utf-8") == index_before
    assert (wiki_root / "wiki" / "log.md").read_text(encoding="utf-8") == log_before
    from researchwiki import mutation as mut
    assert mut.pending_journals() == [], "a completed rollback leaves no journal"


def test_real_promote_commits_the_happy_path(wiki_root):
    from researchwiki import mutation as mut
    res = _promote(wiki_root)

    assert res.promoted is True
    page = wiki_root / "wiki" / "compbio" / "smith-2024-a-paper.md"
    assert page.exists()
    assert (wiki_root / "papers" / "smith-2024-a-paper.pdf").exists()
    assert not (wiki_root / "inbox" / "in.pdf").exists(), "inbox PDF was moved"
    assert "smith-2024-a-paper" in (wiki_root / "wiki" / "index.md").read_text(encoding="utf-8")
    assert mut.pending_journals() == []


def test_real_promote_rolls_back_a_late_failure(wiki_root, monkeypatch):
    """A failure *after* the PDF move — the window WI-1 could only report."""
    from researchwiki.agents import promote as pm
    monkeypatch.setattr(pm, "_append_index_entry", lambda **k: (_ for _ in ()).throw(
        OSError("disk full")))

    with pytest.raises(OSError):
        _promote(wiki_root)

    assert not (wiki_root / "wiki" / "compbio" / "smith-2024-a-paper.md").exists()
    assert not (wiki_root / "papers" / "smith-2024-a-paper.pdf").exists(), \
        "the moved PDF is put back"
    assert (wiki_root / "inbox" / "in.pdf").exists()


def test_rollback_restores_a_backlink_target(wiki_root, monkeypatch):
    """Back-links are spliced into *existing* pages — the rollback case that
    plain 'delete what we created' would get wrong."""
    from researchwiki.agents import promote as pm

    other = wiki_root / "wiki" / "compbio" / "other-2020-paper.md"
    other.write_text("# Other\n\n## Related Papers\n\n- existing bullet\n",
                     encoding="utf-8")
    before = other.read_text(encoding="utf-8")

    class _Cand:
        wikilink = "compbio/other-2020-paper"
        kind = "topical"
        verified = True

    monkeypatch.setattr(pm, "_append_log_entry", lambda **k: (_ for _ in ()).throw(
        OSError("boom at the last step")))

    with pytest.raises(OSError):
        _promote(wiki_root, candidates=[_Cand()])

    assert other.read_text(encoding="utf-8") == before
