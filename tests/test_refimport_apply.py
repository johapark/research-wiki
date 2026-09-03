"""Staging and dispatch (`refimport.apply`) plus the batch change it needs.

The subtle part is the liveness re-check. Pairing and triage are frozen in the
manifest so `apply` cannot reach a different conclusion than the `inspect` a
user read — but whether a paper is *already in the wiki* is a fact about now,
not a decision made then. Without re-checking it, `apply --limit 30` run twice
imports the same 30 papers twice.
"""

import importlib
import json
from pathlib import Path

import pytest

from researchwiki.refimport.apply import dispatch, plan_wave, stage

import_task = importlib.import_module("researchwiki.tasks.import")


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    for d in ("wiki/cgt", "papers", "inbox", ".ingest"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def mk_pdf(tmp_path: Path, name: str, body: bytes | None = None) -> Path:
    """A stand-in PDF. Never parsed by this module — only copied and hashed.

    Content varies with the filename so that two *different* papers are
    distinguishable by bytes, which is how `plan_wave` recognizes a copy of one
    of them already sitting in `inbox/`. Identical content for every fixture PDF
    would make every record look like a leftover copy of every other.
    """
    p = tmp_path / "src" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body or f"%PDF-1.4\n% stand-in for {name}\n".encode())
    return p


def rec(
    tmp_path,
    key="k",
    doi="10.1234/a",
    stem="ada-2024-a-paper-about-things",
    name="paper.pdf",
    verdict="ready",
    pdf=True,
):
    return {
        "key": key,
        "verdict": verdict,
        "doi": doi,
        "derived_stem": stem,
        "title": "A paper about things",
        "year": 2024,
        "primary_pdf": str(mk_pdf(tmp_path, name)) if pdf else None,
        "ingest_args": ["--doi", doi, "--title", "A paper about things"],
    }


def add_page(wiki: Path, stem: str, doi: str | None = None) -> None:
    fm = ["---", "title: Existing", "type: paper", "category: [cgt]"]
    if doi:
        fm.append(f"doi: {doi}")
    fm += ["---", "", "## Summary", "", "Body."]
    (wiki / "wiki" / "cgt" / f"{stem}.md").write_text("\n".join(fm), encoding="utf-8")


# ---------- plan_wave ----------


def test_only_ready_records_are_planned(wiki):
    records = [
        rec(wiki, key="a"),
        rec(wiki, key="b", verdict="review", name="b.pdf"),
        rec(wiki, key="c", verdict="skip", name="c.pdf"),
    ]
    assert [r["key"] for r in plan_wave(records).staged] == ["a"]


def test_limit_takes_the_first_n(wiki):
    records = [
        rec(wiki, key=str(i), doi=f"10.1234/{i}", stem=f"s-2024-{i}", name=f"{i}.pdf")
        for i in range(5)
    ]
    assert len(plan_wave(records, limit=2).staged) == 2


def test_a_record_whose_doi_is_now_in_the_wiki_is_not_re_imported(wiki):
    """Idempotency: this is what makes a second wave import the *next* N."""
    add_page(wiki, "existing-2024-paper", doi="10.1234/a")
    plan = plan_wave([rec(wiki)])
    assert plan.staged == []
    assert plan.already_present[0]["landed_as"] == "cgt/existing-2024-paper"


def test_a_record_whose_stem_is_now_in_the_wiki_is_not_re_imported(wiki):
    add_page(wiki, "ada-2024-a-paper-about-things")
    plan = plan_wave([rec(wiki, doi=None)])
    assert plan.staged == [] and len(plan.already_present) == 1


def test_the_second_wave_takes_the_next_records_not_the_same_ones(wiki):
    """The bug this design exists to prevent: `--limit 2` twice importing the
    same two papers, because the manifest still calls them ready."""
    records = [
        rec(wiki, key=str(i), doi=f"10.1234/{i}", stem=f"s-2024-{i}", name=f"{i}.pdf")
        for i in range(4)
    ]
    first = plan_wave(records, limit=2)
    assert [r["key"] for r in first.staged] == ["0", "1"]

    for r in first.staged:  # simulate the wave landing
        add_page(wiki, r["derived_stem"], doi=r["doi"])

    second = plan_wave(records, limit=2)
    assert [r["key"] for r in second.staged] == ["2", "3"]
    assert len(second.already_present) == 2


def test_a_pdf_deleted_since_inspect_is_reported_not_absorbed(wiki):
    r = rec(wiki)
    Path(r["primary_pdf"]).unlink()
    plan = plan_wave([r])
    assert plan.staged == [] and len(plan.missing_pdf) == 1


# ---------- stage ----------


def test_staging_copies_rather_than_moving_or_linking(wiki):
    """The source is the user's own library and must survive untouched;
    CLAUDE.md's PDF rule is explicit that files enter inbox/ as copies."""
    r = rec(wiki)
    staged = stage([r])
    dest = staged[0][0]
    assert dest.is_file() and not dest.is_symlink()
    assert Path(r["primary_pdf"]).is_file()  # original still there
    assert dest.parent.name == "inbox"


def test_staging_carries_the_per_record_argv(wiki):
    assert stage([rec(wiki)])[0][1] == [
        "--doi",
        "10.1234/a",
        "--title",
        "A paper about things",
    ]


def test_a_name_collision_does_not_clobber_a_paper(wiki):
    """Exports name files `Nature-2026.3.pdf` — no identity at all — so a
    collision in inbox/ is plausible and overwriting would be unrecoverable."""
    a = rec(wiki, key="a", name="Nature-2026.pdf")
    b = rec(wiki, key="b", doi="10.1234/b", stem="s-2024-b", name="Nature-2026.pdf")
    (wiki / "src2").mkdir()
    b["primary_pdf"] = str(mk_pdf(wiki, "Nature-2026.pdf"))
    staged = stage([a, b])
    names = {p.name for p, _ in staged}
    assert len(names) == 2, names
    assert all(p.is_file() for p, _ in staged)


def test_dry_run_copies_nothing(wiki):
    stage([rec(wiki)], dry_run=True)
    assert list((wiki / "inbox").glob("*.pdf")) == []


# ---------- dispatch ----------


def test_dispatch_passes_per_record_argv_to_the_batch(wiki, monkeypatch):
    seen = {}

    def fake_new_batch(
        pdfs,
        subcommand,
        extra_args,
        workers,
        per_input_args=None,
        relay_watch=False,
        workers_explicit=True,
    ):
        seen.update(
            pdfs=pdfs,
            subcommand=subcommand,
            workers=workers,
            per_input_args=per_input_args,
            relay_watch=relay_watch,
            workers_explicit=workers_explicit,
        )
        return 0

    from researchwiki.tasks import _ingest_batch

    monkeypatch.setattr(_ingest_batch, "new_batch", fake_new_batch)

    staged = stage([rec(wiki)])
    assert dispatch(staged, workers=3) == 0
    assert seen["subcommand"] == ["agent", "ingest"] and seen["workers"] == 3
    assert seen["workers_explicit"] is True
    key = str(staged[0][0].resolve())
    assert seen["per_input_args"][key] == [
        "--doi",
        "10.1234/a",
        "--title",
        "A paper about things",
    ]


def test_dispatch_skips_a_path_that_vanished_rather_than_exiting(wiki, monkeypatch):
    """`_ingest_batch._resolve_inputs` calls `sys.exit(1)` on a missing file —
    fine for a CLI, fatal for a caller that should report and continue."""
    from researchwiki.tasks import _ingest_batch

    monkeypatch.setattr(_ingest_batch, "new_batch", lambda pdfs, *a, **k: len(pdfs))
    staged = stage([rec(wiki)])
    staged[0][0].unlink()
    assert dispatch(staged, workers=1) == 1  # nothing usable, no SystemExit


# ---------- the apply phase ----------


def write_manifest(wiki: Path, records: list[dict]) -> Path:
    run = wiki / ".ingest" / "import-20260101T000000"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-01-01T00:00:00",
                "export_path": "x.ris",
                "export_format": "ris",
                "pdf_root": None,
                "category": None,
                "summary": {},
                "unclaimed_pdfs": [],
                "items": records,
            }
        )
    )
    return run


@pytest.fixture
def no_spend(monkeypatch):
    """Neutralize the two things `apply` does that cost money or need a GPU."""
    calls = []
    from researchwiki.tasks import _ingest_batch

    monkeypatch.setattr(
        _ingest_batch, "new_batch", lambda *a, **k: calls.append((a, k)) or 0
    )
    monkeypatch.setattr(import_task, "_embedding_status", lambda: (True, "fake"))
    return calls


def test_apply_requires_an_explicit_run(wiki, no_spend):
    """A bare `apply` silently choosing among several inspect runs is a
    footgun, so `--run` is required rather than defaulted."""
    with pytest.raises(SystemExit):
        import_task.main(["apply"])


def test_apply_dispatches_the_wave(wiki, no_spend, capsys):
    run = write_manifest(wiki, [rec(wiki)])
    assert import_task.main(["apply", "--run", str(run)]) == 0
    assert len(no_spend) == 1
    capsys.readouterr()


def test_apply_with_nothing_ready_returns_1(wiki, no_spend, capsys):
    """The one phase where nothing-to-do is a failure — acting is all it does."""
    run = write_manifest(wiki, [rec(wiki, verdict="skip")])
    assert import_task.main(["apply", "--run", str(run)]) == 1
    assert no_spend == []
    capsys.readouterr()


def test_apply_dry_run_copies_nothing_and_dispatches_nothing(wiki, no_spend, capsys):
    run = write_manifest(wiki, [rec(wiki)])
    assert import_task.main(["apply", "--run", str(run), "--dry-run"]) == 0
    assert no_spend == []
    assert list((wiki / "inbox").glob("*.pdf")) == []
    capsys.readouterr()


# ---------- the chat-relay batch warning reaches this entry point too ----------
#
# `dispatch` calls `_ingest_batch.new_batch` directly, so `agent ingest`'s own
# check never saw an import run — and `import apply` is unavoidably batch-shaped.
# Under chat-relay that combination stalls silently rather than failing: each
# worker's stderr goes to a log file, which is where the pending-prompt notice
# is printed. See tests/test_chat_relay_batch_guard.py for the guard itself.


def test_apply_warns_under_chat_relay(wiki, no_spend, monkeypatch, capsys):
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    run = write_manifest(wiki, [rec(wiki)])
    assert import_task.main(["apply", "--run", str(run)]) == 0, (
        "the guard must warn, not refuse"
    )
    err = capsys.readouterr().err
    assert "chat-relay" in err
    assert "defaults to 1 worker" in err


def test_apply_dry_run_warns_too(wiki, no_spend, monkeypatch, capsys):
    """The preview is where the user decides whether to spend the wave, so a
    warning that only fires on the real run comes one decision too late."""
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    run = write_manifest(wiki, [rec(wiki)])
    assert import_task.main(["apply", "--run", str(run), "--dry-run"]) == 0
    assert "chat-relay" in capsys.readouterr().err


def test_apply_silent_for_an_api_provider(wiki, no_spend, monkeypatch, capsys):
    monkeypatch.setenv("RW_LLM_PROVIDER", "anthropic")
    run = write_manifest(wiki, [rec(wiki)])
    assert import_task.main(["apply", "--run", str(run)]) == 0
    assert "chat-relay" not in capsys.readouterr().err


def test_apply_still_passes_per_input_args_under_chat_relay(
    wiki, no_spend, monkeypatch, capsys
):
    """The warning must not drag `_BATCH_INCOMPATIBLE_FLAGS` along with it.

    `agent ingest` refuses batch mode combined with `--doi/--title/...` because
    one flag cannot describe N papers. `apply` gives each paper *its own* argv
    through `per_input_args`, which is the case that guard was never about — so
    sharing the flag check would forbid the whole feature.
    """
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    run = write_manifest(wiki, [rec(wiki)])
    assert import_task.main(["apply", "--run", str(run)]) == 0
    ((positional, kwargs),) = [(a, k) for a, k in no_spend]
    assert kwargs["per_input_args"], "per-paper argv was dropped"
    assert positional[3] == 1
    assert kwargs["relay_watch"] is True
    assert kwargs["workers_explicit"] is False
    capsys.readouterr()


def test_apply_persists_an_explicit_worker_count(
    wiki, no_spend, monkeypatch, capsys
):
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    run = write_manifest(wiki, [rec(wiki)])
    assert import_task.main(
        ["apply", "--run", str(run), "--workers", "3"]
    ) == 0
    ((positional, kwargs),) = [(a, k) for a, k in no_spend]
    assert positional[3] == 3
    assert kwargs["relay_watch"] is True
    assert kwargs["workers_explicit"] is True
    capsys.readouterr()


def test_apply_limit_stages_exactly_n(wiki, no_spend, capsys):
    records = [
        rec(wiki, key=str(i), doi=f"10.1234/{i}", stem=f"s-2024-{i}", name=f"{i}.pdf")
        for i in range(5)
    ]
    run = write_manifest(wiki, records)
    assert import_task.main(["apply", "--run", str(run), "--limit", "2"]) == 0
    assert len(list((wiki / "inbox").glob("*.pdf"))) == 2
    capsys.readouterr()


def test_apply_on_a_missing_run_returns_1(wiki, no_spend, capsys):
    assert import_task.main(["apply", "--run", str(wiki / "nope")]) == 1
    capsys.readouterr()


def test_apply_hard_fails_when_the_embedding_model_is_unusable(
    wiki, monkeypatch, capsys
):
    """Dry-run is advisory, but a real run must fail before copying anything."""
    from researchwiki.errors import EnvironmentFailure
    from researchwiki.tasks import _ingest_batch

    monkeypatch.setattr(_ingest_batch, "new_batch", lambda *a, **k: 0)
    monkeypatch.setattr(
        import_task,
        "_embedding_status",
        lambda: (False, "RuntimeError: _ARRAY_API not found"),
    )
    run = write_manifest(wiki, [rec(wiki)])
    with pytest.raises(EnvironmentFailure, match="embedding model"):
        import_task.main(["apply", "--run", str(run)])
    assert list((wiki / "inbox").glob("*.pdf")) == []
    capsys.readouterr()


def test_failed_dispatch_does_not_print_success_followups(wiki, monkeypatch, capsys):
    from researchwiki.tasks import _ingest_batch

    monkeypatch.setattr(_ingest_batch, "new_batch", lambda *a, **k: 1)
    monkeypatch.setattr(import_task, "_embedding_status", lambda: (True, "fake"))
    run = write_manifest(wiki, [rec(wiki)])

    assert import_task.main(["apply", "--run", str(run)]) == 1
    captured = capsys.readouterr()
    assert "Next — free" not in captured.out
    assert "Import wave failed" in captured.err


# ---------- the verify phase ----------
#
# Reads the *manifest*, not the batch checkpoint, because the question is about
# records ("did this paper reach the wiki") and the checkpoint only knows about
# files. A record can complete its ingest and still not be in wiki/ — that's the
# sandbox path, and surfacing it is the whole reason this phase exists.


def test_verify_reports_a_landed_record(wiki, capsys):
    r = rec(wiki)
    run = write_manifest(wiki, [r])
    add_page(wiki, r["derived_stem"], doi=r["doi"])
    assert import_task.main(["verify", "--run", str(run), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["landed"] == ["cgt/" + r["derived_stem"]]
    assert out["not_imported"] == []


def test_verify_distinguishes_sandboxed_from_missing(wiki, capsys):
    """A gate-failed page lands in .agent-output/ instead of wiki/. Counting it
    as 'not imported' would hide work that is done and waiting for review."""
    held = rec(wiki, key="held", stem="held-2024-a-paper", doi="10.1234/held")
    never = rec(
        wiki, key="never", stem="never-2024-a-paper", doi="10.1234/never", name="n.pdf"
    )
    run = write_manifest(wiki, [held, never])
    sandbox = wiki / ".agent-output"
    sandbox.mkdir()
    (sandbox / "held-2024-a-paper.md").write_text("draft")

    import_task.main(["verify", "--run", str(run), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["sandboxed"] == ["held-2024-a-paper"]
    assert out["not_imported"] == ["never-2024-a-paper"]


def test_verify_matches_a_doi_only_record_to_its_sandbox_page(wiki, capsys):
    held = rec(wiki, key="held", stem=None, doi="10.1234/doi-only")
    run = write_manifest(wiki, [held])
    sandbox = wiki / ".agent-output"
    sandbox.mkdir()
    (sandbox / "reconciled-2026-title-from-doi.md").write_text(
        "---\ntitle: Reconciled\ntype: paper\ndoi: 10.1234/doi-only\n---\n",
        encoding="utf-8",
    )

    import_task.main(["verify", "--run", str(run), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["sandboxed"] == ["reconciled-2026-title-from-doi"]
    assert out["not_imported"] == []


def test_verify_only_counts_records_that_were_ready(wiki, capsys):
    """`skip` and `review` records were never going to land; counting them as
    'not imported' would make every run look half-failed."""
    run = write_manifest(
        wiki,
        [
            rec(wiki, key="a"),
            rec(wiki, key="b", verdict="skip", name="b.pdf"),
            rec(wiki, key="c", verdict="review", name="c.pdf"),
        ],
    )
    import_task.main(["verify", "--run", str(run), "--json"])
    assert json.loads(capsys.readouterr().out)["ready"] == 1


def test_verify_carries_a_lint_snapshot(wiki, capsys):
    """Shelled out rather than suggested: a bulk import is exactly when nobody
    runs the follow-ups by hand."""
    r = rec(wiki)
    run = write_manifest(wiki, [r])
    add_page(wiki, r["derived_stem"], doi=r["doi"])  # lint needs pages to lint
    import_task.main(["verify", "--run", str(run), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert isinstance(out["lint"], dict)
    assert "invalid_frontmatter" in out["lint"]


def test_verify_reports_no_snapshot_on_an_empty_wiki(wiki, capsys):
    """`lint` prints a human message rather than JSON when there is nothing to
    lint, so the snapshot is legitimately absent — not a failure."""
    run = write_manifest(wiki, [rec(wiki)])
    assert import_task.main(["verify", "--run", str(run), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["lint"] is None


def test_verify_survives_a_lint_failure(wiki, capsys, monkeypatch):
    """`verify` is a report. It must never be the thing that fails."""
    from researchwiki.tasks import lint as lint_task

    monkeypatch.setattr(
        lint_task, "main", lambda argv: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    run = write_manifest(wiki, [rec(wiki)])
    assert import_task.main(["verify", "--run", str(run), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["lint"] is None


def test_verify_on_a_missing_run_returns_1(wiki, capsys):
    assert import_task.main(["verify", "--run", str(wiki / "nope")]) == 1
    capsys.readouterr()


def test_verify_requires_an_explicit_run(wiki):
    with pytest.raises(SystemExit):
        import_task.main(["verify"])


def test_verify_names_the_graph_wiring_followups(wiki, capsys):
    """A bulk import arrives as N disconnected nodes, and nothing else in the
    report measures that."""
    run = write_manifest(wiki, [rec(wiki)])
    import_task.main(["verify", "--run", str(run)])
    out = capsys.readouterr().out
    assert "claim-overlap --backlog" in out
    assert "candidates concepts --bridges" in out


# ---------- dry-run must predict what apply does ----------


def test_dry_run_predicts_the_same_destinations_as_a_real_run(wiki):
    """A dry-run that disagrees with the run it previews is worse than none.

    Uniquification used to test only `dest.exists()`, and under `--dry-run`
    nothing is written — so two sources named `Nature-2026.pdf` both reported
    the same destination while the real run correctly wrote the second to
    `Nature-2026--1.pdf`. Exporters name files without identity, so this is a
    plausible case, not a contrived one.
    """
    a = rec(wiki, key="a", name="Nature-2026.pdf")
    b = rec(wiki, key="b", doi="10.1234/b", stem="s-2024-b")
    b["primary_pdf"] = str(mk_pdf(wiki, "Nature-2026.pdf"))

    previewed = [p for p, _ in stage([a, b], dry_run=True)]
    actual = [p for p, _ in stage([a, b])]
    assert previewed == actual
    assert len(set(previewed)) == 2


def test_three_way_collision_still_yields_three_destinations(wiki):
    recs = []
    for i in range(3):
        r = rec(wiki, key=str(i), doi=f"10.1234/{i}", stem=f"s-2024-{i}")
        r["primary_pdf"] = str(mk_pdf(wiki, "Nature-2026.pdf"))
        recs.append(r)
    assert len({p for p, _ in stage(recs, dry_run=True)}) == 3


# ---------- a wave that already ran must not stage twice ----------
#
# `stage` uniquifies on collision, so a leftover `inbox/X.pdf` from a failed wave
# used to yield `X--1.pdf` and strand `X.pdf` as permanent phantom backlog — which
# the next `agent ingest inbox/*.pdf` would ingest as a separate paper.
# Identity is tested by content: `stage`'s own `--N` scheme means the leftover may
# not even share the source's name, and exporter names carry no identity anyway.


def test_a_byte_identical_copy_in_inbox_is_not_staged_again(wiki):
    r = rec(wiki)
    leftover = wiki / "inbox" / "paper.pdf"
    leftover.write_bytes(Path(r["primary_pdf"]).read_bytes())

    plan = plan_wave([r])
    assert plan.staged == []
    assert len(plan.already_staged) == 1
    assert plan.already_staged[0]["staged_as"] == str(leftover)


def test_the_leftover_is_found_under_the_uniquified_name(wiki):
    """A previous wave may have written it as `X--1.pdf`, so a name comparison
    would miss exactly the case this guards."""
    r = rec(wiki)
    leftover = wiki / "inbox" / "paper--1.pdf"
    leftover.write_bytes(Path(r["primary_pdf"]).read_bytes())

    plan = plan_wave([r])
    assert plan.staged == []
    assert plan.already_staged[0]["staged_as"] == str(leftover)


def test_apply_prints_frozen_metadata_for_an_already_staged_pdf(wiki, no_spend, capsys):
    r = rec(wiki, doi="10.1234/preserved")
    leftover = wiki / "inbox" / "paper.pdf"
    leftover.write_bytes(Path(r["primary_pdf"]).read_bytes())
    run = write_manifest(wiki, [r])

    assert import_task.main(["apply", "--run", str(run)]) == 1
    out = capsys.readouterr().out
    assert "researchwiki agent ingest" in out
    assert "--doi 10.1234/preserved" in out
    assert "--title 'A paper about things'" in out


def test_a_different_pdf_of_the_same_name_still_stages(wiki):
    """The negative case. Two different papers legitimately share an
    exporter-generated name, so name-matching here would silently drop one."""
    r = rec(wiki)
    (wiki / "inbox" / "paper.pdf").write_bytes(b"%PDF-1.4\n% a different paper\n")

    plan = plan_wave([r])
    assert [x["key"] for x in plan.staged] == ["k"]
    assert plan.already_staged == []
    assert [p.name for p, _ in stage(plan.staged)] == ["paper--1.pdf"]


def test_already_staged_does_not_consume_the_limit(wiki):
    """Matching `already_present` and `missing_pdf`: the limit counts work to do."""
    done = rec(
        wiki, key="done", doi="10.1234/done", stem="s-2024-done", name="done.pdf"
    )
    (wiki / "inbox" / "done.pdf").write_bytes(Path(done["primary_pdf"]).read_bytes())
    todo = rec(
        wiki, key="todo", doi="10.1234/todo", stem="s-2024-todo", name="todo.pdf"
    )

    plan = plan_wave([done, todo], limit=1)
    assert [x["key"] for x in plan.staged] == ["todo"]


def test_an_empty_inbox_hashes_nothing(wiki, monkeypatch):
    """Size is the pre-filter, so the common case must not read any PDF."""
    import researchwiki.refimport.apply as apply_mod

    monkeypatch.setattr(
        apply_mod,
        "file_sha256",
        lambda p: pytest.fail(f"hashed {p} with an empty inbox"),
    )
    assert len(plan_wave([rec(wiki)]).staged) == 1


# ---------- the stem lookup is batched ----------


def test_plan_wave_walks_the_wiki_once_regardless_of_record_count(wiki, monkeypatch):
    """`find_stem_collision` re-walks `wiki/` per call, so calling it per record
    is O(records x pages). Measured at 0.5 ms/call over 117 pages — fine today,
    and worse with every paper added. One walk, N lookups."""
    import researchwiki.refimport.apply as apply_mod

    calls = []
    real = apply_mod.read_wiki_stems
    monkeypatch.setattr(apply_mod, "read_wiki_stems", lambda: calls.append(1) or real())

    records = [
        rec(wiki, key=str(i), doi=f"10.1234/{i}", stem=f"s-2024-{i}", name=f"{i}.pdf")
        for i in range(25)
    ]
    plan_wave(records)
    assert calls == [1], f"walked the wiki {len(calls)} times for 25 records"


def test_batched_lookup_still_sees_a_page_written_between_waves(wiki):
    """Not cached across calls: `apply` re-checks between waves and must see
    what the previous wave wrote."""
    r = rec(wiki)
    assert plan_wave([r]).staged  # first wave: importable
    add_page(wiki, r["derived_stem"], doi=r["doi"])
    assert plan_wave([r]).staged == []  # second: already present
