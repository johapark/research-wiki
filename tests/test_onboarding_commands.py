"""Onboarding command contracts: local doctor, add alias, and first receipt."""

from __future__ import annotations

from types import SimpleNamespace

from researchwiki.tasks import add, agent, doctor


def test_add_forwards_every_argument_to_full_agent_ingest(monkeypatch):
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 7

    from researchwiki.tasks import agent as agent_task
    monkeypatch.setattr(agent_task, "main", fake_main)

    rc = add.main(["/tmp/paper.pdf", "--no-semantic", "-n", "1"])

    assert rc == 7
    assert seen["argv"] == [
        "ingest", "/tmp/paper.pdf", "--no-semantic", "-n", "1",
    ]


def test_doctor_default_never_runs_provider_probe(monkeypatch):
    ok = doctor.Check("ok", "test", "ready")
    monkeypatch.setattr(doctor, "_dependency_checks", lambda: [ok])
    monkeypatch.setattr(doctor, "_content_checks", lambda: [])
    monkeypatch.setattr(doctor, "_provider_checks", lambda: [])
    monkeypatch.setattr(doctor, "_semantic_check", lambda: ok)
    monkeypatch.setattr(doctor, "_database_check", lambda: ok)
    monkeypatch.setattr(doctor, "_index_check", lambda: ok)
    monkeypatch.setattr(doctor, "_curl_check", lambda: ok)
    monkeypatch.setattr(
        doctor, "_probe_check",
        lambda: (_ for _ in ()).throw(AssertionError("probe must be explicit")),
    )

    checks = doctor.run_checks()

    assert checks


def test_doctor_reports_blocker_and_fix(monkeypatch, capsys):
    blocked = doctor.Check("block", "Provider", "missing key", "set the key")
    monkeypatch.setattr(doctor, "run_checks", lambda probe=False: [blocked])

    assert doctor.main([]) == 1
    out = capsys.readouterr().out
    assert "BLOCKED" in out
    assert "Fix: set the key" in out


def test_doctor_ready_with_nonblocking_warning(monkeypatch, capsys):
    checks = [
        doctor.Check("ok", "Paths", "ready"),
        doctor.Check("warn", "Semantic model", "not cached"),
    ]
    monkeypatch.setattr(doctor, "run_checks", lambda probe=False: checks)

    assert doctor.main([]) == 0
    out = capsys.readouterr().out
    assert "READY TO INGEST" in out
    assert "Provider connectivity was not tested" in out


def test_success_receipt_names_page_pdf_claims_and_trace(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.chdir(tmp_path)
    page = tmp_path / "wiki" / "other" / "paper-stem.md"
    monkeypatch.setattr(agent, "_indexed_claim_count", lambda stem: 4)
    ctx = SimpleNamespace(
        committed_path=page,
        paper_stem="paper-stem",
        attempt_id="attempt-123",
    )

    agent._print_ingest_receipt(ctx)

    out = capsys.readouterr().out
    assert "✓ Paper added" in out
    assert "Page:   wiki/other/paper-stem.md" in out
    assert "PDF:    papers/paper-stem.pdf" in out
    assert "Claims: 4 indexed" in out
    assert "researchwiki agent trace attempt-123" in out
