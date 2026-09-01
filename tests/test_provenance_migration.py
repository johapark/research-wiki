"""Reviewed migration for the beta author-provenance contract."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import yaml

from researchwiki.fsatomic import write_json_atomic
from researchwiki.migrate import provenance
from researchwiki.tasks import migrate as migrate_task


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    wiki = tmp_path / "wiki"
    (wiki / "ai").mkdir(parents=True)
    (wiki / "synthesis").mkdir()
    ingest = tmp_path / ".ingest"
    monkeypatch.setattr("researchwiki.paths.wiki_dir", lambda: wiki)
    monkeypatch.setattr("researchwiki.wiki.wiki_dir", lambda: wiki)
    monkeypatch.setattr(provenance, "wiki_dir", lambda: wiki)
    monkeypatch.setattr(provenance, "ingest_dir", lambda: ingest)
    monkeypatch.setattr(provenance, "commit_page", lambda _path: None)
    monkeypatch.setattr(
        "researchwiki.tasks.lint.provenance.telemetry_author_models",
        lambda: {},
    )
    return tmp_path, wiki


def _page(wiki: Path, key: str, *, ptype: str = "paper", extra: str = "") -> Path:
    path = wiki / f"{key}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    ingested = "ingested_at: 2026-08-01T10:00:00\n" if ptype == "paper" else ""
    path.write_text(
        f"---\ntitle: T\ntype: {ptype}\nyear: 2024\n{ingested}{extra}---\n\n"
        "## Summary\n\nBody.\n",
        encoding="utf-8",
    )
    return path


def _manifest(run: Path) -> dict:
    return json.loads((run / "manifest.json").read_text(encoding="utf-8"))


def test_plan_is_read_only_and_only_preselects_exact_telemetry(corpus, monkeypatch):
    _root, wiki = corpus
    recoverable = _page(wiki, "ai/recoverable-2024-x")
    unresolved = _page(wiki, "synthesis/unresolved", ptype="synthesis")
    before = {p: p.read_bytes() for p in (recoverable, unresolved)}
    monkeypatch.setattr(
        "researchwiki.tasks.lint.provenance.telemetry_author_models",
        lambda: {"recoverable-2024-x": "gpt-5.6-luna"},
    )

    run, document = provenance.create_plan()

    assert [item["decision"] for item in document["items"]] == [
        "recover", "pending"
    ]
    assert document["items"][0]["telemetry_author_model"] == "gpt-5.6-luna"
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert (run / "manifest.json").is_file()


def test_pending_decision_blocks_before_backup_or_page_writes(corpus):
    _root, wiki = corpus
    page = _page(wiki, "synthesis/unresolved", ptype="synthesis")
    original = page.read_bytes()
    run, _document = provenance.create_plan()

    with pytest.raises(provenance.ProvenanceMigrationError, match="still pending"):
        provenance.apply_plan(run)

    assert page.read_bytes() == original
    assert not (run / "pre-apply.tar.gz").exists()


def test_apply_supports_recover_attest_acknowledge_skip_and_is_idempotent(
    corpus, monkeypatch,
):
    _root, wiki = corpus
    recover = _page(wiki, "ai/recover-2024-x")
    attest = _page(wiki, "synthesis/attest", ptype="synthesis",
                   extra='author_model: "TODO"\n')
    acknowledge = _page(wiki, "synthesis/ack", ptype="synthesis")
    skipped = _page(wiki, "synthesis/skip", ptype="synthesis")
    monkeypatch.setattr(
        "researchwiki.tasks.lint.provenance.telemetry_author_models",
        lambda: {"recover-2024-x": "gpt-5.6-luna"},
    )
    monkeypatch.setattr(provenance, "_now", lambda: __import__("datetime").datetime(
        2026, 9, 1, 12, 0, 0
    ))
    run, document = provenance.create_plan()
    by_key = {item["page"]: item for item in document["items"]}
    by_key["synthesis/attest"].update(
        decision="set-author-model", author_model="claude-opus-4-7"
    )
    by_key["synthesis/ack"]["decision"] = "acknowledge"
    by_key["synthesis/skip"]["decision"] = "skip"
    write_json_atomic(run / "manifest.json", document)

    result = provenance.apply_plan(run)

    assert result["changed"] == 3
    assert result["skipped"] == 1
    assert yaml.safe_load(recover.read_text().split("---", 2)[1])["author_model"] \
        == "gpt-5.6-luna"
    attest_fm = yaml.safe_load(attest.read_text().split("---", 2)[1])
    assert attest_fm["author_model"] == "claude-opus-4-7"
    assert "author_provenance" not in attest_fm
    ack_fm = yaml.safe_load(acknowledge.read_text().split("---", 2)[1])
    assert ack_fm["author_provenance"] == "legacy-unrecorded"
    assert str(ack_fm["provenance_acknowledged_at"]) == "2026-09-01"
    assert "author_model" not in ack_fm
    assert "author_model" not in skipped.read_text()

    backup = Path(result["backup"])
    with tarfile.open(backup, "r:gz") as archive:
        names = set(archive.getnames())
    assert names == {
        "wiki/ai/recover-2024-x.md",
        "wiki/synthesis/attest.md",
        "wiki/synthesis/ack.md",
    }

    again = provenance.apply_plan(run)
    assert again == {
        "run_dir": str(run),
        "changed": 0,
        "already_applied": 3,
        "skipped": 1,
        "backup": None,
    }


def test_hash_drift_blocks_the_whole_apply_before_backup(corpus):
    _root, wiki = corpus
    first = _page(wiki, "synthesis/first", ptype="synthesis")
    second = _page(wiki, "synthesis/second", ptype="synthesis")
    run, document = provenance.create_plan()
    for item in document["items"]:
        item.update(decision="set-author-model", author_model="gpt-5.6-luna")
    write_json_atomic(run / "manifest.json", document)
    second.write_text(second.read_text() + "\nMaintainer edit.\n", encoding="utf-8")

    with pytest.raises(provenance.ProvenanceMigrationError, match="changed since planning"):
        provenance.apply_plan(run)

    assert "author_model" not in first.read_text()
    assert not (run / "pre-apply.tar.gz").exists()


def test_conflicting_recorded_model_is_never_overwritten(corpus):
    _root, wiki = corpus
    page = _page(wiki, "synthesis/page", ptype="synthesis")
    run, document = provenance.create_plan()
    document["items"][0].update(
        decision="set-author-model", author_model="gpt-5.6-luna"
    )
    write_json_atomic(run / "manifest.json", document)
    page.write_text(
        page.read_text().replace("year: 2024\n", "year: 2024\nauthor_model: other\n"),
        encoding="utf-8",
    )

    with pytest.raises(provenance.ProvenanceMigrationError, match="conflicts"):
        provenance.apply_plan(run)
    assert "author_model: other" in page.read_text()


def test_acknowledgment_is_rejected_when_exact_telemetry_exists(corpus, monkeypatch):
    _root, wiki = corpus
    _page(wiki, "ai/recover-2024-x")
    monkeypatch.setattr(
        "researchwiki.tasks.lint.provenance.telemetry_author_models",
        lambda: {"recover-2024-x": "gpt-5.6-luna"},
    )
    run, document = provenance.create_plan()
    document["items"][0]["decision"] = "acknowledge"
    write_json_atomic(run / "manifest.json", document)

    with pytest.raises(provenance.ProvenanceMigrationError, match="exact telemetry"):
        provenance.apply_plan(run)


def test_recover_decision_must_still_match_live_telemetry(corpus, monkeypatch):
    _root, wiki = corpus
    _page(wiki, "ai/recover-2024-x")
    monkeypatch.setattr(
        "researchwiki.tasks.lint.provenance.telemetry_author_models",
        lambda: {"recover-2024-x": "gpt-5.6-luna"},
    )
    run, document = provenance.create_plan()
    document["items"][0]["telemetry_author_model"] = "invented-model"
    write_json_atomic(run / "manifest.json", document)

    with pytest.raises(provenance.ProvenanceMigrationError, match="no longer verifies"):
        provenance.apply_plan(run)


def test_cli_json_plan_and_apply_requires_a_reviewed_run(corpus, capsys):
    _root, wiki = corpus
    _page(wiki, "synthesis/page", ptype="synthesis")

    assert migrate_task.main(["provenance", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pages"] == 1
    assert payload["pending_review"] == 1
    assert Path(payload["manifest"]).is_file()

    assert migrate_task.main(["provenance", "--apply"]) == 1
    assert "requires --run" in capsys.readouterr().err
