"""Agent-native web scouting stores only bounded source provenance."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

from researchwiki.scouting import web, web_cli, web_report


@pytest.fixture
def scout_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("wiki", "papers", "inbox"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(web_report, "wiki_root", lambda: tmp_path)
    monkeypatch.setattr(web, "scout_cache_dir", lambda: tmp_path / ".scout-cache")
    return tmp_path


def _request(scout_root: Path, **kwargs) -> tuple[dict, Path]:
    return web.create_request(
        "protein design news",
        run_id="20260827-protein-design-deadbeef",
        created_at="2026-08-27T12:00:00Z",
        **kwargs,
    )


def _source(
    url: str = "https://example.org/story#section",
    *,
    fetched: bool = True,
    title: str | None = "A source",
    published_at: str | None = None,
) -> dict:
    source = {"url": url, "fetched": fetched}
    if title is not None:
        source["title"] = title
    if published_at is not None:
        source["published_at"] = published_at
    return source


def _receipt(run_id: str, sources: list[dict] | None = None) -> dict:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "harness": "agent-native-web",
        "sources": [_source()] if sources is None else sources,
    }


def _write_receipt(root: Path, receipt: dict) -> Path:
    path = root / "receipt-input.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_request_is_bounded_quarantined_and_provenance_only(scout_root: Path):
    request, path = _request(
        scout_root,
        max_results=8,
        max_fetches=3,
        domains=["Example.org", "example.org"],
        since="2025-01-01",
    )

    assert path == (
        scout_root / ".scout-cache" / "web" / "runs"
        / request["run_id"] / "request.json"
    )
    assert request["schema_version"] == 2
    assert request["evidence_class"] == "discovery-only"
    assert request["constraints"] == {
        "max_results": 8,
        "max_fetches": 3,
        "domains": ["example.org"],
        "since": "2025-01-01",
        "follow_transitive_links": False,
    }
    assert request["receipt_contract"]["required_per_source"] == ["url", "fetched"]
    assert "deliverable" not in request
    assert "findings" not in json.dumps(request)
    assert all(
        not any((scout_root / name).iterdir())
        for name in ("wiki", "papers", "inbox")
    )


@pytest.mark.parametrize("kwargs", [
    {"max_results": 0},
    {"max_results": web.HARD_MAX_RESULTS + 1},
    {"max_results": 2, "max_fetches": 3},
    {"domains": ["localhost"]},
    {"domains": ["intranet"]},
    {"domains": ["127.0.0.1"]},
    {"domains": ["https://example.org/path"]},
    {"since": "yesterday"},
])
def test_request_rejects_invalid_or_private_bounds(scout_root: Path, kwargs):
    with pytest.raises(web.ScoutInputError):
        _request(scout_root, **kwargs)


def test_receipt_normalizes_deduplicates_and_hashes_sources(scout_root: Path):
    request, _ = _request(scout_root, max_results=4, max_fetches=2)
    receipt = _receipt(request["run_id"], [
        _source(fetched=False, title=None),
        _source("https://EXAMPLE.org/story#other", title="Opened source"),
        _source("https://example.net/search-only", fetched=False),
    ])

    manifest, path = web.accept_submission(
        request["run_id"], receipt, recorded_at="2026-08-27T12:02:00Z"
    )

    assert path.name == "manifest.json"
    assert manifest["status"] == "recorded"
    assert manifest["source_count"] == 2
    assert manifest["fetched_count"] == 1
    assert manifest["duplicates_dropped"] == 1
    recorded = json.loads(path.with_name("receipt.json").read_text())
    assert recorded["sources"] == [
        {
            "url": "https://example.net/search-only",
            "fetched": False,
            "title": "A source",
            "published_at": None,
        },
        {
            "url": "https://example.org/story",
            "fetched": True,
            "title": "Opened source",
            "published_at": None,
        },
    ]
    assert recorded["evidence_class"] == "discovery-only"
    assert manifest["request_sha256"] == hashlib.sha256(
        path.with_name("request.json").read_bytes()
    ).hexdigest()
    assert manifest["receipt_sha256"] == hashlib.sha256(
        path.with_name("receipt.json").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "http://localhost/page",
    "http://intranet/page",
    "https://bad_host.example/page",
    "http://127.0.0.1/page",
    "http://169.254.169.254/latest/meta-data",
    "https://user:secret@example.org/page",
])
def test_receipt_rejects_non_public_or_credentialed_urls(scout_root: Path, url: str):
    request, _ = _request(scout_root)
    with pytest.raises(web.ScoutInputError):
        web.accept_submission(request["run_id"], _receipt(request["run_id"], [_source(url)]))


def test_receipt_enforces_domain_source_and_fetch_limits(scout_root: Path):
    request, _ = _request(
        scout_root, domains=["example.org"], max_results=2, max_fetches=1
    )
    with pytest.raises(web.ScoutInputError, match="outside request bounds"):
        web.accept_submission(
            request["run_id"],
            _receipt(request["run_id"], [_source("https://example.net/page")]),
        )
    with pytest.raises(web.ScoutInputError, match="fetched source count"):
        web.accept_submission(request["run_id"], _receipt(request["run_id"], [
            _source("https://a.example.org/one"),
            _source("https://b.example.org/two"),
        ]))
    with pytest.raises(web.ScoutInputError, match="source count"):
        web.accept_submission(request["run_id"], _receipt(request["run_id"], [
            _source("https://a.example.org/one", fetched=False),
            _source("https://b.example.org/two", fetched=False),
            _source("https://c.example.org/three", fetched=False),
        ]))


def test_since_rejects_known_old_dates_but_allows_undated_sources(scout_root: Path):
    request, _ = _request(
        scout_root, since="2026-01-01", max_results=3, max_fetches=1
    )
    with pytest.raises(web.ScoutInputError, match="predates"):
        web.accept_submission(request["run_id"], _receipt(request["run_id"], [
            _source(fetched=False, published_at="2025-12-31")
        ]))

    manifest, _ = web.accept_submission(
        request["run_id"],
        _receipt(request["run_id"], [_source(fetched=False, published_at=None)]),
    )
    assert manifest["source_count"] == 1


def test_record_flags_can_supply_since_dates(scout_root: Path):
    request, _ = _request(scout_root, since="2026-01-01", max_fetches=1)
    manifest, _ = web.record_sources(
        request["run_id"],
        harness="codex-web",
        fetched_urls=["https://example.org/story#section"],
        published_at={"https://example.org/story": "2026-08-20"},
    )
    assert manifest["source_count"] == 1

    second, _ = web.create_request(
        "cli date", run_id="20260827-cli-date-eeeeeeee",
        created_at="2026-08-27T13:00:00Z", since="2026-01-01", max_fetches=1,
    )
    assert web_cli.main([
        "record", second["run_id"], "--harness", "codex-web",
        "--fetched", "https://example.org/story",
        "--published-at", "https://example.org/story", "2026-08-20", "--json",
    ]) == 0


@pytest.mark.parametrize("field", [
    "findings", "coverage_gaps", "brief", "report", "excerpt"
])
def test_receipt_rejects_top_level_research_prose(scout_root: Path, field: str):
    request, _ = _request(scout_root)
    receipt = _receipt(request["run_id"])
    receipt[field] = "research prose"
    with pytest.raises(web.ScoutInputError, match="research prose"):
        web.accept_submission(request["run_id"], receipt)


def test_receipt_rejects_per_source_prose_and_accepts_empty_ledger(scout_root: Path):
    request, _ = _request(scout_root)
    source = _source()
    source["excerpt"] = "prose"
    with pytest.raises(web.ScoutInputError, match="unsupported fields: excerpt"):
        web.accept_submission(request["run_id"], _receipt(request["run_id"], [source]))

    manifest, _ = web.accept_submission(
        request["run_id"], _receipt(request["run_id"], []),
        recorded_at="2026-08-27T12:02:00Z",
    )
    assert manifest["source_count"] == 0


def test_receipt_rejects_unknown_top_level_fields(scout_root: Path):
    request, _ = _request(scout_root)
    receipt = _receipt(request["run_id"])
    receipt["notes"] = "free-form material"
    with pytest.raises(web.ScoutInputError, match="unsupported source-receipt fields"):
        web.accept_submission(request["run_id"], receipt)


def test_receipt_is_idempotent_and_immutable(scout_root: Path):
    request, _ = _request(scout_root)
    receipt = _receipt(request["run_id"])
    first, _ = web.accept_submission(
        request["run_id"], receipt, recorded_at="2026-08-27T12:02:00Z"
    )
    second, _ = web.accept_submission(request["run_id"], receipt)
    assert second == first

    changed = _receipt(
        request["run_id"], [_source("https://example.org/different")]
    )
    with pytest.raises(web.ScoutInputError, match="refusing to overwrite provenance"):
        web.accept_submission(request["run_id"], changed)


def test_equivalent_duplicate_order_has_one_receipt_identity(scout_root: Path):
    request, _ = _request(scout_root, max_results=4, max_fetches=1)
    first = _receipt(request["run_id"], [
        _source("https://example.org/one", fetched=False, title="Z title"),
        _source("https://example.org/one#fragment", fetched=True, title="A title"),
    ])
    second = _receipt(request["run_id"], list(reversed(first["sources"])))

    manifest, _ = web.accept_submission(
        request["run_id"], first, recorded_at="2026-08-27T12:02:00Z"
    )
    replay, _ = web.accept_submission(request["run_id"], second)
    assert replay == manifest
    recorded = json.loads(
        (scout_root / ".scout-cache" / "web" / "runs" / request["run_id"] / "receipt.json")
        .read_text()
    )
    assert recorded["sources"] == [{
        "url": "https://example.org/one",
        "fetched": True,
        "title": "A title",
        "published_at": None,
    }]


def test_report_is_an_inert_source_ledger_without_research_prose(scout_root: Path):
    request, _ = _request(scout_root)
    web.accept_submission(
        request["run_id"], _receipt(request["run_id"], [
            _source(title="<script>unsafe</script>"),
            _source("https://example.net/snippet", fetched=False, title=None),
        ]), recorded_at="2026-08-27T12:02:00Z"
    )

    text, path, summary = web.render_report(request["run_id"])

    assert "DISCOVERY ONLY — NOT WIKI EVIDENCE" in text
    assert "native answer is not stored here" in text
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in text
    assert "<script>" not in text
    assert "harness-reported opened" in text
    assert "search result only; page not opened" in text
    assert "## Research brief" not in text
    assert path == scout_root / "output" / "scout" / request["run_id"] / "report.md"
    assert summary["source_count"] == 2
    assert summary["fetched_count"] == 1
    assert "deliverable" not in summary
    assert all(
        not any((scout_root / name).iterdir())
        for name in ("wiki", "papers", "inbox")
    )


def test_report_marks_undated_since_sources_as_unverified(scout_root: Path):
    request, _ = _request(scout_root, since="2026-01-01")
    web.accept_submission(
        request["run_id"], _receipt(request["run_id"], [
            _source(fetched=False, published_at=None),
            _source("https://example.net/dated", fetched=False, published_at="2026-02-01"),
        ]), recorded_at="2026-08-27T12:02:00Z"
    )
    text, _, _ = web.render_report(request["run_id"])
    assert "Since bound: 2026-01-01" in text
    assert "(1 dated; 1 without a publication date)" in text


def test_report_detects_receipt_and_manifest_tampering(scout_root: Path):
    request, _ = _request(scout_root)
    _, manifest_path = web.accept_submission(
        request["run_id"], _receipt(request["run_id"]),
        recorded_at="2026-08-27T12:02:00Z"
    )
    receipt_path = manifest_path.with_name("receipt.json")
    receipt_path.write_text(receipt_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(web.ScoutInputError, match="manifest hash"):
        web.render_report(request["run_id"])

    recorded = json.loads(receipt_path.read_text())
    receipt_path.write_text(json.dumps(recorded, indent=2), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text())
    manifest["source_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(web.ScoutInputError, match="source_count"):
        web.render_report(request["run_id"])


def test_schema_one_request_is_rejected(scout_root: Path):
    request, request_path = _request(scout_root)
    legacy = json.loads(request_path.read_text())
    legacy["schema_version"] = 1
    legacy.pop("receipt_contract")
    legacy["result_contract"] = {"legacy": True}
    request_path.write_text(json.dumps(legacy, indent=2), encoding="utf-8")

    with pytest.raises(web.ScoutInputError, match="unsupported scout request schema_version"):
        web.load_request(request["run_id"])


def test_request_rejects_unknown_nested_contract_fields(scout_root: Path):
    request, request_path = _request(scout_root)
    changed = json.loads(request_path.read_text())
    changed["receipt_contract"]["brief"] = "research prose"
    request_path.write_text(json.dumps(changed, indent=2), encoding="utf-8")

    with pytest.raises(web.ScoutInputError, match="invalid receipt contract"):
        web.load_request(request["run_id"])


def test_recorded_manifest_rejects_unknown_fields(scout_root: Path):
    request, _ = _request(scout_root)
    _, manifest_path = web.accept_submission(
        request["run_id"], _receipt(request["run_id"]),
        recorded_at="2026-08-27T12:02:00Z",
    )
    changed = json.loads(manifest_path.read_text())
    changed["brief"] = "research prose"
    manifest_path.write_text(json.dumps(changed, indent=2), encoding="utf-8")

    row = web.inspect_run(request["run_id"])
    assert row["state"] == "invalid"
    assert "unsupported fields: brief" in row["error"]
    with pytest.raises(web.ScoutInputError, match="unsupported fields: brief"):
        web.render_report(request["run_id"])


def test_run_lifecycle_is_resumable_and_detects_a_stale_report(scout_root: Path):
    request, _ = _request(scout_root)
    row = web.inspect_run(request["run_id"])
    assert row["state"] == "requested"
    assert "scout web show" in row["next_command"]

    web.record_sources(
        request["run_id"], harness="codex-web",
        fetched_urls=["https://example.org/story"],
        recorded_at="2026-08-27T12:02:00Z",
    )
    row = web.inspect_run(request["run_id"])
    assert row["state"] == "recorded"
    assert row["source_count"] == 1
    assert row["next_command"] is None

    _, report_path, _ = web.render_report(request["run_id"])
    assert web.inspect_run(request["run_id"])["state"] == "recorded"
    report_path.write_text(report_path.read_text() + "manual drift\n", encoding="utf-8")
    assert web.inspect_run(request["run_id"])["state"] == "recorded"


def test_list_runs_filters_states_and_reports_invalid_artifacts(scout_root: Path):
    requested, _ = web.create_request(
        "new request", run_id="20260827-new-aaaaaaaa",
        created_at="2026-08-27T13:00:00Z",
    )
    invalid, invalid_path = web.create_request(
        "broken request", run_id="20260827-broken-bbbbbbbb",
        created_at="2026-08-27T12:00:00Z",
    )
    invalid_doc = json.loads(invalid_path.read_text())
    invalid_doc["constraints"]["follow_transitive_links"] = True
    invalid_path.write_text(json.dumps(invalid_doc), encoding="utf-8")

    rows = web.list_runs()
    assert [row["run_id"] for row in rows] == [requested["run_id"], invalid["run_id"]]
    assert [row["state"] for row in rows] == ["requested", "invalid"]
    assert web.list_runs(state="requested") == [rows[0]]
    with pytest.raises(web.ScoutInputError, match="state must be one of"):
        web.list_runs(state="pending")


def test_run_and_report_directories_must_not_be_symlinks(scout_root: Path):
    external = scout_root / "external-run"
    external.mkdir()
    (external / "request.json").write_text("{}", encoding="utf-8")
    run_id = "20260827-linked-cccccccc"
    run_dir = scout_root / ".scout-cache" / "web" / "runs" / run_id
    run_dir.parent.mkdir(parents=True)
    run_dir.symlink_to(external, target_is_directory=True)
    with pytest.raises(web.ScoutInputError, match="must not be a symlink"):
        web.load_request(run_id)

    output = scout_root / "output"
    output.symlink_to(external, target_is_directory=True)
    with pytest.raises(web.ScoutInputError, match="must not be symlinks"):
        web_report.report_path("safe-run")


def test_cli_supports_shorthand_record_show_list_and_stdin_receipts(
    scout_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    assert web_cli.main([
        "a bounded query", "--max-results", "3", "--max-fetches", "1", "--json"
    ]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["mode"] == "web-agent-handoff"
    assert "deliverable" not in created

    assert web_cli.main(["show", created["run_id"], "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == created["run_id"]

    assert web_cli.main([
        "record", created["run_id"], "--harness", "codex-web",
        "--fetched", "https://example.org/opened", "--json"
    ]) == 0
    assert json.loads(capsys.readouterr().out)["source_count"] == 1
    assert web_cli.main(["list", "--state", "recorded", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert [row["run_id"] for row in listing["runs"]] == [created["run_id"]]

    second, _ = web.create_request(
        "stdin request", run_id="20260827-stdin-bbbbbbbb",
        created_at="2026-08-27T13:00:00Z",
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_receipt(second["run_id"]))))
    assert web_cli.main(["accept", second["run_id"], "-", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["source_count"] == 1


def test_status_surfaces_resumable_web_scout_work(scout_root: Path, monkeypatch, capsys):
    from researchwiki import paths
    from researchwiki.tasks import status as status_task

    request, _ = _request(scout_root)
    monkeypatch.setattr(paths, "wiki_root", lambda: scout_root)
    monkeypatch.chdir(scout_root)

    assert status_task.main([]) == 0
    output = capsys.readouterr().out
    assert "web-scout runs awaiting agent:  1" in output
    assert f"[requested] {request['run_id']}" in output
    assert "resume with `researchwiki scout web list`" in output
