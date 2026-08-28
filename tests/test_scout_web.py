"""Agent-native web scouting stores only bounded source provenance."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

from researchwiki.scouting import web, web_cli


@pytest.fixture
def scout_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("wiki", "papers", "inbox"):
        (tmp_path / name).mkdir()
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


def _receipt(
    run_id: str,
    sources: list[dict] | None = None,
    *,
    discovery_method: str = "search",
) -> dict:
    return {
        "schema_version": 3,
        "run_id": run_id,
        "harness": "agent-native-web",
        "discovery_method": discovery_method,
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
    assert request["schema_version"] == 3
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
    {"domains": ["127.1"]},
    {"domains": ["0x7f.0.0.1"]},
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
    "http://127.1/page",
    "http://0177.0.0.1/page",
    "http://0x7f.0.0.1/page",
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
        discovery_method="search",
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
        "--discovery-method", "search",
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


def test_receipt_requires_an_integer_schema_version(scout_root: Path):
    request, _ = _request(scout_root)
    receipt = _receipt(request["run_id"])
    receipt["schema_version"] = 3.0
    with pytest.raises(web.ScoutInputError, match="schema_version"):
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


def test_show_loads_cached_result_without_creating_a_second_artifact(
    scout_root: Path,
):
    request, _ = _request(scout_root)
    web.accept_submission(
        request["run_id"], _receipt(request["run_id"], [
            _source(title="<script>unsafe</script>"),
            _source("https://example.net/snippet", fetched=False, title=None),
        ]), recorded_at="2026-08-27T12:02:00Z"
    )

    run = web.load_run(request["run_id"])

    assert run["state"] == "recorded"
    assert run["query"] == request["query"]
    cached = run["cached_result"]
    assert cached["manifest"]["source_count"] == 2
    assert cached["manifest"]["fetched_count"] == 1
    assert cached["receipt"]["evidence_class"] == "discovery-only"
    assert {source["title"] for source in cached["receipt"]["sources"]} == {
        None, "<script>unsafe</script>",
    }
    assert "deliverable" not in json.dumps(run)
    assert not (scout_root / "output").exists()
    assert all(
        not any((scout_root / name).iterdir())
        for name in ("wiki", "papers", "inbox")
    )


def test_show_marks_undated_since_sources_as_unverified(
    scout_root: Path, capsys: pytest.CaptureFixture[str],
):
    request, _ = _request(scout_root, since="2026-01-01")
    web.accept_submission(
        request["run_id"], _receipt(request["run_id"], [
            _source(fetched=False, published_at=None),
            _source("https://example.net/dated", fetched=False, published_at="2026-02-01"),
        ]), recorded_at="2026-08-27T12:02:00Z"
    )
    assert web_cli.main(["show", request["run_id"]]) == 0
    text = capsys.readouterr().out
    assert "date-unverified" in text
    assert "2026-02-01" in text


def test_show_detects_receipt_and_manifest_tampering(scout_root: Path):
    request, _ = _request(scout_root)
    _, manifest_path = web.accept_submission(
        request["run_id"], _receipt(request["run_id"]),
        recorded_at="2026-08-27T12:02:00Z"
    )
    receipt_path = manifest_path.with_name("receipt.json")
    receipt_path.write_text(receipt_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(web.ScoutInputError, match="manifest hash"):
        web.load_run(request["run_id"])

    recorded = json.loads(receipt_path.read_text())
    receipt_path.write_text(json.dumps(recorded, indent=2), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text())
    manifest["source_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(web.ScoutInputError, match="source_count"):
        web.load_run(request["run_id"])


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
        web.load_run(request["run_id"])


@pytest.mark.parametrize(("field", "value"), [
    ("schema_version", 3.0),
    ("source_count", True),
    ("fetched_count", 1.0),
])
def test_recorded_manifest_requires_exact_scalar_types(
    scout_root: Path, field: str, value: object,
):
    request, _ = _request(scout_root)
    _, manifest_path = web.accept_submission(
        request["run_id"], _receipt(request["run_id"]),
        recorded_at="2026-08-27T12:02:00Z",
    )
    changed = json.loads(manifest_path.read_text())
    changed[field] = value
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(web.ScoutInputError):
        web.load_run(request["run_id"])


def test_run_lifecycle_is_resumable_from_one_cached_result(scout_root: Path):
    request, _ = _request(scout_root)
    row = web.inspect_run(request["run_id"])
    assert row["state"] == "requested"
    assert "scout web show" in row["next_command"]

    web.record_sources(
        request["run_id"], harness="codex-web", discovery_method="search",
        fetched_urls=["https://example.org/story"],
        recorded_at="2026-08-27T12:02:00Z",
    )
    row = web.inspect_run(request["run_id"])
    assert row["state"] == "recorded"
    assert row["source_count"] == 1
    assert row["next_command"] is None
    run = web.load_run(request["run_id"])
    assert run["cached_result"]["receipt"]["sources"][0]["fetched"] is True
    assert not (scout_root / "output").exists()


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


def test_unreadable_run_artifact_is_one_invalid_row_not_a_dashboard_failure(
    scout_root: Path, monkeypatch: pytest.MonkeyPatch,
):
    """An unreadable artifact must not take `status` down with exit 2.

    `inspect_run` caught only ScoutInputError, so malformed JSON degraded to an
    `invalid` row while a permission error or undecodable byte — both
    ScoutStorageUnavailable — propagated out of `list_runs` and aborted the
    whole dashboard over a subsystem the corpus may never have used.
    """
    healthy, _ = web.create_request(
        "healthy request", run_id="20260827-healthy-dddddddd",
        created_at="2026-08-27T13:00:00Z",
    )
    broken, broken_path = web.create_request(
        "unreadable request", run_id="20260827-unreadable-eeeeeeee",
        created_at="2026-08-27T12:00:00Z",
    )
    broken_path.write_bytes(b"\xff\xfe not utf-8")

    rows = web.list_runs()
    assert [row["run_id"] for row in rows] == [healthy["run_id"], broken["run_id"]]
    assert [row["state"] for row in rows] == ["requested", "invalid"]
    assert rows[1]["error"]

    # Enumerating the runs directory at all is a different failure class: that
    # one is genuinely environmental and still surfaces.
    monkeypatch.setattr(
        Path, "iterdir", lambda self: (_ for _ in ()).throw(PermissionError("denied")),
    )
    with pytest.raises(web.ScoutStorageUnavailable):
        web.list_runs()


def test_show_returns_cached_urls_verbatim_without_markdown_rendering(
    scout_root: Path, capsys: pytest.CaptureFixture[str],
):
    request, _ = _request(scout_root, max_results=2, max_fetches=0)
    hostile = "https://example.com/a?q=[click](https://phish.example)&r=1_2"
    web.record_sources(
        request["run_id"], harness="test-harness", discovery_method="search",
        snippet_urls=[hostile],
        recorded_at="2026-08-27T14:00:00Z",
    )
    assert web_cli.main(["show", request["run_id"], "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["cached_result"]["receipt"]["sources"][0]["url"] == hostile


def test_run_directories_must_not_be_symlinks(scout_root: Path):
    external = scout_root / "external-run"
    external.mkdir()
    (external / "request.json").write_text("{}", encoding="utf-8")
    run_id = "20260827-linked-cccccccc"
    run_dir = scout_root / ".scout-cache" / "web" / "runs" / run_id
    run_dir.parent.mkdir(parents=True)
    run_dir.symlink_to(external, target_is_directory=True)
    with pytest.raises(web.ScoutInputError, match="must not be a symlink"):
        web.load_request(run_id)


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
    shown = json.loads(capsys.readouterr().out)
    assert shown["run_id"] == created["run_id"]
    assert shown["state"] == "requested"
    assert shown["cached_result"] is None

    assert web_cli.main([
        "record", created["run_id"], "--harness", "codex-web",
        "--discovery-method", "user-provided-url",
        "--fetched", "https://example.org/opened", "--json"
    ]) == 0
    assert json.loads(capsys.readouterr().out)["source_count"] == 1
    assert web_cli.main(["show", created["run_id"], "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["state"] == "recorded"
    assert shown["cached_result"]["receipt"]["sources"] == [{
        "url": "https://example.org/opened",
        "fetched": True,
        "title": None,
        "published_at": None,
    }]
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


def test_removed_report_action_points_to_cached_show(
    scout_root: Path, capsys: pytest.CaptureFixture[str],
):
    request, _ = _request(scout_root)
    assert web_cli.main(["report", request["run_id"]]) == 1
    assert "use `show <run-id>`" in capsys.readouterr().err
    assert web.list_runs()[0]["run_id"] == request["run_id"]


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


@pytest.mark.parametrize("method", web.DISCOVERY_METHODS)
def test_receipt_records_each_discovery_method(scout_root: Path, method: str):
    request, _ = _request(scout_root)
    manifest, _ = web.accept_submission(
        request["run_id"],
        _receipt(request["run_id"], discovery_method=method),
        recorded_at="2026-08-27T12:01:00Z",
    )
    assert manifest["discovery_method"] == method
    run = web.load_run(request["run_id"])
    assert run["cached_result"]["receipt"]["discovery_method"] == method
    # `list`/`status` read the method off the manifest, without opening the receipt.
    assert web.inspect_run(request["run_id"])["discovery_method"] == method


def test_receipt_requires_a_known_discovery_method(scout_root: Path):
    request, _ = _request(scout_root)
    for bad in (None, "", "guessed", "SEARCH", True, 3, ["search"]):
        receipt = _receipt(request["run_id"])
        receipt["discovery_method"] = bad
        with pytest.raises(web.ScoutInputError, match="discovery_method must be one of"):
            web.accept_submission(request["run_id"], receipt)
    missing = _receipt(request["run_id"])
    del missing["discovery_method"]
    with pytest.raises(web.ScoutInputError, match="discovery_method must be one of"):
        web.accept_submission(request["run_id"], missing)


@pytest.mark.parametrize("method", ["fetch-only", "user-provided-url"])
def test_searchless_methods_cannot_report_search_only_sources(
    scout_root: Path, method: str
):
    """A harness that never searched has no search hits it declined to open."""
    request, _ = _request(scout_root)
    receipt = _receipt(
        request["run_id"],
        [_source(fetched=True), _source("https://example.net/hit", fetched=False)],
        discovery_method=method,
    )
    with pytest.raises(web.ScoutInputError, match="cannot report search-only sources"):
        web.accept_submission(request["run_id"], receipt)
    # The same sources are legitimate for a harness that did search.
    manifest, _ = web.accept_submission(
        request["run_id"],
        _receipt(
            request["run_id"],
            [_source(fetched=True), _source("https://example.net/hit", fetched=False)],
            discovery_method="search",
        ),
        recorded_at="2026-08-27T12:01:00Z",
    )
    assert manifest["source_count"] == 2
    assert manifest["fetched_count"] == 1


def test_user_provided_url_run_records_without_a_search_harness(scout_root: Path):
    """The CLAUDE.md user-provided-URL case: no search, operator supplies the URL."""
    request, _ = _request(scout_root)
    manifest, _ = web.record_sources(
        request["run_id"],
        harness="claude-code-webfetch",
        discovery_method="user-provided-url",
        fetched_urls=["https://example.org/handed-over"],
        recorded_at="2026-08-27T12:03:00Z",
    )
    assert manifest["discovery_method"] == "user-provided-url"
    assert manifest["source_count"] == manifest["fetched_count"] == 1
    # Still quarantined: a user-supplied URL is no more citable than a search hit.
    assert manifest["evidence_class"] == "discovery-only"


def test_schema_two_receipt_is_rejected(scout_root: Path):
    request, _ = _request(scout_root)
    legacy = _receipt(request["run_id"])
    legacy["schema_version"] = 2
    del legacy["discovery_method"]
    with pytest.raises(web.ScoutInputError, match="schema_version must be 3"):
        web.accept_submission(request["run_id"], legacy)


def test_recorded_receipt_cannot_drop_or_contradict_its_discovery_method(
    scout_root: Path,
):
    """Tampering with a recorded artifact's method is caught on read, like harness."""
    request, _ = _request(scout_root)
    web.accept_submission(
        request["run_id"],
        _receipt(request["run_id"], discovery_method="fetch-only"),
        recorded_at="2026-08-27T12:01:00Z",
    )
    receipt_path = (
        scout_root / ".scout-cache" / "web" / "runs" / request["run_id"] / "receipt.json"
    )
    recorded = json.loads(receipt_path.read_text())
    recorded["discovery_method"] = "search"
    receipt_path.write_text(json.dumps(recorded, indent=2), encoding="utf-8")
    # The manifest still says fetch-only, so the disagreement is caught by
    # the schema check ahead of the hash check, naming the field.
    with pytest.raises(web.ScoutInputError, match="disagrees on discovery_method"):
        web.load_run(request["run_id"])
