"""Dashboard integration: canonical sources, portable exports, local lifecycle."""

from __future__ import annotations

import json
import base64
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from researchwiki import dashboard, dashboard_runtime, dashboard_server, paths
from researchwiki.tasks import dashboard as cli


def page(root, key, body="## Summary\nA local page.", **frontmatter):
    import yaml
    p = root / "wiki" / (key + ".md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("---\n" + yaml.safe_dump(frontmatter) + "---\n" + body, encoding="utf-8")
    return p


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    (tmp_path / "papers").mkdir()
    (tmp_path / "inbox").mkdir()
    page(tmp_path, "cgt/alpha", "## Summary\nSee [[beta#kc-123]] and [[alpha]].", type="paper", title="Alpha", year=2025, ingested_at="2026-11-17T12:00:00", hook="Alpha result")
    page(tmp_path, "cgt/beta", type="paper", title="Beta without date", year=2024)
    page(tmp_path, "ideas/design", type="idea", title="An idea", category=["cgt"], generated_at="2026-10-01", status="open")
    page(tmp_path, "proposals/candidate", type="proposal", title="A candidate", created_at="2026-11-10T09:00:00Z", status="proposed")
    page(tmp_path, "references/protocol", type="protocol", title="A protocol", issuer="Test")
    page(tmp_path, "views", type="dashboard", title="Do not count")
    page(tmp_path, "synthesis/suggested-additions", type="meta", title="Discovery leads")
    (tmp_path / "papers/alpha.pdf").write_bytes(b"%PDF-test")
    (tmp_path / "inbox/new drop.pdf").write_bytes(b"%PDF-new")
    return tmp_path


def test_library_keeps_undated_pages_and_uses_frontmatter_stamps(library):
    data = dashboard.build_dashboard()
    rows = {r["id"]: r for r in data["pages"]}
    assert len(rows) == 5
    assert rows["cgt/beta"]["added"] == ""
    assert "Missing added date" in rows["cgt/beta"]["flags"]
    assert rows["ideas/design"]["added"].startswith("2026-10-01")
    assert rows["ideas/design"]["date_field"] == "generated_at"
    assert rows["proposals/candidate"]["date_field"] == "created_at"
    assert data["meta"]["categories"] == {"cgt": 2}
    assert data["meta"]["n_pdfs"] == 1


def test_wikilinks_resolve_claim_anchors_and_no_self_or_ambiguous_links(library):
    rows = {r["id"]: r for r in dashboard.build_dashboard()["pages"]}
    assert rows["cgt/alpha"]["links"] == ["cgt/beta"]
    assert rows["cgt/beta"]["backlinks"] == ["cgt/alpha"]
    page(library, "other/beta", type="paper", title="Another Beta")
    rows = {r["id"]: r for r in dashboard.build_dashboard()["pages"]}
    assert rows["cgt/alpha"]["links"] == []


def run_dashboard_script(data, script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for dashboard JavaScript tests")
    template = Path(dashboard.__file__).parent / "templates/dashboard.html"
    source = template.read_text().split("const $=", 1)[1].split("document.addEventListener('click'", 1)[0]
    app = "const DATA=JSON.parse(require('fs').readFileSync(0,'utf8'));\nconst $=" + source
    run = subprocess.run([node, "-e", app + "\n" + script], input=json.dumps(data),
                         text=True, capture_output=True)
    assert run.returncode == 0, run.stderr


def test_reader_separates_adjacent_footnotes(library):
    run_dashboard_script(dashboard.build_dashboard(), """
const assert=require('node:assert/strict');
const body='Together.[^sc][^sctype] [^chet] Alone.[^sc] Literal `[^sc][^chet]`';
const html=markdown(body).html;
const citation=id=>`<button class="plain wikilink" data-section="fn-${id}">${id}</button>`;
assert.ok(html.includes('<sup>'+['sc','sctype','chet'].map(citation).join(', ')+'</sup>'));
assert.ok(html.includes('Alone.<sup>'+citation('sc')+'</sup>'));
assert.ok(html.includes('<code>[^sc][^chet]</code>'));
""")


def test_reader_targets_exact_claims_in_rendered_markdown(library):
    from researchwiki.claim_graph.slug import compute_claim_slug

    bullet = "A repeated contribution."
    method = "The second sentence describes a separate measurement."
    row = "Example second row"
    page(library, "cgt/claims", f"""## Summary
Emoji 📚 before the anchors.
## Key Contributions
- **{bullet}**
- **{bullet}**
## Methodology and Architecture
The first sentence describes the measurement procedure. {method}
## Results
| Example | Value |
| --- | --- |
| Example | first row |
| Example | second row |
""", type="paper")
    run_dashboard_script(dashboard.build_dashboard(), """
const assert=require('node:assert/strict');
rebuild();
const page=byId.get('cgt/claims');
const renderClaim=slug=>markdown(page.body,page.claim_offsets[slug]).html;
const marker='<span id="reader-claim"></span>';
const [bullet,method,row]=%s;
assert.ok(renderClaim(bullet+'-1').includes(
 '<li><strong>A repeated contribution.</strong></li><li>'+marker+'<strong>A repeated contribution.</strong></li>'));
assert.ok(renderClaim(method).includes('procedure. '+marker+'The second sentence'));
assert.ok(renderClaim(row).includes('<tr id="reader-claim"><td>Example</td><td>second row</td></tr>'));
""" % json.dumps([compute_claim_slug("key_contributions", bullet),
                  compute_claim_slug("methodology", method), compute_claim_slug("results", row)]))


def test_automatic_refresh_requeries_active_search_on_304(library):
    run_dashboard_script(dashboard.build_dashboard(live=True), """
const assert=require('node:assert/strict');
const elements=new Map();
const document={getElementById(id){
 if(!elements.has(id))elements.set(id,{});
 return elements.get(id);
}};
const requests=[];
global.fetch=async url=>{
 requests.push(url);
 if(url==='/api/dashboard')return {status:304};
 return {ok:true,json:async()=>({hits:[{id:'cgt/beta',score:1}]})};
};
(async()=>{
 state.q='beta';
 searchState.query=state.q;
 await refresh();
 assert.equal(requests.length,2);
 assert.equal(new URL(requests[1],'http://localhost').searchParams.get('q'),'beta');
 assert.ok($('results').innerHTML.includes('data-page="cgt/beta"'));
 state.q='';
 await refresh();
 assert.deepEqual(requests.slice(2),['/api/dashboard']);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_export_paths_support_custom_locations_and_encoded_names(library):
    out = library / "exports/deep/library.html"
    data = dashboard.build_dashboard(out=out, vault="Research vault")
    alpha = next(r for r in data["pages"] if r["stem"] == "alpha")
    assert alpha["pdf_url"] == "../../papers/alpha.pdf"
    assert alpha["markdown_url"] == "../../wiki/cgt/alpha.md"
    assert "vault=Research+vault" in alpha["obsidian_url"]
    assert data["inbox"][0]["url"] == "../../inbox/new%20drop.pdf"
    live = dashboard.build_dashboard(live=True)
    assert live["inbox"][0]["url"] == "/files/inbox/new%20drop.pdf"


def test_corpus_html_cannot_escape_embedded_json(library):
    page(library, "cgt/hostile", "## Summary\n<script>alert(1)</script>", type="paper", title='</script><img src=x onerror="alert(1)">')
    html = dashboard.render_html(dashboard.build_dashboard())
    assert "__DASHBOARD_DATA__" not in html
    assert html.count("</script>") == 2  # one inert data block, one application
    payload = html.split('id="dashboard-data">', 1)[1].split("</script>", 1)[0]
    assert "<" not in payload
    assert any(r["title"].startswith("</script>") for r in json.loads(payload)["pages"])


def test_reader_fonts_are_embedded_offline_and_local_when_live(library):
    offline = dashboard.render_html(dashboard.build_dashboard())
    assert "SIL OPEN FONT LICENSE Version 1.1" in offline
    assert "__FONT_FACES__" not in offline
    for source in offline.split("data:font/woff2;base64,")[1:]:
        assert base64.b64decode(source.split("'", 1)[0]).startswith(b"wOF2")
    assert offline.count("data:font/woff2;base64,") == len(dashboard.FONT_FILES)
    live = dashboard.render_html(dashboard.build_dashboard(live=True))
    assert "data:font/woff2;base64," not in live
    assert all("/assets/fonts/" + filename in live for filename in dashboard.FONT_FILES)
    for name in dashboard.FONT_LICENSES:
        notice = Path(dashboard.__file__).parent.joinpath("templates/fonts", name).read_text(encoding="utf-8")
        assert notice in offline and notice in live
    # Static faces must resolve headings, bold emphasis, and genuine italics.
    for weight in ("400", "600", "700"):
        for style in ("normal", "italic"):
            assert any("font-family:'SNU Jaha'" in face
                       and "font-style:" + style + ";" in face
                       and "font-weight:" + weight + ";" in face
                       for face in live.split("@font-face"))


def test_malformed_type_stays_visible(library):
    page(library, "cgt/malformed", type=["paper"], title="Bad type")
    record = next(r for r in dashboard.build_dashboard()["pages"] if r["stem"] == "malformed")
    assert record["type"] == "unknown"


def test_cli_export_never_overwrites_source_and_has_no_wiki_mutations(library):
    before = {p: p.read_bytes() for p in (library / "wiki").rglob("*.md")}
    assert cli.main(["--out", str(library / "wiki/dashboard.html")]) == 2
    out = library / "output/dashboard.html"
    assert cli.main(["--out", str(out)]) == 0
    assert out.is_file()
    assert before == {p: p.read_bytes() for p in (library / "wiki").rglob("*.md")}
    assert not dashboard_runtime.runtime_path().exists()


def test_server_file_routes_reject_traversal_and_file_symlink_escape(library):
    allowed = dashboard_server.resolve_file("/files/papers/alpha.pdf")
    assert allowed == library / "papers/alpha.pdf"
    for route in ["/files/wiki/../../.env", "/files/wiki/%2e%2e/%2e%2e/.env", "/files/wiki/%00.md", "/files/papers/../alpha.pdf", "/files/.env", "/files/wiki\\cgt/alpha.md"]:
        assert dashboard_server.resolve_file(route) is None
    secret = library / "private.md"
    secret.write_text("private")
    try:
        (library / "wiki/cgt/escape.md").symlink_to(secret)
    except OSError:
        pytest.skip("file symlinks unavailable")
    assert dashboard_server.resolve_file("/files/wiki/cgt/escape.md") is None


def test_directory_symlinks_are_supported(library):
    target = library / "synced-papers"
    (library / "papers").rename(target)
    try:
        (library / "papers").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    assert dashboard_server.resolve_file("/files/papers/alpha.pdf") == target / "alpha.pdf"


def test_live_cache_refreshes_on_page_change_not_each_request(library):
    cache = dashboard_server.DashboardCache()
    data, first = cache.get()
    same, second = cache.get()
    assert same is data and first == second
    page(library, "cgt/new", type="paper", title="A new ingest", ingested_at="2026-11-18")
    changed, third = cache.get()
    assert first != third and len(changed["pages"]) == len(data["pages"]) + 1


def test_local_http_health_refresh_and_authenticated_shutdown(library):
    server = dashboard_server.make_server(0)
    def serve_until_shutdown():
        try:
            server.serve_forever()
        finally:
            server.server_close()
    thread = threading.Thread(target=serve_until_shutdown, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/health", timeout=3) as response:
            assert json.load(response) == server.dashboard_identity
        for filename in dashboard.FONT_FILES:
            with urlopen(base + "/assets/fonts/" + filename, timeout=3) as response:
                assert response.headers["Content-Type"] == "font/woff2"
                assert response.read().startswith(b"wOF2")
        for filename in dashboard.FONT_LICENSES:
            with pytest.raises(HTTPError) as blocked_font:
                urlopen(base + "/assets/fonts/" + filename, timeout=3)
            assert blocked_font.value.code == 404
        with urlopen(base + "/api/dashboard", timeout=3) as response:
            data = json.load(response)
            etag = response.headers["ETag"]
            assert data["meta"]["live"] is True
            assert response.headers["X-Content-Type-Options"] == "nosniff"
        with pytest.raises(HTTPError) as unchanged:
            urlopen(Request(base + "/api/dashboard", headers={"If-None-Match": etag}), timeout=3)
        assert unchanged.value.code == 304
        with pytest.raises(HTTPError) as forbidden:
            urlopen(Request(base + "/api/shutdown", data=b""), timeout=3)
        assert forbidden.value.code == 403
        with pytest.raises(HTTPError) as bad_host:
            urlopen(Request(base + "/api/health", headers={"Host": "attacker.example"}), timeout=3)
        assert bad_host.value.code == 403
        record = {**server.dashboard_identity, "port": server.server_port, "control_token": server.dashboard_control_token, "pid": 123, "session": "test-agent"}
        dashboard_runtime.runtime_path().parent.mkdir()
        dashboard_runtime.runtime_path().write_text(json.dumps(record))
        assert dashboard_runtime.status()["session"] == "test-agent"
        assert dashboard_runtime.stop()
        thread.join(timeout=3)
        assert not thread.is_alive()
    finally:
        # Closing the listener before joining is required for the status probe
        # after shutdown to fail immediately rather than wait on a dead loop.
        server.server_close()
        if thread.is_alive():
            server.shutdown()
            thread.join(timeout=3)


def test_runtime_never_contacts_another_repo_or_an_invalid_port(library):
    assert dashboard_runtime.request_runtime({"port": -1}) == {}
    assert dashboard_runtime.request_runtime({"port": 80, "repository_id": "another-repo"}) == {}


def test_session_owner_exit_stops_server_and_existing_server_is_reused(library, capsys):
    """Exercise actual process ownership, receipt cleanup, and duplicate reuse."""
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(dashboard.__file__).parent.parent)
    script = ("from researchwiki.tasks.dashboard import main; "
              f"raise SystemExit(main(['--serve', '--port', '0', '--session', 'test-chat', '--owner-pid', '{owner.pid}']))")
    server = subprocess.Popen([sys.executable, "-c", script], cwd=library, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            row = dashboard_runtime.status()
            if row["running"] or server.poll() is not None:
                break
            time.sleep(.05)
        assert row["running"], server.communicate(timeout=3)
        assert row["owner_pid"] == owner.pid
        assert row["session"] == "test-chat"
        assert "control_token" not in row
        assert cli.main(["--serve", "--port", "0", "--session", "another-chat"]) == 0
        assert "already running" in capsys.readouterr().out
        assert dashboard_runtime.status()["server_id"] == row["server_id"]
        owner.terminate()
        owner.wait(timeout=3)
        output, error = server.communicate(timeout=6)
        assert server.returncode == 0, (output, error)
        assert not dashboard_runtime.runtime_path().exists()
        assert dashboard_runtime.status()["running"] is False
    finally:
        for child in (owner, server):
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=3)
