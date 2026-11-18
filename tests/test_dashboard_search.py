"""Actual search engines: relevance, offline fidelity, and the CLI HTTP adapter."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

from researchwiki import dashboard_server
from researchwiki.index.types import Document
from researchwiki.search import get_default_backend


def test_minisearch_ranking_phrases_prefixes_and_exact_identifiers():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for browser search engine tests")
    assets = Path(dashboard_server.__file__).parent / "templates"
    script = """
const assert = require('node:assert/strict');
const MiniSearch = require(process.argv[1]);
const DashboardSearch = require(process.argv[2]);
const pages = [
 {id:'title',title:'R2 integration',body:'Targeted insertion'},
 {id:'body',title:'Other study',body:'A mention of R2'},
 {id:'keyword',title:'Insertion platform',keywords:['retrotransposons']},
 {id:'phrase',title:'Prime editing',body:'A method'},
 {id:'separated',title:'Prime research',body:'Editing experiments'},
 {id:'brca1',title:'BRCA1 experiments'},
 {id:'brca2',title:'BRCA2 experiments'},
];
const engine = new DashboardSearch(pages, MiniSearch);
const ids = query => engine.query(query).map(hit => hit.id);
assert.equal(ids('R2')[0],'title');
assert.deepEqual(ids('"prime editing"'),['phrase']);
assert.deepEqual(ids('retrot'),['keyword']);
assert.deepEqual(ids('retrotansposons'),['keyword']);
assert.deepEqual(ids('BRCA1'),['brca1']);
assert.deepEqual(ids('R21'),[]);
assert.deepEqual(ids('R2 insertion'),['title']);
assert.deepEqual(ids('zzznomatch'),[]);
assert.deepEqual(new DashboardSearch([...pages,{id:'new',title:'New R2 result'}],MiniSearch).query('R2').map(x=>x.id).sort(),['body','new','title']);
"""
    run = subprocess.run([node, "-e", script, str(assets / "vendor/minisearch-7.2.0.js"),
                          str(assets / "dashboard-search.js")], text=True, capture_output=True)
    assert run.returncode == 0, run.stderr


def test_http_keyword_search_reuses_cli_index_and_sees_updates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = get_default_backend()
    docs = [Document("alpha", "cgt", "paper", "R2 insertion", "Test", 2026,
                     "R2 insertion", "A retrotransposon platform."),
            Document("beta", "genomics", "paper", "Another study", "Test", 2025,
                     "A mention of R2", "Unrelated methods.")]
    backend.build(docs)
    server = dashboard_server.make_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/api/search?"

    def search(query):
        with urlopen(base + urlencode({"q": query, "limit": 1000}), timeout=5) as response:
            return json.load(response)

    try:
        result = search("R2")
        expected = backend.query("R2", limit=1000)
        assert result["engine"] == "tantivy"
        assert result["hits"] == [{"id": hit.key, "score": hit.score} for hit in expected]
        assert search("zzznomatch")["hits"] == []
        backend.add(Document("gamma", "cgt", "paper", "New R2 result", "Test", 2026,
                             "R2 result", "A new paper."))
        assert "cgt/gamma" in {hit["id"] for hit in search("R2")["hits"]}
        backend.build(docs[:1])
        assert [hit["id"] for hit in search("R2")["hits"]] == ["cgt/alpha"]
        for query in ('"unfinished', "x" * 513):
            with pytest.raises(HTTPError) as invalid:
                search(query)
            assert invalid.value.code == 400
        (backend._path / ".managed.json").unlink()
        with pytest.raises(HTTPError) as missing:
            search("R2")
        assert missing.value.code == 503
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
