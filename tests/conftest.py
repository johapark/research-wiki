"""Suite-wide isolation.

CONTRIBUTING requires the test suite to be hermetic — no network, no LLM calls,
no model downloads. Reading the developer's `.env` breaks that in a subtler way:
it makes results depend on an untracked file on one machine.

`researchwiki.__main__.main` calls `_load_dotenv()` before dispatching, which
`os.environ.setdefault`s every key in the repo's `.env` for the rest of the
process. This repo's `.env` sets `RW_MODELS_CONFIG`, and `model_config` reads
that env var — so a single test that drove the CLI end-to-end silently retargeted
the model-config loader for every test that ran after it, and
`tests/test_model_config.py` failed 18 assertions purely on ordering. Passing
alone, failing in the suite.

Neutralized here rather than per-file because the trap is invisible at the call
site: nothing about `cli.main([...])` suggests it mutates global state. A test
that genuinely wants `.env` semantics should set the variables it needs with
`monkeypatch.setenv`, which unwinds.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_dotenv_leak(monkeypatch):
    from researchwiki import __main__ as cli
    monkeypatch.setattr(cli, "_load_dotenv", lambda *_args, **_kwargs: None)


@pytest.fixture(autouse=True)
def _isolate_provider_environment(monkeypatch):
    """Provider tests opt in explicitly; developer shell credentials never leak."""
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "RW_MODELS_CONFIG",
        "RW_LLM_PROVIDER",
        "RW_LLM_BASE_URL",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _isolate_state_db(tmp_path, monkeypatch):
    """Point `state.db` at a per-test temp file.

    Same class of trap as `_no_dotenv_leak`, and invisible at the call site for
    the same reason. `db_path()` resolves from the *current working directory*
    (`_repo_key()` over `wiki_root()`), so any test that reaches a write path
    lands in the developer's real per-repo DB. `backlinks.append_related_paper`
    is the live example: it calls `commit_page` → `db.upsert_page` whenever it
    actually appends (`backlinks.py:81`), so `test_appends_when_absent` — the one
    backlinks test where the append succeeds — inserted a `papers` row keyed on a
    pytest tmp dir into the real DB on every suite run. That row then showed up
    as `db_drift.extra` in `researchwiki lint` and as a phantom zero-claim paper,
    i.e. the suite manufactured corpus defects.

    `RESEARCHWIKI_DB_PATH` is the documented override and wins over the per-repo
    path (`db/connection.py:91-95`). Nothing memoizes it, so a per-test value
    takes effect immediately. Tests that assert on path *resolution* already
    `monkeypatch.delenv` it (see `tests/test_db_path.py`), and monkeypatch
    unwinds in LIFO order, so this fixture doesn't fight them.
    """
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "_state" / "state.db"))


@pytest.fixture(autouse=True)
def _isolate_claim_embedding_cache(tmp_path, monkeypatch):
    """Point the claim-embedding cache at a per-test temp dir.

    Third instance of the `_isolate_state_db` trap, and it became reachable the
    moment a *writer* did. Until `warm_claim_embeddings` landed, the only
    cache-touching call in a test's reach was `load_cached_claim_embeddings`,
    which is read-only — so nothing needed isolating. `cross_paper` now warms the
    cache, and its tests feed the embedder fake 2-dimensional vectors, so an
    unisolated run would rewrite the developer's real 384-dim claim cache (~19 MB,
    12.4k rows) with a handful of 2-dim fakes. Every later `claim-overlap`,
    `candidates pairs` and `check-coverage` run would then be scoring against
    garbage, and nothing would report an error — the cache is a derived artifact
    that silently rebuilds, so the only symptom is wrong numbers.

    Patched on the *bound* name in `claim_embeddings` rather than on
    `researchwiki.paths.semantic_cache_dir`, because that module does a
    module-level `from ..paths import semantic_cache_dir` — patching the origin
    would leave the already-bound reference untouched. `_paths()` calls it per
    invocation, so a per-test value takes effect immediately.

    Deliberately not `monkeypatch.chdir(tmp_path)`, which would isolate every
    `wiki_root()`-derived path at once: too broad as an autouse default, since
    tests legitimately read `prompts/`, `config/` and fixtures relative to the
    repo root.
    """
    from researchwiki.index import claim_embeddings
    cache = tmp_path / "_semantic-cache"
    monkeypatch.setattr(claim_embeddings, "semantic_cache_dir", lambda: cache)


@pytest.fixture(autouse=True)
def _isolate_edges_db(tmp_path, monkeypatch):
    """Point the claim-graph edge cache at a per-test temp dir.

    Fourth instance of the `_isolate_state_db` trap. `edges.db` holds LLM-judged
    relations *and* human decisions (`status='rejected'` is a person saying "not a
    relation"), so a test that reaches a write path does not just add noise — it
    can plant a `candidate` edge that `claim-graph`, `claim-graph --tensions` and
    `visualize` will then present as a real finding about the corpus.

    Latent until now only by accident: the cross-paper tests seeded claims with a
    NULL `claim_slug`, and `_persist_contradicts_edge` returns early when either
    slug is missing, so the writer was never actually reached. The moment a test
    seeds slugs — which the resumability tests must, since the coverage table is
    slug-keyed — that early return stops shielding the real database.

    Patched on the bound name in `claim_graph.edges` (module-level
    `from ..paths import claim_graph_dir`), for the same reason as the claim-cache
    fixture above.
    """
    from researchwiki.claim_graph import edges
    graph_dir = tmp_path / "_claim-graph"
    monkeypatch.setattr(edges, "claim_graph_dir", lambda: graph_dir)
