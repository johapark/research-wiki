"""Within-category divergence detection in `researchwiki.tasks.suggest_splits`.

The semantic index is monkeypatched to a hand-built (embeddings, rows) tuple
and the LLM judge is replaced with an injected stub, so the tests are
deterministic and never touch a provider. `monkeypatch.chdir(tmp_path)`
isolates `wiki_root()` (which is just `Path.cwd()`) so the decay stamps land
in a temp dir.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from researchwiki.tasks import suggest_splits as s

# Orthonormal axes → identical-axis papers have cosine 1.0 (one component),
# cross-axis papers have cosine 0.0 (separate components) at any threshold in
# (0, 1). Lets us engineer an exact core + separable sub-cluster.
_A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_B = np.array([0.0, 1.0, 0.0], dtype=np.float32)
_C = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _row(cat, i, vec, page_type="paper"):
    return {
        "key": f"{cat}/{cat}-{i}",
        "stem": f"{cat}-{i}",
        "category": cat,
        "page_type": page_type,
        "title": f"{cat} paper {i}",
        "content_hash": f"h{cat}{i}",
    }


def _corpus():
    """A `bio` category with a 12-paper core (axis A) + a 3-paper divergent
    sub-cluster (axis B); a small `chem` category below the min-papers floor;
    two `other` papers that within-category mode must ignore."""
    rows, vecs = [], []
    for i in range(12):
        rows.append(_row("bio", i, _A)); vecs.append(_A)
    for i in range(12, 15):
        rows.append(_row("bio", i, _B)); vecs.append(_B)
    for i in range(5):
        rows.append(_row("chem", i, _C)); vecs.append(_C)
    for i in range(2):
        rows.append(_row("other", i, _A)); vecs.append(_A)
    return np.vstack(vecs).astype(np.float32), rows


@pytest.fixture
def indexed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()  # log() appends to wiki/log.md
    embs, rows = _corpus()
    monkeypatch.setattr(s.pages_semantic, "load_index", lambda: (embs, rows), raising=True)
    # Live category set — detect_divergence_candidates iterates this.
    monkeypatch.setattr(s, "content_categories",
                        lambda: frozenset({"bio", "chem", "other"}), raising=True)
    # is_valid is used by the migration printer / --category guard.
    monkeypatch.setattr(s, "is_valid",
                        lambda c: (c or "").strip().lower() in {"bio", "chem", "other"},
                        raising=True)
    return embs, rows


# --- pure clustering ---------------------------------------------------------

def test_split_candidates_isolates_minority_cluster(indexed):
    embs, rows = indexed
    bio = embs[[i for i, r in enumerate(rows) if r["category"] == "bio"]]
    core, cands = s._split_candidates(bio, threshold=0.70)
    assert len(core) == 12
    assert [len(c) for c in cands] == [3]


def test_split_candidates_cohesive_blob_yields_nothing(indexed):
    embs, rows = indexed
    core_only = embs[[i for i, r in enumerate(rows) if r["category"] == "bio"][:12]]
    # All identical axis-A vectors → one component → no divergence.
    assert s._split_candidates(core_only, threshold=0.70) == ([], [])


# --- structural scan ---------------------------------------------------------

def test_detect_divergence_candidates(indexed):
    cands = s.detect_divergence_candidates(threshold=0.70, min_papers=12)
    assert len(cands) == 1
    c = cands[0]
    assert c["category"] == "bio"
    assert c["n_total"] == 15
    assert [len(sc) for sc in c["subclusters"]] == [3]
    # the 3 divergent stems, not the core
    assert set(c["subclusters"][0]) == {"bio-12", "bio-13", "bio-14"}


def test_detect_respects_min_papers(indexed):
    # Raise the floor above bio's size → nothing qualifies.
    assert s.detect_divergence_candidates(threshold=0.70, min_papers=20) == []


def test_detect_ignores_other_bucket(indexed):
    # `other` is handled by the default mode; within-category scan skips it.
    cands = s.detect_divergence_candidates(threshold=0.70, min_papers=1)
    assert all(c["category"] != "other" for c in cands)


# --- command flow ------------------------------------------------------------

def test_category_mode_split_out_migration(indexed, capsys):
    def stub(rows):
        return {"verdict": "split_out", "slug": "rna-biology",
                "scope": "RNA-focused work", "rationale": "distinct method family"}

    ns = argparse.Namespace(threshold=0.70, category="bio", all_categories=False)
    rc = s._run_category_mode(ns, judge_fn=stub)
    out = capsys.readouterr().out
    assert rc == 0
    assert "SPLIT OUT of `bio` → NEW category `rna-biology`" in out
    assert "mv wiki/bio/bio-12.md wiki/rna-biology/bio-12.md" in out
    assert "category: [bio]` → `category: [rna-biology]" in out


def test_category_mode_threshold_none_uses_mode_default(indexed, capsys):
    # argparse passes threshold=None when --threshold is omitted; the mode must
    # resolve it to CATEGORY_DIVERGENCE_COSINE_THRESHOLD, not pass None into the
    # cosine comparison (which would raise). Synthetic axes split identically at
    # any threshold in (0,1), so the candidate still surfaces.
    def stub(rows):
        return {"verdict": "split_out", "slug": "rna-biology",
                "scope": "sc", "rationale": "r"}

    ns = argparse.Namespace(threshold=None, category="bio", all_categories=False)
    assert s._run_category_mode(ns, judge_fn=stub) == 0
    assert "SPLIT OUT of `bio`" in capsys.readouterr().out


def test_other_mode_threshold_none_uses_mode_default(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()
    embs = np.vstack([_A, _A, _A]).astype(np.float32)
    rows = [_row("other", i, _A) for i in range(3)]
    monkeypatch.setattr(s.pages_semantic, "load_index", lambda: (embs, rows), raising=True)
    monkeypatch.setattr(s, "is_valid", lambda c: c in {"other"}, raising=True)
    ns = argparse.Namespace(threshold=None, category=None, all_categories=False)
    assert s._run_other_mode(ns, judge_fn=lambda r: {"verdict": "stay", "rationale": "x"}) == 0


def test_category_mode_rejects_existing_slug(indexed, capsys):
    # Judge proposes a slug that already exists → downgraded to STAY, no mv.
    def stub(rows):
        return {"verdict": "split_out", "slug": "chem", "rationale": "x"}

    ns = argparse.Namespace(threshold=0.70, category="bio", all_categories=False)
    s._run_category_mode(ns, judge_fn=stub)
    out = capsys.readouterr().out
    assert "already exists" in out
    assert "mv wiki/bio/" not in out


def test_category_mode_unknown_category_errors(indexed, capsys):
    ns = argparse.Namespace(threshold=0.70, category="nope", all_categories=False)
    assert s._run_category_mode(ns, judge_fn=lambda r: None) == 1


def test_all_mode_scans_and_writes_stamp(indexed, tmp_path):
    from researchwiki import categories
    ns = argparse.Namespace(threshold=0.70, category=None, all_categories=True)
    rc = s._run_category_mode(ns, judge_fn=lambda r: {"verdict": "stay", "rationale": "ok"})
    assert rc == 0
    # --all always dismisses the nudge for the decay window.
    assert (tmp_path / categories.CATEGORY_DIVERGENCE_STAMP).exists()


# --- status warning + decay --------------------------------------------------

def test_divergence_warning_fires_then_decays(indexed, tmp_path):
    from researchwiki import categories
    msg = s.divergence_warning(threshold=0.70)
    assert msg is not None
    assert "bio" in msg
    assert "suggest-splits --category bio" in msg
    # Stamp written → within the decay window the warning suppresses.
    assert (tmp_path / categories.CATEGORY_DIVERGENCE_STAMP).exists()
    assert s.divergence_warning(threshold=0.70) is None


def test_divergence_warning_none_when_cohesive(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()
    # One cohesive category, no separable sub-cluster.
    embs = np.vstack([_A] * 14).astype(np.float32)
    rows = [_row("bio", i, _A) for i in range(14)]
    monkeypatch.setattr(s.pages_semantic, "load_index", lambda: (embs, rows), raising=True)
    monkeypatch.setattr(s, "content_categories", lambda: frozenset({"bio", "other"}), raising=True)
    assert s.divergence_warning(threshold=0.70) is None


# --- default (other-bucket) mode still works ---------------------------------

def test_other_mode_unchanged(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()
    embs = np.vstack([_A, _A, _A]).astype(np.float32)
    rows = [_row("other", i, _A) for i in range(3)]
    monkeypatch.setattr(s.pages_semantic, "load_index", lambda: (embs, rows), raising=True)
    monkeypatch.setattr(s, "is_valid", lambda c: c in {"other"}, raising=True)

    def stub(rows):
        return {"verdict": "new_category", "slug": "genomics",
                "scope": "sc", "rationale": "r"}

    ns = argparse.Namespace(threshold=0.70, category=None, all_categories=False)
    rc = s._run_other_mode(ns, judge_fn=stub)
    out = capsys.readouterr().out
    assert rc == 0
    # source category templated as `other`, matching legacy behaviour.
    assert "mv wiki/other/other-0.md wiki/genomics/other-0.md" in out
