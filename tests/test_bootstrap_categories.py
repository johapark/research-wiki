"""Review-gated cold-start taxonomy proposal and application."""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.categories import PAGE_TYPE_DIRS
from researchwiki.errors import EnvironmentFailure
from researchwiki.tasks import bootstrap_categories as bc


def _seed_inbox(root: Path) -> list[Path]:
    inbox = root / "inbox"
    inbox.mkdir()
    pdfs = []
    for i in range(bc.MIN_INBOX_FOR_BOOTSTRAP):
        pdf = inbox / f"paper-{i}.pdf"
        pdf.write_bytes(b"%PDF-1.4\n" + bytes([i]))
        pdfs.append(pdf)
    return pdfs


def _mock_proposer(monkeypatch, calls: list[str]) -> None:
    monkeypatch.setattr(
        bc,
        "_gather_inbox_metadata",
        lambda pdfs: [
            {"title": pdf.stem, "year": 2026, "venue": "Test", "excerpt": "x"}
            for pdf in pdfs
        ],
    )

    def propose(_bag, _max_cats):
        calls.append("provider")
        return {
            "categories": [
                {"slug": "genomics", "scope": "Genome-focused methods."},
                {"slug": "other", "scope": "Unclassified work."},
            ],
            "rationale": "The staged papers form one coherent group.",
        }

    monkeypatch.setattr(bc, "_call_proposer", propose)


@pytest.mark.parametrize("reserved", sorted(PAGE_TYPE_DIRS))
def test_reserved_page_type_names_cannot_satisfy_category_validation(
    reserved, capsys,
):
    parsed = {
        "categories": [
            {"slug": reserved, "scope": "Not a content category."},
            {"slug": "other", "scope": "Fallback."},
        ]
    }

    assert bc._validate_categories(parsed, 3) is None
    assert f"reserved page-type directory `{reserved}`" in capsys.readouterr().err


def test_preview_then_apply_reuses_exact_receipt_without_second_model_call(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _seed_inbox(tmp_path)
    calls: list[str] = []
    _mock_proposer(monkeypatch, calls)
    monkeypatch.setattr(bc, "log", lambda *args, **kwargs: None)

    assert bc.main([]) == 0
    assert calls == ["provider"]
    assert bc._proposal_receipt_path().is_file()
    assert not (tmp_path / "wiki" / "genomics").exists()

    monkeypatch.setattr(
        bc,
        "_call_proposer",
        lambda *_args: pytest.fail("apply must not call the provider"),
    )
    assert bc.main(["--apply"]) == 0
    assert (tmp_path / "wiki" / "genomics").is_dir()
    assert calls == ["provider"]


def test_apply_rejects_inbox_changes_after_review(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    pdfs = _seed_inbox(tmp_path)
    calls: list[str] = []
    _mock_proposer(monkeypatch, calls)

    assert bc.main([]) == 0
    pdfs[0].write_bytes(b"%PDF-1.4\nchanged body")

    assert bc.main(["--apply"]) == 1
    assert not (tmp_path / "wiki" / "genomics").exists()
    assert "inbox/ changed" in capsys.readouterr().err
    assert calls == ["provider"]


def test_apply_without_preview_never_calls_provider(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _seed_inbox(tmp_path)
    monkeypatch.setattr(
        bc,
        "_call_proposer",
        lambda *_args: pytest.fail("apply without a receipt must fail locally"),
    )

    assert bc.main(["--apply"]) == 1
    assert "run `researchwiki bootstrap-categories` first" in capsys.readouterr().err


def test_apply_reports_category_directory_write_failures_as_environment_errors(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    proposal = bc.TaxonomyProposal(
        categories=[
            {"slug": "genomics", "scope": "Genome-focused methods."},
            {"slug": "other", "scope": "Unclassified work."},
        ],
        rationale="One coherent group.",
        n_papers=bc.MIN_INBOX_FOR_BOOTSTRAP,
        max_categories=2,
        inbox_fingerprint=[],
    )

    def deny_category_write(self, *args, **kwargs):
        raise PermissionError("read-only synced folder")

    monkeypatch.setattr(Path, "mkdir", deny_category_write)

    with pytest.raises(EnvironmentFailure, match="cannot create category directory"):
        bc._apply_taxonomy(proposal)


def test_shipped_prompt_forbids_every_page_type_directory():
    prompt = (Path(__file__).resolve().parents[1] / "prompts" /
              "bootstrap-categories-system.md").read_text(encoding="utf-8")
    assert "Reserved names are forbidden" in prompt
    for reserved in PAGE_TYPE_DIRS:
        assert f"`{reserved}`" in prompt
    assert "`ai`, `references`, `other`" not in prompt
