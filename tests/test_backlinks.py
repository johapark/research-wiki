"""`researchwiki.backlinks.append_related_paper` — idempotency across link forms.

A back-link should not be duplicated when the target already links the source
in ANY wikilink form: `[[category/stem]]`, bare `[[stem]]` (the form CLAUDE.md
mandates in tables), an aliased `[[stem|…]]`, or a claim anchor `[[stem#slug]]`.
"""

import pytest

from researchwiki.backlinks import append_related_paper


def _page(tmp_path, body: str):
    p = tmp_path / "target.md"
    p.write_text(body, encoding="utf-8")
    return p


def test_appends_when_absent(tmp_path):
    p = _page(tmp_path, "---\ntype: paper\n---\n\n## Related Papers\n\n- [[cgt/other]] — x\n")
    assert append_related_paper(p, "cgt/smith-2024-x") is True
    assert "[[cgt/smith-2024-x]]" in p.read_text(encoding="utf-8")


def test_idempotent_on_full_key(tmp_path):
    p = _page(tmp_path, "## Related Papers\n\n- [[cgt/smith-2024-x]] — already here\n")
    assert append_related_paper(p, "cgt/smith-2024-x") is False


def test_idempotent_on_bare_stem_in_table(tmp_path):
    # CLAUDE.md mandates bare `[[stem]]` in markdown tables; the full-key
    # back-link must recognize that as already-linked and not duplicate it.
    body = ("Some prose.\n\n"
            "| Paper | Note |\n|---|---|\n| [[smith-2024-x]] | a |\n")
    p = _page(tmp_path, body)
    assert append_related_paper(p, "cgt/smith-2024-x") is False


def test_idempotent_on_claim_anchor(tmp_path):
    p = _page(tmp_path, "Grounded via [[smith-2024-x#kc-9f3a2b1c]] in the prose.\n")
    assert append_related_paper(p, "cgt/smith-2024-x") is False


def test_idempotent_on_aliased_link(tmp_path):
    p = _page(tmp_path, "See [[smith-2024-x|Smith's method]] for details.\n")
    assert append_related_paper(p, "cgt/smith-2024-x") is False


def test_similar_stem_prefix_not_confused(tmp_path):
    # A different paper whose stem is a prefix of the source stem must NOT
    # suppress the back-link — the terminator class ([\]|#]) prevents that.
    p = _page(tmp_path, "## Related Papers\n\n- [[cgt/smith-2024-x-extended]] — other\n")
    assert append_related_paper(p, "cgt/smith-2024-x") is True
