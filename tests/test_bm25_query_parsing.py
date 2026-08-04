"""`more_like_text`'s lenient-parse contract against the installed tantivy.

The bug this pins: `pages_bm25.more_like_text` called the MODULE-level
`tantivy.parse_query_lenient(text, schema, field_names)`, but in tantivy 0.26
that function takes only `(query)` — the field-aware lenient parser is the
`Index` METHOD. So every call raised TypeError and dropped into the strict
`parse_query` fallback, which then died on punctuation-heavy seed text with
`ValueError: Syntax Error: …`. Seen while sweeping See-Also over the corpus:
`hybrid_more_like` propagated it out of the BM25 ranker.

A signature check rather than an end-to-end query, because the defect was
never about retrieval behaviour — the code asked the dependency for a call
signature it does not have. That is what would break again on a tantivy
upgrade, and it needs no index fixture to detect.
"""
from __future__ import annotations

import inspect

import tantivy


def test_index_has_the_field_aware_lenient_parser():
    """`more_like_text` relies on the METHOD taking (query, field_names)."""
    assert hasattr(tantivy.Index, "parse_query_lenient")
    params = list(inspect.signature(tantivy.Index.parse_query_lenient).parameters)
    # `self` is present because this is the unbound method.
    assert params[:2] == ["self", "query"]
    assert "default_field_names" in params


def test_module_level_lenient_parser_cannot_take_fields():
    """Documents why the method is used, and fails loudly if that ever flips.

    If a future tantivy gives the module-level function a field-name
    parameter, the comment in `more_like_text` explaining the choice becomes
    stale and this test is the prompt to revisit it.
    """
    params = list(inspect.signature(tantivy.parse_query_lenient).parameters)
    assert params == ["query"], (
        "tantivy.parse_query_lenient gained parameters; re-check the "
        "method-vs-function choice in pages_bm25.more_like_text"
    )


def test_more_like_text_calls_the_method_not_the_function():
    """Guards the specific line, since both spellings type-check fine."""
    from researchwiki.index import pages_bm25

    src = inspect.getsource(pages_bm25.TantivySearchBackend.more_like_text)
    assert "idx.parse_query_lenient(" in src
    assert "tantivy.parse_query_lenient(" not in src
