"""Reconcile journalled paths from current canonical files, without model calls."""

from pathlib import Path

from .paths import wiki_dir


def reconcile(entries, operation: str, details: dict) -> None:
    """Repair promotion/removal projections, including legacy v1 receipts."""
    if not details.get("repair_projections") and not (
        operation in {"promote", "remove"} and details.get("stem")
    ):
        return
    from .db.rebuild import upsert_page, delete_page
    from .index.incremental import update_page_indexes
    from .mutation import corpus_lock

    root = wiki_dir().resolve()
    paths = [entry.target for entry in entries
             if entry.target.suffix == ".md" and root in entry.target.resolve().parents
             and entry.target.resolve().parent != root]
    if not paths:
        return
    with corpus_lock():
        from .index.pages_semantic import mark_stale
        mark_stale()
        for path in paths:
            if path.exists():
                page, _ = upsert_page(path)
                if page is None:
                    raise ValueError(f"cannot repair malformed page: {path}")
            else:
                # A stale receipt must not delete a page since moved elsewhere.
                matches = list(root.rglob(f"{path.stem}.md"))
                if matches:
                    page, _ = upsert_page(matches[0])
                    if page is None:
                        raise ValueError(f"cannot repair malformed page: {matches[0]}")
                else:
                    delete_page(path.stem)
        stem = details.get("stem")
        if operation == "remove" and stem and not list(root.rglob(f"{stem}.md")):
            from .removal import _delete_db_rows, _delete_edges
            _delete_db_rows(str(stem))
            _delete_edges(str(stem))
        # Recovery refreshes keyword search and explicitly marks the semantic
        # view unavailable. The regular reindex/ingest path refreshes embeddings.
        update_page_indexes([Path(p) for p in paths], semantic=False)
