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
    from .db.verify import complete_markdown_scan
    from .index.incremental import update_page_indexes
    from .mutation import corpus_lock

    root = wiki_dir().resolve()
    paths = [entry.target for entry in entries
             if entry.target.suffix == ".md" and root in entry.target.resolve().parents
             and entry.target.resolve().parent != root]
    if not paths:
        return
    with corpus_lock():
        # A complete scan, or none: a page hidden by an unreadable folder must
        # not read as deleted. Scan failure keeps the receipt for retry.
        by_stem: dict[str, Path] = {}
        for md in complete_markdown_scan(root):
            by_stem.setdefault(md.stem, md)
        for path in paths:
            if path.exists():
                page, _ = upsert_page(path)
                if page is None:
                    raise ValueError(f"cannot repair malformed page: {path}")
            else:
                # A stale receipt must not delete a page since moved elsewhere.
                moved = by_stem.get(path.stem)
                if moved is not None:
                    page, _ = upsert_page(moved)
                    if page is None:
                        raise ValueError(f"cannot repair malformed page: {moved}")
                else:
                    delete_page(path.stem)
        stem = details.get("stem")
        if operation == "remove" and stem and str(stem) not in by_stem:
            from .removal import _delete_db_rows, _delete_edges
            _delete_db_rows(str(stem))
            _delete_edges(str(stem))
        # Keyword search is refreshed here without a model. The semantic view
        # drops outdated rows instead of going dark: marking it stale on every
        # promote/remove switched page KNN off for the rest of a batch.
        from .index.pages_semantic import drop_outdated
        from .wiki import read_page
        present = [page for path in paths if (page := read_page(by_stem.get(path.stem, path)))]
        absent = {path.stem for path in paths if path.stem not in by_stem}
        drop_outdated(present, absent)
        update_page_indexes([Path(p) for p in paths], semantic=False, mark_semantic_stale=False)
