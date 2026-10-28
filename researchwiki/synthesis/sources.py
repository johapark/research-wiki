"""Snapshot and revalidate selected canonical sources, without reading other papers."""
from dataclasses import replace
from pathlib import Path

from ..errors import EnvironmentFailure
from ..fsatomic import file_sha256
from ..wiki import read_page


def snapshot_scope(scope, root=None):
    if root is None:
        scope.check_root()
    root = Path(root or Path.cwd()).resolve()
    revisions = []
    for stem in sorted(scope.stems):
        pages = list((root / "wiki").glob(f"*/{stem}.md"))
        if len(pages) != 1:
            raise EnvironmentFailure(f"paper page missing or ambiguous: {stem}")
        page = read_page(pages[0])
        if page is None or page.page_type != "paper":
            raise ValueError(f"synthesis evidence must be an ingested paper: {stem}")
        try:
            revisions.append((stem, file_sha256(pages[0]), file_sha256(root / "papers" / f"{stem}.pdf")))
        except OSError as exc:
            raise EnvironmentFailure(f"synthesis source unavailable: {stem}") from exc
    current = replace(scope, library_root=str(root), source_revisions=tuple(revisions))
    if scope.source_revisions and scope.source_revisions != current.source_revisions:
        raise ValueError("synthesis source snapshot is stale; resolve the selection again")
    return current
