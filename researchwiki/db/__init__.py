"""SQLite-backed structured index over the wiki — the V1 plan's memory layer.

The DB is a derived index, rebuildable from markdown + caches. Markdown stays
canonical; the DB never holds prose that should be edited by hand. See
schema.sql for the full schema and invariants.

Public API is bound lazily so importing this package doesn't open a
connection or run migrations until something actually needs the DB.
"""

from __future__ import annotations


def get_connection(*args, **kwargs):
    from .connection import get_connection as _impl
    return _impl(*args, **kwargs)


def db_path(*args, **kwargs):
    from .connection import db_path as _impl
    return _impl(*args, **kwargs)


def init_schema(*args, **kwargs):
    from .connection import init_schema as _impl
    return _impl(*args, **kwargs)


def rebuild(*args, **kwargs):
    from .rebuild import rebuild as _impl
    return _impl(*args, **kwargs)


def upsert_page(*args, **kwargs):
    from .rebuild import upsert_page as _impl
    return _impl(*args, **kwargs)


def delete_page(*args, **kwargs):
    from .rebuild import delete_page as _impl
    return _impl(*args, **kwargs)


def find_by_doi(*args, **kwargs):
    from .rebuild import find_by_doi as _impl
    return _impl(*args, **kwargs)


def verify(*args, **kwargs):
    from .verify import verify as _impl
    return _impl(*args, **kwargs)


def write_iteration(*args, **kwargs):
    from .iterations import write_iteration as _impl
    return _impl(*args, **kwargs)


def read_attempt(*args, **kwargs):
    from .iterations import read_attempt as _impl
    return _impl(*args, **kwargs)


def update_paper_stem(*args, **kwargs):
    from .iterations import update_paper_stem as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "get_connection", "db_path", "init_schema", "rebuild",
    "upsert_page", "delete_page", "find_by_doi", "verify",
    "write_iteration", "read_attempt", "update_paper_stem",
]
