"""Add one or more PDFs through the complete research-wiki ingest workflow."""

from __future__ import annotations


def main(argv: list[str]) -> int:
    """Forward to ``agent ingest`` so both commands keep one implementation."""
    from . import agent

    return agent.main(["ingest", *argv])
