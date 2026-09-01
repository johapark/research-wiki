"""Add one or more PDFs through the complete research-wiki ingest workflow."""

from __future__ import annotations


def main(argv: list[str]) -> int:
    """Forward to ``agent ingest`` so both commands keep one implementation.

    `ingest_prog` makes the shared implementation name itself as `researchwiki
    add` in usage lines and error prefixes. Without it a first-run user who
    typed `add` is answered by a command they have never heard of.
    """
    from . import agent

    return agent.main(["ingest", *argv], ingest_prog="researchwiki add")
