"""Scout structured citation metadata for connections and missing papers.

``researchwiki scout`` and ``researchwiki scout citations`` are equivalent.
Citation mode uses only structured Semantic Scholar metadata. Web mode is an
agent-handoff protocol: the CLI itself performs no search and quarantines only
a minimal source receipt as discovery-only.
"""

from __future__ import annotations

from ..scouting import citations
from ..scouting import web_cli


def main(argv: list[str]) -> int:
    if argv and argv[0] == "citations":
        return citations.main(
            argv[1:],
            prog="researchwiki scout citations",
        )
    if argv and argv[0] == "web":
        return web_cli.main(argv[1:])
    return citations.main(argv, prog="researchwiki scout")
