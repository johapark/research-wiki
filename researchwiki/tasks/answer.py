"""Answer a question from an explicit paper selection, with mandatory PDF checks.

Uses answer and answer_judge model phases (API, chat-relay, or mixed routing).
No wiki page or saved output is written. Rejected drafts are hidden in prose;
JSON exposes the complete validation receipt for inspection.
"""
import argparse
import json

from ..answers import answer
from ._scope_file import add_scope_argument, load_scope


def main(argv):
    parser = argparse.ArgumentParser(prog="researchwiki answer", description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--json", action="store_true")
    add_scope_argument(parser)
    args = parser.parse_args(argv)
    load_scope(args, parser)
    if args.evidence_scope is None:
        parser.error("answer requires --scope-file with an explicit paper selection")
    try:
        result = answer(args.question, args.evidence_scope)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result.to_dict(), indent=2) if args.json else result.markdown(), end="\n")
    return 0 if result.status == "verified" else 1
