"""Generate and verify a complete scoped synthesis; preview only, with no wiki writes.

Both API and chat-relay use synthesis_author/synthesis_support. --json returns
a version 1 provenance receipt. Use publish-synthesis for explicit publication.
"""
import argparse
import json

from ..synthesis import compose
from ._scope_file import add_scope_argument, load_scope


def main(argv):
    parser = argparse.ArgumentParser(prog='researchwiki compose', description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--title', required=True)
    parser.add_argument('--json', action='store_true')
    add_scope_argument(parser)
    args = parser.parse_args(argv)
    load_scope(args, parser)
    if args.evidence_scope is None:
        parser.error('compose requires an explicit --scope-file')
    try:
        result = compose(args.title, args.question, args.evidence_scope)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result.to_dict(), indent=2) if args.json else result.markdown())
    return 0 if result.status == 'verified' else 1
