"""Explicitly publish a verified synthesis receipt after fresh source and page checks."""
import argparse
import json
from pathlib import Path

from ..errors import EnvironmentFailure
from ..grade.generated import strict_json
from ..synthesis import SynthesisReceipt
from ..synthesis.publication import publish
from ._scope_file import add_scope_argument, load_scope


def main(argv):
    parser = argparse.ArgumentParser(prog='researchwiki publish-synthesis', description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--guard-file', type=Path)
    parser.add_argument('--guard-sha256')
    parser.add_argument('--json', action='store_true')
    add_scope_argument(parser)
    args = parser.parse_args(argv)
    load_scope(args, parser)
    if args.evidence_scope is None:
        parser.error('publication requires the current --scope-file')
    try:
        raw = args.receipt.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as exc:
        raise EnvironmentFailure('cannot read the synthesis receipt') from exc
    try:
        data = strict_json(raw)
        receipt = SynthesisReceipt.from_dict(data)
        result = publish(receipt, args.evidence_scope, args.slug,
                         guard_path=args.guard_file, guard_sha256=args.guard_sha256)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2) if args.json else f"{result['publication']}: {result['path']} ({result['projections']})")
    return 0
