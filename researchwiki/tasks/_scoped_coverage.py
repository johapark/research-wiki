"""Scope-aware local counterpart of the whole-library advisory coverage task."""
import json
from pathlib import Path

from ..synthesis.coverage import coverage


def run(args, frontmatter):
    path = Path(args.path).resolve()
    seed = frontmatter['topic_seed']
    hits = coverage(path.read_text(encoding='utf-8'), seed, args.evidence_scope, top_n=args.top_n)
    report = {'path': str(path), 'topic_seed': seed, 'top_n': args.top_n, 'unreferenced': hits,
              'scope': {**args.evidence_scope.to_dict(), 'fingerprint': args.evidence_scope.fingerprint}}
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f'Coverage: {len(hits)} unreferenced selected papers (bounded local retrieval).')
        for hit in hits:
            print(f"  [[{hit['key']}]]")
    return 1 if hits else 0
