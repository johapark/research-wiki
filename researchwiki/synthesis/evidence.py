"""All selected papers get a bounded, source-checked evidence card."""
from dataclasses import replace
from pathlib import Path

from ..answers.retrieval import retrieve
from ..errors import EnvironmentFailure
from .receipt import Source


def sources_for(scope, root=None):
    sources = []
    for stem in sorted(scope.stems):
        paths = list((Path(root or Path.cwd()) / 'wiki').glob(f'*/{stem}.md'))
        if len(paths) != 1:
            raise EnvironmentFailure(f'synthesis source page missing or ambiguous: {stem}')
        sources.append(Source(stem=stem, key=f'{paths[0].parent.name}/{stem}'))
    return tuple(sources)


def gather(question, scope):
    evidence = []
    for stem in sorted(scope.stems):
        selected = replace(scope, stems=frozenset({stem}),
                           source_revisions=tuple(r for r in scope.source_revisions if r[0] == stem))
        for hit in retrieve(question, selected):
            evidence.append(replace(hit, evidence_id=f'e{len(evidence)}'))
    return tuple(evidence)
