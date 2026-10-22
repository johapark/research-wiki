"""Deterministic academic page rendering; model responses never supply Markdown."""
from collections import Counter
import json

import yaml


def render(receipt):
    draft = receipt.draft
    if draft is None or not receipt.models:
        raise ValueError("synthesis has no attributable draft")
    evidence = {e.evidence_id: e for e in receipt.evidence}
    sources = {s.stem: s for s in receipt.sources}
    used = sorted({evidence[eid].stem for unit in draft.units() for eid in unit.evidence_ids})
    categories = Counter(sources[stem].key.split('/')[0] for stem in used)
    frontmatter = {"title": receipt.title, "type": "synthesis", "category": [categories.most_common(1)[0][0]],
                   "generated_at": receipt.created_at[:10],
                   "topic_seed": receipt.question, "tags": ["synthesis"], "hook": draft.hook.text,
                   "source_fingerprint": receipt.scope.fingerprint, "generation_receipt": receipt.digest}
    header = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).rstrip()
    header += '\nauthor_model: ' + json.dumps(receipt.models[0].model)
    lines = ["---", header, "---", "",
             "## Question", "", receipt.question.rstrip('?') + '?', ""]

    def paragraph(unit):
        refs = list(dict.fromkeys(evidence[eid].citation for eid in unit.evidence_ids))
        anchors = [ref for ref in refs if '#' in ref]
        stems = sorted({evidence[eid].stem for eid in unit.evidence_ids})
        citations = ' '.join(anchors) + ''.join(f'[^{stem}]' for stem in stems)
        return f"{unit.text} {citations.strip()}"

    for section in draft.sections:
        if section.heading == 'Cross-cutting insights':
            lines.extend(['## Findings', ''])
            for theme in draft.themes:
                lines.extend([f'### {theme.title.text}', ''])
                for unit in theme.paragraphs:
                    lines.extend([paragraph(unit), ''])
        lines.extend([f'## {section.heading}', ''])
        for unit in section.paragraphs:
            lines.extend([paragraph(unit), ''])
        if section.heading == 'Organizing framework':
            lines.extend(['| Paper | Theme |', '| --- | --- |'])
            for stem in used:
                themes = [theme.title.text for theme in draft.themes if any(
                    evidence[eid].stem == stem for unit in theme.paragraphs for eid in unit.evidence_ids)]
                label = '; '.join(themes) or 'Context'
                lines.append(f'| [[{stem}]] | {label.replace("|", " ")} |')
            lines.append('')
    lines.extend(['## References', ''])
    lines.extend(f'[^{stem}]: [[{sources[stem].key}]]' for stem in used)
    return '\n'.join(lines).rstrip() + '\n'
