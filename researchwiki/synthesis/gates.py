"""Run the established page gates on the exact rendered draft, without ML downloads."""
from pathlib import Path
import tempfile

from ..errors import EnvironmentFailure
from ..grade import grounding
from ..grade.fidelity import grade_synthesis
from ..tasks.lint.synthesis_contract import check_page
from .coverage import coverage
from .receipt import Gates


def page_gates(text, question, scope):
    anchors = grounding.extract_claim_anchors(text)
    try:
        resolved = grounding._resolve_claim_anchors(anchors)
    except grounding.ClaimDBUnavailable as exc:
        raise EnvironmentFailure('state.db unavailable; synthesis claim anchors could not be verified') from exc
    # Resolve every anchor explicitly. A valid paper footnote must not mask a
    # dangling inline claim anchor in the same paragraph (or a short unit).
    structural = grounding.check(text, permissive=False, valid_anchors=resolved)
    with tempfile.TemporaryDirectory(prefix='researchwiki-synthesis-gates-') as tmp:
        path = Path(tmp) / 'draft.md'
        path.write_text(text, encoding='utf-8')
        # The same fidelity gate as `grade synthesis --no-semantic`. Mandatory
        # qualitative support is checked separately for every structured unit.
        fidelity = grade_synthesis(path, semantic=False)
    violations = check_page(Path('wiki/synthesis/draft.md'), text)
    omissions = coverage(text, question, scope, top_n=max(20, len(scope.stems)))
    return Gates(grounding_total=structural.total_claims, ungrounded=len(structural.ungrounded_units),
                 fidelity_graded=fidelity.n_claims,
                 misattributed=fidelity.n_misattributed + fidelity.n_anchor_misattributed + fidelity.n_inference_ungradable,
                 unresolved=fidelity.n_uncited + sum(bool(c.unresolved_citations) for c in fidelity.claims),
                 weak=fidelity.n_weak, contract_violations=tuple(v['kind'] for v in violations),
                 dangling_anchors=len(set(anchors) - resolved),
                 coverage_omissions=tuple(hit['stem'] for hit in omissions))
