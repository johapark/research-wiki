# Proposal scoring handbook — rubric v2

Status: **agent-calibrated, not human-validated**. The user delegated the initial
calibration to the active chat agent. This is a reasoned editorial policy, not
measured agreement with the user's preferences. See [calibration record](./CALIBRATION.md).

## What is being scored

A request returns zero to three proposals. Score each proposal, then the request
as a set. A synthesis should explain cross-paper relationships and identify what
would change its conclusion. An idea should propose a mechanism and a rejecting
test. A transfer idea must additionally connect a supplied source operation to a
target operation and explain where the analogy could fail.

The benchmark measures justified next steps, not eventual experimental success
or worldwide novelty. Novelty is assessed relative to the supplied existing work.

## Review procedure

Review anonymized outputs in randomized order. Read the input and evaluator
criteria, then identify the proposed contribution and its load-bearing factual
premises. Check those premises against claim IDs and supplied PDF text. A premise
is load-bearing if removing it undermines the central argument. Distinguish a
hypothesis from an assertion of an already-observed result.

For every axis, record an integer 0–2 plus a reason referencing output passages
and evidence IDs where applicable. Record a direct shortlist/defer/reject decision
and its reason independently. Compare with existing-page bodies, not only titles.
Use evaluator examples as possibilities, never as exact-match answer keys.

### Questions to ask in every review

1. What useful decision is missing from the request and existing pages?
2. What does this proposal add beyond the framing already supplied in the prompt?
3. Which factual premise carries the argument, and does its cited evidence support it?
4. What is the smallest credible next action? What observation would change the decision?
5. Would a simpler alternative answer the same question? What added benefit justifies complexity?
6. Does the response actually obey scope and feedback, rather than merely acknowledge them?
7. Shortlist, defer, or reject as submitted—and what specific action or blocker explains that verdict?

Record the direct verdict before calculating the total. Assess the actual text,
not an improved proposal the reviewer could write. Distinguish insight supplied
by the request from insight contributed by the response: repeating a requested
comparison or experimental axis alone earns no insight credit. This suite measures
question-conditioned ideation, not autonomous discovery of research questions.

| Axis | 0 | 1 | 2 |
|---|---|---|---|
| Evidence | Central premise unsupported, contradicted, or misattributed | Relevant support but attribution, inference, or limits remain ambiguous | Load-bearing premises supported; uncertainty and hypothesis status explicit |
| Insight | Topic grouping, restatement, or empty terminology | Plausible connection with an underdeveloped explanatory consequence | Specific explanatory distinction, mechanism, or testable implication |
| Contribution | Repeats existing work or ignores a material decision | A modest difference without a clear remaining question | A specific unresolved question with a defensible distinction |
| Decision value | No consequence or way to assess it | General comparison, experiment, or update condition | Discriminating evidence would support, reject, or revise the proposal |
| Proportionality | Complexity without justification or an infeasible next step | Costs/alternatives acknowledged but tradeoff underspecified | Credible simpler alternative and feasible scope; added complexity must earn its cost |

Full decision-value credit for a synthesis does not require an experiment.
Clear discriminators, boundary conditions, and updating evidence can suffice.
A small proposal does not need elaborate cost accounting to earn proportionality
credit. Do not reward word count, numbers, field count, or sophisticated terminology.

An experiment's named metric and baseline do not guarantee decision value. Check
that the comparison isolates the claimed benefit, costs use compatible units (or
separate resource limits), and the rejection rule can be applied without logical
contradiction. Matching a simpler baseline without an offsetting benefit is not
evidence that extra complexity is worthwhile. A comparison may remain inconclusive;
failure to detect an effect does not by itself establish equivalence.

Record two explicit flags:

- `critical_factual_failure`: an unsupported load-bearing factual assertion.
- `constraint_failure`: violating an explicit request restriction, such as
  proposing outside a closed scope or claiming a transfer with no source method.

A correctly labeled but uncertain design hypothesis is neither flag. Peripheral
imprecision can lower evidence quality without being critical. Unsupported
performance claims that justify the entire design are critical. Invalid JSON is
an operational error, not automatically a factual one.

## Shortlist, defer, reject

- **Shortlist:** merits a concrete next action, such as scoping a page or designing
  a bounded experiment. It need not already warrant publication.
- **Defer:** potentially useful but needs a specific missing source, clarification,
  or smaller feasibility test before that next action is justified.
- **Reject:** no useful contribution in scope, a fatal unsupported premise, or a
  constraint violation.

A shortlist is permission to scope, not automatic permission to draft a full page
or deploy a method. State the next action in `decision_reason`. Missing details
that can be chosen during scoping are tolerable; unresolved contribution, an
incoherent comparison, or a missing load-bearing source warrants defer or reject.
A design flaw is not automatically a factual failure: keep methodological defects
in decision value/proportionality unless a false factual assertion causes them.

The provisional numerical rule is total ≥8/10, no zero axis, and neither critical
flag. During calibration, human decisions define primary utility. The rule is a
diagnostic proxy and its disagreements are reported separately. Agent-only reviews
instead yield **agent-assessed utility**, with `reviewer_kind: agent`; they must
never be described as human judgments. Do not pool reviewer kinds. Before release
evaluation, resolve disagreements on development cases and freeze the resulting
policy. Do not adjust a rule after viewing held-out model performance.

If a strong unexpected solution contradicts an abstention label, adjudicate the
case rather than force the output to fail. The scorer refuses that contradiction.
If an opportunity lacks credible positive approaches after review, revise the
fixture and rerun all candidates on the new version. Until resolved, do not
publish a complete suite score. Curator examples are not proof of feasibility.

## Metric definitions

Compute metrics separately for each candidate, repeat, split, and execution mode.
Never pool fixed-packet and end-to-end runs. The review scorer accepts one block
at a time and requires every selected case exactly once.

| Metric | Numerator | Denominator |
|---|---|---|
| Useful proposal yield | Opportunity requests with a processable, shortlisted output | All opportunity requests |
| Proposal precision | Processable shortlisted proposals | All emitted proposals, including diagnostically reviewed malformed outputs |
| Appropriate abstention | Abstention-required requests with a valid empty output | All abstention-required requests |
| Unnecessary abstention | Opportunity requests with a valid empty output | All opportunity requests |
| Critical factual failure | Emitted proposals with the critical factual flag | All emitted proposals |
| Feedback compliance | Successful revision requests satisfying every listed requirement and required parent linkage | All feedback cases |
| Operational success | Requests passing output validation | All requests |

Zero denominators are `null`, never a perfect score. A transport or parser failure
does not count as abstention. It remains in request denominators, and no raw
proposal from an operational failure contributes to useful yield or precision's
numerator. Keep raw content scores for diagnosis. If no interpretable proposals
were returned, use an empty proposal list with operational_success=false.

The v2 suite has nine opportunity cases and three required-abstention cases;
development has five opportunities and three abstentions, heldout four
opportunities. Heldout abstention and feedback metrics are therefore unavailable.
Do not claim held-out coverage of those behaviors. The next expansion should add
new-topic restraint/feedback cases without copying development topics.

Inspect within-set duplicates and required output type during review. Reworded
copies receive low contribution scores; returning three versions of one idea
does not constitute three useful ideas. Compare requests at the same maximum
proposal count and comparable input/output budgets. Preserve actual input/output
tokens, retries, latency, and monetary cost separately; do not sum incompatible
units into an undefined budget.

## Feedback and paired cases

For each listed feedback requirement, record passed=true/false and a reason.
Acknowledging feedback is insufficient; the mechanism, scope, or proposed test
must change when required. F01 and F02 explicitly request revisions, so every
returned revision must carry the supplied parent ID. Independent proposals in
other cases need not manufacture a parent.

X01/X02 tests the effect of removing source-method evidence. I01/R02 changes scope
as well as context, and I01/F02 introduces a specific prior design and constraint.
Do not describe those latter differences as isolated effects of a single field.

## Calibration and release comparison

Start with two independent human ratings on a small development sample that
includes good, plausible-but-weak, unsupported, and correctly abstaining outputs.
Where a second reviewer is unavailable, disclose single-reviewer calibration;
do not manufacture consensus. Reconcile score disagreements with reasons and
record any handbook changes. A model judge may later assist, but first measure
its agreement and failure cases against these human judgments. In particular,
check verbose unsupported answers and terse valid answers for style bias.

When the user delegates calibration to an agent, that agent may perform a local
editorial review instead. Preserve raw input/output receipts, source hashes,
per-axis reasons, verdicts, and explicit reviewer provenance. Self-review is not
independent validation. Constructed counterexamples may clarify rubric boundaries,
but label them synthetic and exclude them from model-performance counts. Historical
outputs with different packets cannot stand in for missing current-suite runs.

Freeze eight development cases before evaluating the four new held-out groups.
Use the same model and fixed packets for a simple prompt baseline versus the
production proposal generator. This measures generation quality, not retrieval
or search planning; evaluate those separately in end-to-end mode. Three runs per
case are a variance check, not three independent topics.
Report counts, per-case outcomes, mean/range across repeats, and blind pairwise
preferences with ties permitted. Small heldout counts do not justify a broad
statistical superiority claim.

Hard release checks should include output validity and no unresolved critical
factual/constraint regressions. Quality acceptance should compare the full
workflow with its baseline on yield and precision, inspect every regression, and
keep costs visible. Do not declare an arbitrary target percentage empirically
calibrated before this first human review round.
