# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versioning aims at [Semantic Versioning](https://semver.org/spec/v2.0.0.html) but is
not governed by it yet: SemVer's PATCH/MINOR/MAJOR clauses are each guarded `| x > 0`,
so at `0.y.z` only clause 4 applies ("anything MAY change at any time"). While 0.x this
project shifts the whole line down a slot — **breaking → MINOR, additive → PATCH** —
which is what `^0.4.0` and `~=0.4.0` already assume. See
[`CONTRIBUTING.md` § Releasing](./CONTRIBUTING.md#releasing).

**What counts as a breaking change here** is not a Python API — it's the CLI surface,
the exit-code contract, the `--json` key contracts, the page frontmatter contract, and
the persisted phase-role strings. See [`CONTRIBUTING.md` § Releasing](./CONTRIBUTING.md#releasing)
for the policy and the bump procedure.

Entries are curated, not generated: commit bodies in this repo carry the design
rationale, and a subject-line dump would throw exactly that away. Read `git log` for
the reasoning behind any line below.

## [Unreleased]

### Fixed

- **Two library modules were advertised as CLI commands and crashed when run.**
  `_discover_tasks` registers every non-underscore module under
  `researchwiki.tasks` — by name, without importing — so `claim_discover.py`
  (whose `discover_pairs()` backs `candidates pairs`) and `pair_dismissals.py`
  (the store behind `--decline`) both appeared in `researchwiki --help` and both
  died on dispatch with an `AttributeError: no attribute 'main'`, surfaced as exit
  3, "internal bug". Accurate, and useless to a caller: nothing about their command
  line was fixable.

  The leading-underscore convention was the intended signal and stays valid; what
  was missing is the invariant behind it, so `__main__._is_entry_point` now checks
  it at the two points the module object already exists. No extra imports on the
  hot path — `main()` imports exactly one module and checks that one. Naming a
  library module is now a user error (1) listing the real commands, not a crash,
  and neither appears in `--help`. A test pins the invariant rather than the two
  names, so a helper dropped into `tasks/` tomorrow cannot become a broken command
  by accident.

- **`WORKFLOW.md` documented neither the discovery tier nor five real commands.**
  It had zero occurrences of "discovery" or "bottom-up" — the entire
  proposal-generating half of the framework — and its all-commands table was
  missing `remove`, `visualize` and `figures` while listing `candidates` as
  `<concepts|synthesis>`, with no `pairs`. The `lint` row did not mention
  `--cross-paper`, its one LLM-costing surface. All 39 real commands are now
  present and the table is checked against the CLI rather than eyeballed.

- **Cross-paper contradiction scan: a 611 MB allocation and a corpus re-embed, on
  every ingest.** `find_cross_paper_contradictions` called `embed_texts` over every
  graded claim per invocation — that function has no cache of its own, so it was
  ~29 s at 12.4k claims — then built a full N×N similarity matrix (611 MB, measured)
  and walked its upper triangle in pure Python (78 M iterations, ~11 s). None of
  that was confined to the opt-in `lint --cross-paper` surface:
  `alert_after_ingest` is unconditional in the agent path, and it applied its
  `only_stem` filter *after* the scan, so every ingest paid the whole corpus cost
  and discarded 99.8% of the result.

  Three changes. Vectors now come from a new append-only claim-cache path, so a run
  embeds only what is genuinely new. The scan is blocked at `_BLOCK = 512`, the same
  fix `tasks/claim_discover` already carries for the identical cliff. And
  `only_stem` is pushed *into* the scan rather than applied after it, which is the
  one that matters for ingest: the pairs that survive are one paper's claims against
  everything else, so the work scales with the new paper instead of the corpus.
  **Ingest path: ~40 s and 611 MB → 0.07 s and 13 MB.** A full corpus sweep peaks at
  62 MB. Output is unchanged — pinned by a parametrized equivalence test against a
  brute-force reference at four block sizes, a cosine-ordering test (the transplanted
  loop ranks by IDF mass and needed the sort re-added, which would otherwise have
  silently changed *which* pairs get judged), and a pushdown-equivalence test.

- **The claim-embedding cache could be evicted by any caller with a narrow row set.**
  `get_claim_embeddings` ends in a persist that rewrites the cache to *exactly* the
  rows it was handed, so a caller asking about one section dropped every other claim
  — the bug that silently evicted 489 limitations claims (3,027 → 2,538) during the
  semantic-member calibration. New `warm_claim_embeddings` persists the union
  instead, so a narrow caller cannot shrink the cache, and short-circuits the
  SentenceTransformer construction (~3 s) entirely when every row is a hit. Both
  properties are pinned by tests, because both failures are silent: the cache is a
  derived artifact that rebuilds on demand, so corruption shows up as wrong numbers
  rather than an error. `claim-overlap` still uses the old writer; migrating the
  cache's only current writer forces a compaction question and is left alone.

  Side effect worth knowing: the first run filled the 143 claims the cache had been
  missing, so it is now 100% covered. Every cache-only consumer
  (`candidates pairs`, `semantic_members`, `check-coverage`) had been scanning 98.9%
  of the corpus and reporting the result as complete.

- **Test suite had no isolation for the claim-embedding cache or `edges.db`.** Both
  resolve from the working directory, so a test reaching a write path lands in the
  developer's real data. Latent only by accident — the cross-paper tests seeded a
  NULL `claim_slug` and the edge writer returns early without one — and the first
  test to seed slugs would have written fake 2-dimensional vectors over the real
  19 MB claim cache and planted `candidate` edges that `claim-graph --tensions` and
  `visualize` present as real findings about the corpus. Two autouse fixtures now sit
  beside `_isolate_state_db`, which is the same trap for the third and fourth time.

- **Concept-hub discovery: five defects found by using it.** The discovery tier
  (tracked at the time in `PLAN-bottom-up-synthesis.md`; the calibration figures
  now live in WORKFLOW.md under *Bottom-up discovery*) was exercised end to end
  for the first time,
  authoring `wiki/concepts/parameter-efficient-fine-tuning.md` — the first hub
  proposed by the corpus rather than by a question put to it. Recall worked: the
  semantic pass recovered scArches' "architectural surgery" as
  parameter-efficient fine-tuning in `single-cell`, vocabulary no lexical search
  reaches, lifting the hub from span 2 to span 3. Every defect below is in the
  advice *around* that recall, and two of them actively misled the author.

  **Aliases are mined from the distinguishing word, not the class noun.**
  `_head_token` returned the longest word, so `frozen embeddings` mined off
  "embeddings" and printed a paste-ready `--aliases "positional
  embeddings,format-specific embeddings,fm embeddings,projected
  embeddings,elmo embeddings"` — following it would have filled the hub with
  ALiBi and ELMo papers. Class nouns are dropped first, then length decides. A
  rarity rule was measured and rejected: over 10,501 contribution claims
  "mechanism" (df 50) is rarer than "attention" (df 155), which inverts
  `attention mechanism` and breaks the case that has pinned this since the pass
  shipped. No corpus statistics needed.

  **Hub membership is attributed to the alias that found it.** Five plausible
  aliases took that hub from 5 members to 17 across 4 categories, admitting a
  spherical-Bayesian-optimization paper and Feynman's restaurant problem on
  substring hits inside "low-rank" and "adapter", with nothing saying which
  alias did it. `find_members` already knew and discarded the answer; it now
  reports it, `run` aggregates it into a new additive `alias_hits` `--json` key,
  and the CLI prints a `members by matching term` table when `--aliases` is
  passed. Deliberately a diagnostic and not a cap — this tier's rule is
  propose-never-decide, and a ceiling would block legitimately broad hubs.

  **Alias qualifiers filter the whole closed class of function words.** The
  filter carried determiners and the common prepositions but not relative
  pronouns, the remaining prepositions, quantifiers or auxiliaries, so real
  claim text proposed `whose mixture`, `over expression` and `without
  supervision` — 27 distinct function words leaked across the live corpus, and
  `mixture model`, the calibration case, was itself proposing `whose mixture`.
  `single`, `multiple` and `few` are deliberately still absent: they read as
  generic and are not (single-cell, multiple sequence alignment, few-shot), and
  filtering a word that carries a domain concept costs a real alias, which is
  worse than an obviously wrong one a reviewer ignores.

  **`--dry-run` no longer demands a `--thesis`.** The thesis test asks *why is
  this a concept and not glossary*, which depends on the member list a dry run
  exists to show — so requiring it in order to look was circular, and the value
  was discarded immediately after. Three inspections during the hub build passed
  `--thesis "provisional"` to get past it. The write path is unchanged.

  **Spokes are no longer anchored to a claim that never mentioned the term.**
  `find_members` fell back to the paper's *first* key_contributions claim for
  keyword-only members, citing seqLens as "Introduced seqLens, a DeBERTa-v2
  based gLM family…" on a hub about parameter-efficient fine-tuning, and
  indistinguishably from a real match. `best_slug` now stays `None`, rendered as
  a bare `[[stem]]` — the form CLAUDE.md prescribes for citing a paper as a
  whole, and what `concepts --upgrade-spokes` exists to fill in. An unrelated
  anchor is invisible; a bare one asks to be fixed. `_term_claim_hint` also
  stopped running its own case-sensitive, unfiltered scan and now shares the
  anchor's query, so hint and anchor cannot disagree. `attach_after_ingest`
  keeps its fallback: there the anchor doubles as the membership test, so
  removing it changes what auto-attaches at ingest — a separate question.

### Added

- **`WORKFLOW.md` gains *Bottom-up discovery — when the wiki proposes the page*.**
  Documents the four proposal tiers and, more importantly, carries the calibration
  evidence behind their thresholds: why literal member search fails hardest on
  bridge terms, why the true/false membership margin of 0.003 forces
  propose-never-decide, why lowering a cosine floor makes 80% of all paper pairs
  "related" at 0.70, why IDF-weighted rare-term overlap beats cosine in the band
  (cosine measures register, rare-term overlap measures subject), why semantic
  proposals at ingest time would return nothing, and why ≥85% of the 0.85
  contradiction pool is already judged at ~1-in-900.

  These numbers previously existed only in a tracking doc that is being retired.
  Two provenance facts are stated with them, because they are not all measured on
  the same corpus: the 2026-08-19 figures come from a 117-paper / 3,027-claim
  corpus on another machine, and the papers they cite — including the
  `mixture-model` hub the whole workstream opened around — **are not in this
  wiki**, so those cases are not reproducible here even though the mechanisms
  transfer. The 2026-08-20 figures are this corpus. Test docstrings and the
  module map now point here instead of at the retiring doc.

- **`cross_paper_judgements` — the contradiction judge now records what it cleared.**
  Its only trace was a `contradicts` edge, written for the two disagreement verdicts
  alone, so `agree` and `different_topic` — the overwhelming majority — left nothing
  behind. A re-run therefore re-paid for every pair it had already dismissed, and
  "has this pool been judged?" had no answer; deciding whether the 0.85 pool was
  worth a judged sweep had to be inferred from a pairs-per-paper distribution
  because no record could answer it. Every verdict is now recorded, so a repeat run
  judges only what the last one never reached (`lint --cross-paper-rejudge`
  overrides). Keyed on the two claim slugs and nothing else: slugs are
  content-addressed, so editing a claim gives it a new slug and its pairs stop
  matching — self-invalidating on exactly the change that should invalidate it, the
  same property `pair_dismissals` relies on.

- **`lint --json` gains `cross_paper_stats`** — `null` unless `--cross-paper` ran,
  then `{pool, judged, skipped_already_judged, disagreements, sim_threshold}`. `pool`
  is filled before the `max_pairs` slice, which makes
  `--cross-paper-max-pairs 0` a zero-cost way to size a sweep before paying for one.
  Additive; no existing key renamed or removed. The prose report also prints a zero
  section when the check ran and found nothing, because silence there was
  indistinguishable from never having run — the ambiguity the new table exists to
  remove.

### Changed

- **Dropped the planned tension-hunting discovery tier.** The last open item in
  the discovery workstream was a generator that proposes a page because two papers
  argue with each other. Measured before building: the existing contradiction judge
  confirms real errors at ~1 in 900, and softening the target to "papers pulling in
  opposite directions" reaches ~1 in 15 across the top 60 cross-category pairs
  (hand-classified, zero tokens). All four survivors were then rejected on
  inspection — none had one paper saying the other was wrong, two dissolved once
  scope was checked (one paper supplied the very evidence the other asked for), and
  three of the near-misses were papers *agreeing* while criticizing some third
  method neither used.

  Genuine inter-paper disagreement is rare, so the tier would spend a judge call
  per pair to surface mostly shared vocabulary, with the reviewer absorbing the
  cost of separating them. The reasoning is recorded in WORKFLOW.md under
  *Bottom-up discovery → Decided not to build*, including the two leads worth
  keeping if it is ever revisited, so the question is closed rather than merely
  unfinished.

- **`candidates concepts` counts membership over contribution sections only.**
  It advertised `direction of effect` as "4 paper pages, 3 categories —
  concept-ready (bridge)"; the scaffolder then found 2 and refused it. Both read
  claims; the detector applied no section filter while `find_members` restricts
  to `key_contributions`/`results`/`methodology`, on the stated ground that a
  term in a paper's limitations is a mention and not an instantiation. Two of
  that term's four "paper pages" were limitations, and the bridge tier is
  exactly the list a reviewer is told to act on. `weighted` and the `sections`
  key still see every section, so the 0.5 `SECTION_WEIGHTS` signal survives and
  the author can still see why a term looks bigger than its member count.

  Four of thirty candidates drop below the `>= 3` floor — `direction of effect`,
  `attention mechanism`, `target profiling`, `AMP`, two of them previously
  labelled concept-ready (bridge) — and `status`'s bridge count falls 10 -> 8.
  Every remaining bridge candidate was checked to return `>= 3` members when
  scaffolded. The detector can still under-count relative to the scaffolder,
  which also has a keyword signal (`foundation models`: 5 advertised, 20
  members); that direction is harmless, since the advertised number becomes a
  floor rather than an over-promise.

## [0.4.1] - 2026-08-20

### Added

- **`candidates pairs` — a zero-token discovery tier below the auto-link
  threshold.** `claim-overlap` judges claim pairs above cosine 0.83 and writes
  reciprocal bullets; that precision is right for *writing* and wrong for
  *finding*, where a missed connection is invisible and permanent.

  Lowering the threshold does not fix it, and the measurement is the point: on a
  117-paper corpus, cosine ≥ 0.70 admits **5,415 of 6,786 possible paper pairs —
  80%**. Meanwhile the relation that motivated this work (Parks 2018 vs van
  Iterson 2017 disagreeing over whether a mixture model can serve as an empirical
  null) peaks at **0.743**, so reaching it by threshold costs ~2,400 pairs.

  So this ranks a cosine *band* (0.72–0.83, ceiling = the auto-link threshold, so
  it never re-surfaces judged pairs) by **IDF-weighted shared-term mass**. A
  384-dimension embedding compresses away the rare vocabulary marking two claims
  as being about the same specific thing: cosine measures register, rare-term
  overlap measures subject. Same hybrid `search` already trusts, applied to claim
  pairs. That puts the motivating pair at rank 74 of 400 and surfaces, at rank 4,
  a paper discussing the NTLA-2001 interim report alongside the trial it reports
  on — same six patients, same transient D-dimer elevations, previously unlinked.

  Judges nothing, writes nothing, costs no tokens: 0.22 s over 3,027 claims.
  `--cross-category` narrows to pairs no other structure in the wiki connects.
  `--json` emits `{cos_lo, cos_hi, cross_category_only, pairs[]}`. It sits beside
  `candidates concepts`/`synthesis` rather than on `claim-overlap`, whose job is
  cross-linking one paper — corpus-wide discovery is a different question.

- **`candidates pairs --decline A B --reason "…"`** (plus `--undecline`,
  `--list-declined` — the same vocabulary `candidates concepts` already uses). The discovery queue is stateless and would re-propose a
  rejected pair at the same rank forever. Order-independent, stored in
  `.pair-dismissals.json`, mirroring `.concept-declines.json`. Permanent rather
  than decay-stamped: two papers do not become related because time passed.

  Each entry is fingerprinted on the **evidence it was judged against** — a hash
  of both papers' contribution-claim slugs. Those slugs are already
  content-addressed (`blake2s(normalize(text))`), which gives exactly the right
  sensitivity for free: stable across `db rebuild`, changed by a rewritten,
  added or removed claim. When the fingerprint no longer matches, the pair
  returns to the list and the entry is marked `[STALE]` in `--list-declined`.
  Entries predating the field — and any pair whose fingerprint cannot be
  computed — stay valid, since suppression is the safe failure mode for a list
  of human decisions.

- **`check-coverage` reports the claim behind each hit.** The gate ranked whole
  pages against `topic_seed`, so its score said nothing about *why* a paper was
  retrieved: on the mixture-model hub it put `van-iterson-2017` at rank 2 among
  hits scoring 4.47 and 4.24, and learning that it was the page's most valuable
  omission meant querying its claims by hand.

  Each hit whose contribution claims also match the seed is now annotated with
  that claim and sorted first. Replaying the pre-alias page surfaces both
  previously-missed papers at the top with their evidence, while two
  higher-scoring "foundation **model**" hits fall below them for having no claim
  match. The pass only annotates rows the page-level ranking already produced —
  it never adds one — so it cannot introduce a false positive, and the `--json`
  contract is unchanged apart from the optional per-hit `claim_*` fields.

- **`status` surfaces the discovery queue** once ≥15 cross-category pairs are
  unreviewed, on a 14-day decay stamp — a higher bar and slower cadence than the
  claim-overlap backlog nudge, because a coverage gap means something is wrong
  and an opportunity queue does not.

- **Both discovery surfaces now report an empty claim substrate instead of an
  empty result.** Claim extraction matches H2 headings verbatim, so a corpus
  imported from an older release or another generator can yield zero claims —
  at which point every discovery surface returns nothing, and "this wiki has no
  claims" is indistinguishable from "nothing matched". Measured: a cold
  embedding cache already logged its cause and its fix; zero claims logged
  nothing. Both now name the cause and point at `lint`'s `zero_claim_papers`.
  `candidates pairs` also stopped asserting the cold-cache explanation on an
  empty result, since that is only one of three reasons it can be empty.

  Worth knowing for a migrated corpus: discovery needs `db rebuild` (free, and
  it assigns the `claim_slug` the queries filter on) plus one `claim-overlap`
  run to warm the cache — that call embeds the whole corpus, not just its own
  stem. It does **not** need grading; neither query reads `last_graded_at`.

- **Semantic recall tier on concept-hub member discovery.** `find_members`
  matches claim text lexically, which fails hardest on exactly the terms worth
  hub-building: a term is a *bridge* when fields name the same thing
  differently. Scaffolding `mixture model` returned 4 members / span 2; the
  finished hub has 7 / span 4, and all three missed members were alias-only
  ("Gaussian Mixture Models", "three-component normal mixture", "normal-mixture
  mean estimation").

  The scaffolder now reports papers whose contribution claims are semantically
  near the term but lexically invisible, with the alias mined from the wording
  that matched — so the author converts a candidate by re-running with
  `--aliases`, through the already-trusted lexical path. **Proposes, never
  adds**: on the calibration case every true member outranked the first false
  positive by 0.003, and a usable floor admits 31 papers for a term like
  "ATAC-seq". Also surfaced in the `min_members` failure, where a thin hub is
  usually a vocabulary problem rather than an absence. `--no-semantic` opts out.

- **`lint --fix` recovers `ingested_at` / `author_model` from the ingest log.**
  Both are recorded facts — every run writes `ingest_iterations` rows carrying the
  model it used and when it committed — so this reconstructs rather than infers,
  and the recovered values are marked `# recovered from ingest_iterations` in the
  YAML so a reader can tell them from stamped ones. On this corpus it reaches 35
  pages, which is what makes `wiki/views.md` sortable by real ingest order.

  Deliberately not a new subcommand: it is a one-shot cleanup for pages that
  predate the fields, so it lives where the gap is already reported. The rule is
  the last committed attempt **that used a real model** — `agents.llm` records
  `stub:{model}` for placeholder runs, and `asai-2023`'s two newest commits are
  both stubs while its prose came from a real attempt eleven days earlier.

  A migrated wiki gets nothing and is told so. `migrate` never writes
  `ingest_iterations`, and there is no honest fallback — a file timestamp is not
  an ingest date, since back-link splicing resets both mtime and birthtime. Pages
  the log has never seen are counted in the report rather than skipped silently,
  so a part-migrated wiki shows how much is out of reach. Existing values are
  never overwritten: a stem can have many committed attempts (`cui-2024-scgpt`
  has 14), so the newest is a sound basis for filling a blank and a poor one for
  correcting an assertion.

- **`lint`: `missing_author_model`** — paper pages that carry `ingested_at:` but
  no `author_model:`. That field is the only one naming which LLM wrote a page's
  prose, and `okfexport._actor_for` reads it alone: absent it, an OKF export
  omits the page's whole `generated` block rather than inventing an actor, so
  the page ships with no provenance for its own text and nothing said so.

  Scoped to pages the pipeline wrote, which is the point rather than a
  limitation. The field is optional by contract and 31 of this corpus's 120
  pages predate it, so an unscoped check would bury the 11 real gaps under
  legacy noise and train the reader to skip the section. `ingested_at:` is the
  scope test because `promote._build_frontmatter` emits both fields in the same
  call. No autofix, for the same reason the OKF exporter omits the block:
  nothing on disk records the model after the fact, and guessing would assert an
  author nobody knows.

- **`lint` → `idea_contract_violations`** — the idea-page analogue of
  `concept_contract_violations`, and the only check in the toolchain that reads an
  idea page's headings. Both mandatory page gates parse *units* — paragraphs and
  bullets — so a page whose Verdict prose sits above the first H2 with no
  `## Verdict` heading passes `check-grounding` and `grade synthesis` green. That
  is not cosmetic: CLAUDE.md §4 fixes the five-section order precisely because
  grounding policy differs per section (Verdict/Background/Caveats strict,
  Opportunities/Plans admit `*(model prior)*`), so a displaced section puts prose
  under a policy that wasn't written for it.
  `grounding.py`'s `_PERMISSIVE_IDEA_SECTION_RE` was mistaken for this check
  before — it matches only `^(opportunities|plans)\b`, because its job is to
  locate the model-prior-eligible ranges, not to validate the contract.
  Checks: missing section, wrong order, unexpected H2 (`## Related Papers` is a
  paper-page section three idea pages carried as an empty stub), missing YAML
  `verdict:`, `verdict:`-vs-section-label disagreement, unparseable label, and
  footnote hygiene. Advisory — reported, never exit-code-flipping — matching
  `concept_contract`'s staging.
  The label parser is anchored on the opening `**` so a parenthetical qualifier
  cannot hijack the match: `**Strength: incremental (with strong upside…)**`
  reads as `incremental`, which a substring search gets wrong. Footnotes are
  split into two findings rather than one, because definitions that merely sit
  outside a `## References` H2 still resolve in Obsidian — only a ref with no
  definition anywhere is a broken citation, and conflating them cries wolf on a
  page carrying 25 working footnotes.
  First run over the corpus surfaced 8 real findings across 5 of 8 idea pages,
  four of them a missing `verdict:` field that no other check could see.

### Changed

- **`CLAUDE.md` and `prompts/` now describe eight things the code already did.**
  Each was a silent gap rather than a wrong statement — an agent working from the
  contract alone would not have known the behavior existed:

  - `lint --json`'s published key list was missing `orphan_prompts` and
    `broken_prompt_pointers`. Both are emitted, and the release policy calls a
    removed `--json` key breaking, so the list has to be complete to mean anything.
  - `concept_thesis:` is required — `tasks/concepts` refuses to scaffold without it
    and exits 1 non-interactively — but the concept-page YAML enumeration omitted it
    and the worked command was the bare form that fails. Every `researchwiki concepts`
    example now carries `--thesis`.
  - Idea pages take `## References` (and `## What would update this page`) alongside
    the five required sections; `idea_contract` allows both. The closed-list rationale
    read as forbidding them, while the same section mandates footnote citation.
  - `meta` / `dashboard` were missing from the page-type enumeration, and `wiki/views.md`
    — which `init` scaffolds and three modules recognize — appeared in neither the
    repository tree nor the meta-pages list.
  - `ungraded_papers` had no remedy anywhere outside `prompts/migration-backfill.md`,
    behind a bulk-import trigger. It now names `grade regression --missing-only` where
    the finding is reported.
  - `neighbors` calls the Semantic Scholar Graph API and was absent from Rule 1's
    whitelist table, which is meant to be the authoritative allowlist. It and `attach`
    are now documented — `attach` beside the supplementary layout it creates, which
    CLAUDE.md described without ever saying what produces it.
  - `candidates concepts --triage` spends LLM calls and auto-writes to the decline
    list, and was documented nowhere despite being the scalable answer to the
    candidate noise the same bullet describes.
  - `prompts/export-bibliography.md` covered three of the four `--format` values.
    OKF has a different scope (every page type, not just published ones), a different
    `--json` contract, and a mandatory `--out`, and the prompt asserted a
    bibliography-only exclusion as though it were universal.

- **`prompts/ingest-digest.md`'s page contract was missing five fields the agent path
  always writes** — `type:`, `hook:`, `ingested_at:`, `venue:`, `short_name:`. A page
  built from that template lands on `lint`'s `missing_type` and `missing_hook` by
  construction, and `missing_type` is the finding CLAUDE.md itself calls invisible to
  every other check. The template also now states that the four graded H2 names are
  matched exactly, since the digest path is where a renamed heading actually happens.

- Smaller corrections in the same sweep: `share-page` told the author to read
  `referenced_papers:` off pages that don't carry it; `prompt_lib`'s docstring listed
  a `clinical-trial` author prompt that was never added; `attach`'s docstring still
  called `--supplementary` unshipped; the repository tree omitted `prompts/` and
  `config/`; and the synthesis "the middle is yours" guidance now notes that renaming
  the middle section drops it from the semantic index, which `thin_index_text` will
  not catch when the required sections still match.

- **The module-size gate counts code, not lines** — `MAX_CODE_LINES = 800`,
  excluding docstrings, comment-only lines and blanks, replacing a same-numbered
  cap on *physical* lines (a far tighter bound, since this package is ~29% prose). This package is ~57% code and ~29% prose, so the old metric
  taxed documentation, and it had its own ranking inverted: `agents/llm.py` was
  pinned as debt at 902 lines while holding 470 lines of code (39% prose — well
  explained, not complex), while `tasks/lint/report.py` passed comfortably at 603
  lines while holding 549, more code than four pinned modules. Raising the
  ceiling would not have fixed that ordering, which is what a budget is for.

  The old rationale — "small enough for an agent to hold in context" — no longer
  carried it either: the largest module is ~13,500 tokens against a 200k window.
  What a size cap buys is cohesion, and that scales with code, not with how well
  the code is described.

  The ceiling is permissive by design — only `agents/runner.py` (817 code lines)
  exceeds it — and the per-module ratchet is now a *separate* bound rather than a
  consequence of it. All nine `_DEBT` pins (502-817) bind whether or not their
  module sits under the ceiling, and retire at `RATCHET_RELEASE` (500) rather
  than when the ceiling moves. That split exists because keying retirement on the
  ceiling meant raising it deleted eight pins and handed those modules ~1,700
  lines of unratcheted growth, though none had shrunk by a line. A string bound
  to a name still counts as code, so the rule can't be dodged by moving text into
  a triple-quoted constant.

- **Extraction-noise repair moved to `researchwiki/pdf/repair.py`** (`repair_text`,
  formerly `text._repair_ligatures`). `pdf/text.py` is about getting bytes out of
  a PDF; how those bytes come out wrong is a separate subject with its own
  judgement calls, and the one most likely to need arguing with.

### Fixed

- **Four statements in `CLAUDE.md`/`prompts/` described behavior the code does not
  have.** All four would mislead an agent that trusted them, and none was
  detectable by the reachability check `lint` already runs:

  - **`agent ingest` was documented as accepting `--category`.** It defines no
    such argument and uses a strict `parse_args`, so the suggested invocation is
    a usage error. `--category` belongs to the digest-path `researchwiki ingest`.
  - **The ingest section still claimed pages are tagged `ingested-via-agent`,**
    contradicting the Page Types rule three sections above it and
    `promote._build_frontmatter`, which writes no `tags:` on paper pages and
    carries a comment saying why.
  - **The commentary guard's rule was stale on `Editorial`,** listing it as a
    weak label needing structural corroboration. It was promoted to the strong
    tier precisely because the case that motivated it — a 2-page editorial
    depositing 9 references — can never be corroborated structurally, so the
    documented rule predicted that PDF was *not* caught. The `10.1038/d…`
    news-DOI tier and the `Books & Arts` / `In this issue` weak labels were
    missing outright.
  - **`prompts/ask-system.md` mandated `claim_id:NNN` citations,** which the
    claims-DB corollary forbids by name (row ids are reassigned on
    `db rebuild`), and named five tools against `mcp-serve`'s actual three —
    two of which, `wiki_get_page` and `db_query`, have no MCP exposure at all.
    Rewritten against `search` / `claims` / `check_grounding` and the
    `[[stem#claim_slug]]` form.

  Also corrected: `eval.pointers` and its test asserted every `-system` prompt is
  loaded through `prompt_lib`. `ask-system` is not — it is the system prompt an
  MCP *client* runs, so nothing in this package loads it. The orphan exemption is
  still right, but for a different reason than the comment gave.

- **`researchwiki ingest` crashed on every PDF it was supposed to move.** The
  `journal-upgrade` branch of `process_one` carried a function-local
  `import shutil`, which makes `shutil` local to the *whole* function — so the
  final `shutil.move(src_pdf, pdf_dest)` raised
  `UnboundLocalError: cannot access local variable 'shutil'` on precisely the
  common path, where no stem collision sends control through that branch and the
  name therefore never gets bound. The module already imports `shutil` at the
  top; the shadowing local is now just `import uuid`.

  Worth noting how this hid: the failure needed the branch *not* to be taken, so
  a stem-collision test exercising `journal-upgrade` binds the local and passes,
  and the digest path is the documented fallback for recovery and unextractable
  PDFs rather than routine use. `--no-move` also skips the call entirely.

- **Ligature repair rewrote scientific prose into plausible nonsense.** Mode B
  infers damage from a failed dictionary lookup alone, and over a 20-paper
  sample 166 of its 187 insertions landed on an acronym or a hyphenated term:
  `p-values` → `ftp-values` (21×), `UNG` → `flUNG`, `OOD` → `flOOD`,
  `HLA-U` → `HLA-flU`, `re-produced` → `fire-produced`. Every one was legal
  under the old rule, which accepted a hyphenated candidate whenever each part
  was a word (`ftp` + `values`) and matched acronyms case-insensitively — and
  this text is what the author phase writes pages from and what the claim
  grader scores against.

  Frequency cannot separate them (`ftp` and `floor` are common words), so the
  new guards are structural, and repair is judged per hyphen-part rather than
  per token. Loading the wordlist as `{word: rank}` also fixes two failures in
  the other direction: `ne-grained` stayed damaged because `neff` made `fine`
  look ambiguous, and `rst` → `first` never fired because the list contains
  `rst`. Mode A (C1-byte repair) is untouched — a control byte is hard evidence
  of damage, and these guards would reject cases it handles correctly.

- **A genuine hyphen falling at a line break was welded away.** The control byte
  stands for two different hyphens, and eliding both destroyed 221 of 2,367 in
  the sample — `off-target` → `offtarget` in a corpus where that term is
  central, after which the chunk index holds a token no reader's query matches.
  The wordlist now decides, with the rule order load-bearing: a repairable
  welded form wins before the both-halves-are-words test, so `dif-cult` becomes
  `difficult` and `therefore` does not sprout a hyphen.

- **`_load_dictionary` caught only `FileNotFoundError`**, so an unreadable
  wordlist broke every extraction instead of degrading to no-repair; the empty
  result is now cached rather than re-attempted per word.

- **`page_for_offset` promised `None` when out of range** but only guarded the
  low end. It takes an optional `total_len` and the chunker passes it.

- **`figures.page_texts` returned raw control bytes**, so a soft-hyphenated
  `Fig\x02ure 3 | …` missed the caption pattern and printed the byte to the
  terminal when it matched.

- **Journal recovery could delete the working directory.** A journal document
  with no `backup_dir` key fell back to `Path("")` — which is `.` — and cleanup
  then ran `shutil.rmtree` on the wiki root, silently (`ignore_errors=True`).
  One stray or truncated `.json` under `.mutation/` was the whole trigger, and
  `recover_pending` auto-runs at ingest start. Recovery now removes a backup
  directory only when the recorded path is strictly inside `.mutation/`; the
  journal itself is still drained. It also leaves journals from a *newer*
  schema version in place (draining one with old code could lose its rollback)
  and reports orphaned backup directories instead of ignoring them.

- **`remove --apply` failed for any paper with supplementary files.** The
  removal plan declares directories (`papers/{stem}.supp/`, figure and grade
  caches, evolution-proposal dirs), but the snapshot backed paths up with
  `shutil.copy2`, which raises on a directory — so the removal died before
  writing anything and stranded a journal-less backup dir in `.mutation/`.
  Snapshots now back directories up with `copytree` and rollback restores or
  removes them whole; a snapshot that fails mid-copy cleans up after itself.

- **The `RW_MUTATION_JOURNAL=0` escape hatch crashed at every commit point.**
  The passthrough snapshot's journal path is `/dev/null`, and `mark_committed()`
  still persisted — dying on `PermissionError: /dev/null.tmp` *after* the
  caller's work had landed, so a successful promote or removal reported
  failure. Persist and cleanup are now no-ops on a disabled snapshot.

- **`remove` could edit a stem-collision neighbour's bullets.** The index-bullet
  and concept-spoke patterns matched the stem as a suffix or substring inside a
  wikilink, so removing `lee-2025-…` also deleted `garcia-lee-2025-…`'s entries
  — the shape a hyphenated surname produces. All removal patterns now anchor
  the stem the way `backlinks.remove_related_paper` always did. The commentary
  probe likewise no longer flags a commentary whose *body* merely mentions the
  removed stem (`primary_paper:` matching ran to end-of-file under `DOTALL`).

- **OKF export duplicated two mapped fields.** `source_url` (→ `resource`) and
  `generated_at` (→ `generated.at`) were missing from the mapped-key set, so
  they also leaked through the `x_researchwiki_` passthrough.

- **`visualize` collapsed a `promoted` claim edge down to its weakest sibling.**
  Only `confirmed` was special-cased when parallel edges collapse per page
  pair; the strongest live status (`promoted` > `confirmed` > `candidate`) now
  wins. The graph page is also written atomically, matching every other writer.

- **`export --format okf` with a file at `--out` now exits 2, not 1** — an
  unwritable output path is an environment failure per the documented contract,
  not a malformed argument.

- `prompts/idea-page-author.md` claimed the five section names were "matched by
  the linter", naming a regex that matches two of them, and still called the
  contract "four-section" a dozen lines after listing five — a leftover from
  before Verdict existed. It also presented the five H2s as exhaustive while
  telling authors to use `[^id]` footnotes, which require a `## References`
  section to define.

- **The dashboard ranked recently *touched* files as recent additions.** `views.md`'s
  Dataview blocks fell back to `file.ctime` / `file.mtime` when a page carried no
  `ingested_at` / `generated_at`. Back-link splicing resets birthtime and mtime moves
  on any edit, so ingesting one paper — which spliced 12 reciprocal back-links — stamped
  the 7 unstamped targets with the ingest second and sorted them *above* the paper just
  ingested: *Recent papers* was headed by foldseek, MMseqs2, geNomad, AlphaFold 3 and
  Boltz-2, none of it recently added. The blocks now require the stamp and sort on it,
  excluding undated pages rather than mis-ranking them — the rule `lint`'s staleness
  checks and provenance recovery already follow. Two dead `length(referenced_papers)`
  columns go with it (neither synthesis nor idea pages carry that field; it is real only
  on concept pages), and the `researchwiki init` wizard template carried every one of
  these defects, so it would have regenerated them for the next fresh wiki.

## [0.4.0] - 2026-08-14

### Added

- **`researchwiki visualize`** — renders the corpus as a self-contained interactive
  graph at `output/graph.html`. No CDN, no build step, no server: one file that
  opens from `file://`. Draws both edge kinds — `[[wikilinks]]` and the typed
  claim edges from `.claim-graph/edges.db` — with `contradicts` styled loud,
  because `claim-graph --tensions` can list tensions but cannot show you that
  four of them land on one paper. Filters by page type, free-text, and a
  tensions-only mode; click a node for its claim relations; a table view carries
  the same data non-visually. `--json` emits the graph instead of the page.

  Only live claim statuses are drawn by default — `stale` means the claim the
  edge was judged against has changed, so it is an assertion nobody currently
  stands behind. Parallel claim pairs collapse per page pair, which is why the
  argument is semantic and not about volume: widening to `stale` on a corpus
  holding 13,535 of them adds only ~20 visible edges.

  Page type is encoded by **shape as well as hue** so identity never rests on
  color alone. The palette is validated all-pairs against the render surface
  rather than eyeballed — which caught a neutral-vs-aqua pair at CVD ΔE 2.0,
  indistinguishable to a deuteranope, that looked perfectly fine on screen.

- **`researchwiki export --format okf`** — emits the corpus as an Open Knowledge
  Format bundle (OKF v0.2), folded into the existing `export` command rather than
  added as a subcommand. Writes a directory tree, so it requires `--out` and
  refuses a non-empty directory that isn't already one of its bundles.

  **The two formats have different scope on purpose.** The bibliography carries
  only pages describing somebody else's publication, because a BibTeX entry for a
  synthesis page would assert a publication that does not exist. OKF's unit is a
  *concept* — explicitly including abstract ideas with no underlying resource — so
  it carries every page type and omits `resource` where nothing is published. A
  page absent from the `.bib` and present in the bundle is correct in both; the
  `export` docstring says so at length, since "aligning" the two lists is the
  tempting wrong fix.

  Two mappings worth calling out. The wiki's `[^id]:` footnote labels *are* OKF's
  `sources[].id`, arrived at independently for the same stated reason (agents
  rewrite these documents, so a positional reference misattributes silently), so
  per-claim attribution carries over with the body unchanged. And `verified` is
  emitted **only** for graded paper pages, from `claims.last_graded_at`;
  synthesis/idea/concept pages get none, because `check-grounding` and
  `grade synthesis` persist nothing and a trust tier with no record behind it is
  the exact falsehood those gates exist to prevent. The report counts them so
  "unverified" is not read as "ungraded".

- **`lint`'s `missing_type` check** (new `--json` key; additive) — pages carrying no
  `type:` at all. Invisible to every other check by construction: consumers read
  the field as `fm.get("type", "paper")`, so a page that lost it behaves as a
  paper and `page_type_mismatches` — applying the same default — cannot see it.
  The stakes are attribution: a commentary without `type` has its claims
  extracted and credited to the commentator, the exact failure `type: commentary`
  exists to prevent. 23 pages in the maintainer's corpus had none.

  Also one of only three OKF conformance criteria, so this is a prerequisite for
  emitting a conformant Open Knowledge Format bundle.

- **`tests/test_module_size.py`** — 800-line cap per module, since a file an agent
  can't hold in context is one it edits blind. Nine existing modules are
  grandfathered **at their current size** (`path -> ceiling`), so they may shrink
  but not grow; the usual bare exemption set is how `agents/runner.py` reached
  1214 lines with nothing objecting. A companion check fails on stale entries, so
  the list shows remaining debt rather than accumulating excuses.

- **`researchwiki eval triggers`** — check that CLAUDE.md's prompt pointers fire
  when they should. Each `prompts/*.md` is reachable only through the sentence
  in CLAUDE.md that gates it, which makes that sentence load-bearing, and nothing
  tested it across 23 prompt files.

  A generator reads one prompt's gating text *and its body* and writes N requests
  that should route to it plus N near-misses that shouldn't; a grader then routes
  each using only the gating text of *every* prompt, so a competing trigger can
  steal and picking the wrong prompt is observable rather than scored as a pass.
  Graders run bounded-concurrent, and a grading that errors is excluded from the
  denominators rather than counted as a failure — without that, one timeout
  silently depresses a pass rate. `--dry-run` prices the run and spends nothing.
  The gating text is the enclosing CLAUDE.md section rather than the link's own
  line, because CLAUDE.md is loaded whole and the link is not always in the
  sentence stating the trigger. Method adapted from OpenKB's
  `skill/evaluator.py` (Apache-2.0).

  It shares a command with the existing classifier eval rather than adding a
  top-level name: **`researchwiki eval classifier`** is the same leave-one-out
  category evaluation as before, and `eval-classifier` still works as a
  deprecated alias. The eval family is one command, not two.

- **`lint` reports unreachable prompts** — `orphan_prompts` (a `prompts/*.md`
  no CLAUDE.md pointer reaches, so the agent has no condition under which it
  would read it) and `broken_prompt_pointers` (a pointer whose file is missing).
  The same class of check as `broken_wikilinks`, one layer up. Free and
  deterministic, so it belongs in the health check that already runs rather than
  behind the paid eval above. Prompts whose name carries `-system` are exempt:
  those are LLM system prompts loaded by code through `prompt_lib` and are
  correctly absent from CLAUDE.md.

- **`researchwiki remove <stem>`** — retract a paper and every generated trace
  of it. Deleting a page by hand used to strand the PDF, the `index.md` bullet,
  inbound back-links, `[[stem#slug]]` anchors on synthesis pages, concept
  spokes, claim-graph edges and four tables' worth of rows; `lint` reported the
  wreckage and nothing cleaned it up.

  **Generated text is removed, authored text is reported.** Back-link bullets
  and the catalogue entry were written by `promote`, so they go. A sentence on a
  synthesis or idea page citing the paper was written by a human and has passed
  both gates, so it is listed with file and line for the reviewer to resolve —
  no rewrite rule is safe there, since stripping the citation leaves an
  unsupported claim and deleting the sentence can remove a conclusion several
  papers jointly carried. Expect `lint` to report `dangling_claim_anchors` on
  exactly the listed pages afterwards: that is the to-do queue, not a defect.
  A concept hub gets both treatments — its generated spoke registry is cleaned
  (and `concept_span` recomputed), its authored Definition is not.

  Dry run is the default; `--apply` is required to write. `--keep-pdf` retains
  `papers/{stem}.pdf` when the page is wrong but the paper should be re-ingested.
  The whole removal runs inside the mutation journal below, so a failure
  part-way through rolls back rather than half-removing. `log.md` is
  append-only, so a removal appends an entry rather than editing history.

- **`promote_to_wiki` is transactional.** Its five steps — page write + DB row,
  PDF move, back-links into N existing pages, `index.md`, `log.md` — were each
  individually atomic with nothing binding them, so a failure after the PDF move
  left a paper half-landed. They now run inside a write-ahead journal under
  `.mutation/` (gitignored): either every step lands or the tree is restored,
  including back-link targets the mutation had already modified and the inbox
  PDF the move had already consumed.

  A crash leaves a journal rather than a mess. `researchwiki status` reports it
  under *Workflow state*; the next `agent ingest` drains it before starting —
  recovery on the next run rather than a repair command nobody remembers exists.
  A rollback that fails five times stops retrying and says so instead of
  spinning on every subsequent run.

  Two things worth knowing. The commit point is explicit, and cleanup runs
  *after* it, so a failure while discarding backups can never undo committed
  work. And the transaction spans two storage systems: file state is journalled,
  the `state.db` row is not. An in-process rollback deletes the row explicitly; a
  crash-recovered one can't, so it removes the page and leaves the row for the
  next `db rebuild`, which is the right way round given markdown is canonical and
  the DB is derived.

  `RW_MUTATION_JOURNAL=0` bypasses journalling entirely — an escape hatch for one
  release if this interacts badly with something, not a supported mode.

  Adapted from OpenKB's `mutation.py` (Apache-2.0), minus its hardlink backups:
  those are an optimisation for whole-tree snapshots and break on exactly the
  cloud-synced volumes `wiki/` and `papers/` tend to live on here. A promote
  touches a handful of files, so plain copies are fine.

- **PDF chunks carry their page and section.** A claim's `supporting_text` was
  an anonymous 250-word window — nothing could say it came from §Results, p. 7,
  which is the first thing a reader wants when deciding whether a weak grade is
  the claim's fault or retrieval's. `Chunk` and `RetrievedChunk` now carry
  `page_start` / `page_end` / `section`, and `claims --include-context`,
  `pdf-search` and `grade --weakest` print the label.

  Note the two `section` axes, which are different things: `claims.section`
  names the *wiki page's* H2 (key_contributions / results / limitations /
  methodology), while this one names the *PDF's* own section. Both are useful
  and neither replaces the other.

  Labels are omitted rather than guessed. A paper with no detectable headings
  gets `None`, because a wrong section on displayed evidence is worse than no
  section. The `supporting_provenance` column is likewise blank on claims
  graded before this existed; they fill in on the next `grade`.

  Two supporting changes worth knowing about. `pdf.text.extract_pdf_page_texts`
  returns per-page text whose join reproduces `extract_pdf` byte for byte —
  the invariant page offsets are measured against, pinned by a test.
  `pdf.sections.section_spans` returns the segmentation `anchor_sections` had
  always computed internally and discarded, so the boundaries used for labelling
  and the text used for claim extraction cannot drift apart.

- `.grade-cache/` indexes now record a `cache_version`. `build_pdf_index`
  returns early when the directory exists, so without one this release's schema
  change would have left every existing index in place — missing the new fields,
  reporting no error. A version mismatch rebuilds instead. Expect a one-time
  rebuild of the per-paper chunk indexes (and their embeddings) on first use
  after upgrading: a few minutes across a ~100-paper corpus.

- **`researchwiki figures <stem>`** — list a paper's figure and table captions;
  render one page when a question actually turns on it. Rule 3 tooling for the
  case `pdf-search` can't serve, where the passage says "see Fig. 4" and Fig. 4
  is where the number lives.

  Listing is free and is the default mode — captions carry the quantitative
  results often enough to answer the question outright, which is why
  `sections.py` extracts them for the claim pipeline in the first place.
  `--figure N` (or `--page N`, the escape hatch when a caption isn't detected)
  renders one page to `.figures-cache/{stem}/`, gitignored and safe to delete.
  More than one page needs an explicit `--pages`: rendering is free local
  compute, but reading a PNG costs context in proportion to its pixel area.

  Rendering, not object extraction, because a PDF image object is a *placed
  raster* — the vector paths that every matplotlib/R/Illustrator plot is made of
  are invisible to an image-object walk in any library. Measured over 12 random
  corpus papers: 498 image objects against 73,294 path objects.

  No new dependency: pypdfium2 and numpy were already required, and PNG
  encoding is ~30 lines of stdlib `zlib`+`struct` rather than pulling in Pillow
  for `PdfBitmap.to_pil()`. Nothing runs at ingest, nothing is backfilled, and
  no page field or `lint` check was added.

  Caption detection spans the separator styles the corpus actually uses —
  `Fig. 1 | T` (Nature), `Figure 1: T` (preprints), `Fig 1. T` (PLOS),
  `Fig. 1 T` (BMC, none at all) and `Figure 1- T` / `Figure 2 - T`
  (accepted manuscripts). Validated across all 115 corpus PDFs plus the five
  bundled benchmark fixtures: 1,108 captions, no body-prose false positives.

  A caption is not always on the same page as its figure, and two layouts in
  this corpus put them apart: accepted manuscripts that collect every caption
  onto one page with the plates a few pages later (`fonseca-2026`), and
  preprints that run the whole manuscript and append every plate at the end
  (`aygun-2026` — legends p15-17 and p30-31, plates p34-44). Rendering the
  caption page shows text and no figure.

  Handled in two steps. Where the plate prints the figure label as well — as
  appended Extended Data plates usually do — the label's later occurrence is
  preferred over its first, so `--figure "ed 1"` renders the plate rather than
  the legend. That is evidence rather than a guess: the destination page
  carries the same label. Where no repeat exists, the command says the page it
  is about to render has no artwork and names the pages that do, without
  rendering them.

  Artwork is measured as **fraction of page area covered**, not object count.
  Counting objects gets it wrong in both directions: a page holding one
  full-page raster figure has a single image object, while a plain text page
  with a header rule and a logo has two. Across this corpus, caption-only and
  prose pages measure ~1.5% while every real figure page ran 15-98%. Tables are
  exempt from the check entirely — a table is text with rules, so low coverage
  on its page is correct rather than a symptom.

### Changed

- **`output/` is now the umbrella for everything the repo emits outward**, and
  `share-page` writes to `output/share/<slug>.md` instead of `share/`. Three
  generators had begun scattering outbound artifacts across two top-level
  directories (`share/`, plus `output/graph.html` and `output/okf/` from this
  release); collecting them means one gitignore rule covers the category, so the
  next generator needs no `.gitignore` change to stay out of git. The old `share/`
  path stays ignored — a stale copy on a second synced machine should not surface
  as untracked just because the convention moved.

- **`lint`'s emitters moved to `tasks/lint/report.py`.** `_emit_json` and
  `_emit_prose` were 58% of the package `__init__` and decide nothing — both take
  finished results as kwargs and only choose how to print. The dispatcher now
  reads as a list of checks (831 → 261 lines) and left the module-size grandfather
  list entirely — the ratchet above asked for the split on its first encounter
  rather than being handed a raised ceiling. No behaviour change; the JSON
  contract is unchanged apart from the additive key above.

- **`mutation.py` reimplemented independently.** This package is MIT and the
  module was originally written against a third-party Apache-2.0 file, so it was
  re-expressed to remove any question of foreign-licensed code in an MIT tree.
  Measured before and after: 24.6% -> 19.3% identical normalized lines, with the
  remainder being this module's own public API vocabulary (`rollback`, `discard`,
  `mark_committed`, `backup_dir`, `journal_path`, `MAX_ROLLBACK_ATTEMPTS`) and
  idioms with no second spelling (`shutil.rmtree(..., ignore_errors=True)`). No
  shared class name, no shared helper name, and no shared multi-line logic in
  either version — the two designs already differed substantially (this one has a
  context manager, `also_undo` hooks and an env bypass; the other has directory
  and hardlink backups, and opposite cap semantics).

  **Behaviour, the public API and the on-disk journal schema are unchanged**, the
  last of these deliberately: a journal left by an interrupted promote must still
  be drainable after an upgrade, or the abandoned mutation becomes permanently
  half-landed with nothing reporting it. Two tests now pin that — one replays a
  journal hard-coded in v0.4.0's exact key set (not one produced by the current
  writer, which would pass even if both ends drifted together), and one pins the
  serialized key set for the release that will read it next.

  The in-memory representation did change, and for the better: entries were a
  `{target: backup_or_None}` mapping where `None` overloaded "no backup because
  the file did not exist", which is what rollback branches on. That is now a
  `BackedUpPath` record with an `existed_before` property, so the distinction
  lives in the type rather than in a comment.

- A batch containing a duplicate PDF now exits non-zero where it previously exited
  0. Same for a resume that finds an input no longer on disk. Wrapper scripts that
  branch on the exit status will see this.

### Fixed

- **Clicking a node in `visualize` threw the whole graph around.** `mousedown`
  resolved a press to a drag before anything had moved, so the first `mousemove`
  took the drag branch and floored `alpha` at 0.55 — which on a settled 477-node
  layout permits ~32px of movement per node per tick, so a few pixels of hand
  jitter re-annealed the entire graph. `mouseup` then cleared the drag but not
  `pinned`, leaving the clicked node nailed down for the layout to re-settle
  around. A press is now provisional until it clears 4px, and the drag path uses a
  gentler `nudge()` (alpha floor 0.10): conflating the two re-heat levels was the
  root of it, since `kick()` is for a change that invalidates every position and
  exactly one node had moved. Verified by driving synthetic events at a settled
  layout — jittery click 0.550 → 0.000, drag 0.550 → 0.100, selection still works.

- **`visualize`'s layout no longer thrashes on load or on `reset view`.** Four
  causes, all in the annealing rather than the input handling that the earlier click
  fix addressed. The per-tick displacement ceiling was `K * 0.6` — about 32px per
  node per tick on a 477-page corpus, ~1900px/s at 60fps. Nodes were seeded on a
  tight disc, so every one started inside every other one's repulsion range and
  tick 1 was the most violent frame of the run. `reset view` set `alpha = 1`,
  re-deriving a layout that was already correct, which made a *camera* button the
  most violent action in the UI. And the camera was fitted twice mid-settle, which
  reads as shake even when the layout is fine.

  The hot phase now runs in `warmup()` before the first paint — 260 ticks, 96ms, so
  the graph appears roughly arranged and eases into place. Alpha decays slower
  (0.992) because total relaxation scales as 1/(1-decay) and those extra iterations
  are spent off-screen where they cost nothing to watch. Measured over the frames a
  viewer actually sees:

  | | before | after |
  |---|---|---|
  | worst per-tick node movement | 27.2px | **0.8px** |
  | mean per-tick movement | 10.3px | **0.4px** |
  | ticks moving >15px (of 200) | 50 | **0** |
  | after `reset view` (60 ticks) | 20.0px mean | **0.5px mean** |

  Convergence quality held: mean nearest-neighbour spacing 35.0px against 37.6px
  before, and zero overlapping node pairs. An intermediate version was calmer but
  visibly under-relaxed (30.2px, 4 overlaps), which is why the warmup budget and
  decay were raised rather than just damping the motion.

- **23 paper pages had no `type:`** and now declare `type: paper`. Each was
  verified paper-shaped first (no `primary_paper`, no `issuer`, not in a
  page-type dir, has a `## Summary`) rather than defaulted, since the failure this
  guards against is precisely a non-paper page being treated as one.

- **Seven drifted claim-section headings on six paper pages**, which were yielding
  zero claims from sections that plainly had them — extraction matches H2 names
  exactly. `migrate.sections.canonical_for` now also strips a trailing
  parenthetical qualifier (`## Key Contributions (as a Review)`) and a slashed
  alternative (`## Results / Findings`), and tolerates an inflected architecture
  suffix (`## Methodology and Architecting`). The ambiguity guard runs against both
  the decorated and undecorated form, so `## Discussion (results)` still refuses to
  become Results. Recovered **38 citable claims**; `migrate` gains the same
  tolerance for future imports.

- **`eval classifier` reported 0% abstention on runs that abstained.** `other`
  is both a real content category and the abstention bucket, and
  `suggest_category_llm` expresses "I don't know" by returning
  `category="other"`. The eval counted an abstention only when the classifier
  returned `None` at all, so every abstain-to-`other` was recorded as an
  ordinary prediction — a number whose label did not describe what it measured.

  `Suggestion.abstained` now carries the decision the classifier already made,
  and the report separates two things that were conflated. **Placement** is
  where the paper ends up on disk: an abstention still files it under `other`,
  so it stays in the confusion matrix under `other` rather than being hidden,
  because hiding it would under-report how `other` fills up. **Commitment** is
  whether the classifier named a category or declined. An abstention onto a
  genuinely-`other` paper is correct placement *and* a declined commitment at
  once, which one counter could not express.

  Per-category output now also notes how much of a category's predicted volume
  arrived by abstention — the number that says whether `other` is a judgement or
  a shrug. And the confidence column is labelled as the classifier's own
  self-report, since `suggest_category` is LLM-first and the figure is not a
  neighbour-vote share.

- **A promote that didn't complete reported success.** `promote_to_wiki` is five
  multi-file steps with no transaction binding them — page write + DB commit → PDF
  move → back-links → `index.md` → `log.md` — and returns `promoted=False` rather
  than raising when a step after the page write fails. Nothing read that flag: the
  only `.promoted` reads in the package were `gate.promoted`, a different object.
  Execution fell through to `decision = "committed-to-wiki"`.

  This was reachable without a crash. `_move_pdf` refuses a stem collision that
  isn't a preprint→journal upgrade, i.e. **any duplicate PDF in a batch**, which
  left a page on disk and in `state.db` with the PDF still in `inbox/`, no
  back-links, no index bullet and no log entry — while the process exited 0 and the
  checkpoint recorded `completed`. The warning did reach the log; the exit code was
  what lied.

  The commit phase now records a `promote-failed` iteration and raises
  `PromoteFailed`, which the CLI maps to exit **2** (retryable — deleting the
  duplicate makes the retry work) with an inspection checklist and no stack trace.

- **`--resume` re-queued inputs whose PDF had already moved.** The batch checkpoint
  is only written once a worker subprocess returns, so a worker that died
  mid-promote recorded nothing, and `resume_batch` re-queued its input path by
  name — a path `_move_pdf` had already `shutil.move`d out of `inbox/`. The re-run
  was dead on arrival and the half-landed paper was never repaired.

  Such inputs are now classified `unresumable` in `checkpoint.json`, reported with
  what to check, and never re-queued. Whether the worker ever started is recovered
  from the existence of its log file — `_worker` opens that before
  `subprocess.run`, so no new bookkeeping was needed — which distinguishes "died
  mid-promote" from "the user moved the file". The check also covers retryable
  (exit 2) failures, whose exit code says retry but whose input is gone.

## [0.3.1] - 2026-08-13

### Added

- Two warnings for config states the provider-resolution layering can produce but
  could not previously report. Precedence itself was never ambiguous —
  `RW_MODELS_CONFIG` selects the file, the file merges over `_FALLBACK_ROLES`,
  `RW_LLM_PROVIDER` and `RW_LLM_BASE_URL` override last — but two of those layers
  can still combine into something unrunnable:

  - **`RW_LLM_PROVIDER` replaces the provider and not the model.** The two halves
    of a routing decision arrive from different layers, so
    `RW_MODELS_CONFIG=models.chatgpt.yaml RW_LLM_PROVIDER=anthropic` resolved to
    `anthropic/gpt-5.6-terra` and failed at the API on an unknown model, naming
    neither the env var nor the config. The pre-existing mixing banner returns
    early unless the config declares ≥2 providers, so it was silent on exactly
    this shape — a *uniform* config whose provider the env var replaces wholesale.
    The check compares config-provider against forced-provider rather than model
    name families, so `models.glm.yaml` running `glm-4.7-flash` through
    `provider: anthropic` (z.ai's Anthropic-compatible endpoint) is not flagged.
    `chat-relay` is exempt — it treats the model string as a label.

  - **An OpenAI-compatible role with no `base_url:` silently means localhost.**
    `base_url()` returns None and `call_openai_compatible` reads None as the LM
    Studio default, so a cloud config missing one key becomes a local one. The
    asymmetry that hid it: a *missing* config file falls back to OpenAI, while a
    *present* file with no `base_url:` falls back to `http://localhost:1234/v1`.
    Fires on the merged view, so a partial config inheriting the all-OpenAI
    fallback roles is caught too. Silent when `RW_LLM_BASE_URL` supplies the
    endpoint. No shipped template trips it, and a test now pins that.

  Both fire once per process on stderr, like the existing banner.

### Changed

- `researchwiki init` now recommends OpenAI/ChatGPT as the default provider, matching
  README and `model_config._FALLBACK_ROLES`. Its menu had offered Anthropic as entry 1
  labelled `default, ~$0.10/paper` — so the human-driven setup path steered new users
  onto a ~10x-dearer provider and told them it was the default, while the LLM-driven
  path (which reads the README) set them up on OpenAI at ~$0.01/paper. Every provider
  is still offered; the menu now mirrors README's *Providers* table in its order, and
  has five entries rather than four, since plain OpenAI and "other OpenAI-compatible
  cloud" have different setup steps and were previously conflated into one.

  Picking OpenAI writes **no** `config/models.yaml`: the built-in fallback already
  routes every role there, which is what makes that path zero-config. It is
  deliberately not `models.chatgpt.yaml` — that template puts author/critic/judge on
  gpt-5.6-terra (~$0.071/paper by its own header) against the fallback's gpt-5.6-luna
  (~$0.009/paper), so copying it would have made the recommended choice ~7x dearer
  than choosing nothing. A leftover `models.yaml` from a previous run is offered for
  removal, because otherwise it silently overrides the choice just made.

- Invalid input at either wizard menu re-prompts instead of resolving to entry 1. The
  provider menu's old fallback printed `defaulting to Anthropic` and continued, so one
  slipped keystroke picked the dearest provider on the list; the category menu compared
  the raw string to `"1"` and treated everything else as "manual".

- The wizard's readiness check is now the same provider-aware check `agent ingest`
  preflights with, so its verdict and the first ingest's outcome cannot disagree. It
  previously used `has_synchronous_llm()` — "is any key set anywhere" — which printed
  a ✓ for an Anthropic key against an OpenAI-routed config, the exact mix-up the step
  exists to catch.

- The wizard restated the bootstrap PDF threshold as 5 against the real
  `MIN_INBOX_FOR_BOOTSTRAP = 3`, so users with 3–4 PDFs were told bootstrap was
  unavailable when it would have worked. It now imports the constant.

- `db rebuild` is no longer documented as a required post-ingest step. Per-page
  commits keep `papers`/`claims` current, leaving rebuild as the reconciler of last
  resort — deletion detection, `claim-graph reconcile`, and edits made outside the
  package. `reindex` is still required after every batch, since the Tantivy and
  semantic indexes are rebuilt wholesale rather than incrementally.
- `db/rebuild.py`'s module docstring claimed rebuild was "mtime-aware on the second
  pass". There is no mtime fast-path — every page is re-parsed and re-upserted
  unconditionally, which is what makes the reproducibility property hold. Corrected
  the docstring rather than adding the optimization: a full rebuild of a ~500-page
  corpus runs in under a second.

- The version-bump rule in `CONTRIBUTING.md` § *Releasing* now reads the
  breaking-change surface table in both directions: a change that **adds** to a listed
  surface takes MINOR, and one that only alters behaviour behind an existing surface
  takes PATCH. It replaces "any `feat` → MINOR", which keyed the number to a
  Conventional-Commits label chosen per commit rather than to SemVer's actual MINOR
  clause — new functionality introduced to *the public API*. This release is the
  demonstration: its one `feat` adds two stderr warnings inside `model_config.py` and
  no command, flag, `--json` key or frontmatter field, so under the old rule it would
  have shipped as 0.4.0 — the same slot 0.3.0 used for `import` and `export`. The
  table stays the single arbiter, so the choice is still mechanical.

### Fixed

- Documentation corrections, all of which described behaviour the code no longer has:

  - README presented `config/models.chatgpt.yaml` as the zero-config default routing
    every role to `gpt-5.6-luna` at ~$0.01/paper. It is neither: the template puts
    author/critic/judge on `gpt-5.6-terra` (~$0.07/paper by its own header), while the
    *fallback* — used when no `config/models.yaml` exists — is the all-luna table. So
    the documented `cp config/models.chatgpt.yaml config/models.yaml` made a reader ~7x
    more expensive than the default it claimed to be. The two are now described
    separately, and the copy example points at a template that actually changes
    something.
  - `researchwiki/categories.py` has no `VALID_CATEGORIES` set — categories are derived
    from the filesystem — but README and `prompts/init.md` both told users (and the LLM
    driving setup) to edit it. `mkdir wiki/<slug>/` is the whole operation.
  - `bootstrap-categories --apply` does not rewrite the CLAUDE.md table or
    `categories.py`; it creates directories. Both docs claimed otherwise.
  - README's category table was labelled "Shipped defaults". A fresh `init` creates only
    `other` plus the four page-type dirs, so following it produced a rejected
    `--category`. Now labelled as one author's example. The same claim is corrected in
    `prompts/bootstrap-categories-system.md`, where it was steering the taxonomy proposer.
  - README said `models.anthropic.yaml` routes to Sonnet 4.6; it routes to Sonnet 5.
  - "Your first ingest (~5 min, ~$0.05)" contradicted the ~$0.01/paper figure four lines
    below it.

- `agent ingest` now checks that the configured provider has usable credentials
  *before* extracting the PDF and reconciling metadata, instead of discovering it
  in the author phase at the end of that work. The old failure was the worst one
  a new user could hit on their first command: an uncaught `RuntimeError` escaped
  as exit **3** — "internal bug, file a report" — for what is a plain
  configuration error, and because `call_openai_compatible` substitutes the
  literal string `lm-studio` for an unset `OPENAI_API_KEY`, the diagnostic read
  `Incorrect API key provided: lm-studio`, quoting a value the user had never
  typed. It is now exit **2** with the missing variable, the endpoint that needs
  it, and the config file actually in force.

  The check resolves the endpoint through the same precedence as `llm.call`
  (`RW_LLM_BASE_URL` → the config's `base_url:` → the LM Studio default), so a
  preflight verdict can't disagree with what the call site would do. Loopback
  endpoints and `chat-relay` need no credentials and pass; `--stub` skips the
  check entirely. It is deliberately stricter than `has_synchronous_llm()`,
  which answers "is any key set anywhere" and so waves through the case README
  and `prompts/init.md` both name as the one that actually happens — an
  Anthropic key set, the config copy skipped, every role still routed to
  OpenAI. That case now fails with the `cp config/models.anthropic.yaml` hint.

  Argv is still validated first: a path that doesn't exist remains exit 1, so a
  typo isn't reported as missing credentials.

- Wiki-page writers that bypassed `wiki.commit_page` and left the state DB stale
  until the next `db rebuild`. Seven write sites across four modules: the
  memory-evolve applier (`agents/phases/evolution.py`), the claim-graph promotion
  applier (`claim_graph/promote.py`), `attach`'s two `supplementary:` branches, and
  `backfill`'s three frontmatter helpers. Each rewrote frontmatter the DB mirrors —
  `doi` and `venue` are dedicated columns, `supplementary:` and `generated_at:` live
  in `raw_frontmatter` — so `status`, `db query`, and `claims` could read a stale row
  and `db verify` reported the page as `stale`. Every site now reconciles at write
  time; `tests/test_db_write_paths_commit.py` pins the behaviour per branch, with a
  negative control asserting an uncommitted raw write *is* still detected as stale.

## [0.3.0] - 2026-08-11

### Added

- `researchwiki import`, for the corpus most new users actually arrive with:
  an existing library in Zotero, Paperpile, Mendeley or ReadCube. Four phases —
  `preflight` → `inspect` → `apply` → `verify` — of which only `apply` spends tokens
  or writes pages; the rest write nothing outside their run directory. `migrate`
  explicitly refuses this case — it imports markdown pages from an older wiki, while
  these users have PDFs and metadata and no prose — so until now they were told what
  the tool was *not* for and given nowhere to go.

  The export is the asset. A reference manager carries curated DOI, title, authors
  and year, which is exactly what `agent ingest` otherwise rediscovers through its
  most failure-prone stretch (PDF extract → DOI hunt → S2 lookup → LLM reconcile →
  `metadata_sanity`) and the stretch that produces every `unknown-` stem and
  wrong-but-resolving DOI. `inspect` records the exact `--doi/--title/--authors/
  --year` argv each record contributes, so importing becomes a lookup rather than a
  rediscovery.

  Parsing is tolerant by design, because the files that break a strict parser are the
  ones users have. Validated against a real 532-item ReadCube library exported in both
  RIS and BibTeX, reproducing every count exactly from each: 532 records, 521 usable
  DOIs, 529 titles, 517 author lists, 521 years. That file carries a 4-character
  `PMID` RIS tag where the convention is 2, an always-empty `XX` tag on 385 records,
  citekeys containing `:` (55) and non-ASCII (16) that strict BibTeX forbids outright,
  and CRLF endings that make a literal `"\r\n"` split return one giant record.

  Triage is three verdicts — `ready`/`review`/`skip` — with the detail in reason
  strings. Two gates are worth naming. `no-text-layer` is the silent one: a scanned
  PDF extracts to nothing, ingest logs a warning nobody reads, and the page then
  passes every later gate on grounding that does not exist. `superseded-by-journal`
  is invisible to DOI-level dedupe — the real library held 10 preprint/published
  pairs and **zero** duplicate DOIs, so only title comparison finds them, and the
  survivor is chosen deliberately because the two exports listed those pairs in
  different orders. `duplicate-doi` is its complement, for the same paper under one
  DOI twice: that library had none, but a concatenated or merged export produces
  exact duplicates readily, and nothing downstream would catch them — `apply`
  re-checks the wiki, not the manifest, so both records dispatch in one wave.
  Both gates pick their survivor by a total order rather than by input position.

  `<pdf-root>` is optional, and the report always lists every record that clears
  every gate except having a file, with its DOI. For a cloud-hosted library that
  fetch list is the most useful thing the command produces: without it, a
  metadata-only run reports a count and nothing actionable.

  `researchwiki import apply --run <dir> --limit N` copies a wave into `inbox/` and
  hands it to `agent ingest` with each paper's own `--doi/--title/--authors/--year`.
  It is the only phase that spends tokens or writes pages, and the only one where
  "nothing to do" is a failure rather than a result.

  There is no journal and no staging directory, because nothing here can be left
  half-done: the only mutation is copying a PDF, and everything after belongs to
  `_ingest_batch`, which already keeps a crash-safe checkpoint. Recovery is
  `agent ingest --resume <batch-dir>`, the path users already know.

  One fact is deliberately **not** frozen in the manifest. Pairing, verdicts, stems
  and argv are, so `apply` cannot conclude something different from the `inspect` a
  user read — but whether a paper is *already in the wiki* is a fact about now.
  Re-checking it per record is what makes `--limit 30` mean "the next 30 still
  pending" instead of "the first 30, again", so waves compose instead of repeating.

  `researchwiki import verify --run <dir>` closes the loop: landed / sandboxed /
  not-yet-imported per record, plus the `lint` keys an import can break and the
  graph-wiring follow-ups. It reads the manifest rather than the batch checkpoint,
  because the question is about records ("did this paper reach the wiki") and the
  checkpoint only knows about files — a record can finish its ingest and still sit
  in `.agent-output/`, which is finished work awaiting review, not a failure.

- `researchwiki export` — the corpus as BibTeX / RIS / CSL-JSON, which is the
  inverse of `import` and the answer to "can I get my data out". Zero tokens, no
  network, and byte-identical across runs, so a `.bib` can live in version control
  and diff meaningfully. One phase; nothing is written without `--out`.

  **The citekey is the page stem.** Long, and the only key that cannot change
  under you: a short key has to disambiguate collisions with a letter suffix — 47
  pages collide across 19 naive `surname+year` keys on the real corpus — and that
  suffix is recomputed every run, so ingesting one more paper by the same author
  in the same year renumbers keys already sitting in a manuscript. There is no
  `--citekey` flag, because shipping an unsafe alternative beside the safe one
  invites exactly that failure.

  **Names are not parsed for BibTeX or RIS**, which both understand
  `First von Last` themselves. That is what makes 58 nobiliary-particle names
  (`A. van der Graaf`) and 76 four-token names impossible to corrupt: no boundary
  is ever guessed. Only CSL-JSON wants structured `family`/`given`, and there
  anything ambiguous becomes a CSL `literal` — the format's own construct for a
  name with no given/family structure, so declining to split is a faithful record
  rather than a fallback.

  Gaps are downgraded and reported, never invented. 53 venue-less papers become
  `@misc` rather than `@article` with an empty `journal`, because that shape makes
  `bibtex` merely *warn* — surfacing weeks later in a LaTeX log instead of in the
  bibliography and the report. Three venue values that are really typesetting
  furniture (`Journal of LaTeX Class Files`, `preprint`) are suppressed; that is
  the only place this command could have printed a falsehood. Eight DOIs recorded
  as `https://doi.org/…` normalize through the importer's own `clean_doi`, so they
  are neither emitted as URLs nor dropped by a stricter validator. `document_id`
  values like `Nature 654:324-326` *contain* a volume and page range and are still
  passed through to `note` whole, because the field is free text and FDA guidance
  numbers use it differently. 248 of 421 titles carry a token a title-lowercasing
  style would destroy, so titles are brace-protected per word.

  Synthesis, idea and concept pages are excluded with no flag to include them.
  They have no DOI, venue or year of record, so an entry for one would assert a
  publication that does not exist once pasted into a manuscript — a
  citation-integrity problem rather than a formatting one. Sharing analysis is
  what `prompts/share-page.md` is for.

  Round-trips: 421 records, all three formats, zero mismatches on title, authors,
  DOI, venue or year back through `refimport`, and `import preflight` on the
  emitted file reports the same counts. The one documented loss is that the 149
  `@misc` entries return as `preprint` (96) and `other` (53) rather than
  `article`; that information genuinely is not in the corpus, and a test asserts
  the asymmetry rather than hiding it.

- Author-name parsing lives in one module (`researchwiki/names.py`), shared by
  stem derivation and the exporter. `stems.first_author_surname` held the only
  real parser — `et al.` stripping, consortium detection, and a
  nobiliary-particle walk whose floor is what keeps `Bin Liu` and `Di Liu` correct
  even though `bin` and `di` are particles — and the exporter needed the same
  boundary for CSL. Behaviour-preserving, gated on it rather than asserted: output
  is byte-identical across all 432 distinct author strings in the corpus and the
  documented test shapes.

- `refimport.clean_doi` is public. DOI wrappers arrive from both directions, so
  the normalizer belongs to the package rather than to `parse`'s internals.

- `prompts/export-shareable.md` → `prompts/share-page.md`. It triggered on the
  word "export", which now names a command that does something else; the output
  has always gone to `share/`, so this aligns the name with the destination.

- `_ingest_batch.new_batch` accepts `per_input_args`, mapping an absolute input path
  to flags for that PDF alone. The CLI still refuses `--doi`/`--title`/`--authors`/
  `--year` in batch mode and should: one `--doi` has no meaning across N PDFs. But a
  programmatic caller holding a *different* DOI for every input is the case that
  guard was never about, and it is exactly what importing a reference-manager
  library is. Persisted in `plan.json` as an additive key, read back with a default,
  so batch directories written before this still resume.

### Fixed

- Stem derivation no longer drops a nobiliary particle from a name written
  `Family, Given`. `van der Graaf, A.` derived `der-graaf`: the part before the
  comma is *already* the surname, but the particle walk ran on it anyway, and its
  floor — the rule that keeps `Bin Liu` from being read as particle + surname —
  stopped it one token early. Also affected `De Winter, S.` and
  `van den Berg, L.`.

  The comma is the hard part, not the walk. It separates `Family, Given` in a
  bibliographic export and it separates *authors* in this wiki's own `authors:`
  field, so `van der Graaf, A.` is one person and `Akari Asai, Zeqiu Wu` is two.
  Treating every comma as an inversion is the obvious fix and it changes **349**
  first-author surnames on the real corpus, because a byline's leading given name
  is so often an initial or a particle lookalike (`Di Liu, Bin Wang`). Two signals
  are now required together: the part before the comma must be surname-shaped
  (every token but the last a particle), and the part after must be a given-name
  run. Measured across every corpus byline *and* every raw `authors:` field —
  756 distinct inputs — exactly the three broken cases change and no existing
  stem does. Callers are still expected to split a byline before asking for its
  first surname; this keeps the function robust when one doesn't, because the
  failure mode is a silent wrong filename.

- Stem derivation folds Unicode dashes to ASCII `-`, so a paper whose title is set
  with U+2010 no longer derives a different stem than the same paper spelled with a
  plain hyphen. Publisher-set titles use the Unicode forms freely, and the
  `[^a-z0-9-]` pass *deletes* an unfolded dash instead of preserving the word
  boundary — welding `ATAC‐seq` into `atacseq` while the PDF's own spelling gives
  `atac-seq`. The fold already existed in `slugify_phrase` and was missing from
  `normalize_title_word`, which is the shape of the bug: a normalization every
  caller had to remember separately. It now lives in `strip_diacritics`, which both
  paths share, so slugs and stems cannot drift apart again. Measured against a real
  532-item reference-manager library: 15 of 516 stems were affected. U+2212 MINUS
  SIGN is deliberately excluded — it is category `Sm`, and titles use it as a
  mathematical operator (`CD4−`), not as punctuation.
- Suspended compounds no longer leave a dangling separator in a stem. A title like
  *"epigenome- and transcriptome-wide"* produced `…-in-epigenome-`, and
  *"long- and short-read"* produced `…-long--and-short-read` — trailing and doubled
  hyphens no other stem carries. Interior hyphens are untouched, so CLAUDE.md's rule
  that a hyphenated term counts as one word (`Cas-OFFinder`) still holds. 3 of 516
  stems in the same library. The defect is cosmetic rather than breaking:
  `STEM_PREFIX_RE` only anchors the `{surname}-{year}` prefix, so it matches a
  doubled-hyphen stem, and the one such page in this corpus
  (`chang-2025-rapid-accurate-long--and-short-read`) parses and grades normally with
  41 claims.

  Neither fix renames anything: existing pages keep the stems they were created
  with, per the stem-stability rule in CLAUDE.md § *Disambiguation & updates*. Seven
  pages in this corpus would derive a different title-part if re-ingested — six from
  the dash fold (`…-buchwaldhartwig`, `…-crisprnet`, and four `…-crisprcas9` stems,
  all from en dashes or U+2010 in the published title) and one from the hyphen repair
  (`…-long--and-short-read`). Leaving them is the deliberate call: a stem is the
  filename and every inbound `[[wikilink]]`, so the corpus will carry both
  conventions until a paper is re-ingested for some other reason. Worth knowing
  before a recovery re-ingest of any of the seven, which will land the new form.

## [0.2.1] - 2026-08-10

### Added

- `RW_RELAY_TIMEOUT` (seconds) overrides the chat-relay poll deadline, previously a
  hard-coded 600 s. The deadline starts when the prompt is written rather than when
  the responder notices it, so it does not survive concurrency: with several
  ingests in flight each holds its own 600 s clock, and a responder answering them
  serially can lose workers it has not reached. Settable per run instead of raising
  the floor for everyone, which would also make a genuinely abandoned run hang ten
  times longer. Resolved per call, not as a default argument, so the value is not
  frozen at import; an unusable value falls back to 600 rather than failing.
- `agent ingest` warns when batch mode is combined with the `chat-relay` provider,
  the one provider/mode pairing that fails silently: each worker's stderr is
  redirected into `.ingest/batch-<ts>/worker-*.log`, and that is where the relay
  prints its pending-prompt notice, so the responder is never told a prompt is
  waiting and every worker eventually fails its 600 s timeout. The warning names
  the fix (one foreground invocation per responder) and the alternative (keep batch
  mode, poll `.llm-relay/pending/`). It warns rather than refuses, because batch
  mode is legitimate here for a responder that polls. Detection is
  `model_config.uses_chat_relay()`, which checks the `RW_LLM_PROVIDER` override
  *and* the resolved per-phase providers — a `models.yaml` can route individual
  phases to chat-relay while the rest stay on an API provider.
- Chat-relay prompt payloads carry `stem` and `pdf`, so a responder answering
  several ingests at once can tell whose prompt it is holding without parsing the
  prompt body. `stem` is null on the `reconcile` prompt — the phase that derives
  it — and `pdf` is set before the first call, so every prompt is attributable.
  Additive and nullable, so `schema_version` stays `1` and the response shape is
  unchanged; neither field feeds the op_id hash, because op_id is the cache key
  and folding identity into it would invalidate responses already on disk.

### Fixed

- `researchwiki init --scaffold-only` now creates `wiki/index.md`. It created every
  content directory but not the catalog, and `promote._append_index_entry` returns
  False when that file is absent — so on a fresh clone the *first* paper ingested
  never got its catalog line while every later one did. Existence is the whole
  requirement: a missing `## <category>` section is created by the splice, not
  refused. Idempotent, and it never rewrites an existing catalog. The
  accompanying warning no longer offers "category section absent" as a possible
  cause, since that case is handled — it now names the missing path and the fix.
- Reconcile no longer adopts a PDF's `/Title` when that field holds production
  furniture rather than a title. Oxford stamps it with an internal job code and the
  page range — minimap2's is `OP-CBIO180195 3094..3100`, 24 characters beginning with
  nothing on the banned-prefix list, so the length-and-prefix check took it and
  short-circuited the first-page text scan. It then became the Semantic Scholar
  title-match query (three 404s, retried) and the last-resort page title. 9 of 345
  `/Title` values in the corpus are furniture of this shape. A deny-list cannot cover
  it because the failure is a *shape*, so titles are now checked for one: at least
  three tokens that look like words. Rejection falls through to the text scan, which
  CLAUDE.md already names as the source of truth ("the PDF's first page text, not
  `reader.metadata`"). Measured over every corpus PDF: 336 of 345 kept, and all 9
  rejected are exactly the furniture cases — no real title is refused.
- The first-page title scan skips journal mastheads. With furniture `/Title` values
  now falling through to it, the first qualifying line was often the masthead instead
  — bae-2014's reads "Vol. 30 no. 10 2014, pages 1473–1475 BIOINFORMATICS
  APPLICATIONS NOTE doi:10.1093/bioinformatics/btu048", which passes both the prefix
  and authorish checks. Volume/issue citations, explicit page ranges, inline DOIs,
  "Advance Access publication" running heads and bare `19: 1655-1664` volume:page
  citations never occur inside a title, so those lines are skipped. 7 of the 8
  affected PDFs still on disk now yield their real title.
- Reconcile no longer takes Semantic Scholar's year over the document's when S2 has
  merged a preprint into the journal record. The existing `_s2_record_is_preprint`
  guard needs S2 to *admit* it is describing a preprint by naming a preprint venue;
  S2 also merges the other way, keeping the journal's venue and the preprint's year.
  For minimap2's journal DOI `10.1093/bioinformatics/bty191` it returns `year=2017`
  `venue='Bioinform.'` (its `ArXiv: 1708.01492` was posted 2017-08) while the PDF
  prints "accepted on May 4, 2018" throughout — and since S2 outranks the LLM's
  reading of the document by two places in the chain, a correctly-extracted 2018 was
  discarded and the stem came out `li-2017-…`. When S2 and the document disagree,
  Crossref now arbitrates: its record for a journal DOI is the journal's own, with no
  preprint to merge. Fires only on disagreement, so the common path adds no request;
  no Crossref answer leaves the prior behaviour intact. Does **not** cover the
  distinct case of an arXiv-DOI preprint whose version year differs from S2's (arXiv
  DOIs are not in Crossref) — CLAUDE.md's "version year on the document" rule still
  governs there, by hand.
- `prompts/chat-relay.md` no longer documents a lock that does not exist. It
  claimed the relay grabs `.llm-relay/lock` so parallel ingests serialize on
  chat-relay phases; there is no such lock (the only `flock`s guard `index.md` and
  back-link writes), and relay calls are isolated per op_id, so concurrent ingests
  can be answered in parallel. It also gave the op_id formula as
  `sha1(phase|prompt|stem)`, where the code hashes `phase|prompt` plus a
  `retry_of` discriminator. Documented the real constraint in its place: batch mode
  parallelizes correctly but runs each worker with `stderr=subprocess.STDOUT` into
  a log file, which is where the relay's pending-prompt notice goes — so under
  chat-relay a batch run looks like a hang and then times out. Parallelize
  chat-relay with one foreground single-PDF invocation per responder instead; noted
  as the documented exception to CLAUDE.md's "never fan out one Bash task per file".
- Numeric-drift no longer glues adjacent numeric table columns into one impossible
  number. `parse_claims._extract_table_rows` joins cells with spaces, so a Results
  row arrives as prose like `74,514 559`; `collapse_spaced_thousands` then read
  `514` as a fresh thousands lead and produced `74,514559`, a value nobody wrote and
  nothing can match, so the claim drifted by construction. `_SPACED_THOUSANDS_RE`'s
  lookbehind now also excludes a comma — the mirror of the trailing `(?!,\d)` guard
  added for superscript markers, on the same reasoning: a lead group immediately
  preceded by a comma is the tail of an already-comma-delimited number and cannot
  also begin a space-delimited one. Punctuation is unaffected (`In 2020, 300 000`
  still joins, since the character before `300` is the space). Measured across all
  11,907 claims: 7 claims cleared, 0 newly flagged, 11,899 untouched. Rows of three
  or more numeric columns still corrupt (`1,173 864 619 486` → `864619486`); that
  case is genuinely ambiguous against a real space-delimited 864,619,486 and needs
  claim provenance threaded into the grader, which is separate work.
- Numeric-drift no longer fires on values a paper only ever writes with a letter
  prefix. `NUMERIC_TOKEN_RE`'s `(?<![\w.])` lookbehind — which correctly stops
  `K562` from contributing `562` — also hid Phred/QV notation, so Hansen 2026's
  "QV increased from Q63.1 ... to Q68.9" put neither value in the evidence set and a
  page correctly claiming "QV from 63.1 to 68.9" was flagged as drift and refused
  auto-promotion. `63.1` escaped only because that PDF also says "the initial QV was
  63.1" unprefixed, so whether the veto fired came down to luck. Evidence-side
  matching now also admits `<letter><decimal>` forms; the decimal point is the
  discriminator, so letter+integer identifiers (`K562`, `HG002`, `GRCh38`,
  `rs45512696`) stay excluded. Additive to the value-set, so it can only suppress
  false drift, never mask a rounding — `grade paper` on the affected page goes from
  1 drift claim to 0 with every other value still checked.
- `pip install 'researchwiki[mcp]'` no longer installs a broken `mcp-serve`. The
  extra was unbounded (`mcp>=1.0`) and mcp 2.0 removed `mcp.server.fastmcp`, which
  the server is built on, so a fresh install of the extra resolved to a version the
  server couldn't import. Capped to `<2`; the port to the 2.x server API is
  separate work.

### Changed

- The ingest log's `pdf_claims=N` counter is gone, along with the
  `extract ⚠ zero PDF-side claims extracted` warning it drove. It counted lines
  starting with `-`/`*` in the PDF's Methods/Results — markdown bullets, in typeset
  prose, which essentially never appear: it read 0 for 6 of 8 papers in one session,
  and the sole nonzero reading was two samtools/pbmm2 command-line flags
  (`--secondary=no -s 25000 -K 15G`) mistaken for claims. Nothing consumed it
  (`ctx.claims_count` was assigned and never read), so its only effect was a warning
  that fired on nearly every ingest and implied the drift check had been weakened
  when that check reads the full PDF text and was never affected. The warning now
  fires from the `target_claims` phase, which actually extracts PDF-side claims
  (18–35 on those same papers). `extract_sections()` returns
  `(sections, full_text)`; `ctx.claims_count` is removed.
- CI installs the `mcp` extra, so the MCP server's tests actually run. They had
  been `importorskip`-ing on every CI run — which is how the breakage above stayed
  invisible.

## [0.2.0] - 2026-08-07

### Added

- `migrate` — bulk-import one-paper-per-PDF markdown from an older release or a simpler
  LLM wiki, without re-authoring prose. Phases `preflight` → `inspect` → `apply` →
  `verify`, all zero-token; normalizes H2 headings and frontmatter so claim extraction
  works on the imported pages.
- **Commentary page type** (`type: commentary`) with a structural ingest guard: a
  Research Highlight or News & Views is no longer auto-promoted as the paper it
  describes, so the primary authors' results stop being credited to the commentator.
- Concept-hub tooling: a permanent suppression list for declined candidates
  (`candidates concepts --decline/--undecline/--list-declined`), batch LLM triage, and
  aggregation of near-duplicate candidates.
- `backfill hook` (catalog glosses + short names) and `backfill doi --verify`, which
  checks the DOIs already recorded rather than filling missing ones — a wrong-but-
  resolving DOI was previously invisible to every other check.
- `lint` checks: `duplicate_claim_sets`, `zero_claim_papers`, `venue_suspect`,
  `none_placeholders`, `thin_index_text`.
- Dated API pricing table (`config/pricing.yaml`, carrying `as_of:` and its sources)
  behind the cost rollups in `status` and `insights`.
- A recall signal for the critic, so omission stops being rewarded.
- Versioning system: this changelog, a single source of truth for the version, release
  invariants pinned by `tests/test_version.py`, and a GitHub Release built on tag push.

### Changed

- **DOI adoption is gated on identity, not resolvability**: a DOI is adopted only when
  the resolved record *is* this paper (first-author surname, year within ±1, ≥50% title
  overlap), and the URL-DOI hunt no longer reads past the References heading.
- Verified cross-links are written in **both** directions at promote time.
- `claim-overlap` is batched with per-stem coverage tracking instead of running per
  ingest — it confirmed a link on roughly one paper in ten, so per-ingest cost bought
  little.
- Catalog glosses (`hook:`) are derived from page YAML, so `index.md` is regenerable
  rather than hand-maintained.
- `tags:` dropped from paper and commentary frontmatter; `keywords:` already carried the
  vocabulary, and `tags:` was provenance noise on 334 of 391 pages.
- `init` creates the content directories in code instead of shipping `.gitkeep`
  scaffolding.
- The exit-code contract is now decidable, not merely consistent.
- Model routing: the chatgpt config's quality roles (author/critic/judge) moved to
  `gpt-5.6-terra`; `target_claims` routed to the cheaper extractor role.
- Docs consolidated — `docs/` removed, its one file folded into
  `prompts/concept-page-author.md`.

### Fixed

- **Cross-link integrity**: back-links no longer assert citations that were never
  checked, `lint --fix` mirrors the citation direction when inserting one, and a
  `measures_same` verdict no longer earns a Related Papers bullet.
- **Staleness detection** ages synthesis pages on source ingest date, never file mtime —
  which had made every maintenance pass look like a source change.
- **Counting and graph coverage**: DOI-less papers and cross-page citations are no
  longer dropped from the cross-link graph; root meta pages are excluded from orphan,
  db-drift and broken-link noise; `status` counts every page.
- **Stem derivation**: nobiliary particles stay in the first-author surname, and Latin
  letters NFKD cannot decompose are transliterated instead of dropped.
- **Grading and numeric parsing**: space-separated thousands, superscript citation
  markers no longer glued into numbers, claim-anchor citations routed correctly, and the
  abstract tier no longer scores salience against extraction junk.
- **Providers**: Semantic Scholar cache writes are atomic; ORCID and PubMed negative
  caches have a TTL.
- LLM calls negotiate `reasoning_effort` instead of failing on HTTP 400.
- `page_key` no longer crashes when `wiki/` is a directory symlink.
- `attach` consumes its `inbox/` source instead of leaving it as backlog.
- Year is no longer taken from Semantic Scholar when its record is the preprint's.
- Test-suite hygiene: the suite no longer writes to the real `state.db`, and passes from
  any working directory.

### Removed

- The unused `source_collection` frontmatter field.
- `gpt-5.4-mini` from model routing.

## [0.1.0] - 2026-07-18

Initial tagged release.

[Unreleased]: https://github.com/johapark/research-wiki/compare/v0.4.0...HEAD
[0.4.1]: https://github.com/johapark/research-wiki/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/johapark/research-wiki/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/johapark/research-wiki/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/johapark/research-wiki/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/johapark/research-wiki/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/johapark/research-wiki/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/johapark/research-wiki/releases/tag/v0.1.0
