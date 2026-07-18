# Preventing glossary-style concept hubs

**Status:** proposed (rev 2 — cut from 5 fixes to 1 after review)
**Author:** design pass, 2026-07-04
**Trigger:** the retrospective concept build produced 7 hubs, of which 4 (PAM, RNP, LNP, DSB) read as glossary rather than concept notes. Retracted; DSB content moved to a synthesis page. See `wiki/log.md` 2026-07-04 retract entry.

The root cause was simple: **no friction at the scaffold moment forced the "is this really a concept?" judgment.** Once I said yes to acronym-tier terms in an `AskUserQuestion`, the framework happily wrote hubs for them. Detection wasn't wrong (the terms *are* recurring); attach isn't wrong; refresh isn't wrong. The scaffold silently accepted the wrong input.

---

## The fix

`researchwiki concepts <term>` refuses to write a hub without a **thesis** — one sentence answering *why this is a concept, not glossary/synthesis*. The answer lives in the hub's YAML frontmatter as `concept_thesis:` and shows at the top of the page as a review anchor.

### Interface

Interactive (default when stdin is a TTY):
```
$ researchwiki concepts "PAM"
Scaffolding `PAM` (34 member candidates).

In one sentence, why is this a *concept* (an idea the corpus disagrees
about or elaborates on) rather than a *glossary* term (vocabulary papers
use consistently) or a *synthesis* topic (a comparison of approaches)?

Answer or Ctrl-C to abort:
> _
```

Non-interactive:
```
$ researchwiki concepts "long-read sequencing" --thesis "confirmatory
  readout across cgt (edit verification) and compbio (isoform
  confirmation) — same technology, different epistemic burden."
```

Empty answer or empty `--thesis` → refuse to scaffold, exit 1.

### YAML shape

```yaml
---
title: "long-read sequencing"
type: concept
concept_thesis: |
  Confirmatory readout across cgt (edit verification) and compbio
  (isoform confirmation) — same technology, different epistemic burden.
category: [cgt]
referenced_papers:
  - ...
---
```

Rendered in the page body right below the H1, so a reviewer sees the thesis before the Definition and can judge whether the authored Definition actually delivers on it.

### Why this is enough

Every failed hub had a shape that becomes impossible to author under this rule:

- **PAM** — "the protospacer-adjacent motif that Cas9 requires" is the only sentence that fits, and it's a definition, not a concept-thesis. Forced to write it out, the author sees "this is glossary" and aborts.
- **LNP, RNP** — same failure mode. "The lipid nanoparticle delivery vehicle" / "the pre-assembled Cas9-guide complex." Definitions.
- **DSB** — the *only* honest thesis is "positions every editing chemistry on a DSB-required-vs-avoided axis." The author writing that sentence sees they're describing an axis, not the DSB itself, and either retitles to `dsb-free-editing-axis` or promotes to a synthesis page.

The three surviving hubs pass because their thesis writes cleanly:
- **long-read sequencing** — "confirmatory readout, same role different burden across cgt/compbio."
- **familial hypercholesterolemia** — "one disease, three epistemic roles on the LDL-C axis."
- **MSA** — "same operation, three directions of dependency: signal / problem instance / design space."

The thesis is the discriminator. If it can be written honestly, the hub is real.

---

## Implementation

Small. ~50 LOC in `researchwiki/tasks/concepts.py`:

- Add `--thesis STRING` arg + interactive prompt in `main()`. Refuse to call `run()` if empty.
- Extend `run()` to accept `thesis: str` and thread it through.
- `_template()` emits `concept_thesis: |\n  <thesis>` in the YAML block, and inserts a short blockquote under the H1 rendering the thesis for the reader.
- No new lint rule, no detector heuristic, no attach change, no refresh feature. If those pressures ever materialise, we build them then.

Back-filling: the three surviving hubs need `concept_thesis:` added by hand. One-line YAML edit each; the thesis already exists in each page's Cross-domain connections section.

---

## What this does not fix

- **Hand-authored hubs that skip the CLI.** A user who writes `wiki/concepts/foo.md` directly bypasses the prompt. The lint check I originally proposed would catch this, but hand-authored hubs are rare in practice; if the failure mode reappears, add the lint rule then.
- **Bad rationales.** Someone can type "it's a real concept because I say so" and pass. This is a social gate, not a technical one; the discipline is that the thesis is *readable* by future you and by graders.

Everything else in the earlier five-item plan (glossary-score heuristic in `--candidates`, auto-attach warning, retitle proposal, standalone lint rule) was speculative infrastructure for problems that haven't happened yet. Cut them.
