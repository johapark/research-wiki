"""The one exception family the CLI funnel maps to exit code 2.

The exit-code contract (CLAUDE.md, and `AGENTS.md` symlinks it to every other
agent tool) distinguishes 2 "environment error — go look at the machine" from
3 "internal bug — file a report". Deciding which one a failure is cannot be
done at the top of the call stack: by the time an exception reaches
`__main__.main`, a locked `state.db` and a `KeyError` in the grader are both
just `Exception`.

So the classification has to be made where the failure originates, and carried
up in the exception's *type*. `EnvironmentFailure` is that carrier. Raise a
subclass at the boundary that touched the unreliable thing — the DB, the search
index, a provider socket — and the funnel maps it to 2. Everything else reaches
the generic handler and gets 3 plus a traceback, which is the right outcome for
a bug.

Before this existed the boundary was drawn by whether a task author happened to
wrap `main` in a `try/except` at all: 17 of 33 task modules didn't, so an
unreachable provider inside `search`, `db`, or `reindex` reported 3 = "internal
bug". The 16 that *did* wrap used `except Exception: return 2`, which fails the
other way — a genuine `KeyError` reported as "environment error", traceback
swallowed. Both directions came from asking the question in the wrong place.

Subclasses live next to the resource they describe, not here:
  - `db.connection.StateDBUnavailable`     — state.db unopenable / unreadable
  - `grade.grounding.ClaimDBUnavailable`   — claims table unreachable
  - `index.types.SearchBackendUnavailable` — index not built yet
  - `agents.llm.ProviderUnavailable`       — configured LLM has no credentials

`EnvironmentFailure` subclasses `RuntimeError` so that pre-existing
`except RuntimeError` handlers (and the graceful-degradation `except Exception`
blocks that fall back rather than exit) keep behaving exactly as before. This
type only changes what happens to an exception nobody catches.

That last sentence is the whole difficulty, so the three call-site categories
below are the house rules. A phase added later inherits its behaviour from
which category its *caller* is in, not from whichever broad `except` happens to
sit above it. `tests/test_environment_failure.py` pins all three.

1. **Phase wrappers propagate.** A function whose job is "call one model phase
   and parse the answer" must re-raise `EnvironmentFailure` ahead of any
   `except Exception` fallback, so the fallback only ever absorbs a parse miss
   or a malformed reply. Swallowing it produces the worst outcome available:
   the command exits 0 having quietly written a degraded artefact (a page with
   `keywords: []`, a paper filed from provider metadata with the extraction
   step skipped) that looks indistinguishable from a good one. Write it as
   `except EnvironmentFailure: raise` immediately before `except Exception`.
   `agents.judge.run_llm_judge` is the reference implementation.

2. **Optional work records a skip.** Work the command's success does not depend
   on catches `EnvironmentFailure` deliberately and logs a skip instead of
   failing. The test is whether the artefact is already canonical: once
   `promote_to_wiki` has landed a page, PDF, back-links and log entry, no
   later phase may retract that by raising. This is the same contract
   `agents.budget.BudgetExhausted` already has at those sites, and it must
   cover `EnvironmentFailure` for the same reason — see
   `agents.runner_support.run_post_promote_memory_evolution`.

3. **Loops over independent items stop and report.** A command that judges N
   items catches `EnvironmentFailure` at the *loop* boundary, stops iterating,
   and returns what it accumulated together with the reason it stopped. Do not
   instead let each item swallow it: the failure modes are highly correlated
   (an absent credential, or a chat responder who walked away), so per-item
   tolerance turns one 600 s timeout into N of them with no output. Do not let
   it unwind past the loop either — the caller has already-computed results
   that cost real tokens. `tasks.lint.cross_paper` and
   `tasks.claim_overlap` are the reference implementations.
"""

from __future__ import annotations


class EnvironmentFailure(RuntimeError):
    """The machine is at fault, not the command line — exit code 2.

    Message text reaches the user verbatim as
    `researchwiki <command>: <message>`, with no traceback, so write it as a
    diagnostic they can act on. Name the resource and the remedy where one
    exists: "search index not built — run `researchwiki reindex`" rather than
    "backend unavailable".
    """
