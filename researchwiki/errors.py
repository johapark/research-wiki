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
