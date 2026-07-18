# Chat-relay protocol

This document tells you (a chat-platform agent: Claude Code, Codex, Cursor,
Continue, Aider, or anything else with file-read/write tools) how to fill
LLM prompts on behalf of the `researchwiki` CLI when the user has it
running in `chat-relay` mode.

You don't need to remember any of this — this file lives at
`prompts/chat-relay.md` and you can re-read it whenever you encounter a
pending prompt.

## When this protocol fires

The user runs a `researchwiki` command (e.g. `agent ingest`, `synthesize`,
`evolve`) with chat-relay enabled — usually via:

```bash
RW_LLM_PROVIDER=chat-relay researchwiki agent ingest inbox/paper.pdf
```

…or persistently via `provider: chat-relay` in `config/models.yaml`.
Whenever the CLI hits an LLM step, it writes a JSON prompt file to
`.llm-relay/pending/` and blocks polling `.llm-relay/completed/` for your
response. You'll see a stderr line like:

```
📨 LLM relay pending [classifier] → .llm-relay/pending/abc1d2e3f4.prompt.json
   Awaiting response at .llm-relay/completed/abc1d2e3f4.response.json (timeout 600s)
```

## Your job — the loop

When you see a pending file, or when the user asks you to "respond to the
pending prompt" / "watch for prompts" / similar:

1. **List `.llm-relay/pending/*.prompt.json`.** If there are no pending
   files, the CLI hasn't asked for anything yet — just wait or do other
   work.
2. **Read the pending file** (it's plain JSON; see schema below).
3. **Produce a response** that follows the `system` + `prompt` fields. If
   `schema` is non-null, your response **must** be a JSON value that
   validates against it (you'll be asked to retry otherwise).
4. **Write the response** atomically to
   `.llm-relay/completed/{op_id}.response.json` (write to a `.tmp`
   sibling, then rename). Most file-write tools do this implicitly.
5. **Do NOT delete the pending file.** The CLI deletes both files when it
   consumes the response.
6. **Don't fabricate prompts.** Only respond to pending files that exist.
   Don't write speculative responses to `completed/` for prompts that
   haven't been asked.
7. Loop back to step 1 if the user is running a multi-phase command (one
   `agent ingest` fires 5–8 prompts).

## Pending file schema (what you read)

`.llm-relay/pending/{op_id}.prompt.json`:

```json
{
  "schema_version": 1,
  "op_id": "abc1d2e3f4",
  "phase": "author" | "classifier" | "judge" | "critic" | ...,
  "model_hint": "claude-sonnet-4-6",   // advisory; you can use any model
  "system": "<system prompt or null>",
  "prompt": "<the actual user prompt — usually a long block>",
  "schema": { /* JSON Schema, or null */ },
  "temperature": 0.5,
  "max_tokens": 2500,
  "retry_of":      null | "<prior op_id>",
  "retry_feedback": null | "<concrete error from the validator>"
}
```

Field notes:

- **`phase`** — what kind of work this is. See the phase glossary below
  for the expected output shape per phase.
- **`model_hint`** — *advisory only*. Whatever model the chat platform is
  using is fine; the `via` field in your response carries traceability.
- **`schema`** — when non-null, your response **must** validate. You'll
  see retries if it doesn't.
- **`retry_of` / `retry_feedback`** — non-null when this is a retry. The
  CLI's previous attempt failed validation; `retry_feedback` is a
  concrete JSONPath-style error message like `"$.items[0]: missing
  required field 'keywords'"`. **Read it carefully** and fix exactly that
  problem in your new response.

## Response file schema (what you write)

`.llm-relay/completed/{op_id}.response.json`:

```json
{
  "schema_version": 1,
  "op_id": "<echo the op_id from the pending file>",
  "via": "<your platform/model identifier, e.g. claude-code/opus-4.7>",
  "response":   "<free-text response>",      // OR
  "structured": { /* arbitrary JSON value */ }
}
```

Exactly **one** of `response` or `structured` per file:

- **`response`** for free-text prompts (most `author`, `critic` work).
- **`structured`** for JSON-output prompts. **Required** when the pending
  file has a non-null `schema` — the CLI rejects schema-mode replies that
  use the `response` text field, since that means the structured-output
  contract was ignored.

The `via` field is optional but recommended — it helps the user audit
which platform/model produced which response if the relay archive is
enabled.

### Atomic write

Write to `…response.json.tmp`, then rename to `…response.json`. This
avoids the CLI's polling loop reading a half-written file. Most agent
file-write tools (Claude Code's `Write`, Codex's editor, etc.) do this
implicitly under the hood; you usually don't need to think about it.

## Phase glossary — what's expected per phase

Each phase has a stable output shape. The system prompt in the pending
file always has the authoritative instruction; this glossary is a quick
mental model for what you're producing.

| Phase             | Output kind | Shape                                                             |
|---|---|---|
| `author`          | free text   | A complete wiki page in markdown (Summary / Key Contributions / Methodology and Architecture / Results / Limitations / Related Papers, with YAML frontmatter) |
| `critic`          | free text   | Revision notes for the author draft                               |
| `judge`           | structured  | Cross-link verdicts, synthesis judgments, or memory-evolution proposals depending on the call site (the schema field tells you exactly) |
| `evolve`          | free text   | Rewritten draft of the wiki page incorporating critic notes       |
| `debug`           | free text   | Same as `author` but with lower temperature                       |
| `classifier`      | structured  | `{categories: […]}` for bootstrap, `{verdict: "stay"\|"new_category"\|"reassign", …}` for suggest-splits |
| `proposer`        | free text   | Short outputs: one-line short-name handles, etc.                  |
| `keywords`        | structured  | `{items: [{key, keywords: [string]}]}` for batched calls or `{keywords: [string]}` for single-paper |
| `link_generation` | structured  | `{verdicts: [{wikilink, verdict, rationale}]}` (citation judge)   |
| `memory_evolution`| structured  | `{verdict, rationale, patch}` evolution proposal                  |
| `synthesis_judge` | structured  | `{verdicts: [{key, verdict: "in_scope"\|"tangential"\|"out_of_scope", rationale}]}` |
| `reconcile`       | structured  | Paper metadata: `{title, year, doi, all_authors, abstract, …}`    |

If the pending file's `schema` field is set, treat that as the strict
contract — the glossary is just orientation.

## Schema mode — validation and retry

When `schema` is non-null in the pending file:

1. Your response **must** put the result in the `structured` field, not
   `response`.
2. The structured value **must** validate against the schema (the CLI
   uses the `jsonschema` package if installed, falls back to a
   lightweight type+required+enum check otherwise).
3. If validation fails, the CLI writes a *new* pending file with:
   - A fresh `op_id` (deterministic — derived from the failing op_id).
   - `retry_of:` set to the prior op_id.
   - `retry_feedback:` set to the validator's error, in JSONPath form
     like `"$.verdicts[0].verdict: 'MAYBE' is not one of ['in_scope', 'tangential', 'out_of_scope']"`.

   Read the feedback carefully and fix exactly that. Don't rewrite from
   scratch — the rest of your previous response was probably fine.
4. After 3 failed attempts, the CLI gives up and raises with the full
   op-id chain in the error message. You won't see further retries.

## What NOT to do

- **Don't write to `pending/`.** Only the CLI writes pending files.
- **Don't delete `pending/{op_id}.prompt.json` after responding.** The CLI
  deletes it on consumption. If you delete it, the CLI keeps polling and
  eventually times out.
- **Don't put both `response` and `structured` in one response file.** The
  CLI accepts the first non-null and ignores the other; ambiguous shape
  means the user can't tell which path was taken.
- **Don't speculate.** If the pending file references a PDF or wiki page
  you can't read with confidence, prefer null/empty fields where the
  schema allows over fabricating content. The system prompts for phases
  like `reconcile` and `evolve` explicitly call this out — emit `null`
  when not confident.
- **Don't change `op_id`.** Echo the input `op_id` in your response file
  exactly. The CLI keys the response by filename, and a mismatch will
  confuse archive/audit tooling.

## Multiple pending prompts (multi-phase commands)

A single `researchwiki agent ingest` fires 5–8 LLM prompts in sequence:
reconcile → author → critic → judge → link_generation → keywords →
short_name. The CLI fires them one at a time — when you fill prompt N's
response, the CLI consumes it and writes prompt N+1. So you'll see
exactly one pending file at a time (during a single ingest).

Two `agent ingest` calls running in parallel can produce two
simultaneous pending files. The relay grabs `.llm-relay/lock` while
holding a single chat-relay call, so the two ingests serialize on
chat-relay phases — but you may still see both ingests' pending files in
flight if they're at different non-relay steps. Just respond to whichever
is pending; the CLI infers ownership by op_id.

## Caching — re-runs reuse prior responses

`op_id` is deterministic: `sha1(phase|prompt|stem)[:12]`. If a CLI run
crashes mid-ingest after you wrote a response but before it consumed,
re-running picks up the same response file (because the same prompt
re-derives the same op_id). You don't need to do anything — the CLI just
skips writing a new pending file when an existing response is on disk.

If the user wants to force regenerate (e.g. you produced something they
want to redo), they pass `RW_RELAY_FRESH=1` which appends a pid+counter
to the op_id, breaking the cache for that one command.

## Stale pending files

If `.llm-relay/pending/` accumulates files older than ~1 hour, those are
probably abandoned (the CLI process died and the user moved on). Don't
respond to them speculatively — the user can clean them up with
`researchwiki relay clean --ttl 1h` (when phase 3 lands) or `rm` them
manually.

The filesystem protocol is the only interface — poll `.llm-relay/pending/`
and write responses there directly. There is no server or tool plugin to
register.
