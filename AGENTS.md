# AGENTS.md

Onboarding pointer for any coding agent that lands in this repo — Codex CLI, Cursor, Aider, Continue, Gemini CLI, Sourcegraph Cody, Claude Code, or anything else that reads a repo-level agent instruction file.

The authoritative project contract for LLM behavior in this repo is [**CLAUDE.md**](./CLAUDE.md). Read it before touching anything. It sets the four grounding rules (no web search for prose, wiki-first answers, PDF fallback, "say I don't have the paper" when true), the page-type contract, and the ingest / query / lint operations.

Task-specific procedures live under [`prompts/`](./prompts/) and are gated on triggers listed at the top of each file:

- [`prompts/init.md`](./prompts/init.md) — first-time setup on a fresh clone (install → provider → categories → dashboard → first ingest).
- [`prompts/ingest-digest.md`](./prompts/ingest-digest.md) — digest-path fallback when the agent-driven `researchwiki agent ingest` won't work.
- [`prompts/recovery.md`](./prompts/recovery.md) — re-ingest after a broken ingest (bad DOI, wrong stem, drifted metadata).
- [`prompts/recategorize.md`](./prompts/recategorize.md) — move a paper to a different category.
- [`prompts/idea-page-author.md`](./prompts/idea-page-author.md) / [`prompts/concept-page-author.md`](./prompts/concept-page-author.md) — author idea / concept pages (both require Verdict → Background → Opportunities → Plans → Caveats or Definition → How-it-appears order and both grounding gates).
- [`prompts/lookups.md`](./prompts/lookups.md) — whitelist API wrappers (`retraction-check`, `preprint-check`, `orcid-lookup`).
- [`prompts/cross-link-discovery.md`](./prompts/cross-link-discovery.md) — cross-link discovery for manually-written multi-topic pages.
- [`prompts/audit-refresh.md`](./prompts/audit-refresh.md) — refresh `wiki/synthesis/suggested-additions.md` from the S2 audit.
- [`prompts/export-shareable.md`](./prompts/export-shareable.md) — export a wiki page as a self-contained shareable.
- [`prompts/chat-relay.md`](./prompts/chat-relay.md) — protocol for filling LLM prompts on behalf of `researchwiki` when it runs in `chat-relay` provider mode (specialized worker role; only fires when the user has set `RW_LLM_PROVIDER=chat-relay`).

If your agent tool auto-loads a specific filename (`CLAUDE.md` for Claude Code, `.cursorrules` for Cursor, `GEMINI.md` for Gemini CLI, etc.), point it at `CLAUDE.md` — that's the single source of truth. Everything else in `prompts/` is trigger-gated and loaded on demand.
