---
name: repo-audit
description: Audit a repo for simplification and consolidation — dead code, path hygiene, duplicated logic, smells. Use when the user asks to audit, clean up, simplify, consolidate, or de-clutter a repo or folder, or mentions dead code, unused files, duplicate logic, hardcoded paths, scattered figures, or repo hygiene. Report-first — present numbered candidates and wait for the user to pick before touching files.
---

# Repo Audit

Surface cleanup opportunities across four concerns. The aim is a leaner repo where the next reader (human or AI) can locate behaviour quickly and trust that what remains is load-bearing.

## Vocabulary

Use these terms exactly. Drifting into "stale," "old," "legacy code" loses precision.

- **Dead** — zero inbound references from entry points or tests.
- **Transitive dead** — only referenced by other dead code.
- **Drift risk** — the *same* transform implemented in two places; results can silently diverge.
- **Off-path output** — a file produced outside its canonical home (`figures/<analysis>/`, `data/processed/<pipeline>/`, `data/generated/`).
- **Legacy** — moved-but-kept code under `legacy/` mirroring its original path. Never deleted on first pass.

## Diagnostics

- **Grep test** — every "dead" candidate must survive a repo-wide grep for both its name *and* every callable spelling (`from foo import bar`, `foo.bar`, string references in configs). One tool is not enough.
- **Deletion test for duplication** — if implementation B vanished, would its callers naturally migrate to A? If not, the two aren't truly duplicates — they just look similar.
- **Convention contract** — once a canonical layout is agreed, record it. The next audit must not re-litigate it.

## Process

### 1. Explore
Default scope: entire repo from cwd. If the user names a folder, restrict to it. Read `CLAUDE.md` and `audit/CONVENTIONS.md` first. Use the Agent tool with `subagent_type=Explore` for the call/import graph and grep sweeps — do not read every file into your own context.

### 2. Present candidates
Write `audit/REPORT_<YYYY-MM-DD>.md` with four numbered lists (one per concern: dead, paths, duplication, smells). For each candidate:

- **Files** — paths and line numbers
- **Problem** — one sentence
- **Solution** — what would change (not the diff)
- **Confidence** — high / med / low. Low requires the user to verify.

Do not apply fixes. Ask the user which candidates to take.

### 3. Approval loop
For each chosen candidate: sketch the change, name affected callers, confirm before editing. Apply one section at a time, smallest blast radius first.

- Dead → `git mv` to `legacy/` mirroring source path.
- Paths → rewrite to canonical homes. If no canonical layout is recorded, agree one with the user and write it to `audit/CONVENTIONS.md` immediately.
- Duplication → repoint callers to the canonical implementation; move the loser to `legacy/`.
- Smells → propose the refactor; do not bundle with other passes.

Run tests after each section. Commit per section.

### 4. Record rejections
If the user rejects a candidate with a load-bearing reason ("looks dead but cron invokes it," "duplicate is intentional because units differ"), append a one-line entry to `audit/CONVENTIONS.md` so future audits skip it. Skip ephemeral rejections ("not now").

## Hard rules
Never modify `data/raw/`. Never rewrite git history. Never bundle unrelated changes. If the repo has no tests, say so loudly before applying anything destructive. If a section would touch more than 20 files, stop and split it.
