# CLAUDE.md — working agreement for AI agents in this repo

Guidance for Claude Code (and any AI agent) working in UncertaintyV1Project.
The aim is to keep the repo's git history clean by working **directly on
`main`** — no branches, no worktrees, no per-session PRs.

> **Start every session by reading [`PROJECT_LOG.md`](PROJECT_LOG.md)** — the
> single hub indexing all notes/handoffs, the active open threads, and the
> reverse-chronological session log. **End every session** by adding a dated
> entry at the top of its Session log (and folding pitfalls into `GOTCHAS.md`,
> durable facts into `memory/`).

## No branches — work on `main`

- **Never create a branch or worktree.** Do not run `git checkout -b`,
  `git switch -c`, `git worktree add`, or open PRs. Work directly on `main`
  and commit there. (Per-session branches caused branch/worktree sprawl; we've
  stopped that — one trunk, `main`.)
- **If you're not on `main`, get back on it** (`git checkout main`) before
  committing. Don't leave work stranded on a stray branch.
- **Commit directly to `main`** when a coherent piece of work is done, with a
  clear message. Group related changes into one commit; don't leave the session
  with your own work uncommitted.
- **Don't commit training outputs.** `nn_decoder/results/` (`.mat`, per-run
  `config.yaml`, `.pt` checkpoints) and `nn_decoder/figures/` are gitignored —
  keep it that way. If you see result/figure files staged, unstage them.
- **Only stage what you changed.** If unrelated files are already modified or
  untracked in the working tree, leave them alone — stage just the files for
  the change at hand.
- **Push only when the user asks** (`git push` to `origin/main`).

## Logging — keep `PROJECT_LOG.md` current (every session)

`PROJECT_LOG.md` is the single cross-session source of truth. Treat updating it
as part of finishing the work, not an optional extra.

- **Read it first.** At the start of a session, read `PROJECT_LOG.md` (the
  index, the active open threads, the most recent log entries) before acting.
- **Log at the end of every substantive session** — any time you changed code,
  ran/queued an analysis, produced figures, or made a decision. Add a **dated
  entry at the top** of the Session log (newest first). Trivial one-off chats
  don't need an entry; everything else does.
- **Entry contents:** date + short title, a 3–6 line summary of what changed and
  why, the key files touched, and concrete **open items / next steps**. If you
  wrote a detailed handoff file, link it; otherwise the entry IS the record.
- **Update the surrounding sections in the same pass:** add new **Active open
  threads** and prune done ones; keep the live plan
  (`nn_decoder/PLAN_*`) pointer current.
- **Route durable knowledge to its home:** persistent methodological pitfalls →
  `GOTCHAS.md`; durable cross-session facts → auto-memory (`memory/`). Don't bury
  those only in a session entry.
- **Write a separate handoff doc** (`documents/session_<date>_*.md` or
  `nn_decoder/HANDOFF_*.md`) only for large sessions that need depth; small
  sessions live fully in the log entry. Either way, link it from the log.

## Plotting — always emit PNG **and** SVG; only ever preview PNG

- **Every figure must be saved in both formats.** When plotting anything, write a
  `.png` **and** a `.svg` of the same figure (same basename). SVG is the
  vector copy for the manuscript / research vault; PNG is the raster copy used
  for previewing. A `_save(fig, out_dir, stem)`-style helper that loops
  `for ext in ('png', 'svg')` is the standard pattern — reuse it. If you call a
  plotting routine that only writes SVG (e.g. some `decoder_plotting_utils`
  functions), also emit a PNG of the same figure.
- **Keep PNGs previewable: ≤ 2000 px on every side (aim ≤ 1600).** The Claude
  Code image reader rejects any image whose width *or* height exceeds 2000 px, so
  a figure you can't preview is a figure you can't check. At the default
  `dpi=140`, **2000 px ÷ 140 ≈ 14 in is the hard ceiling for `figsize` in either
  dimension** — and tall multi-row grids blow past it fast (e.g. a 6-row grid at
  ~2 in/row is already too tall). Rules:
    - Cap `figsize` so `max(width, height) × dpi ≤ 2000`; for big grids, lower
      `dpi` (100–110) and/or split into multiple figures rather than one giant one.
    - The `_save(fig, out_dir, stem)` helper should enforce this — after saving the
      full-res SVG, **rasterise the PNG at a dpi chosen so the longest side is
      ≤ 1600 px** (compute it from the figure's inches). Do not hand-tune per call.
    - The SVG keeps full detail for the manuscript; only the PNG needs capping.
- **Only ever open / preview PNG inside Claude Code, never SVG.** To inspect a
  figure, Read its `.png`. If a PNG is still too large to load, downscale a copy
  (≤1500 px) and preview that — never retry the original, never reach for the SVG.

## Core-module caution

`nn_decoder/nn_classifier.py`, `run_experiment.py`, `training/config.py`,
`training/run.py`, and `time_binned_ppc.py` are load-bearing. After touching
them, run the relevant tests before trusting the change:

```bash
python -m pytest tests/ -k "pca or time_binned or ppc or training or early" -q
```

New `Config` fields must default to the no-op value (so production runs are
unchanged) and be carried through `Config.to_legacy_dict` →
`run_experiment` `training_params`.
