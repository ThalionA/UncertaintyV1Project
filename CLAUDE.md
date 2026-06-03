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
