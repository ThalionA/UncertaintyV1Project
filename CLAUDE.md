# CLAUDE.md — working agreement for AI agents in this repo

Guidance for Claude Code (and any AI agent) working in UncertaintyV1Project.
The aim is to keep the repo's git history clean and avoid the branch /
worktree sprawl that builds up when every session spins off its own branch
and leaves work uncommitted.

## Branch hygiene

- **Reuse, don't multiply.** When continuing related work, commit onto the
  existing feature branch for that work rather than opening a new
  `claude/<random>` branch per session. Check `git branch -a` first; if a
  branch already covers the task, use it.
- **One branch per logical change.** Group a coherent piece of work on a
  single branch. Don't scatter the same feature across several branches that
  later fight over the same files (this repo has hit three branches editing
  `nn_classifier.py` / `run_experiment.py` / `training/config.py` at once).
- **Name branches for the work**, e.g. `feat/early-stopping`,
  `kl-js-entropy-sweep` — not for the session.
- **Don't create orphan branches.** Always branch from an up-to-date `main`
  (`git fetch && git checkout -b <name> origin/main`) so the branch shares
  history with `main` and can be diffed / merged normally.

## Commit & PR discipline

- **Never end a session with uncommitted work.** Before stopping, either
  commit (to a branch) or explicitly note why not. Uncommitted piles on a
  detached/feature branch are how work gets silently lost on cleanup.
- **Don't commit training outputs.** `nn_decoder/results/` (`.mat`, per-run
  `config.yaml`, `.pt` checkpoints) is gitignored — keep it that way. If you
  see result files staged, unstage them.
- **Push branches you want kept.** Local-only branches are one disk failure
  from gone; `git push -u origin <branch>` once the work is worth keeping.
- **Open a PR when a branch is ready**, and **delete the branch once merged**
  (locally `git branch -d`, on the remote via the GitHub UI or
  `git push origin --delete <branch>`). A merged branch left around is clutter.
- **Reconcile overlaps deliberately.** Before merging a branch that edits the
  same core files as another open branch/PR, diff them and decide which
  version wins — don't merge both blindly.

## Periodic cleanup

Run from your local machine (the sandbox usually can't delete remote refs):

```bash
git fetch --prune                       # drop refs to deleted remote branches
git checkout main && git pull           # make 'merged' checks accurate
git branch --merged main | grep -vE '^[*+]|main$'   # branches safe to delete
git branch -d <branch>                  # delete each merged one (safe form)

git worktree list                       # Claude Code worktrees live in .claude/worktrees/
git worktree prune                      # drop bookkeeping for removed worktree dirs
```

To delete a merged branch on the remote (GitHub): use the branches page, or
`git push origin --delete <branch>`.

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
