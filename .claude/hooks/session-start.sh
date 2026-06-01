#!/bin/bash
# SessionStart hook: install the minimal Python deps the IO-HMM model + its
# test suite need (numpy, scipy, pytest) so tests run in Claude Code on the web.
# Synchronous + idempotent; only does work in the remote (web) environment.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"
python3 -m pip install --quiet --disable-pip-version-check -r requirements.txt
