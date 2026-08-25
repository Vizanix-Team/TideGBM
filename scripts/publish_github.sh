#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REMOTE="${REMOTE:-git@github-vizanix:Vizanix/TideGBM.git}"
BRANCH="${BRANCH:-main}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository. Run from TideGBM root." >&2
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REMOTE"
else
  git remote set-url origin "$REMOTE"
fi

echo "Remote: $(git remote get-url origin)"
echo "Branch: $BRANCH ($(git rev-parse --short HEAD))"
echo "Testing GitHub SSH..."
ssh -T git@github.com 2>&1 | head -1 || true
echo "Pushing..."
git push -u origin "$BRANCH"
echo "Done: https://github.com/Vizanix/TideGBM"
