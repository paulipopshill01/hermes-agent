#!/usr/bin/env bash
# F-3 (Viktor review, t_79e44797 / t_20438626): fork-pin permanently disables
# automatic upstream sync (hermes_cli/update_cmd_git.py:413-419 skips the
# upstream merge whenever `origin` carries unique commits over `upstream`).
# That is correct behaviour -- it is what stops `hermes update` from
# re-wiping the local-only CWE-312 scrub patch -- but it means security
# fixes landing on NousResearch/hermes-agent now require a human to notice
# and manually `git pull upstream main && git push origin main`.
#
# This script answers one question: "how far behind upstream/main is our
# fork/main, and is that a problem?" It never writes anything -- read-only,
# safe to run from any checkout, any time, with no side effects.
#
# RECOMMENDED CADENCE: run this weekly (manual, or wire into a cron job via
# `cronjob_manage` on the ops profile). It is intentionally NOT auto-merging
# anything -- upstream commits must be reviewed before they land on the fork,
# same as any other change to a security-relevant checkout.
#
# Usage: scripts/check_upstream_divergence.sh [repo-root]
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

if ! git remote get-url upstream >/dev/null 2>&1; then
  echo "No 'upstream' remote configured in $REPO_ROOT -- nothing to compare." >&2
  exit 0
fi

echo "Fetching upstream (read-only, no merge)..." >&2
git fetch --quiet upstream main

BRANCH="$(git symbolic-ref --quiet --short HEAD || echo main)"
AHEAD_BEHIND="$(git rev-list --left-right --count "upstream/main...${BRANCH}")"
UPSTREAM_ONLY="$(echo "$AHEAD_BEHIND" | awk '{print $1}')"
FORK_ONLY="$(echo "$AHEAD_BEHIND" | awk '{print $2}')"
UPSTREAM_LATEST="$(git log -1 --format='%h %cI %s' "upstream/main")"

echo "fork branch:           $BRANCH"
echo "commits only on fork:  $FORK_ONLY   (expected: the local-only scrub + regression tests)"
echo "commits only upstream: $UPSTREAM_ONLY   (0 = fully caught up; >0 = review needed)"
echo "upstream/main HEAD:    $UPSTREAM_LATEST"

if [ "$UPSTREAM_ONLY" -gt 0 ]; then
  echo
  echo "ACTION: $UPSTREAM_ONLY commit(s) on upstream/main are not on $BRANCH." >&2
  echo "Review them (security fixes especially), then:" >&2
  echo "  git pull upstream main   # merge, resolve conflicts, keep the scrub patch" >&2
  echo "  git push origin $BRANCH  # push the merge to the fork" >&2
  exit 2
fi

echo
echo "Up to date with upstream/main -- no action needed."
