# Fork-pin upstream sync — ops runbook

**Context:** `t_79e44797` re-pointed this checkout's `origin` remote at Paul's
private fork (`paulipopshill01/hermes-agent`) to permanently close a 3x
recurring CWE-312 regression (`hermes update` resetting the checkout to
`origin/main` and discarding the local-only terminal-snapshot secret scrub
patch, because the patch never lived on any branch a reset could land on).
`NousResearch/hermes-agent` is now the `upstream` remote, read-only.

**Side effect (Viktor review F-3, Info/CWE-1104):** `hermes update`'s
auto-sync-with-upstream step (`hermes_cli/update_cmd_git.py:413-419`)
permanently skips merging `upstream/main` whenever the fork has unique
commits — which it always does now (the scrub patch + regression tests are
fork-only commits). This is *intentional and correct*: it's what stops the
update flow from reverting the scrub. The cost is that **upstream security
fixes no longer arrive automatically.**

## Manual cadence

Run weekly (or after any NousResearch/hermes-agent security advisory):

```bash
scripts/check_upstream_divergence.sh
```

This is read-only (fetch only, no merge). Exit code `2` means upstream has
commits the fork doesn't — review them, then:

```bash
git pull upstream main     # merge; resolve conflicts, keep the scrub patch
# re-run the guard + scrub test suites before pushing:
.venv/bin/python -m pytest tests/tools/test_snapshot_secret_scrub.py \
    tests/hermes_cli/test_post_update_guard.py
git push origin main       # push the merge to the fork
```

**Never** `git reset --hard upstream/main` or otherwise fast-forward past
the fork's unique commits — that is the exact action that would silently
reopen the CWE-312 regression this runbook exists to prevent.

## Automating the cadence (optional)

If/when this should run unattended, wire
`scripts/check_upstream_divergence.sh` into a Hermes cron job (`cronjob_manage`)
on the ops/Dave profile with weekly schedule and notify-on-exit-2. Left as a
manual step for now (LOW priority, deferred per `t_20438626`) — automating
the *check* is safe (read-only), but the merge step must stay a human
decision; do not auto-merge upstream commits into a security-pinned branch.

## Related

- `hermes_cli/post_update_guard.py` — detect + pinned-blob auto-restore for
  the scrub guard itself (defense-in-depth against any out-of-band wipe,
  including a wipe caused by ignoring this runbook).
- `/root/security-reviews/hermes-update-reset-forkpin-20261107.md` — the
  full Viktor review this runbook's F-3 finding came from.
