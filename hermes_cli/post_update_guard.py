"""Post-update assertion that the local security patches survived the update.

``hermes update`` resets a diverged checkout to ``origin/main`` and thereby deletes
every un-pushed local commit (2026-11-03: it discarded the terminal-snapshot secret
scrub, CWE-312, and the live tree went back to publishing credentials at-rest). The
patch cannot be pushed (it is local-only by decision), so the update flow itself has
to prove the guard is still in the live tree after every run — a checklist is not an
enforcement point.

Every ``hermes update`` that succeeds calls ``enforce_snapshot_scrub`` from the
``else`` branch of ``cmd_update`` in ``hermes_cli/main.py``.  When the guard is found
absent the function prints a loud banner to stdout (the gateway streams this live to
the user's platform) and to the update log, writes a durable marker file that
``hermes status`` and the post-update gateway watcher can surface, and logs an ERROR
to the ``hermes`` logger.  The update itself is NOT failed — the code is committed,
and the notice is the enforcement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Path within the install root where the scrub guard lives.
SNAPSHOT_SCRUB_REL = "tools/environments/base_session_env.py"

# --- Option 2 defense-in-depth: pinned-blob auto-restore (t_20438626) -----
#
# Deferred from t_79e44797 (fork-pin closed the *recurrence vector*; this is
# extra depth against an out-of-band wipe, e.g. a manual `git reset --hard
# upstream/main` run outside `hermes update`). Dave's design constraints
# (t_79e44797 decision comment) that this code must honour:
#
#   1. The blob is hash-verified (sha256 PINNED below) -- never trusted on
#      path/name alone.
#   2. It must NEVER silently restore stale/wrong content. Any hash mismatch
#      OR version drift (the blob no longer satisfies every sentinel the
#      *current* code requires) => FAIL LOUD, restore nothing.
#   3. The existing detect-only evidence path (banner + marker + log) fires
#      unchanged on BOTH branches -- restore success does not make the
#      incident quieter.
#   4. Restore never fails the update itself -- same notice-only semantics.
#
# MAINTENANCE: if ``tools/environments/base_session_env.py`` or ``SENTINELS``
# legitimately change, the pinned blob at PINNED_BLOB_REL and the hash below
# must be updated together in the same commit, or every future restore
# attempt will correctly refuse (version drift) until they are realigned.
PINNED_BLOB_REL = "hermes_cli/_security_pins/base_session_env.pinned"
PINNED_BLOB_SHA256 = (
    "3f1f481b4a3127dda2fddd38528599e86491d3cf8af39b42f2b170c0cf5314b7")

# Sentinels that must be present on the live file for the guard to be considered
# functional.  A grep on the symbol name is the right level: a white-box scan
# that passes an empty function body on a name match is acceptable because the
# runtime also exercises the guard (the test suite is gate-gated).
SENTINELS: tuple[str, ...] = (
    "_snapshot_publish_guard",
    "_SECRET_NAME_SHAPES",
    "_SECRET_VALUE_SHAPES",
)

# Durable marker written when the guard is missing.  The gateway watcher and
# ``hermes status`` can surface this; removing it when the guard is restored is
# the self-healing path.
MISSING_MARKER: Path | None = None  # resolved lazily via _marker_path()


def _marker_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "state" / "snapshot_scrub_missing.json"

# Banner printed to stdout (mirrored to update.log and .update_output.txt) so the
# gateway streams it to the subscribing user.
_BANNER = """\
╔══════════════════════════════════════════════════════════════════════════════╗
║  SECURITY GUARD MISSING — snapshot secret scrub is NOT in the live tree.     ║
║                                                                              ║
║  ``hermes update`` reset a diverged checkout and discarded the local-only     ║
║  CWE-312 patch.  Every terminal session snapshot is now publishing            ║
║  credential-shaped exports at-rest.  Reapply the patch with ``hermes agent    ║
║  restore-security-guard`` or ask Dave.                                       ║
║                                                                              ║
║  This notice repeats on every ``hermes update`` until the guard is present.   ║
╚══════════════════════════════════════════════════════════════════════════════╝"""

_LOG_LINE = "SECURITY GUARD MISSING: _snapshot_publish_guard absent from {path}"


def sanitise_path(root: str | Path) -> Path:
    """Normalise to a real ``Path``; never return the string handle."""
    return Path(root).expanduser().resolve()


def _scrub_path(root: Path) -> Path:
    return root / SNAPSHOT_SCRUB_REL


def _pinned_blob_path(root: Path) -> Path:
    return root / PINNED_BLOB_REL


def _load_pinned_blob(root: Path) -> tuple[bytes | None, str]:
    """Load the pinned restore blob and verify it against ``PINNED_BLOB_SHA256``.

    Returns ``(content, "ok")`` only on an exact hash match. Any other
    outcome returns ``(None, reason)`` -- the caller must treat every
    non-"ok" reason as "restore nothing, fail loud".
    """
    path = _pinned_blob_path(root)
    if not path.is_file():
        return None, "pinned blob file not found"
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, f"cannot read pinned blob: {exc}"
    digest = hashlib.sha256(data).hexdigest()
    if digest != PINNED_BLOB_SHA256:
        return None, (
            "pinned blob FAILED hash verification (got "
            f"{digest[:8]}, expected {PINNED_BLOB_SHA256[:8]}) -- refusing "
            "to restore unverified/tampered content")
    return data, "ok"


def _attempt_pinned_restore(root: Path) -> dict:
    """Try to restore the scrub file from the pinned, hash-verified blob.

    Never restores on anything short of an exact hash match, and never
    restores content that itself fails to satisfy every sentinel the
    *current* code requires (version drift: the pin fell behind a legitimate
    change to ``SENTINELS`` or the guarded file). Both failure modes are
    "fail loud, restore nothing" -- per Dave's named concern, a silently
    wrong restore is worse than the existing loud detect-only notice.

    Never raises; always returns a result dict for the evidence trail.
    """
    result: dict = {"restored": False, "reason": None}
    blob, status = _load_pinned_blob(root)
    if blob is None:
        result["reason"] = status
        return result

    text = blob.decode("utf-8", errors="replace")
    missing_in_blob = [s for s in SENTINELS if s not in text]
    if missing_in_blob:
        result["reason"] = (
            "pinned blob is missing sentinel(s) required by the current "
            f"guard ({', '.join(missing_in_blob)}) -- treated as version "
            "drift, restore refused")
        return result

    target = _scrub_path(root)
    tmp = target.with_name(target.name + ".restoring")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(blob)
        tmp.replace(target)
    except OSError as exc:
        result["reason"] = f"restore write failed: {exc}"
        return result
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    # Re-verify from disk -- never trust the in-memory write.
    post_problems = snapshot_scrub_problems(root)
    if post_problems:
        result["reason"] = (
            "restore written but post-write verification still reports "
            f"problems: {post_problems}")
        return result
    try:
        written_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError as exc:
        result["reason"] = f"restore written but re-read failed: {exc}"
        return result
    if written_digest != PINNED_BLOB_SHA256:
        result["reason"] = "restore written but on-disk hash does not match pin"
        return result

    result["restored"] = True
    result["reason"] = f"restored from pinned blob (sha256:{PINNED_BLOB_SHA256[:8]})"
    return result


def snapshot_scrub_problems(root: Path) -> list[str]:
    """Return human-readable lines for every missing requirement.

    ``root`` is the install root (``PROJECT_ROOT`` in the update CLI).
    An empty list means the guard is present and functional as far as a
    static check can determine.
    """
    problems: list[str] = []
    path = _scrub_path(root)
    if not path.is_file():
        problems.append(f"File not found: {path}")
        return problems
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        problems.append(f"Cannot read {path}: {exc}")
        return problems
    for sentinel in SENTINELS:
        if sentinel not in content:
            problems.append(f"Sentinel '{sentinel}' absent from {path}")
    return problems


def enforce_snapshot_scrub(
    root: str | Path,
    *,
    stream: object | None = None,
) -> bool:
    """Check the live tree; if the guard is missing, raise a loud notice.

    *stream* is the update's mirrored stdout (``_UpdateOutputStream``) when
    called from ``cmd_update``, or ``None`` for standalone/test callers.
    Returns ``True`` when the guard looks sane -- either because it was
    already present, or because the pinned-blob auto-restore (Option 2,
    t_20438626) verified and repaired it. Returns ``False`` when it is
    absent and no restore could be safely performed.

    A ``False`` return **must not fail the update** -- the code is committed
    (or auto-restorable) and the notice is the remediation. Auto-restore is
    pure defense-in-depth: it NEVER replaces the loud evidence trail below,
    it only additionally repairs the tree when the pinned blob is an exact,
    hash-verified match for what the current code requires.
    """
    root_path = sanitise_path(root)
    problems = snapshot_scrub_problems(root_path)
    ok = not problems
    if ok:
        _heal_marker()
        return True

    restore_result = _attempt_pinned_restore(root_path)
    # Evidence fires on BOTH branches -- restore success must not make the
    # incident quieter than before (Dave's constraint, carried from
    # t_79e44797). The marker/log always record what was actually missing,
    # plus whether auto-restore fixed it.
    _shout(problems, root_path, stream, restore_result)
    return bool(restore_result["restored"])


def _shout(
    problems: list[str],
    root: Path,
    stream: object | None,
    restore_result: dict | None = None,
) -> None:
    """Emit every loud channel at once."""
    path = _scrub_path(root)
    restore_result = restore_result or {"restored": False, "reason": None}
    if restore_result["restored"]:
        restore_line = f"  ✔ AUTO-RESTORE SUCCEEDED: {restore_result['reason']}"
    else:
        restore_line = (
            "  ✘ AUTO-RESTORE REFUSED (no change made): "
            f"{restore_result['reason']}")
    # 1. Banner to stdout (mirrored to update.log + .update_output.txt).
    print(_BANNER, file=sys.stdout, flush=True)
    for p in problems:
        print(f"  • {p}", file=sys.stdout)
    print(restore_line, file=sys.stdout, flush=True)
    # 2. Also stderr in case the stdout mirror isn't active (standalone call).
    print(_BANNER, file=sys.stderr, flush=True)
    for p in problems:
        print(f"  • {p}", file=sys.stderr)
    print(restore_line, file=sys.stderr, flush=True)
    # 3. Write directly to the update log file handle if available.
    try:
        from hermes_cli.update_cmd import _log_only_write
        _log_only_write(_BANNER + "\n")
        for p in problems:
            _log_only_write(f"  • {p}\n")
        _log_only_write(restore_line + "\n")
    except ImportError:
        pass  # standalone call outside the updater
    # 4. Durable marker file.
    _write_marker(problems, root, restore_result)
    # 5. Logger (gateway.log when running under the gateway).
    logger.error(_LOG_LINE.format(path=path))
    for p in problems:
        logger.error("  %s", p)
    logger.error(restore_line.strip())


def _write_marker(problems: list[str], root: Path, restore_result: dict | None = None) -> None:
    marker = _marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    restore_result = restore_result or {"restored": False, "reason": None}
    data = {
        "missing": True,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "root": str(root),
        "reasons": problems,
        "fix": "Re-apply the local CWE-312 patch; see t_c91d51c5.",
        "auto_restore_attempted": True,
        "auto_restored": bool(restore_result["restored"]),
        "auto_restore_detail": restore_result["reason"],
    }
    marker.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _heal_marker() -> None:
    """Remove the durable marker when the guard is found present."""
    try:
        _marker_path().unlink(missing_ok=True)
    except OSError:
        pass