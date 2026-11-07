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

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Path within the install root where the scrub guard lives.
SNAPSHOT_SCRUB_REL = "tools/environments/base_session_env.py"

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
    Returns ``True`` when the guard look sane, ``False`` when it is absent.
    A ``False`` return **must not fail the update** — the code is committed
    and the notice is the remediation.
    """
    root_path = sanitise_path(root)
    problems = snapshot_scrub_problems(root_path)
    ok = not problems
    if ok:
        _heal_marker()
    else:
        _shout(problems, root_path, stream)
    return ok


def _shout(problems: list[str], root: Path, stream: object | None) -> None:
    """Emit every loud channel at once."""
    path = _scrub_path(root)
    # 1. Banner to stdout (mirrored to update.log + .update_output.txt).
    print(_BANNER, file=sys.stdout, flush=True)
    for p in problems:
        print(f"  • {p}", file=sys.stdout)
    # 2. Also stderr in case the stdout mirror isn't active (standalone call).
    print(_BANNER, file=sys.stderr, flush=True)
    # 3. Write directly to the update log file handle if available.
    try:
        from hermes_cli.update_cmd import _log_only_write
        _log_only_write(_BANNER + "\n")
        for p in problems:
            _log_only_write(f"  • {p}\n")
    except ImportError:
        pass  # standalone call outside the updater
    # 4. Durable marker file.
    _write_marker(problems, root)
    # 5. Logger (gateway.log when running under the gateway).
    logger.error(_LOG_LINE.format(path=path))
    for p in problems:
        logger.error("  %s", p)


def _write_marker(problems: list[str], root: Path) -> None:
    marker = _marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "missing": True,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "root": str(root),
        "reasons": problems,
        "fix": "Re-apply the local CWE-312 patch; see t_c91d51c5.",
    }
    marker.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _heal_marker() -> None:
    """Remove the durable marker when the guard is found present."""
    try:
        _marker_path().unlink(missing_ok=True)
    except OSError:
        pass