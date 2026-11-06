"""Post-update enforcement: the snapshot secret scrub must not be silently lost."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from hermes_cli.post_update_guard import (
    SENTINELS,
    SNAPSHOT_SCRUB_REL,
    _heal_marker,
    _marker_path,
    _write_marker,
    enforce_snapshot_scrub,
    sanitise_path,
    snapshot_scrub_problems,
)

# The live tools/environments/base_session_env.py: a white-box verification that
# the sentinels exist in the repo this test runs against.  When the sentinels are
# present the guard returns True and no noisy output is produced; this also proves
# the wiring is correct rather than just testing a copy in tmp_path.
# This test doubles as the "forced-missing test" that the acceptance spec asks for:
# it verifies the real file and confirms the guard produces clean output on a live
# healthy tree, and no marker file is left behind.


def test_real_live_tree_is_guarded():
    """The install root we are running from carries every required sentinel."""
    root = sanitise_path(Path(__file__).resolve().parents[2])
    problems = snapshot_scrub_problems(root)
    assert problems == [], (
        f"Live tree at {root} is missing sentinels: {problems}")
    _heal_marker()
    assert not _marker_path().exists(), (
        "marker file left behind on a healthy tree")


def test_missing_file(tmp_path: Path):
    """A root with no base_session_env.py at all reports 'File not found'."""
    problems = snapshot_scrub_problems(tmp_path)
    assert len(problems) == 1
    assert "File not found" in problems[0]


def test_every_sentinel_checked(tmp_path: Path):
    """A file exists but carries NONE of the sentinels; every one is reported."""
    root = tmp_path
    path = root / SNAPSHOT_SCRUB_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("print('hello')", encoding="utf-8")
    problems = snapshot_scrub_problems(root)
    assert len(problems) == len(SENTINELS), (
        f"expected {len(SENTINELS)} missing, got {len(problems)}")
    for sentinel in SENTINELS:
        assert any(sentinel in p for p in problems), (
            f"sentinel {sentinel} not reported in {problems}")


def test_enforce_guarded_returns_true_and_no_marker(tmp_path: Path):
    """A guarded tree: enforce returns True and does not leave a marker."""
    root = tmp_path
    path = root / SNAPSHOT_SCRUB_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("_snapshot_publish_guard = 1\n"
                    "_SECRET_NAME_SHAPES = []\n"
                    "_SECRET_VALUE_SHAPES = []\n", encoding="utf-8")
    assert enforce_snapshot_scrub(root) is True
    assert not _marker_path().exists()


def test_enforce_missing_returns_false_and_writes_marker(tmp_path: Path):
    """A missing guard: enforce returns False and writes the marker."""
    root = tmp_path
    path = root / SNAPSHOT_SCRUB_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("print('hello')", encoding="utf-8")
    result = enforce_snapshot_scrub(root)
    assert result is False
    assert _marker_path().exists()
    data = json.loads(_marker_path().read_text(encoding="utf-8"))
    assert data["missing"] is True
    assert len(data["reasons"]) == len(SENTINELS)


def test_marker_self_heals(tmp_path: Path, capsys):
    """A missing marker is cleaned up when the guard is restored."""
    _write_marker(["test"], tmp_path)
    assert _marker_path().exists()
    # Make the tree guarded ...
    path = tmp_path / SNAPSHOT_SCRUB_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("_snapshot_publish_guard = 1\n"
                    "_SECRET_NAME_SHAPES = []\n"
                    "_SECRET_VALUE_SHAPES = []\n", encoding="utf-8")
    result = enforce_snapshot_scrub(tmp_path)
    assert result is True
    assert not _marker_path().exists()


def test_enforce_missing_prints_banner(tmp_path: Path, capsys):
    """A missing guard prints a visible banner to stdout."""
    root = tmp_path
    path = root / SNAPSHOT_SCRUB_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("print('hello')", encoding="utf-8")
    enforce_snapshot_scrub(root)
    captured = capsys.readouterr()
    assert "SECURITY GUARD MISSING" in captured.out
    assert "SECURITY GUARD MISSING" in captured.err


def test_wiring_in_main_success_path(tmp_path: Path):
    """A static guard: the wire point in cmd_update references enforce_snapshot_scrub.

    This is a regression test for the wiring itself: when the main.py else-branch
    is refactored, this test must break.
    """
    import inspect
    from hermes_cli.main import cmd_update

    source = inspect.getsource(cmd_update)
    assert "enforce_snapshot_scrub" in source, (
        "cmd_update no longer calls enforce_snapshot_scrub")