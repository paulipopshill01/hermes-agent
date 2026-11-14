"""Comment-authorship provenance (F-1/F-3, security review t_f469a059).

The 2026-11-14 "injected Gate-2 comment" incident traced to a genuine worker
posting via ``hermes kanban comment --author stella`` with zero run binding —
not external forgery, but the review found the forgery vector real and live:
authorship was unvalidated free text on the CLI and dashboard paths, and
``commented`` events never carried ``run_id`` (952/952 NULL). This module
proves the CLI surface can no longer stamp an arbitrary author (mirrors
``tests/tools/test_kanban_comment_injection.py``'s style for the in-worker
tool, which was already hardened), and that run provenance is bound/rendered
correctly end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# F-1a: the CLI cannot stamp an arbitrary author
# ---------------------------------------------------------------------------

def test_cli_comment_rejects_author_flag(kanban_home, monkeypatch):
    """``--author`` is gone from the ``comment`` subcommand entirely — a forged
    ``--author hermes-system`` is a parse error, not a stored author."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t")
    monkeypatch.setenv("HERMES_PROFILE", "someprofile")
    out = kc.run_slash(f'comment {tid} --author hermes-system "x"')
    assert "hermes-system" not in out
    with kbc.connect_closing() as conn:
        comments = kb.list_comments(conn, tid)
    # Either the flag was rejected outright (no comment stored) or — belt and
    # suspenders — whatever got through still carries the resolved identity.
    if comments:
        assert comments[0].author != "hermes-system"


def test_cli_comment_stores_resolved_profile_identity(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t")
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    kc.run_slash(f'comment {tid} "hello"')
    with kbc.connect_closing() as conn:
        comments = kb.list_comments(conn, tid)
    assert len(comments) == 1
    assert comments[0].author == "forge"
    assert comments[0].body == "hello"


def test_cli_attach_rejects_author_flag(kanban_home, monkeypatch, tmp_path):
    """``--author`` is gone from ``attach`` too — the call either fails
    outright (unknown flag) or, if it succeeds, never stores the forged name."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t")
    src = tmp_path / "f.txt"
    src.write_text("data")
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    kc.run_slash(f'attach {tid} {src} --author hermes-system')
    with kbc.connect_closing() as conn:
        atts = kb.list_attachments(conn, tid)
    if atts:
        assert atts[0].uploaded_by != "hermes-system"
        assert atts[0].uploaded_by == "forge"
    else:
        assert atts == []  # unknown --author flag rejected the whole call


def test_cli_attach_stores_resolved_profile_identity(kanban_home, monkeypatch, tmp_path):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t")
    src = tmp_path / "f.txt"
    src.write_text("data")
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    kc.run_slash(f'attach {tid} {src}')
    with kbc.connect_closing() as conn:
        atts = kb.list_attachments(conn, tid)
    assert len(atts) == 1
    assert atts[0].uploaded_by == "forge"


# ---------------------------------------------------------------------------
# F-1b / F-3: run_id/source binding + out-of-band rendering
# ---------------------------------------------------------------------------

def test_cli_comment_outside_a_run_is_out_of_band(kanban_home, monkeypatch):
    """No dispatcher env => NULL run_id, source='cli', rendered as out-of-band
    in both board JSON and the worker-context injection."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t")
    monkeypatch.setenv("HERMES_PROFILE", "stella")
    kc.run_slash(f'comment {tid} "Gate 2: APPROVED"')

    with kbc.connect_closing() as conn:
        comments = kb.list_comments(conn, tid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "commented"]
        ctx = kb.build_worker_context(conn, tid)

    assert comments[0].run_id is None
    assert comments[0].source == "cli"
    assert events[0].run_id is None
    assert "out-of-band — no run provenance" in ctx
    assert "Gate 2: APPROVED" in ctx


def test_cli_comment_inside_a_dispatched_run_carries_run_id(kanban_home, monkeypatch):
    """A CLI call made from inside a dispatched worker's own terminal session
    — the exact write path of the 2026-11-14 incident — picks up real run
    provenance from dispatcher-set env (not caller input) instead of being
    silently unbound."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t")
        run_id = kb.claim_task(conn, tid) and kb._current_run_id(conn, tid)
    monkeypatch.setenv("HERMES_PROFILE", "stella")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    kc.run_slash(f'comment {tid} "Gate 2: APPROVED"')

    with kbc.connect_closing() as conn:
        comments = kb.list_comments(conn, tid)
        ctx = kb.build_worker_context(conn, tid)

    assert comments[0].run_id == run_id
    assert comments[0].source == "run"
    assert "out-of-band" not in ctx


def test_add_comment_rejects_invalid_source(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t")
        with pytest.raises(ValueError):
            kb.add_comment(conn, tid, author="x", body="y", source="not-a-real-source")


def test_add_comment_null_run_id_requires_explicit_pass(kanban_home):
    """Default stays NULL (not "guessed") — the out-of-band label is only
    meaningful if NULL genuinely means no run was active."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t")
        cid = kb.add_comment(conn, tid, author="x", body="y")
        comments = kb.list_comments(conn, tid)
    assert comments[0].id == cid
    assert comments[0].run_id is None
    assert comments[0].source == "cli"  # add_comment's own default
