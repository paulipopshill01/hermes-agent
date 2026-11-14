"""N-1 (CWE-345, security review t_87b48aca / t_57070ca4): ``run_id`` must be
validated against the run's recorded ``profile`` at BOTH comment sinks —
``add_comment`` and ``_insert_comment`` — not just trusted verbatim off
``HERMES_KANBAN_RUN_ID`` env. A forged/stale/mismatched run_id must downgrade
honestly to NULL/``source="cli"`` rather than render as authoritative in-run
provenance for a comment the env owner never earned.

Covers both sinks: ``add_comment`` (CLI ``comment``/dashboard) and
``_insert_comment`` (specify/decompose audit comments via
``specify_triage_task``/``decompose_triage_task``, the stranded F-1/F-2 half).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Sink 1: add_comment
# ---------------------------------------------------------------------------

def test_add_comment_downgrades_nonexistent_run_id(kanban_home):
    """A run_id naming no real run at all (the acceptance-spec probe:
    HERMES_KANBAN_RUN_ID=999999 on a scratch board with no such run)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t")
        cid = kb.add_comment(conn, tid, author="forge", body="x", run_id=999999, source="run")
        comments = kb.list_comments(conn, tid)
    assert comments[0].id == cid
    assert comments[0].run_id is None
    assert comments[0].source == "cli"


def test_add_comment_downgrades_run_id_belonging_to_a_different_profile(kanban_home):
    """A real run exists, but its profile doesn't match the claimed author —
    env-forged impersonation of someone else's genuine run."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="alice")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = kb._current_run_id(conn, tid)
        assert run_id is not None
        cid = kb.add_comment(conn, tid, author="mallory", body="x", run_id=run_id, source="run")
        comments = kb.list_comments(conn, tid)
    assert comments[0].id == cid
    assert comments[0].run_id is None
    assert comments[0].source == "cli"


def test_add_comment_keeps_run_id_when_profile_matches_author(kanban_home):
    """Legitimate in-run comment: the run's profile matches the author — keeps
    binding (acceptance spec's second bullet)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="alice")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = kb._current_run_id(conn, tid)
        cid = kb.add_comment(conn, tid, author="alice", body="x", run_id=run_id, source="run")
        comments = kb.list_comments(conn, tid)
    assert comments[0].id == cid
    assert comments[0].run_id == run_id
    assert comments[0].source == "run"


def test_add_comment_non_run_source_with_mismatched_run_id_stays_its_own_source(kanban_home):
    """Downgrade only rewrites 'run' -> 'cli'; a non-'run' source that smuggled
    in a bogus run_id (shouldn't happen via any real caller, but the helper
    must not corrupt an already-valid source value) keeps its own source."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t")
        cid = kb.add_comment(conn, tid, author="forge", body="x", run_id=999999, source="api")
        comments = kb.list_comments(conn, tid)
    assert comments[0].id == cid
    assert comments[0].run_id is None
    assert comments[0].source == "api"


# ---------------------------------------------------------------------------
# Sink 2: _insert_comment (specify/decompose audit path)
# ---------------------------------------------------------------------------

def test_specify_audit_comment_downgrades_forged_run_id(kanban_home):
    """The acceptance-spec probe against specify_triage_task: a run_id for a
    run that doesn't exist must not render as source='run'."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", triage=True)
        kb.specify_triage_task(
            conn, tid, title="Specified", body="b",
            author="forge", run_id=999999, source="run",
        )
        comments = kb.list_comments(conn, tid)
    assert len(comments) == 1
    assert comments[0].run_id is None
    assert comments[0].source == "cli"


def test_specify_audit_comment_keeps_run_id_for_matching_profile(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", triage=True)
        other = kb.create_task(conn, title="runner", assignee="forge")
        kb.claim_task(conn, other)
        run_id = kb._current_run_id(conn, other)
        kb.specify_triage_task(
            conn, tid, title="Specified", body="b",
            author="forge", run_id=run_id, source="run",
        )
        comments = kb.list_comments(conn, tid)
    assert len(comments) == 1
    assert comments[0].run_id == run_id
    assert comments[0].source == "run"


def test_decompose_audit_comment_downgrades_forged_run_id(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="root", triage=True)
        decompose_triage_task(
            conn, tid, root_assignee=None, children=[{"title": "child"}],
            author="forge", run_id=999999, source="run",
        )
        comments = kb.list_comments(conn, tid)
    assert len(comments) == 1
    assert "Decomposed into" in comments[0].body
    assert comments[0].run_id is None
    assert comments[0].source == "cli"


def test_decompose_audit_comment_keeps_run_id_for_matching_profile(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="root", triage=True)
        other = kb.create_task(conn, title="runner", assignee="forge")
        kb.claim_task(conn, other)
        run_id = kb._current_run_id(conn, other)
        decompose_triage_task(
            conn, tid, root_assignee=None, children=[{"title": "child"}],
            author="forge", run_id=run_id, source="run",
        )
        comments = kb.list_comments(conn, tid)
    assert len(comments) == 1
    assert comments[0].run_id == run_id
    assert comments[0].source == "run"


def test_insert_comment_direct_downgrade(kanban_home):
    """Unit-level proof on the raw sink itself: no caller can bypass the
    helper by calling ``_insert_comment`` straight."""
    import time

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t")
        with kb.write_txn(conn):
            kb._insert_comment(
                conn, tid, "forge", "raw sink probe", int(time.time()),
                run_id=999999, source="run",
            )
        comments = kb.list_comments(conn, tid)
    assert len(comments) == 1
    assert comments[0].run_id is None
    assert comments[0].source == "cli"
