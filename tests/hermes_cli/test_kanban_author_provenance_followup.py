"""Residual author/attribution surfaces (R-2/R-3, security review t_bd48901c).

Non-gating follow-up to F-1/F-3 (49166fbda2, security review t_f469a059):
``--author``/``--created-by`` free text on ``specify``/``decompose``/``create``/``swarm``
reached ``task_comments`` (triage audit notes, blackboard updates) with the same
authorship-forgery exposure comment/attach already closed. Same treatment here:
drop the free-text override, always use the caller's resolved profile identity,
and thread real run_id/source provenance through the audit comment this writes.
"""

from __future__ import annotations

import argparse
import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban as kanban_cli
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


def _run_cli(*argv: str) -> int:
    root = argparse.ArgumentParser()
    subp = root.add_subparsers(dest="cmd")
    kanban_cli.build_parser(subp)
    ns = root.parse_args(["kanban", *argv])
    return kanban_cli.kanban_command(ns)


def _patch_aux_client(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return patch("agent.auxiliary_client.call_llm", MagicMock(return_value=resp))


# ---------------------------------------------------------------------------
# R-2: --author is gone from specify/decompose
# ---------------------------------------------------------------------------

def test_cli_specify_rejects_author_flag(kanban_home):
    with pytest.raises(SystemExit):
        _run_cli("specify", "t_doesnotmatter", "--author", "hermes-system")


def test_cli_decompose_rejects_author_flag(kanban_home):
    with pytest.raises(SystemExit):
        _run_cli("decompose", "t_doesnotmatter", "--author", "hermes-system")


def test_cli_specify_audit_comment_uses_resolved_profile_identity(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    content = jsonlib.dumps({"title": "Refined", "body": "**Goal**\nbody"})
    with _patch_aux_client(content):
        rc = _run_cli("specify", tid)
    assert rc == 0
    with kbc.connect_closing() as conn:
        comments = kb.list_comments(conn, tid)
    assert len(comments) == 1
    assert comments[0].author == "forge"
    assert comments[0].run_id is None
    assert comments[0].source == "cli"


def test_cli_decompose_audit_comment_uses_resolved_profile_identity(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    content = jsonlib.dumps({"title": "Refined", "body": "body", "fanout": False})
    with _patch_aux_client(content):
        rc = _run_cli("decompose", tid)
    assert rc == 0
    with kbc.connect_closing() as conn:
        comments = kb.list_comments(conn, tid)
    assert len(comments) == 1
    assert comments[0].author == "forge"
    assert comments[0].source == "cli"


# ---------------------------------------------------------------------------
# R-3: --created-by is gone from create/swarm
# ---------------------------------------------------------------------------

def test_cli_create_rejects_created_by_flag(kanban_home):
    with pytest.raises(SystemExit):
        _run_cli("create", "t", "--created-by", "hermes-system")


def test_cli_swarm_rejects_created_by_flag(kanban_home):
    with pytest.raises(SystemExit):
        _run_cli(
            "swarm", "goal here", "--worker", "researcher:do thing",
            "--verifier", "v", "--synthesizer", "s", "--created-by", "hermes-system",
        )


def test_cli_create_uses_resolved_profile_identity(kanban_home, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    rc = _run_cli("create", "t", "--json")
    assert rc == 0
    task = jsonlib.loads(capsys.readouterr().out)
    assert task["created_by"] == "forge"


def test_cli_swarm_blackboard_update_uses_resolved_profile_identity(kanban_home, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    rc = _run_cli(
        "swarm", "goal here", "--worker", "researcher:do thing",
        "--verifier", "v", "--synthesizer", "s", "--json",
    )
    assert rc == 0
    created = jsonlib.loads(capsys.readouterr().out)
    with kbc.connect_closing() as conn:
        comments = kb.list_comments(conn, created["root_id"])
    # post_blackboard_update's topology comment — author anchored to the
    # resolved CLI identity, never a free-text --created-by override.
    assert any(c.author == "forge" for c in comments)
    assert all(c.author != "hermes-system" for c in comments)
