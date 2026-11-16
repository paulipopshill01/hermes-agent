"""Alert-suppression-on-auto-heal (CWE-778): a cron job that fails and self-heals before its
failure alert is confirmed delivered must still produce one observable notification.

Root cause: ``close_incidents_for_recovered_job`` (cron/incidents.py:242) flips any ``detected``/
``alerted`` incident straight to ``resolved`` unconditionally — it never checks whether
``alerted_at`` was ever stamped. ``_mark_incident_alerted`` only stamps it after the ASYNC
delivery queue (``drain_delivery_queue``'s ``send()`` callback, scheduler.py:1369-1377) confirms
the original failure notice actually left the process. If the job's next run succeeds before that
confirmation lands, the incident resolves silently and the notification is lost for good —
independent of ``failure_deliver`` being configured correctly (Finding 1's sweep #50 scope).

Fix: ``_send_recovery_notice_for_unalerted_incidents`` (scheduler.py) runs synchronously in the
success-path callback, BEFORE ``_compose_run_delivery``'s call to
``close_incidents_for_recovered_job``, and sends one backstop notice (bypassing the async queue
entirely) for any incident still missing ``alerted_at`` — stamping it at that instant so the
incident is never silently lost, without needing a new reconciler/limbo-state mechanism.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron import incidents
import cron.scheduler as sched


def _point_db(monkeypatch, tmp_path):
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return incidents


def _job(**overrides):
    job = {"id": "job-1", "name": "j1", "deliver": "slack:alerts", "failure_deliver": None}
    job.update(overrides)
    return job


# ── Task 1: reproduce and pin down the exact race window ───────────────────


def test_unfixed_mechanism_still_silently_resolves_when_called_directly(monkeypatch, tmp_path):
    """Documents the underlying race the fix sits in front of: calling the LOW-LEVEL
    ``close_incidents_for_recovered_job`` directly (bypassing the new synchronous guard, exactly
    as the pre-fix code path always did) still reproduces resolved + alerted_at IS NULL from the
    exact call sequence in the plan's Baseline. This is the bug Task 2's integration guards
    against — not something Task 2 removes from the underlying function, which must stay
    unconditional for ``close_incidents_for_recovered_job``'s own direct callers (e.g.
    ``_resolve_incidents_for_recovered_job``) to keep working exactly as before Task 2 wired the
    new guard in front of it in ``_save_compose_deliver``."""
    inc = _point_db(monkeypatch, tmp_path)

    # 1. A failing run creates a `detected` incident.
    inc_id, is_new = inc.upsert_incident("job-1", "Provider timeout: read timed out")
    assert is_new is True
    assert inc.get_incident(inc_id)["state"] == "detected"
    assert inc.get_incident(inc_id)["alerted_at"] is None

    # 2. Before the delivery queue drains (mocked here as simply never happening — the race
    #    condition being reproduced), the NEXT run succeeds and resolves incidents directly.
    resolved_count = inc.close_incidents_for_recovered_job("job-1")

    # 3. Reproduces the exact defect: resolved with alerted_at still NULL.
    assert resolved_count == 1
    row = inc.get_incident(inc_id)
    assert row["state"] == "resolved"
    assert row["alerted_at"] is None, (
        "documents the alert-suppression-on-auto-heal race: the incident resolved before its "
        "alert was ever confirmed delivered"
    )


# ── Task 2: the new guard closes the gap ────────────────────────────────────


def test_recovery_notice_fires_and_stamps_alerted_before_resolve(monkeypatch, tmp_path):
    """The integrated fix: a never-alerted incident gets one synchronous recovery notice, stamped
    alerted BEFORE close_incidents_for_recovered_job runs, so it ends up resolved WITH alerted_at
    populated — the new guaranteed-alert behavior."""
    inc = _point_db(monkeypatch, tmp_path)
    job = _job(failure_deliver="slack:alerts")
    inc_id, _ = inc.upsert_incident("job-1", "Provider timeout: read timed out")
    assert inc.get_incident(inc_id)["alerted_at"] is None

    sent = []
    monkeypatch.setattr(
        sched, "_resolve_delivery_targets",
        lambda j, for_failure=False: [{"platform": "slack", "chat_id": "alerts"}],
    )
    monkeypatch.setattr(
        sched, "_deliver_result",
        lambda j, content, adapters=None, loop=None, for_failure=False: (
            sent.append(content) or None  # None == success, matching _deliver_result's contract
        ),
    )

    sched._send_recovery_notice_for_unalerted_incidents(job, adapters=None, loop=None)

    assert len(sent) == 1
    assert "job 1" not in sent[0]  # sanity: not a garbled template
    assert "recovered" in sent[0] and "never confirmed" in sent[0]
    assert "Provider timeout" in sent[0]
    row = inc.get_incident(inc_id)
    assert row["state"] == "alerted"
    assert row["alerted_at"] is not None

    # Now the existing unconditional resolve runs (as it always has) and the incident lands
    # resolved WITH alerted_at populated — the guaranteed-alert path.
    inc.close_incidents_for_recovered_job("job-1")
    row = inc.get_incident(inc_id)
    assert row["state"] == "resolved"
    assert row["alerted_at"] is not None


def test_already_alerted_incident_gets_no_second_notice(monkeypatch, tmp_path):
    """No double-alert regression: an incident whose alert already confirmed delivered must not
    get a second recovery notice just because it's still open when the job recovers."""
    inc = _point_db(monkeypatch, tmp_path)
    job = _job(failure_deliver="slack:alerts")
    inc_id, _ = inc.upsert_incident("job-1", "Provider timeout: read timed out")
    sched._mark_incident_alerted(inc_id)
    assert inc.get_incident(inc_id)["alerted_at"] is not None

    sent = []
    monkeypatch.setattr(
        sched, "_resolve_delivery_targets",
        lambda j, for_failure=False: [{"platform": "slack", "chat_id": "alerts"}],
    )
    monkeypatch.setattr(
        sched, "_deliver_result",
        lambda j, content, adapters=None, loop=None, for_failure=False: sent.append(content),
    )

    sched._send_recovery_notice_for_unalerted_incidents(job, adapters=None, loop=None)

    assert sent == [], "already-alerted incident must not get a second recovery notice"


def test_quiet_by_design_job_stays_silent(monkeypatch, tmp_path):
    """A job whose failure_deliver resolves to no live channel (local/absent — Finding 1's
    deliberate quiet-by-design jobs) must stay silent: no recovery notice, and the incident simply
    resolves via the existing unconditional path with alerted_at left NULL (unchanged behavior for
    those jobs, respecting the existing convention rather than overriding it)."""
    inc = _point_db(monkeypatch, tmp_path)
    job = _job(failure_deliver="local")
    inc_id, _ = inc.upsert_incident("job-1", "Provider timeout: read timed out")

    sent = []
    monkeypatch.setattr(
        sched, "_resolve_delivery_targets", lambda j, for_failure=False: [],  # 'local' -> no targets
    )
    monkeypatch.setattr(
        sched, "_deliver_result",
        lambda j, content, adapters=None, loop=None, for_failure=False: sent.append(content),
    )

    sched._send_recovery_notice_for_unalerted_incidents(job, adapters=None, loop=None)

    assert sent == [], "quiet-by-design job must not receive a recovery notice"
    assert inc.get_incident(inc_id)["alerted_at"] is None


def test_no_pending_incidents_is_a_cheap_noop(monkeypatch, tmp_path):
    """A healthy job with no open incidents must not even attempt delivery-target resolution."""
    _point_db(monkeypatch, tmp_path)
    job = _job(failure_deliver="slack:alerts")

    calls = []
    monkeypatch.setattr(
        sched, "_resolve_delivery_targets",
        lambda j, for_failure=False: calls.append(1) or [{"platform": "slack", "chat_id": "x"}],
    )

    sched._send_recovery_notice_for_unalerted_incidents(job, adapters=None, loop=None)

    assert calls == [], "no pending incidents: must return before resolving delivery targets"


def test_delivery_failure_is_non_fatal_and_leaves_alerted_at_null(monkeypatch, tmp_path):
    """A notification failure must never fail the job's own success status, and must not falsely
    stamp alerted_at — the incident stays eligible for a future recovery notice attempt."""
    inc = _point_db(monkeypatch, tmp_path)
    job = _job(failure_deliver="slack:alerts")
    inc_id, _ = inc.upsert_incident("job-1", "Provider timeout: read timed out")

    monkeypatch.setattr(
        sched, "_resolve_delivery_targets",
        lambda j, for_failure=False: [{"platform": "slack", "chat_id": "alerts"}],
    )
    monkeypatch.setattr(
        sched, "_deliver_result",
        lambda j, content, adapters=None, loop=None, for_failure=False: "slack 500",
    )

    sched._send_recovery_notice_for_unalerted_incidents(job, adapters=None, loop=None)  # must not raise

    row = inc.get_incident(inc_id)
    assert row["state"] == "detected"
    assert row["alerted_at"] is None


def test_incident_store_unavailable_does_not_raise(monkeypatch, tmp_path):
    """Store errors are swallowed at debug level — delivery/the job's own success is unaffected."""
    job = _job(failure_deliver="slack:alerts")

    def _boom(job_id):
        raise RuntimeError("db locked")

    monkeypatch.setattr("cron.incidents.list_unalerted_incidents_for_job", _boom)

    sched._send_recovery_notice_for_unalerted_incidents(job, adapters=None, loop=None)  # must not raise


# ── Task 3 support: list_unalerted_incidents_for_job itself ────────────────


def test_list_unalerted_incidents_for_job_scoping(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    detected_id, _ = inc.upsert_incident("job-1", "error A")
    alerted_id, _ = inc.upsert_incident("job-1", "error B")
    sched._mark_incident_alerted(alerted_id)
    other_job_id, _ = inc.upsert_incident("job-2", "error A")
    closed_id, _ = inc.upsert_incident("job-1", "error C")
    inc.ack_incident(closed_id)

    pending = inc.list_unalerted_incidents_for_job("job-1")

    ids = {row["id"] for row in pending}
    assert ids == {detected_id}, "only this job's un-alerted, non-closed incidents"
    assert alerted_id not in ids
    assert other_job_id not in ids
    assert closed_id not in ids
