"""R-01: exact-match fleet-secret redaction boundary (Gate 1 plan,
/root/.hermes/plans/2026-11-01-r01-secret-redaction-boundary.md).

Covers the plan's Task 2-7 acceptance criteria plus Stella's Gate-1 amendments:
  (amendment 1) Task 6 override-path redaction test
  (amendment 2) Task 2(b) negative test against Hermes's own debug/trace logging
"""
import json
import logging
import os
import sqlite3

import pytest

from agent import redact
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


SENTINEL_A = "FLEETSECRETTESTVALUEAAA1234567890abcdef"
SENTINEL_B = "FLEETSECRETTESTVALUEBBB9876543210fedcba"


@pytest.fixture(autouse=True)
def _clean_registry():
    """Fleet-secret registry is module-level + profile-scoped: always start clean and always
    clean up, regardless of which scope a test used."""
    redact._FLEET_SECRET_VALUES.clear()
    redact._FLEET_SECRET_SEEDED_SCOPES.clear()
    yield
    redact._FLEET_SECRET_VALUES.clear()
    redact._FLEET_SECRET_SEEDED_SCOPES.clear()


@pytest.fixture
def scoped_home(tmp_path):
    """A throwaway Hermes-home scope: context-local override, never touches os.environ."""
    token = set_hermes_home_override(str(tmp_path))
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


class TestTask2Registry:
    def test_seed_from_fixture_dotenv_never_touches_os_environ_or_proc_environ(self, scoped_home, monkeypatch):
        (scoped_home / ".env").write_text(f"MY_FLEET_SECRET={SENTINEL_A}\nSHORT=ab\n")
        # Poison os.environ with a DIFFERENT value for the same name: if seeding ever fell back
        # to os.environ, this sentinel (never written to the fixture file) would get registered.
        poison = "POISONED_FROM_OS_ENVIRON_NOT_DOTENV_9999999999"
        monkeypatch.setenv("MY_FLEET_SECRET", poison)
        # Also assert the registration path never shells out to read /proc/*/environ.
        import subprocess
        orig_run = subprocess.run
        def _guard_run(*a, **kw):
            cmd = a[0] if a else kw.get("args")
            if isinstance(cmd, (list, tuple)) and any("/proc" in str(c) and "environ" in str(c) for c in cmd):
                raise AssertionError("seeding must never read /proc/*/environ")
            return orig_run(*a, **kw)
        monkeypatch.setattr(subprocess, "run", _guard_run)

        count = redact.seed_fleet_secrets_from_dotenv(force=True)
        assert count >= 1
        assert poison not in redact._FLEET_SECRET_VALUES.get(str(scoped_home), {})
        assert SENTINEL_A in redact._FLEET_SECRET_VALUES.get(str(scoped_home), {})

    def test_registry_contents_never_logged_or_persisted(self, scoped_home, caplog):
        """Amendment 2: explicit negative test against Hermes's own debug/trace logging, with
        debug logging enabled at runtime, not just a DB check."""
        (scoped_home / ".env").write_text(f"MY_FLEET_SECRET={SENTINEL_B}\n")
        with caplog.at_level(logging.DEBUG):
            redact.seed_fleet_secrets_from_dotenv(force=True)
            redact.register_fleet_secret_value(SENTINEL_B)
            redact.redact_registered_fleet_secrets(f"leaked: {SENTINEL_B}")
        for record in caplog.records:
            assert SENTINEL_B not in record.getMessage(), (
                f"fleet secret value reached a log record: {record.name} {record.getMessage()!r}")
        # Also grep the actual registration/seeding source for any logger.* call that could
        # receive the raw value (static confirmation to pair with the runtime assertion above).
        import inspect
        src = inspect.getsource(redact.register_fleet_secret_value) + inspect.getsource(redact.seed_fleet_secrets_from_dotenv)
        for line in src.splitlines():
            if "logger." in line or "logging." in line:
                assert "value" not in line and "values" not in line, (
                    f"a logger call in the registration/seeding path references a value variable: {line!r}")

    def test_profile_scoped_isolation_no_cross_profile_leakage(self, tmp_path):
        home_a = tmp_path / "profile_a"
        home_b = tmp_path / "profile_b"
        home_a.mkdir(); home_b.mkdir()
        (home_a / ".env").write_text(f"SECRET_A={SENTINEL_A}\n")
        (home_b / ".env").write_text(f"SECRET_B={SENTINEL_B}\n")

        token_a = set_hermes_home_override(str(home_a))
        try:
            redact.seed_fleet_secrets_from_dotenv(force=True)
        finally:
            reset_hermes_home_override(token_a)

        token_b = set_hermes_home_override(str(home_b))
        try:
            redact.seed_fleet_secrets_from_dotenv(force=True)
            # Profile B's registry must not see profile A's secret.
            assert redact.redact_registered_fleet_secrets(SENTINEL_A) == SENTINEL_A
            assert redact.redact_registered_fleet_secrets(SENTINEL_B) != SENTINEL_B
        finally:
            reset_hermes_home_override(token_b)

        token_a = set_hermes_home_override(str(home_a))
        try:
            assert redact.redact_registered_fleet_secrets(SENTINEL_B) == SENTINEL_B
            assert redact.redact_registered_fleet_secrets(SENTINEL_A) != SENTINEL_A
        finally:
            reset_hermes_home_override(token_a)

    def test_root_layer_scrubbed_under_profile_scope_f1(self, tmp_path):
        """F-1: a root-home .env value must be registered (and scrubbed) under a PROFILE scope too
        — not just under the root scope itself — while profile-own values stay isolated."""
        root_home = tmp_path / "root_home"
        profiles_dir = root_home / "profiles"
        profile_a = profiles_dir / "profile_a"
        profile_b = profiles_dir / "profile_b"
        profile_a.mkdir(parents=True)
        profile_b.mkdir(parents=True)
        root_secret = "ROOTLAYERSECRETVALUEZZZ1928374655aabbccdd"
        (root_home / ".env").write_text(f"FLEET_WIDE_ROOT_SECRET={root_secret}\n")
        profile_a_secret = "PROFILEAOWNSECRETVALUE0987654321deadbeef"
        profile_b_secret = "PROFILEBOWNSECRETVALUE1122334455feedface"
        (profile_a / ".env").write_text(f"PROFILE_A_SECRET={profile_a_secret}\n")
        (profile_b / ".env").write_text(f"PROFILE_B_SECRET={profile_b_secret}\n")

        token_a = set_hermes_home_override(str(profile_a))
        try:
            redact.seed_fleet_secrets_from_dotenv(force=True)
            # Root-layer value IS scrubbed under profile_a's scope (the F-1 fix).
            assert redact.redact_registered_fleet_secrets(root_secret) != root_secret
            # Profile_a's own value is registered too.
            assert redact.redact_registered_fleet_secrets(profile_a_secret) != profile_a_secret
            # Profile_b's own value must NOT leak into profile_a's registry.
            assert redact.redact_registered_fleet_secrets(profile_b_secret) == profile_b_secret
        finally:
            reset_hermes_home_override(token_a)

        token_b = set_hermes_home_override(str(profile_b))
        try:
            redact.seed_fleet_secrets_from_dotenv(force=True)
            # Root-layer value is ALSO scrubbed under profile_b's independent scope.
            assert redact.redact_registered_fleet_secrets(root_secret) != root_secret
            assert redact.redact_registered_fleet_secrets(profile_b_secret) != profile_b_secret
            # Profile_a's own value must NOT leak into profile_b's registry.
            assert redact.redact_registered_fleet_secrets(profile_a_secret) == profile_a_secret
        finally:
            reset_hermes_home_override(token_b)

    def test_profile_value_wins_root_on_same_key_f1(self, tmp_path):
        """A same-named key set in both layers: the profile's own value must still be the one
        registered as THAT key's value (profile layer applied on top of root)."""
        root_home = tmp_path / "root_home2"
        profiles_dir = root_home / "profiles"
        profile_a = profiles_dir / "profile_a"
        profile_a.mkdir(parents=True)
        root_value = "ROOTDEFAULTVALUEFORSHAREDKEYAA00112233"
        profile_value = "PROFILEOVERRIDEVALUEFORSHAREDKEYBB99887766"
        (root_home / ".env").write_text(f"SHARED_KEY={root_value}\n")
        (profile_a / ".env").write_text(f"SHARED_KEY={profile_value}\n")

        token = set_hermes_home_override(str(profile_a))
        try:
            redact.seed_fleet_secrets_from_dotenv(force=True)
            assert redact.redact_registered_fleet_secrets(profile_value) != profile_value
        finally:
            reset_hermes_home_override(token)

    def test_same_key_different_value_both_scrubbed_f1(self, tmp_path):
        """Stella Gate 2 spot-check (t_f2a00395): root and a profile define the SAME key name with
        a GENUINELY DIFFERENT value (e.g. each agent's own distinct BUZZ_PRIVATE_KEY shadowing the
        root fleet-wide one). A plain key-keyed dict.update() merge lets the profile's value evict
        root's from the registration set entirely, so root's live secret is never scrubbed under
        that profile's scope — exactly the real-world collision that slipped past the two separate
        single-purpose fixtures above. Both values must be independently registered and scrubbed."""
        root_home = tmp_path / "root_home3"
        profiles_dir = root_home / "profiles"
        profile_a = profiles_dir / "profile_a"
        profile_a.mkdir(parents=True)
        root_value = "ROOTSHAREDKEYDIFFERENTVALUEzz9988776655"
        profile_value = "PROFILESHAREDKEYDIFFERENTVALUEyy1122334455"
        (root_home / ".env").write_text(f"BUZZ_PRIVATE_KEY={root_value}\n")
        (profile_a / ".env").write_text(f"BUZZ_PRIVATE_KEY={profile_value}\n")

        token = set_hermes_home_override(str(profile_a))
        try:
            redact.seed_fleet_secrets_from_dotenv(force=True)
            # Both the root-layer value AND the profile-layer value under the SAME key name must
            # be scrubbed under this profile's scope -- neither evicts the other.
            assert redact.redact_registered_fleet_secrets(root_value) != root_value
            assert redact.redact_registered_fleet_secrets(profile_value) != profile_value
        finally:
            reset_hermes_home_override(token)


class TestTask2RegistrationFloorF3:
    def test_short_or_low_entropy_values_not_registered(self, scoped_home):
        """F-3: a <12-char value and a low-entropy value must NOT be registered even though they
        would have passed the old >=8-char-only floor; a genuine long, high-entropy secret must
        still be registered."""
        genuine_secret = "Tr0ub4dor&3xtraRandomSuffix99!"  # len 31, high entropy
        short_value = "shortval12"  # len 10 < 12
        low_entropy_value = "aaaaaaaaaaaaaaaaaaaa"  # len 20, entropy ~0
        (scoped_home / ".env").write_text(
            f"GENUINE_SECRET={genuine_secret}\n"
            f"SHORT_CONFIG={short_value}\n"
            f"LOW_ENTROPY_CONFIG={low_entropy_value}\n"
        )
        redact.seed_fleet_secrets_from_dotenv(force=True)
        bucket = redact._FLEET_SECRET_VALUES.get(str(scoped_home), {})
        assert genuine_secret in bucket
        assert short_value not in bucket
        assert low_entropy_value not in bucket

    def test_benign_config_word_not_clipped_f3(self, scoped_home):
        """Direct regression for the materialized FP: an 8-char low-entropy config value used to
        get registered under the old floor and clip a legitimate word containing it as a
        substring. Under the new floor it must not even be registered."""
        low_entropy_8 = "disabled"  # len 8, low entropy common word
        (scoped_home / ".env").write_text(f"SOME_MODE={low_entropy_8}\n")
        redact.seed_fleet_secrets_from_dotenv(force=True)
        out = redact.redact_registered_fleet_secrets("service is disabled now")
        assert out == "service is disabled now"


class TestTask3BoundaryOnePerTool:
    def test_marker_format_distinct_from_vault_and_shape_masks(self, scoped_home):
        redact.register_fleet_secret_value(SENTINEL_A)
        out = redact.redact_registered_fleet_secrets(f"value is {SENTINEL_A} here")
        assert SENTINEL_A not in out
        assert out.startswith("value is [REDACTED:") or "[REDACTED:" in out
        assert "«redacted-vault-secret»" not in out

    def test_read_file_tool_redacts_registered_fleet_secret(self, scoped_home, tmp_path):
        redact.register_fleet_secret_value(SENTINEL_A)
        target = tmp_path / "plain.txt"
        target.write_text(f"line one\nsecret={SENTINEL_A}\nline three\n")
        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool(str(target)))
        assert SENTINEL_A not in result.get("content", "")
        assert "line one" in result.get("content", "")  # non-secret content unaffected

    def test_search_files_tool_redacts_registered_fleet_secret(self, scoped_home, tmp_path):
        redact.register_fleet_secret_value(SENTINEL_A)
        target = tmp_path / "plain.txt"
        target.write_text(f"secret={SENTINEL_A}\nbenign=unrelated-value\n")
        from tools.file_tools import search_tool
        result = json.loads(search_tool(pattern="secret|benign", target="content", path=str(tmp_path)))
        blob = json.dumps(result)
        assert SENTINEL_A not in blob
        assert "unrelated-value" in blob  # non-secret content unaffected

    def test_patch_tool_diff_redacts_registered_fleet_secret(self, scoped_home, tmp_path):
        redact.register_fleet_secret_value(SENTINEL_A)
        target = tmp_path / "f.txt"
        target.write_text(f"old secret={SENTINEL_A}\n")
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="replace", path=str(target),
                                       old_string=f"old secret={SENTINEL_A}", new_string="new secret=rotated"))
        assert SENTINEL_A not in json.dumps(result)

    def test_skill_view_redacts_registered_fleet_secret(self, scoped_home, tmp_path, monkeypatch):
        redact.register_fleet_secret_value(SENTINEL_A)
        skills_dir = tmp_path / "skills" / "demo"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            f"---\nname: demo\ndescription: test\n---\nSecret in fixture: {SENTINEL_A}\n")
        import tools.skills_tool as st
        monkeypatch.setattr(st, "skill_search_dirs", lambda: ([("profile", tmp_path / "skills")], tmp_path / "skills"))
        result = st.skill_view("demo")
        assert SENTINEL_A not in result


class TestTask4BoundaryTwoPersistence:
    def _make_db(self, tmp_path):
        import uuid
        from hermes_state import SessionDB
        db_path = tmp_path / "state_for_test.db"
        db = SessionDB(db_path=db_path)
        session_id = str(uuid.uuid4())
        db.create_session(session_id, "test")
        return db, db_path, session_id

    @pytest.mark.parametrize("field", ["content", "reasoning", "reasoning_content", "reasoning_details", "api_content"])
    def test_each_field_scrubbed_from_row_and_fts(self, scoped_home, tmp_path, field):
        redact.register_fleet_secret_value(SENTINEL_A)
        db, db_path, session_id = self._make_db(tmp_path)
        kwargs = {"session_id": session_id, "role": "assistant", "content": "hello"}
        kwargs[field] = f"carries secret {SENTINEL_A} inline" if field != "content" else f"content carries {SENTINEL_A}"
        db.append_message(**kwargs)

        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT content, reasoning, reasoning_content, reasoning_details, api_content "
                "FROM messages WHERE session_id = ?", (session_id,)).fetchone()
            assert row is not None
            for value in row:
                if value:
                    assert SENTINEL_A not in str(value), f"{field}: raw secret found in messages row"
            # FTS mirrors inherit via SQL triggers off this same row.
            for fts_table in ("messages_fts", "messages_fts_trigram"):
                try:
                    hits = conn.execute(
                        f"SELECT COUNT(*) FROM {fts_table} WHERE {fts_table} MATCH ?",
                        (SENTINEL_A,)).fetchone()[0]
                except sqlite3.OperationalError:
                    continue  # table/trigram tokenizer not present on this build; skip rather than false-fail
                assert hits == 0, f"{fts_table}: FTS mirror matched the raw secret"
        finally:
            conn.close()


class TestTask5SideChannels:
    def test_hook_output_spill_redacts_before_writing_to_disk(self, scoped_home, tmp_path):
        redact.register_fleet_secret_value(SENTINEL_A)
        from tools.hook_output_spill import spill_if_oversized
        big_text = (f"leaked secret {SENTINEL_A}\n" * 2000)
        result = spill_if_oversized(big_text, session_id="r01-test", source="hook",
                                    config={"enabled": True, "max_chars": 100,
                                            "preview_head": 50, "preview_tail": 50,
                                            "directory": str(tmp_path / "hook_outputs")})
        assert SENTINEL_A not in result
        for f in (tmp_path / "hook_outputs").rglob("*.txt"):
            assert SENTINEL_A not in f.read_text()


class TestTask6DenyByDefaultGlobs:
    def test_env_and_backup_globs_default_to_metadata_only(self, tmp_path):
        from agent.file_safety import secret_bearing_glob_reason
        assert secret_bearing_glob_reason(str(tmp_path / "config.yaml.backup")) is not None
        assert secret_bearing_glob_reason(str(tmp_path / "docker-compose.prod.yml")) is not None
        assert secret_bearing_glob_reason(str(tmp_path / "state.db")) is not None
        assert secret_bearing_glob_reason(str(tmp_path / "request_dump_abc_20261101.json")) is not None
        assert secret_bearing_glob_reason(str(tmp_path / "normal_source_file.py")) is None

    def test_env_star_glob_covers_noncanonical_variants_f2(self, tmp_path):
        """F-2: plan's .env* glob was omitted; non-canonical .env variants must now be caught."""
        from agent.file_safety import secret_bearing_glob_reason
        for name in (
            ".env.old", ".env.save", ".env.dist", ".env.prod",
            ".env.production.local", "app.env", "secrets.env",
        ):
            assert secret_bearing_glob_reason(str(tmp_path / name)) is not None, name

    def test_env_example_and_template_stay_fully_readable_f2(self, tmp_path):
        """Documented sanctioned substitute (the hard-block message itself names .env.example)
        must stay exempt from the new .env* glob — otherwise its one legitimate purpose breaks."""
        from agent.file_safety import secret_bearing_glob_reason
        assert secret_bearing_glob_reason(str(tmp_path / ".env.example")) is None
        assert secret_bearing_glob_reason(str(tmp_path / ".env.template")) is None

    def test_read_file_default_metadata_only_for_glob_match(self, tmp_path):
        target = tmp_path / "app.backup"
        target.write_text("SOME_SECRET=do-not-leak-me\n")
        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool(str(target)))
        assert result.get("metadata_only") is True
        assert "do-not-leak-me" not in json.dumps(result)

    def test_read_file_explicit_override_still_passes_through_boundary_one(self, scoped_home, tmp_path):
        """Amendment 1: full-content override against a glob-matched path must still complete AND
        demonstrably pass through Boundary 1 — plant a registered secret, confirm
        [REDACTED:<fp>] comes back, not raw content."""
        redact.register_fleet_secret_value(SENTINEL_A)
        target = tmp_path / "app.backup"
        target.write_text(f"SOME_SECRET={SENTINEL_A}\n")
        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool(str(target), full_content_override=True))
        assert result.get("metadata_only") is not True
        assert SENTINEL_A not in result.get("content", "")
        assert "[REDACTED:" in result.get("content", "")

    def test_search_files_explicit_override_still_passes_through_boundary_one(self, scoped_home, tmp_path):
        redact.register_fleet_secret_value(SENTINEL_A)
        target = tmp_path / "app.backup"
        target.write_text(f"SOME_SECRET={SENTINEL_A}\n")
        from tools.file_tools import search_tool
        blocked = json.loads(search_tool(pattern="SOME_SECRET", target="content", path=str(tmp_path)))
        assert "metadata only" in json.dumps(blocked)
        assert SENTINEL_A not in json.dumps(blocked)
        allowed = json.loads(search_tool(pattern="SOME_SECRET", target="content", path=str(tmp_path),
                                         full_content_override=True))
        assert SENTINEL_A not in json.dumps(allowed)
        assert "[REDACTED:" in json.dumps(allowed)


class TestTask7RotationReseed:
    def test_rotation_then_reseed_cycle_no_restart(self, scoped_home):
        env_path = scoped_home / ".env"
        env_path.write_text(f"ROTATING_SECRET={SENTINEL_A}\n")
        redact.seed_fleet_secrets_from_dotenv(force=True)
        assert redact.redact_registered_fleet_secrets(SENTINEL_A) != SENTINEL_A

        # Rotate: new value on disk.
        env_path.write_text(f"ROTATING_SECRET={SENTINEL_B}\n")
        redact.reseed_fleet_secrets()

        # New value protected without a restart.
        assert redact.redact_registered_fleet_secrets(SENTINEL_B) != SENTINEL_B
        # Old value evicted immediately (Forge's documented design choice: no retention window).
        assert redact.redact_registered_fleet_secrets(SENTINEL_A) == SENTINEL_A


class TestNoOverRedactionRegression:
    def test_unrelated_content_in_same_fixture_unaffected(self, scoped_home, tmp_path):
        redact.register_fleet_secret_value(SENTINEL_A)
        target = tmp_path / "mixed.txt"
        target.write_text(f"secret={SENTINEL_A}\nbenign line unrelated to secrets\n42\n")
        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool(str(target)))
        content = result.get("content", "")
        assert SENTINEL_A not in content
        assert "benign line unrelated to secrets" in content
        assert "42" in content
