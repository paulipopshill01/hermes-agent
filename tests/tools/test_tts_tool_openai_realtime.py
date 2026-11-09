"""Unit tests for the opt-in OpenAI Realtime TTS backend (``tts.openai.api: "realtime"``).

All tests are mocked/recorded against the connection protocol -- no live paid OpenAI call is
required or made. Covers the plan's Task 1 acceptance criteria 1-6
(``2026-11-09-openai-tts-realtime-migration.md``).
"""

import base64
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools import tts_tool
from tools.tts_tool_openai_realtime import _generate_openai_realtime_tts


def _event(type_: str, **kw):
    return SimpleNamespace(type=type_, **kw)


class _FakeConnectionCM:
    """Context manager standing in for ``client.realtime.connect(model=...)``."""

    def __init__(self, events=(), raise_on_enter=None):
        self._events = list(events)
        self.session = MagicMock()
        self.response = MagicMock()
        self.conversation = MagicMock()
        self.conversation.item = MagicMock()
        self._raise_on_enter = raise_on_enter

    def __enter__(self):
        if self._raise_on_enter is not None:
            raise self._raise_on_enter
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self._events)


def _fake_client(connection_cm, *, close_calls=None):
    client = MagicMock()
    client.realtime.connect.return_value = connection_cm
    if close_calls is not None:
        client.close.side_effect = lambda: close_calls.append(True)
    return client


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.fixture(autouse=True)
def _direct_credentials(monkeypatch):
    """Route every test through the direct (non-managed) credential path by default."""
    monkeypatch.setattr(
        "tools.tts_tool_openai_realtime._resolve_openai_audio_client_config",
        lambda: ("sk-test-key", "https://api.openai.com/v1", False))


class TestDefaultPathUnchanged:
    """Task 1 AC1: tts.openai.api unset/"speech" must not touch the Realtime backend at all."""

    def test_dispatcher_does_not_call_realtime_when_api_unset(self, monkeypatch):
        called = {"realtime": False, "speech": False}
        monkeypatch.setattr(tts_tool, "_generate_openai_realtime_tts",
                             lambda *a, **k: called.__setitem__("realtime", True))
        monkeypatch.setattr(tts_tool, "_generate_openai_tts",
                             lambda *a, **k: called.__setitem__("speech", True))
        tts_tool._synthesize_builtin("openai", "hi", "/tmp/x.mp3", {"openai": {}}, None)
        assert called == {"realtime": False, "speech": True}

    def test_dispatcher_does_not_call_realtime_when_api_is_speech(self, monkeypatch):
        called = {"realtime": False, "speech": False}
        monkeypatch.setattr(tts_tool, "_generate_openai_realtime_tts",
                             lambda *a, **k: called.__setitem__("realtime", True))
        monkeypatch.setattr(tts_tool, "_generate_openai_tts",
                             lambda *a, **k: called.__setitem__("speech", True))
        tts_tool._synthesize_builtin("openai", "hi", "/tmp/x.mp3", {"openai": {"api": "speech"}}, None)
        assert called == {"realtime": False, "speech": True}

    def test_dispatcher_calls_realtime_only_when_explicitly_selected(self, monkeypatch):
        called = {"realtime": False, "speech": False}
        monkeypatch.setattr(tts_tool, "_generate_openai_realtime_tts",
                             lambda *a, **k: called.__setitem__("realtime", True))
        monkeypatch.setattr(tts_tool, "_generate_openai_tts",
                             lambda *a, **k: called.__setitem__("speech", True))
        tts_tool._synthesize_builtin("openai", "hi", "/tmp/x.mp3", {"openai": {"api": "realtime"}}, None)
        assert called == {"realtime": True, "speech": False}


class TestRealtimeHappyPath:
    def test_short_text_produces_valid_audio_file(self, tmp_path):
        pcm = b"\x01\x02" * 2400  # arbitrary fixture PCM, not real speech
        events = [
            _event("response.output_audio.delta", delta=_b64(pcm[: len(pcm) // 2])),
            _event("response.output_audio.delta", delta=_b64(pcm[len(pcm) // 2 :])),
            _event("response.output_audio.done"),
            _event("response.done"),
        ]
        conn_cm = _FakeConnectionCM(events)
        client = _fake_client(conn_cm)
        output_path = str(tmp_path / "out.wav")

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            result = _generate_openai_realtime_tts(
                "Hello from a fixture", output_path, {"openai": {}}, voice="alloy")

        assert result == output_path
        data = open(output_path, "rb").read()
        assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
        assert len(data) > 44  # header + actual PCM payload, not an empty shell
        conn_cm.session.update.assert_called_once()
        session_kwargs = conn_cm.session.update.call_args.kwargs["session"]
        assert session_kwargs["tools"] == []
        assert session_kwargs["tool_choice"] == "none"
        assert session_kwargs["audio"]["output"]["voice"] == "alloy"

    def test_voice_and_speed_honored_in_session_config(self, tmp_path):
        events = [_event("response.output_audio.delta", delta=_b64(b"\x00\x00" * 100)),
                   _event("response.done")]
        conn_cm = _FakeConnectionCM(events)
        client = _fake_client(conn_cm)
        output_path = str(tmp_path / "out.wav")

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            _generate_openai_realtime_tts(
                "Hi", output_path, {"openai": {}}, voice="cedar", speed=1.25)

        session_kwargs = conn_cm.session.update.call_args.kwargs["session"]
        assert session_kwargs["audio"]["output"]["voice"] == "cedar"

    def test_long_text_sent_as_a_single_turn_no_chunking_needed(self, tmp_path):
        """Task 2 matrix: long text. The Realtime API has no documented small per-turn character
        cap (128K-token context) unlike some providers, so a single conversation.item.create call
        is sufficient -- this test proves the long input reaches that single call unmangled."""
        long_text = ("This is a long-form fixture sentence used only to exercise chunking "
                     "behaviour. " * 80)
        assert len(long_text) > 3000
        events = [_event("response.output_audio.delta", delta=_b64(b"\x00\x00" * 100)),
                   _event("response.done")]
        conn_cm = _FakeConnectionCM(events)
        client = _fake_client(conn_cm)

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            _generate_openai_realtime_tts(long_text, str(tmp_path / "out.wav"), {"openai": {}})

        assert conn_cm.conversation.item.create.call_count == 1
        item_kwargs = conn_cm.conversation.item.create.call_args.kwargs["item"]
        assert item_kwargs["content"][0]["text"] == long_text

    def test_special_characters_and_instructions_forwarded(self, tmp_path):
        """Task 2 matrix: punctuation/special characters, and the `instructions` parameter where
        the Realtime API exposes an equivalent (session-level `instructions`)."""
        text = "Café \u2014 na\u00efve \U0001F916 \"quoted\" & <tags> \u2764\ufe0f 100% \u00bd-done?"
        events = [_event("response.output_audio.delta", delta=_b64(b"\x00\x00" * 100)),
                   _event("response.done")]
        conn_cm = _FakeConnectionCM(events)
        client = _fake_client(conn_cm)

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            _generate_openai_realtime_tts(
                text, str(tmp_path / "out.wav"), {"openai": {}}, instructions="Speak warmly.")

        item_kwargs = conn_cm.conversation.item.create.call_args.kwargs["item"]
        assert item_kwargs["content"][0]["text"] == text
        session_kwargs = conn_cm.session.update.call_args.kwargs["session"]
        assert session_kwargs["instructions"] == "Speak warmly."


class TestManagedGatewayRejection:
    """Task 1 AC3: managed-gateway + realtime must raise explicitly, never silently coerce."""

    def test_managed_gateway_raises_without_sending_a_request(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "tools.tts_tool_openai_realtime._resolve_openai_audio_client_config",
            lambda: ("managed-token", "https://gateway.example/v1", True))
        connect_was_called = {"value": False}

        def _poison_client(**kw):
            connect_was_called["value"] = True
            raise AssertionError("must never construct a client for a rejected managed request")

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = _poison_client
            with pytest.raises(ValueError, match="managed Nous audio gateway"):
                _generate_openai_realtime_tts("Hi", str(tmp_path / "out.wav"), {"openai": {}})
        assert connect_was_called["value"] is False
        assert not (tmp_path / "out.wav").exists()


class TestTimeoutAndCancellation:
    """Task 1 AC4: a session exceeding its timeout is cancelled cleanly, no partial file."""

    def test_timeout_raises_value_error_and_leaves_no_file(self, tmp_path):
        def _never_arrives():
            time.sleep(0.2)
            return
            yield  # pragma: no cover - unreachable, marks this a generator

        conn_cm = _FakeConnectionCM(_never_arrives())
        client = _fake_client(conn_cm)
        output_path = str(tmp_path / "out.wav")

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            with pytest.raises(ValueError, match="timed out"):
                _generate_openai_realtime_tts(
                    "Hi", output_path, {"openai": {"realtime_timeout_s": 0.05}})

        assert not (tmp_path / "out.wav").exists()

    def test_error_event_raises_value_error_no_file(self, tmp_path):
        events = [_event("error", error=SimpleNamespace(message="rate limited"))]
        conn_cm = _FakeConnectionCM(events)
        client = _fake_client(conn_cm)
        output_path = str(tmp_path / "out.wav")

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            with pytest.raises(ValueError, match="rate limited"):
                _generate_openai_realtime_tts("Hi", output_path, {"openai": {}})
        assert not (tmp_path / "out.wav").exists()

    def test_connect_failure_raises_value_error_not_leaked_raw(self, tmp_path):
        conn_cm = _FakeConnectionCM(raise_on_enter=RuntimeError("401 Unauthorized: sk-leaked-xyz"))
        client = _fake_client(conn_cm)
        output_path = str(tmp_path / "out.wav")

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            with pytest.raises(ValueError):
                _generate_openai_realtime_tts("Hi", output_path, {"openai": {}})
        assert not (tmp_path / "out.wav").exists()


class TestNoSensitiveLogging:
    """Task 1 AC5: only model/voice/selector fields are logged, never credentials/audio/transcript."""

    def test_no_api_key_or_transcript_in_log_output(self, tmp_path, caplog):
        events = [_event("response.output_audio.delta", delta=_b64(b"\x00\x00" * 50)),
                   _event("response.done")]
        conn_cm = _FakeConnectionCM(events)
        client = _fake_client(conn_cm)
        output_path = str(tmp_path / "out.wav")
        secret_key = "sk-realtest-should-never-be-logged-99887766"
        secret_text = "the quiet fox jumps over the lazy reindeer at midnight"

        with caplog.at_level("DEBUG"), patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            _generate_openai_realtime_tts(
                secret_text, output_path, {"openai": {}}, api_key=secret_key)

        log_text = caplog.text
        assert secret_key not in log_text
        assert secret_text not in log_text


class TestCredentialChainReused:
    """No new credential variable invented: direct chain + explicit api_key override both work."""

    def test_explicit_api_key_bypasses_resolver(self, tmp_path, monkeypatch):
        resolver_called = {"value": False}
        monkeypatch.setattr(
            "tools.tts_tool_openai_realtime._resolve_openai_audio_client_config",
            lambda: resolver_called.__setitem__("value", True) or ("x", "y", False))
        events = [_event("response.output_audio.delta", delta=_b64(b"\x00\x00" * 50)),
                   _event("response.done")]
        conn_cm = _FakeConnectionCM(events)
        client = _fake_client(conn_cm)

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            _generate_openai_realtime_tts(
                "Hi", str(tmp_path / "out.wav"), {"openai": {}}, api_key="explicit-key")

        assert resolver_called["value"] is False


class TestManagedGuardPredicateBroadened:
    """F-2 (CWE-693): the rejection must fire on any of the three adjacent shapes, not just the
    resolver's ``is_managed`` flag -- and must still raise before any client is constructed."""

    def test_explicit_api_key_does_not_bypass_nous_selection(self, tmp_path, monkeypatch):
        """An explicit api_key kwarg used to skip resolution (and so the old is_managed-only
        guard) entirely. The persisted "tts" selection being the managed row must still reject."""
        monkeypatch.setattr(
            "tools.tts_tool_openai_realtime.read_selection", lambda section: "nous")
        connect_was_called = {"value": False}

        def _poison_client(**kw):
            connect_was_called["value"] = True
            raise AssertionError("must never construct a client for a rejected managed request")

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = _poison_client
            with pytest.raises(ValueError, match="managed Nous audio gateway"):
                _generate_openai_realtime_tts(
                    "Hi", str(tmp_path / "out.wav"), {"openai": {}}, api_key="explicit-key")
        assert connect_was_called["value"] is False

    def test_direct_key_at_gateway_origin_base_url_rejected(self, tmp_path, monkeypatch):
        """A direct API key + ``tts.openai.base_url`` pointed at the managed gateway's own origin
        resolves as non-managed (resolver never consulted) but must still be rejected: the
        gateway's speech route silently coerces unsupported models, and v1/realtime WS support
        through it is unproven."""
        from tools.managed_tool_gateway import ManagedToolGatewayConfig

        monkeypatch.setattr(
            "tools.tts_tool_openai_realtime.resolve_managed_tool_gateway",
            lambda vendor: ManagedToolGatewayConfig(
                vendor=vendor, gateway_origin="https://openai-audio-gateway.nousresearch.com",
                nous_user_token="nous-token", managed_mode=True))
        connect_was_called = {"value": False}

        def _poison_client(**kw):
            connect_was_called["value"] = True
            raise AssertionError("must never construct a client for a rejected managed request")

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = _poison_client
            with pytest.raises(ValueError, match="managed Nous audio gateway"):
                _generate_openai_realtime_tts(
                    "Hi", str(tmp_path / "out.wav"),
                    {"openai": {"base_url": "https://openai-audio-gateway.nousresearch.com/v1"}},
                    api_key="sk-direct-key-not-managed")
        assert connect_was_called["value"] is False

    def test_direct_key_at_unrelated_base_url_still_allowed(self, tmp_path, monkeypatch):
        """Negative control: a direct key against a base_url that is NOT the managed gateway's
        origin must still go through -- the broadened predicate must not over-reject."""
        from tools.managed_tool_gateway import ManagedToolGatewayConfig

        monkeypatch.setattr(
            "tools.tts_tool_openai_realtime.resolve_managed_tool_gateway",
            lambda vendor: ManagedToolGatewayConfig(
                vendor=vendor, gateway_origin="https://openai-audio-gateway.nousresearch.com",
                nous_user_token="nous-token", managed_mode=True))
        monkeypatch.setattr("tools.tts_tool_openai_realtime.read_selection", lambda section: None)
        events = [_event("response.output_audio.delta", delta=_b64(b"\x00\x00" * 50)),
                   _event("response.done")]
        conn_cm = _FakeConnectionCM(events)
        client = _fake_client(conn_cm)

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            result = _generate_openai_realtime_tts(
                "Hi", str(tmp_path / "out.wav"),
                {"openai": {"base_url": "https://api.openai.com/v1"}},
                api_key="sk-direct-key-not-managed")
        assert result == str(tmp_path / "out.wav")


class TestExceptionTextScrubbed:
    """F-1 (CWE-209/CWE-532): secret-shaped substrings and the resolved key's literal value must
    never survive into a raised ValueError's text, and the embedded text is capped at ~200 chars.
    Distinct from ``test_connect_failure_raises_value_error_not_leaked_raw`` above, whose fixture
    is already pre-redacted -- these fixtures carry a live secret-shaped string in ``str(exc)``."""

    def test_connect_failure_scrubs_secret_shaped_and_resolved_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "tools.tts_tool_openai_realtime._resolve_openai_audio_client_config",
            lambda: ("sk-resolved-secret-0123456789", "https://api.openai.com/v1", False))
        live_key = "sk-resolved-secret-0123456789"
        conn_cm = _FakeConnectionCM(raise_on_enter=RuntimeError(f"401 Unauthorized: key={live_key}"))
        client = _fake_client(conn_cm)
        output_path = str(tmp_path / "out.wav")

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            with pytest.raises(ValueError) as excinfo:
                _generate_openai_realtime_tts("Hi", output_path, {"openai": {}})

        message = str(excinfo.value)
        assert live_key not in message
        assert "sk-resolved-secret-0123456789" not in message
        assert "***REDACTED***" in message

    def test_pump_exception_scrubs_secret_shaped_text(self, tmp_path):
        class _BoomIterable:
            def __iter__(self):
                raise RuntimeError("socket closed, auth header was Bearer sk-abcdefghijklmnop")

        conn_cm = _FakeConnectionCM()
        conn_cm._events = _BoomIterable()  # bypass __init__'s list(events) eager consumption
        client = _fake_client(conn_cm)
        output_path = str(tmp_path / "out.wav")

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            with pytest.raises(ValueError) as excinfo:
                _generate_openai_realtime_tts("Hi", output_path, {"openai": {}})

        message = str(excinfo.value)
        assert "sk-abcdefghijklmnop" not in message
        assert "***REDACTED***" in message

    def test_embedded_exception_text_capped_at_200_chars(self, tmp_path):
        huge = "x" * 5000
        conn_cm = _FakeConnectionCM(raise_on_enter=RuntimeError(huge))
        client = _fake_client(conn_cm)
        output_path = str(tmp_path / "out.wav")

        with patch("tools.tts_tool_openai_realtime._origin") as origin:
            origin.return_value._import_openai_client.return_value = lambda **kw: client
            with pytest.raises(ValueError) as excinfo:
                _generate_openai_realtime_tts("Hi", output_path, {"openai": {}})

        assert len(str(excinfo.value)) < 300  # capped text + the fixed wrapper prefix
