"""OpenAI Realtime-based TTS backend for ``tools.tts_tool`` (opt-in, ``tts.openai.api: "realtime"``).

Sibling to ``tts_tool_openai.py`` (kept untouched) because ``gpt-realtime-2.1-mini`` and successors
to the deprecated ``gpt-4o-mini-tts``/``tts-1``/``tts-1-hd`` speech models do **not** serve
``v1/audio/speech`` at all -- only ``v1/realtime`` (OpenAI model page, fetched 2026-11-09; see the
migration plan this module implements: ``2026-11-09-openai-tts-realtime-migration.md``). This is a
distinct session/transport (stateful WebSocket, not a single HTTP call), not a model-ID substitution.

Selected explicitly via ``tts.openai.api`` -- never inferred from the configured model name, since
OpenAI's naming does not reliably signal endpoint support (this very deprecation is the proof).
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any, Optional
from urllib.parse import urlparse

from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, read_selection
from tools.tts_tool_delivery import _origin, _section, _wrap_pcm_as_wav, _write_wav_bytes_as
from tools.tts_tool_openai import _resolve_openai_audio_client_config
from tools.tts_tool_providers import _scrub_secret_text, _tts_response_format_from_path

logger = logging.getLogger("tools.tts_tool")

DEFAULT_OPENAI_REALTIME_MODEL = "gpt-realtime-2.1-mini"
# Only format the Realtime API serves (OpenAI Realtime audio output types): 24kHz mono PCM16.
# Matches tts_tool_delivery's GEMINI_TTS_SAMPLE_RATE/CHANNELS/SAMPLE_WIDTH defaults exactly, so the
# existing _wrap_pcm_as_wav()/_write_wav_bytes_as() container pipeline is reused unmodified.
_REALTIME_AUDIO_FORMAT = {"type": "audio/pcm", "rate": 24000}
DEFAULT_REALTIME_TIMEOUT_S = 30.0


def _base_url_is_managed_gateway_origin(effective_base_url: Optional[str]) -> bool:
    """True when *effective_base_url* resolves to the managed Nous ``openai-audio`` gateway's
    origin (scheme+host[:port]), regardless of path/trailing slash. F-2 (CWE-693): a direct API
    key pointed at the gateway's own origin must be caught the same as the resolver's
    ``is_managed`` flag -- the gateway's speech route silently coerces unsupported models, and a
    v1/realtime WS aimed at it is unproven, so this shape must reject exactly like the managed
    selection does.
    """
    if not effective_base_url:
        return False
    gateway = resolve_managed_tool_gateway("openai-audio")
    if gateway is None or not gateway.gateway_origin:
        return False
    try:
        requested = urlparse(effective_base_url)
        managed = urlparse(gateway.gateway_origin)
    except ValueError:
        return False
    return bool(requested.netloc) and (requested.scheme, requested.netloc) == (managed.scheme, managed.netloc)


def _generate_openai_realtime_tts(
    text: str, output_path: str, tts_config: dict[str, Any], *, api_key: Optional[str] = None,
    base_url: Optional[str] = None, model: Optional[str] = None, voice: Optional[str] = None,
    speed: Optional[float] = None, instructions: Optional[str] = None) -> str:
    """Generate audio via the OpenAI Realtime API (``client.realtime.connect(...)``, WebSocket).

    Mirrors ``_generate_openai_tts``'s public contract (same args, returns ``output_path``, raises
    ``ValueError`` on any resolvable failure) so the dispatcher in ``tools.tts_tool`` can call either
    backend uniformly. Opens one short-lived session per call: sends the approved reply text as a
    single user turn with ``tools: []``/no function-calling exposed (this path only ever needs to
    speak pre-approved text, never to reason or call tools), requests audio-only output, collects
    the streamed ``audio/pcm`` deltas, and wraps them into the container ``output_path``'s extension
    implies via the same WAV->ffmpeg pipeline Gemini's PCM backend already uses.
    """
    fallback_base: Optional[str] = None
    is_managed = False
    if api_key is None:
        api_key, fallback_base, is_managed = _resolve_openai_audio_client_config()
    oai_config = _section(tts_config, "openai")
    config_base_url = oai_config.get("base_url")
    effective_base_url = base_url or config_base_url or fallback_base or "https://api.openai.com/v1"
    # F-2 (CWE-693): reject on ANY of the resolver's is_managed flag, the raw "tts" selection
    # being the managed row, or the effective base_url pointing at the managed gateway's own
    # origin -- independent of how the credential got here. The explicit-api_key kwarg (sole call
    # site never passes one; tests enshrine the skip) used to bypass this guard entirely by
    # skipping resolution, and a gateway-origin base_url + a direct key resolved as "not managed"
    # even though it still aims an unproven v1/realtime session at the gateway, which silently
    # coerces unsupported models on its speech route. Raised before any client is constructed.
    if (is_managed or read_selection("tts") == NOUS_MANAGED_PROVIDER
            or _base_url_is_managed_gateway_origin(effective_base_url)):
        raise ValueError(
            "tts.openai.api='realtime' is not supported through the managed Nous audio gateway "
            "(not proven to proxy v1/realtime WebSocket sessions). Set tts.openai.api_key or "
            "VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY to use the Realtime backend directly, or leave "
            "tts.openai.api unset/'speech' to keep using the managed gateway.")
    if model is None:
        model = oai_config.get("realtime_model", DEFAULT_OPENAI_REALTIME_MODEL)
    if voice is None:
        voice = oai_config.get("voice") or "alloy"
    base_url = effective_base_url
    if speed is None:
        speed_default = tts_config.get("speed", 1.0) if isinstance(tts_config, dict) else 1.0
        speed = float(oai_config.get("speed", speed_default))
    timeout_s = float(oai_config.get("realtime_timeout_s", DEFAULT_REALTIME_TIMEOUT_S))
    if not instructions:
        instructions = oai_config.get("instructions") or None

    client = _origin()._import_openai_client()(api_key=api_key, base_url=base_url)
    audio_chunks: list[bytes] = []
    done_event = threading.Event()
    error_holder: list[str] = []

    def _pump(connection) -> None:
        """Read server events until response.done/output_audio.done or an error event arrives."""
        try:
            for event in connection:
                etype = getattr(event, "type", None)
                if etype in ("response.output_audio.delta", "response.audio.delta"):
                    delta = getattr(event, "delta", None)
                    if delta:
                        audio_chunks.append(base64.b64decode(delta))
                elif etype == "error":
                    err = getattr(event, "error", None)
                    message = getattr(err, "message", None) or str(err) or "unknown Realtime error"
                    error_holder.append(message)
                    done_event.set()
                    return
                elif etype in ("response.done", "response.output_audio.done", "response.audio.done"):
                    done_event.set()
                    if etype == "response.done":
                        return
        except Exception as exc:  # connection dropped / parse failure mid-stream
            error_holder.append(_scrub_secret_text(f"{type(exc).__name__}: {exc}", api_key))
            done_event.set()

    try:
        with client.realtime.connect(model=model) as connection:
            connection.session.update(session={
                "type": "realtime",
                "output_modalities": ["audio"],
                "audio": {"output": {"format": _REALTIME_AUDIO_FORMAT, "voice": voice}},
                # No function-calling/tools exposed to this turn (plan Risk #2): this path only
                # ever speaks pre-approved reply text, never reasons or takes action.
                "tools": [],
                "tool_choice": "none",
                **({"instructions": instructions} if instructions else {}),
            })
            connection.conversation.item.create(item={
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": text}],
            })
            connection.response.create(response={"output_modalities": ["audio"]})
            pump_thread = threading.Thread(target=_pump, args=(connection,), daemon=True)
            pump_thread.start()
            done_event.wait(timeout=timeout_s)
            # Whether done_event fired or we timed out, exiting the `with` block closes the
            # WebSocket, which unblocks _pump's connection.recv() (ConnectionClosed) so the daemon
            # thread exits promptly; bound the join so a wedged thread can never hang this call.
            pump_thread.join(timeout=2.0)
    except Exception as exc:
        raise ValueError(
            _scrub_secret_text(f"OpenAI Realtime TTS session failed: {type(exc).__name__}: {exc}", api_key)
        ) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    if not done_event.is_set():
        # Explicit timeout bound (plan Task 1 AC4): no partial/corrupt file is ever written on a
        # hung session -- audio_chunks (if any) are discarded, nothing touches output_path.
        raise ValueError(
            f"OpenAI Realtime TTS timed out after {timeout_s:.0f}s waiting for a response "
            "(no file written; increase tts.openai.realtime_timeout_s if this recurs).")
    if error_holder:
        raise ValueError(_scrub_secret_text(f"OpenAI Realtime TTS error: {error_holder[0]}", api_key))
    if not audio_chunks:
        raise ValueError("OpenAI Realtime TTS returned no audio data")

    pcm_bytes = b"".join(audio_chunks)
    _tts_response_format_from_path(output_path)  # validated for parity; container chosen below
    return _write_wav_bytes_as(_wrap_pcm_as_wav(pcm_bytes), output_path)
