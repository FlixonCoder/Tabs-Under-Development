"""
TTS Module — Kokoro primary, Piper offline fallback
====================================================
Primary:  Kokoro TTS  — high-quality neural voice, streams audio chunk-by-chunk.
Fallback: Piper TTS   — used ONLY when Kokoro is unavailable (import error /
                         model load failure), i.e. effectively the "offline" path
                         where the Kokoro package itself can't be loaded.

Both engines synthesize fully locally — no cloud calls, no API keys.
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import re

# ─── Sample rate fallback if neither engine loads ─────────────────────────────
_SAMPLE_RATE_KOKORO = 24_000   # Kokoro native output

# ─── Try to load Kokoro (primary) ────────────────────────────────────────────

_kokoro_pipeline = None
_kokoro_available = False

try:
    from kokoro import KPipeline
    print("⏳ Loading Kokoro TTS pipeline...")
    _kokoro_pipeline = KPipeline(lang_code="a")   # 'a' = American English
    _kokoro_available = True
    print("✅ Kokoro TTS loaded.")
except Exception as _kokoro_err:
    print(f"⚠️  Kokoro TTS unavailable ({_kokoro_err}) — will use Piper fallback.")

# ─── Try to load Piper (offline fallback) ────────────────────────────────────
# Piper is only initialised when Kokoro failed to load.

_piper_voice = None
_piper_sample_rate = 22_050   # typical Piper sample rate (updated on load)

if not _kokoro_available:
    import os as _os

    _MODELS_DIR = _os.path.join(_os.path.dirname(__file__), "tests", "models")
    _PIPER_DEFAULT_VOICE = "en_US-john-medium"
    _PIPER_MODEL_PATH = _os.path.join(_MODELS_DIR, f"{_PIPER_DEFAULT_VOICE}.onnx")

    try:
        from piper import PiperVoice
        print(f"⏳ Loading Piper fallback voice '{_PIPER_DEFAULT_VOICE}'...")
        _piper_voice = PiperVoice.load(_PIPER_MODEL_PATH)
        _piper_sample_rate = _piper_voice.config.sample_rate
        print(f"✅ Piper fallback loaded. Sample rate: {_piper_sample_rate} Hz")
    except FileNotFoundError:
        print(
            f"⚠️  Piper model not found at {_PIPER_MODEL_PATH}.\n"
            f"    Download it with:\n"
            f"    python -m piper --download-voice {_PIPER_DEFAULT_VOICE} "
            f"--data-dir {_MODELS_DIR}"
        )
    except Exception as _piper_err:
        print(f"⚠️  Piper fallback also unavailable: {_piper_err}")


# ─── Text cleaner (shared) ────────────────────────────────────────────────────

def _clean_for_tts(text: str) -> str:
    """Remove markdown, code blocks, URLs, and emoji — they sound weird spoken."""
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)           # bold / italic
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)    # headers
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)         # code blocks
    text = re.sub(r"`([^`]+)`", r"\1", text)                       # inline code
    text = re.sub(r"https?://\S+", "", text)                       # URLs
    text = re.sub(                                                  # emoji
        r"[\U0001F300-\U0001F9FF\U00002700-\U000027BF\U0000FE00-\U0000FE0F]",
        "", text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Kokoro speak (primary) ───────────────────────────────────────────────────

_KOKORO_VOICE = "af_heart"   # default Kokoro voice

def _speak_kokoro(text: str, voice: str = _KOKORO_VOICE, speed: float = 1.0) -> None:
    """
    Synthesize with Kokoro and stream audio chunks to the speaker as they arrive.
    Reduces time-to-first-sound significantly.
    """
    audio_queue: queue.Queue = queue.Queue()

    def _synthesize():
        try:
            generator = _kokoro_pipeline(
                text,
                voice=voice,
                speed=speed,
                split_pattern=r"[.!?]+",   # split on sentence boundaries
            )
            for _, result in enumerate(generator):
                audio = result.audio
                if audio is not None:
                    if hasattr(audio, "cpu"):
                        audio = audio.cpu().numpy()
                    audio_queue.put(audio.astype(np.float32))
        except Exception as exc:
            print(f"\n⚠️  Kokoro synthesis error: {exc}")
        finally:
            audio_queue.put(None)   # sentinel

    synth_thread = threading.Thread(target=_synthesize, daemon=True)
    synth_thread.start()

    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        sd.play(chunk, samplerate=_SAMPLE_RATE_KOKORO)
        sd.wait()

    synth_thread.join(timeout=10)


# ─── Piper speak (offline fallback) ──────────────────────────────────────────

def _speak_piper(text: str) -> None:
    """
    Synthesize with Piper and stream raw int16 PCM directly to sounddevice.
    Only called when Kokoro is unavailable.
    """
    if _piper_voice is None:
        print("⚠️  No TTS engine available — cannot speak.")
        return

    audio_queue: queue.Queue = queue.Queue(maxsize=32)

    def _synthesize():
        try:
            for raw_bytes in _piper_voice.synthesize_stream_raw(text):
                if raw_bytes:
                    audio_queue.put(np.frombuffer(raw_bytes, dtype=np.int16))
        except Exception as exc:
            print(f"\n⚠️  Piper synthesis error: {exc}")
        finally:
            audio_queue.put(None)   # sentinel

    synth_thread = threading.Thread(target=_synthesize, daemon=True)
    synth_thread.start()

    with sd.OutputStream(
        samplerate=_piper_sample_rate,
        channels=1,
        dtype="int16",
    ) as stream:
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            stream.write(chunk)

    synth_thread.join(timeout=10)


# ─── Public API ───────────────────────────────────────────────────────────────

import threading
_tts_lock = threading.Lock()

def speak(text: str, voice: str = _KOKORO_VOICE, speed: float = 1.0) -> None:
    """
    Convert text to speech.

    Uses Kokoro (primary).  Falls back to Piper automatically when Kokoro was
    not available at import time (offline / package missing).

    Args:
        text:  Text to speak.
        voice: Kokoro voice name (ignored when Piper fallback is active).
        speed: Speed multiplier — Kokoro only (default 1.0).
    """
    if not text or not text.strip():
        return

    clean = _clean_for_tts(text)
    if not clean:
        return

    with _tts_lock:
        if _kokoro_available:
            _speak_kokoro(clean, voice=voice, speed=speed)
        else:
            _speak_piper(clean)


def speak_streaming(text_generator, voice: str = _KOKORO_VOICE, min_chunk_len: int = 6) -> None:
    """
    Accept a text token generator (e.g. from LLM streaming) and speak each
    sentence as it completes, minimising time-to-first-audio.

    Args:
        text_generator: Iterator yielding string tokens.
        voice:          Kokoro voice (ignored when Piper fallback is active).
        min_chunk_len:  Minimum character count before speaking a fragment.
    """
    buffer = ""
    sentence_endings = {".", "!", "?", "\n"}

    for token in text_generator:
        buffer += token
        print(token, end="", flush=True)

        if any(buffer.rstrip().endswith(end) for end in sentence_endings):
            chunk = buffer.strip()
            if len(chunk) >= min_chunk_len:
                speak(chunk, voice=voice)
            buffer = ""

    # Tail — any remaining text that didn't end with punctuation
    if buffer.strip() and len(buffer.strip()) >= min_chunk_len:
        speak(buffer.strip(), voice=voice)

    print()   # newline after full response


# ─── CLI smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = "Kokoro" if _kokoro_available else "Piper (fallback)"
    print(f"\n🔊 TTS smoke test — engine: {engine}\n")
    speak("Hello! I am TABS, your fully local AI voice assistant. How can I help you today?")
    print("✅ Done.\n")
