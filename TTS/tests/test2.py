"""
Piper TTS — Streaming Voice Pipeline
=====================================
100% local, zero cloud calls. Uses Piper's synthesize_stream_raw() to pipe
raw int16 PCM chunks directly to sounddevice in real time — no WAV file,
no buffering the whole utterance, first audio plays in ~150ms.

Voices available (models/):
  en_US-amy-medium         (female, clear)
  en_US-bryce-medium       (male, deep)
  en_US-hfc_female-medium  (female, neutral)
  en_US-hfc_male-medium    (male, neutral)
  en_US-john-medium        (male, conversational)
  en_US-kathleen-low       (female, lightweight)

Usage:
  python test2.py                     # run built-in demo
  python test2.py --voice bryce       # pick a voice by short name
"""

import os
import sys
import re
import queue
import threading
import argparse
import numpy as np
import sounddevice as sd
from piper import PiperVoice

# ─── Voice registry ───────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

VOICES = {
    "amy":        "en_US-amy-medium",
    "bryce":      "en_US-bryce-medium",
    "hfc_female": "en_US-hfc_female-medium",
    "hfc_male":   "en_US-hfc_male-medium",
    "john":       "en_US-john-medium",
    "kathleen":   "en_US-kathleen-low",
}

DEFAULT_VOICE = "john"

# ─── Model loader (cached singleton per voice) ────────────────────────────────

_loaded_voices: dict[str, PiperVoice] = {}

def _load_voice(short_name: str = DEFAULT_VOICE) -> PiperVoice:
    """Load and cache a PiperVoice by short alias."""
    if short_name in _loaded_voices:
        return _loaded_voices[short_name]

    full_name = VOICES.get(short_name)
    if full_name is None:
        raise ValueError(
            f"Unknown voice '{short_name}'. Choose from: {list(VOICES.keys())}"
        )

    model_path = os.path.join(MODELS_DIR, f"{full_name}.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Download it with:\n"
            f"  python -m piper --download-voice {full_name} --data-dir {MODELS_DIR}"
        )

    print(f"⏳ Loading Piper voice '{short_name}' ({full_name})...")
    voice = PiperVoice.load(model_path)
    _loaded_voices[short_name] = voice
    print(f"✅ Piper voice loaded. Sample rate: {voice.config.sample_rate} Hz")
    return voice


# ─── Text cleaner ─────────────────────────────────────────────────────────────

def _clean_for_tts(text: str) -> str:
    """Strip markdown and emoji that sound weird when spoken aloud."""
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)          # bold/italic
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)   # headers
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)        # code blocks
    text = re.sub(r"`([^`]+)`", r"\1", text)                      # inline code
    text = re.sub(r"https?://\S+", "", text)                      # URLs
    text = re.sub(                                                 # emoji
        r"[\U0001F300-\U0001F9FF\U00002700-\U000027BF\U0000FE00-\U0000FE0F]",
        "", text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Core streaming speak ─────────────────────────────────────────────────────

def speak(text: str, voice_name: str = DEFAULT_VOICE) -> None:
    """
    Synthesize `text` with Piper and stream audio to the speaker in real time.

    Piper's synthesize_stream_raw() yields raw int16 PCM byte-chunks as they
    are generated. We push each chunk into a sounddevice.OutputStream immediately
    — first audio arrives within ~150 ms, well before the full utterance is done.

    Args:
        text:       Text to speak. Markdown/emoji cleaned automatically.
        voice_name: Short voice alias (e.g. 'john', 'amy'). See VOICES dict.
    """
    if not text or not text.strip():
        return

    clean = _clean_for_tts(text)
    if not clean:
        return

    voice = _load_voice(voice_name)
    sample_rate = voice.config.sample_rate

    # Queue bridges the synthesis thread and the playback loop
    audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=32)

    def _synthesize():
        """Background thread: push PCM chunks into the queue."""
        try:
            for raw_bytes in voice.synthesize_stream_raw(clean):
                if not raw_bytes:
                    continue
                # Piper yields int16 bytes → convert to numpy int16 array
                chunk = np.frombuffer(raw_bytes, dtype=np.int16)
                audio_queue.put(chunk)
        except Exception as exc:
            print(f"\n⚠️  Piper synthesis error: {exc}")
        finally:
            audio_queue.put(None)   # sentinel — signals playback to stop

    synth_thread = threading.Thread(target=_synthesize, daemon=True)
    synth_thread.start()

    # Open stream at Piper's native sample rate; dtype int16 matches raw output
    with sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    ) as stream:
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            stream.write(chunk)

    synth_thread.join(timeout=10)


# ─── LLM-streaming helper ─────────────────────────────────────────────────────

def speak_streaming(
    text_generator,
    voice_name: str = DEFAULT_VOICE,
    min_chunk_len: int = 6,
) -> None:
    """
    Accept a token generator (e.g. from an LLM streaming call) and speak each
    sentence as it completes, minimising time-to-first-audio in a conversation.

    Args:
        text_generator: Iterator that yields string tokens one at a time.
        voice_name:     Piper voice alias passed through to speak().
        min_chunk_len:  Ignore sentence fragments shorter than this (chars).
    """
    buffer = ""
    sentence_endings = {".", "!", "?", "\n"}

    for token in text_generator:
        buffer += token
        print(token, end="", flush=True)

        if any(buffer.rstrip().endswith(end) for end in sentence_endings):
            chunk = buffer.strip()
            if len(chunk) >= min_chunk_len:
                speak(chunk, voice_name=voice_name)
            buffer = ""

    # Tail — any remaining text that didn't end with punctuation
    if buffer.strip() and len(buffer.strip()) >= min_chunk_len:
        speak(buffer.strip(), voice_name=voice_name)

    print()


# ─── CLI demo ─────────────────────────────────────────────────────────────────

def _demo(voice_name: str) -> None:
    sentences = [
        "Hello! My name is TABS, your fully local AI voice assistant.",
        "I run entirely on your own hardware — no internet, no cloud, no data leaving your machine.",
        "Piper streams my voice in real time, so you hear me speaking almost instantly.",
        "How can I help you today?",
    ]

    print(f"\n🔊 Piper TTS streaming demo — voice: '{voice_name}'\n")
    for sentence in sentences:
        print(f"  → {sentence}")
        speak(sentence, voice_name=voice_name)

    print("\n✅ Demo complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piper TTS streaming demo")
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        choices=list(VOICES.keys()),
        help=f"Voice to use (default: {DEFAULT_VOICE})",
    )
    args = parser.parse_args()
    _demo(args.voice)