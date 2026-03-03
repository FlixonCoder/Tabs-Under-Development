"""
TTS Module — Kokoro-based Text-to-Speech with streaming playback
Plays audio chunks as they are generated to minimize time-to-first-sound.
"""

import numpy as np
import sounddevice as sd
import threading
import queue
from kokoro import KPipeline

SAMPLE_RATE = 24000   # Kokoro's native output sample rate
VOICE = "af_heart"    # Default voice
LANG_CODE = "a"       # 'a' = American English

print("⏳ Loading Kokoro TTS pipeline...")
_pipeline = KPipeline(lang_code=LANG_CODE)
print("✅ Kokoro TTS loaded.")


def _play_audio_chunk(audio: np.ndarray):
    """Play a numpy float32 audio chunk on default output device, blocking."""
    if audio is None or len(audio) == 0:
        return
    # Ensure float32
    audio = audio.astype(np.float32)
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()


def speak(text: str, voice: str = VOICE, speed: float = 1.0):
    """
    Convert text to speech using Kokoro, streaming audio chunks as they arrive.
    This reduces time-to-first-sound significantly — the first chunk plays before
    the rest of the text finishes synthesizing.

    Args:
        text: Text to speak
        voice: Kokoro voice name (default: af_heart)
        speed: Speed multiplier (default: 1.0)
    """
    if not text or not text.strip():
        return

    # Strip markdown artifacts that don't sound natural
    clean_text = _clean_for_tts(text)
    if not clean_text.strip():
        return

    audio_queue: queue.Queue = queue.Queue()
    synthesis_done = threading.Event()

    def synthesize():
        """Generate audio chunks in background thread."""
        try:
            generator = _pipeline(
                clean_text,
                voice=voice,
                speed=speed,
                split_pattern=r"[.!?]+",   # split on sentence boundaries for lower latency
            )
            for _, result in enumerate(generator):
                audio = result.audio
                if audio is not None:
                    # Convert tensor to numpy if needed
                    if hasattr(audio, "cpu"):
                        audio = audio.cpu().numpy()
                    audio_queue.put(audio.astype(np.float32))
        except Exception as e:
            print(f"\n⚠️ TTS error: {e}")
        finally:
            audio_queue.put(None)  # sentinel to signal done
            synthesis_done.set()

    # Start synthesis in background
    synth_thread = threading.Thread(target=synthesize, daemon=True)
    synth_thread.start()

    # Play chunks as they arrive (streaming playback)
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        _play_audio_chunk(chunk)

    synth_thread.join(timeout=10)


def speak_streaming(text_generator):
    """
    Accept a text token generator (e.g., from LLM streaming) and speak as
    complete sentences accumulate. This further reduces latency by piping LLM
    streaming output directly into TTS without waiting for the full response.

    Args:
        text_generator: Iterator yielding text token strings
    """
    buffer = ""
    sentence_endings = {".", "!", "?", "\n"}

    for token in text_generator:
        buffer += token
        print(token, end="", flush=True)

        # Speak once a sentence is complete
        if any(buffer.rstrip().endswith(end) for end in sentence_endings):
            chunk_text = buffer.strip()
            if len(chunk_text) > 5:  # avoid speaking tiny fragments
                speak(chunk_text)
            buffer = ""

    # Speak any remaining text
    if buffer.strip():
        speak(buffer.strip())

    print()  # newline after full response


def _clean_for_tts(text: str) -> str:
    """Remove markdown/code/emoji that would sound weird when spoken."""
    import re
    # Remove markdown bold/italic
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove emoji (basic range)
    text = re.sub(
        r"[\U0001F300-\U0001F9FF\U00002700-\U000027BF\U0000FE00-\U0000FE0F]",
        "",
        text,
    )
    # Collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    speak("Hello! I am your AI voice assistant. How can I help you today?")
