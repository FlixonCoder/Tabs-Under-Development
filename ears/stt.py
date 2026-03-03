"""
STT Module — Whisper-based Speech-to-Text
Uses Voice Activity Detection (webrtcvad) to auto-detect speech start/end.
Falls back to fixed-duration recording if VAD is unavailable.
"""

import whisper
import sounddevice as sd
import numpy as np
import tempfile
import os
import scipy.io.wavfile as wav
import threading
import queue

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30           # VAD frame size in ms (10, 20, or 30)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # samples per frame
SILENCE_TIMEOUT_SEC = 1.5        # seconds of silence before stopping
MAX_RECORD_SEC = 30              # hard cap on recording duration

# Load Whisper once at module level — expensive operation
print("⏳ Loading Whisper model...")
_whisper_model = whisper.load_model("base")
print("✅ Whisper model loaded.")


def _record_with_vad() -> np.ndarray:
    """
    Record audio using Voice Activity Detection.
    Returns float32 audio array at SAMPLE_RATE.
    """
    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)  # aggressiveness 0-3; 2 is balanced
        use_vad = True
    except ImportError:
        use_vad = False

    audio_frames = []
    silence_frames = 0
    max_frames = int(MAX_RECORD_SEC * 1000 / FRAME_DURATION_MS)
    silence_limit = int(SILENCE_TIMEOUT_SEC * 1000 / FRAME_DURATION_MS)
    speech_started = False

    # Audio comes in as float32; VAD needs int16 PCM bytes
    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    print("🎤 Listening... (speak now)")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE,
        callback=callback,
    ):
        frame_count = 0
        while frame_count < max_frames:
            chunk = audio_q.get()
            audio_frames.append(chunk)
            frame_count += 1

            if use_vad:
                # Convert to int16 for VAD
                pcm_int16 = (chunk[:, 0] * 32767).astype(np.int16).tobytes()
                try:
                    is_speech = vad.is_speech(pcm_int16, SAMPLE_RATE)
                except Exception:
                    is_speech = True  # if VAD fails, assume speech

                if is_speech:
                    speech_started = True
                    silence_frames = 0
                elif speech_started:
                    silence_frames += 1
                    if silence_frames >= silence_limit:
                        break  # silence detected — stop recording
            else:
                # No VAD: collect until max duration
                pass

    audio_np = np.concatenate(audio_frames, axis=0).flatten()
    return audio_np


def listen_and_transcribe() -> str:
    """
    Main entry point. Records audio using VAD, transcribes via Whisper.
    Returns transcribed text string.
    """
    audio = _record_with_vad()

    # Normalize audio
    if audio.max() > 0:
        audio = audio / np.max(np.abs(audio))

    # Save to temp WAV (Whisper needs a file)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        wav.write(tmp_path, SAMPLE_RATE, np.int16(audio * 32767))

    try:
        result = _whisper_model.transcribe(tmp_path, fp16=False, language="en")
        text = result["text"].strip()
    finally:
        os.unlink(tmp_path)

    return text


if __name__ == "__main__":
    # Quick test
    text = listen_and_transcribe()
    print(f"Transcribed: '{text}'")
