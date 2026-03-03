# 🤖 TABS — AI Voice Agent

> **A complete, fully local AI voice agent with real-time capabilities.**  
> No cloud. No API keys. No data leaves your machine.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Qwen2.5-purple)](https://ollama.com/)
[![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-green)](https://github.com/openai/whisper)
[![Kokoro](https://img.shields.io/badge/TTS-Kokoro-orange)](https://github.com/remsky/Kokoro-FastAPI)
[![Piper](https://img.shields.io/badge/TTS-Piper%20(fallback)-ff6b35)](https://github.com/rhasspy/piper)
[![ChromaDB](https://img.shields.io/badge/RAG-ChromaDB-red)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## ✨ What is TABS?

**TABS** is a production-grade, **fully offline AI voice assistant** that runs entirely on your local hardware. It combines state-of-the-art open-source models for speech recognition, language understanding, document retrieval, persistent memory, and neural text-to-speech into a single real-time conversational pipeline.

Think of it as your private, self-hosted, voice-first AI — with the intelligence of a modern LLM, the recall of a RAG knowledge base, and the memory of a long-term companion.

---

## 🏗️ Architecture Overview

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                        TABS Pipeline                            │
 │                                                                 │
 │  🎤 Microphone                                                  │
 │       │                                                         │
 │       ▼                                                         │
 │  ┌─────────┐    VAD-gated      ┌──────────────────────┐        │
 │  │  ears/  │──────────────────▶│  Whisper (base)      │        │
 │  │  stt.py │  auto start/stop  │  Speech-to-Text      │        │
 │  └─────────┘                   └──────────┬───────────┘        │
 │                                           │ transcript          │
 │                                           ▼                     │
 │       ┌───────────────────────────────────────────────┐         │
 │       │            main.py  — Orchestrator            │         │
 │       │                                               │         │
 │       │   ┌──────────┐   ┌──────────┐   ┌─────────┐ │         │
 │       │   │  memory/ │   │   rag/   │   │  brain/ │ │         │
 │       │   │ Short +  │   │ ChromaDB │   │  Qwen   │ │         │
 │       │   │ Long term│   │  + MiniLM│   │  2.5:3b │ │         │
 │       │   └──────────┘   └──────────┘   └────┬────┘ │         │
 │       └────────────────────────────────────────┼──────┘         │
 │                                                │ streamed tokens │
 │                                                ▼                │
 │                                      ┌──────────────────┐      │
 │                                      │    TTS/tts.py    │      │
 │                                      │  Kokoro Neural   │      │
 │                                      │  Text-to-Speech  │      │
 │                                      └────────┬─────────┘      │
 │                                               │                 │
 │                                               ▼                 │
 │                                         🔊 Speakers            │
 └─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

| Feature | Details |
|---|---|
| **100% Local & Private** | Every model runs on your own hardware — no internet required after setup |
| **Real-Time Voice Pipeline** | VAD-driven recording → Whisper STT → LLM streaming → Kokoro TTS, sentence by sentence |
| **Dual TTS Engines** | **Kokoro** (primary, `TTS/tts.py`) + **Piper** (100% local fallback, `TTS/tests/test2.py`) — both stream audio chunk-by-chunk |
| **Minimal Latency** | LLM tokens are streamed; TTS speaks each sentence as it's generated, not after the full response |
| **Agentic RAG** | Indexes your PDFs and text files into ChromaDB; the agent decides *when* to retrieve context |
| **Persistent Memory** | Short-term in-session context + long-term semantic memory via embeddings across sessions |
| **Auto Conversation Logging** | Every session is saved as a `.txt` file and hot-indexed into the RAG store for future recall |
| **Text Fallback Mode** | Run with `--text` flag for keyboard input — no microphone needed |
| **Modular Architecture** | Each component (`ears`, `brain`, `rag`, `memory`, `TTS`) is an independent Python module |

---

## 📁 Project Structure

```
local_bot/
│
├── main.py                   # 🎯 Main orchestration loop
│
├── ears/                     # 👂 Speech-to-Text (STT)
│   ├── stt.py                #    Whisper + WebRTC VAD recording
│   └── __init__.py
│
├── brain/                    # 🧠 Language Model (LLM)
│   ├── llm.py                #    Qwen2.5:3b via Ollama streaming API
│   ├── chatting_model.py     #    Conversational model wrapper
│   ├── vision_model.py       #    Vision model wrapper
│   └── __init__.py
│
├── TTS/                      # 🔊 Text-to-Speech (TTS)
│   ├── tts.py                #    Kokoro neural TTS, sentence-streaming playback (primary)
│   └── tests/
│       ├── test2.py          #    Piper TTS streaming pipeline (local fallback, no cloud)
│       └── models/           #    Piper .onnx voice models (6 voices)
│
├── rag/                      # 📚 Retrieval-Augmented Generation
│   ├── rag_engine.py         #    ChromaDB + LangChain + MiniLM embeddings
│   └── __init__.py
│
├── memory/                   # 🗂️ Memory Management
│   ├── memory_manager.py     #    Short-term context + long-term semantic recall
│   └── __init__.py
│
├── Agentic-RAG/              # 📂 RAG Knowledge Base
│   └── data/                 #    PDF/TXT documents + auto-saved conversation logs
│
├── data/                     # 🗄️ Persistent vector stores
│   ├── chroma_rag/           #    RAG document embeddings (ChromaDB)
│   └── chroma_memory/        #    Long-term memory embeddings (ChromaDB)
│
├── bin/                      # 🗑️ Archived scratch files (excluded from git)
└── requirements.txt          # 📦 Python dependencies
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- A microphone (or use `--text` mode)
- `ffmpeg` in your system PATH (required by Whisper)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/local_bot.git
cd local_bot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note for Windows users:** `webrtcvad` may require [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Alternatively: `pip install webrtcvad-wheels`

### 4. Pull the LLM Model via Ollama

```bash
ollama pull qwen2.5:3b
```

### 5. (Optional) Add Your Documents to the RAG Store

Drop PDF or `.txt` files into `Agentic-RAG/data/`. They will be indexed automatically on first use.

---

## ▶️ Running TABS

### Voice Mode (default)

```bash
# Make sure Ollama is running first
ollama serve

# In a new terminal, start TABS
python main.py
```

### Text Mode (no microphone)

```bash
python main.py --text
```

### Exit Commands

Say or type: `exit`, `quit`, `goodbye`, `bye`, or `stop`

---

## 🧩 Component Deep Dive

### 👂 STT — `ears/stt.py`
- Uses **OpenAI Whisper** (`base` model) for transcription
- **WebRTC VAD** automatically detects when you start and stop speaking — no push-to-talk needed
- Falls back to fixed-duration recording if VAD is unavailable
- 1.5 second silence timeout, 30 second hard cap per utterance

### 🧠 LLM — `brain/llm.py`
- Powered by **Qwen2.5:3b** running locally via **Ollama**
- Full **streaming token output** for minimum time-to-first-audio
- Supports multi-turn conversation context via the memory system

### 📚 RAG — `rag/rag_engine.py`
- **ChromaDB** persistent vector store with **`all-MiniLM-L6-v2`** embeddings
- Supports PDF and plain text ingestion via LangChain
- **Agentic retrieval**: the LLM decides whether a query needs document context, avoiding irrelevant injections
- Conversation logs are hot-indexed at session end for future recall

### 🗂️ Memory — `memory/memory_manager.py`
- **Short-term memory**: recent conversation turns injected as context
- **Long-term memory**: semantic search over all past sessions using ChromaDB embeddings
- Shares the embedding model with RAG to avoid loading it twice

### 🔊 TTS — Dual Engine Design

**Primary: `TTS/tts.py` (Kokoro)**
- **Kokoro** `af_heart` voice, 24 kHz neural TTS — runs fully on-device
- **Sentence-by-sentence streaming**: synthesis and playback are pipelined via a background thread + audio queue — first sound within ~200 ms
- Automatically strips markdown/emoji before speaking

**Fallback: `TTS/tests/test2.py` (Piper)**
- **Piper** ONNX models — zero cloud calls, zero API credits, zero rate limits
- Uses `synthesize_stream_raw()` → raw `int16` PCM chunks piped directly into a `sounddevice.OutputStream` — **~150 ms to first audio**
- 6 bundled voices: `amy`, `bryce`, `hfc_female`, `hfc_male`, `john`, `kathleen`
- Exposes the same `speak(text)` + `speak_streaming(token_generator)` API — drop-in compatible with the main pipeline
- Run standalone: `python TTS/tests/test2.py --voice john`

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `ollama` | Local LLM server client |
| `openai-whisper` | Speech-to-text transcription |
| `sounddevice` | Microphone input capture |
| `webrtcvad` | Voice activity detection |
| `kokoro>=0.9.4` | Neural TTS — primary engine |
| `piper` | Neural TTS — local fallback engine (ONNX, no cloud) |
| `langchain` + `langchain-community` | RAG pipeline orchestration |
| `chromadb` | Persistent vector database |
| `sentence-transformers` | Text embedding model |
| `pypdf` | PDF parsing |
| `torch` | ML backend for Kokoro & embeddings |

---

## 🔒 Privacy

TABS is designed with **privacy-first** principles:

- ✅ All processing is done **on-device**
- ✅ No API calls to external services
- ✅ Your voice, conversations, and documents **never leave your machine**
- ✅ All vector stores are persisted locally

---

## 🗺️ Roadmap

- [ ] Wake-word activation ("Hey TABS")
- [ ] Multi-language STT support
- [ ] GUI / web dashboard
- [ ] Vision capabilities (image understanding)
- [ ] Plugin / tool-use system (calendar, web search, etc.)
- [ ] Quantized model support for lower VRAM usage

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using open-source AI — runs entirely on your own hardware.
</p>
