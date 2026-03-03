"""
TABS — AI Voice Agent
Main orchestration loop integrating STT, LLM, RAG, Memory, and TTS.

Run with: python main.py
Ensure Ollama is running: ollama serve
"""

import sys
import os
import time
import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# ─── Banner ───────────────────────────────────────────────────────────────────

def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║        TABS — AI Voice Agent  (Local & Private)      ║")
    print("║    STT: Whisper  |  LLM: Qwen2.5:3b  |  TTS: Kokoro  ║")
    print("║         RAG: ChromaDB  |  Memory: Persistent         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


# ─── Startup: load all models ─────────────────────────────────────────────────

def load_all_modules():
    """Load all modules and return them. Fails loudly if a dependency is missing."""

    print("🚀 Starting TABS Voice Agent...\n")

    # 1. STT
    print("[1/5] Loading Speech-to-Text (Whisper)...")
    from ears.stt import listen_and_transcribe
    print()

    # 2. TTS
    print("[2/5] Loading Text-to-Speech (Kokoro)...")
    from TTS.tts import speak, speak_streaming
    print()

    # 3. LLM
    print("[3/5] Loading LLM (Qwen2.5:3b via Ollama)...")
    from brain.llm import LLMEngine
    llm = LLMEngine()
    print()

    # 4. RAG
    print("[4/5] Loading RAG Engine (ChromaDB)...")
    from rag.rag_engine import RAGEngine
    rag = RAGEngine()
    print()

    # 5. Memory
    print("[5/5] Loading Memory Manager...")
    from memory.memory_manager import MemoryManager
    # Share the embedding model from RAG to avoid loading it twice
    memory = MemoryManager(embedding_model=rag.embedding_model)
    print()

    print("\u2705 All systems ready!\n")
    return listen_and_transcribe, speak, speak_streaming, llm, rag, memory


# ─── Conversation export + RAG indexing ─────────────────────────────────────────────

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "Agentic-RAG", "data")


def save_and_index_conversation(session_log: list, rag) -> None:
    """
    Save the current session's conversation to a timestamped .txt file inside
    Agentic-RAG/data/ and immediately hot-index it into the RAG store so that
    future sessions can search and recall what was discussed.

    session_log: list of (role, text) tuples accumulated during this session.
    """
    if not session_log:
        print("(No conversation to save.)")
        return

    os.makedirs(DATA_FOLDER, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"conversation_{timestamp}.txt"
    filepath = os.path.join(DATA_FOLDER, filename)

    lines = [f"TABS Conversation Log — {timestamp}\n", "=" * 50 + "\n"]
    for role, text in session_log:
        label = "You" if role == "user" else "TABS"
        lines.append(f"[{label}]: {text}\n")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"\n💾 Conversation saved → {filename}")
    except Exception as e:
        print(f"\n⚠️ Could not save conversation: {e}")
        return

    # Immediately index the new file so the NEXT session can recall it
    print("📂 Indexing conversation into RAG...")
    rag.index_file(filepath)


# ─── Prompt Builder ───────────────────────────────────────────────────────────

def build_messages(user_query: str, rag: object, memory: object) -> list:
    """
    Build the full message list for the LLM:
    1. Recent conversation turns (short-term memory)
    2. Relevant long-term memory injected as a system note
    3. RAG document context (if applicable)
    4. The current user query
    """
    messages = memory.get_short_term()

    # Long-term memory recall
    past_memories = memory.search_long_term(user_query)
    if past_memories:
        memory_note = (
            f"[Relevant past context from memory]:\n{past_memories}"
        )
        # Inject as a system-style message before the user turn
        messages = [{"role": "system", "content": memory_note}] + messages

    # RAG retrieval
    use_rag = rag.should_use_rag(user_query)
    if use_rag:
        print("🔍 Searching documents...")
        context = rag.retrieve_context(user_query)
        if context:
            rag_note = f"[Relevant document context]:\n{context}"
            messages.append({"role": "system", "content": rag_note})
            print("📄 Document context injected.")

    # Current user query
    messages.append({"role": "user", "content": user_query})
    return messages


# ─── Voice pipeline ───────────────────────────────────────────────────────────

def voice_turn(
    listen_and_transcribe,
    speak,
    speak_streaming,
    llm,
    rag,
    memory,
    session_log: list,
) -> bool:
    """
    Execute one full voice conversation turn.
    Returns False if the user wants to exit, True to continue.
    Appends each (role, text) pair to session_log.
    """

    # Step 1: Listen and transcribe
    print("\n" + "─" * 54)
    try:
        user_text = listen_and_transcribe()
    except KeyboardInterrupt:
        return False
    except Exception as e:
        print(f"⚠️ STT error: {e}")
        return True

    if not user_text or len(user_text.strip()) < 2:
        print("(No speech detected)")
        return True

    print(f"\n👤 You: {user_text}")

    # Step 2: Exit detection
    exit_phrases = {"exit", "quit", "goodbye", "bye", "stop", "shut down"}
    if any(phrase in user_text.lower() for phrase in exit_phrases):
        farewell = "Goodbye, Sir. It was a pleasure assisting you."
        print(f"\n🤖 TABS: {farewell}")
        speak(farewell)
        return False

    # Step 3: Build LLM prompt (with memory + RAG)
    messages = build_messages(user_text, rag, memory)

    # Step 4: Stream response — pipe tokens into TTS sentence-by-sentence
    print(f"\n🤖 TABS: ", end="", flush=True)

    full_response = ""
    sentence_buffer = ""
    sentence_endings = {".", "!", "?"}

    for token in llm.stream_response(messages):
        print(token, end="", flush=True)
        full_response += token
        sentence_buffer += token

        # Speak each sentence as it completes for minimum latency
        if any(sentence_buffer.rstrip().endswith(end) for end in sentence_endings):
            chunk = sentence_buffer.strip()
            if len(chunk) > 8:
                speak(chunk)
            sentence_buffer = ""

    # Speak any trailing text that didn't end with punctuation
    if sentence_buffer.strip() and len(sentence_buffer.strip()) > 3:
        speak(sentence_buffer.strip())

    print()  # newline after streamed output

    # Step 5: Persist this turn to memory and session log
    memory.add_turn("user", user_text)
    memory.add_turn("assistant", full_response)
    session_log.append(("user", user_text))
    session_log.append(("assistant", full_response))

    return True


# ─── Text fallback mode ───────────────────────────────────────────────────────

def text_turn(speak, speak_streaming, llm, rag, memory, session_log: list) -> bool:
    """
    Text-only mode — type instead of speaking.
    Appends each (role, text) pair to session_log.
    """
    print("\n" + "─" * 54)
    try:
        user_text = input("📝 You: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False

    if not user_text:
        return True

    exit_phrases = {"exit", "quit", "goodbye", "bye", "stop"}
    if user_text.lower() in exit_phrases:
        farewell = "Goodbye, Sir. It was a pleasure assisting you."
        print(f"\n🤖 TABS: {farewell}")
        speak(farewell)
        return False

    messages = build_messages(user_text, rag, memory)

    print(f"\n🤖 TABS: ", end="", flush=True)

    full_response = ""
    sentence_buffer = ""
    sentence_endings = {".", "!", "?"}

    for token in llm.stream_response(messages):
        print(token, end="", flush=True)
        full_response += token
        sentence_buffer += token

        if any(sentence_buffer.rstrip().endswith(end) for end in sentence_endings):
            chunk = sentence_buffer.strip()
            if len(chunk) > 8:
                speak(chunk)
            sentence_buffer = ""

    if sentence_buffer.strip() and len(sentence_buffer.strip()) > 3:
        speak(sentence_buffer.strip())

    print()

    memory.add_turn("user", user_text)
    memory.add_turn("assistant", full_response)
    session_log.append(("user", user_text))
    session_log.append(("assistant", full_response))

    return True


# ─── Main entry point ─────────────────────────────────────────────────────────

def main():
    print_banner()

    # Allow --text flag for text-only mode (no microphone)
    text_only = "--text" in sys.argv

    try:
        listen_and_transcribe, speak, speak_streaming, llm, rag, memory = load_all_modules()
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    mode_label = "text" if text_only else "voice"
    startup_msg = (
        f"Hello Sir! TABS is ready. Running in {mode_label} mode. "
        "How can I help you today?"
    )
    print(f"🤖 TABS: {startup_msg}")
    speak(startup_msg)

    if not text_only:
        print("\n💡 Tip: Say 'exit' or 'quit' to stop. Use --text flag for text mode.\n")

    # ── Main loop ──
    session_log: list = []   # accumulates (role, text) tuples for this session
    running = True
    while running:
        try:
            if text_only:
                running = text_turn(speak, speak_streaming, llm, rag, memory, session_log)
            else:
                running = voice_turn(
                    listen_and_transcribe, speak, speak_streaming, llm, rag, memory, session_log
                )
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user.")
            break
        except Exception as e:
            print(f"\n⚠️ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)  # brief pause before retrying

    # ── Save and index this session's conversation ──
    save_and_index_conversation(session_log, rag)

    print("\n✅ TABS shut down. Goodbye!\n")


if __name__ == "__main__":
    main()
