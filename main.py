"""
TABS — AI Voice Agent
Main orchestration loop integrating STT, LLM, RAG, Memory, TTS, and Reminders.

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
    print("║  RAG: ChromaDB  |  Memory: Persistent  |  Reminders  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


# ─── Startup: load all models ─────────────────────────────────────────────────

def load_all_modules():
    """Load all modules and return them. Fails loudly if a dependency is missing."""

    print("🚀 Starting TABS Voice Agent...\n")

    # 1. STT
    print("[1/6] Loading Speech-to-Text (Whisper)...")
    from ears.stt import listen_and_transcribe
    print()

    # 2. TTS
    print("[2/6] Loading Text-to-Speech (Kokoro)...")
    from TTS.tts import speak, speak_streaming
    print()

    # 3. LLM
    print("[3/6] Loading LLM (Qwen2.5:3b via Ollama)...")
    from brain.llm import LLMEngine
    llm = LLMEngine()
    print()

    # 4. RAG
    print("[4/6] Loading RAG Engine (ChromaDB)...")
    from rag.rag_engine import RAGEngine
    rag = RAGEngine()
    print()

    # 5. Memory
    print("[5/6] Loading Memory Manager...")
    from memory.memory_manager import MemoryManager
    # Share the embedding model from RAG to avoid loading it twice
    memory = MemoryManager(embedding_model=rag.embedding_model)
    print()

    # 6. Reminder System
    print("[6/6] Loading Reminder Engine...")
    from reminders.reminder_store     import ReminderStore
    from reminders.reminder_parser    import ReminderParser
    from reminders.reminder_engine    import ReminderEngine
    from reminders.reminder_responder import ReminderResponder

    reminder_store     = ReminderStore()
    reminder_parser    = ReminderParser(llm)
    reminder_engine    = ReminderEngine(reminder_store, speak)
    reminder_responder = ReminderResponder(reminder_store)
    reminder_engine.start()
    print(f"   📅 {len(reminder_store)} reminder(s) loaded.")
    print()

    print("\u2705 All systems ready!\n")
    return (
        listen_and_transcribe, speak, speak_streaming,
        llm, rag, memory,
        reminder_store, reminder_parser, reminder_engine, reminder_responder,
    )


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


# ─── Reminder Dispatcher ─────────────────────────────────────────────────────

def handle_reminder_if_needed(
    user_text: str,
    parser,
    engine,
    responder,
    store,
    speak,
    memory,
) -> str:
    """
    Check whether user_text is a reminder-related request.
    If yes, handle it completely and return the spoken response string.
    If no, return an empty string so the caller falls through to the LLM.
    """
    now = datetime.datetime.now()

    # ── Management: delete / edit / list ──────────────────────────────────────
    mgmt = parser.is_reminder_management(user_text)
    if mgmt:
        action, raw = mgmt

        if action == "delete":
            keyword = parser.extract_title_keyword(raw)
            deleted = store.delete_by_title(keyword)
            if deleted:
                response = responder.confirm_deleted(deleted)
            else:
                response = responder.confirm_not_found(keyword)
            return response

        if action == "edit":
            keyword  = parser.extract_title_keyword(raw)
            matches  = store.find_by_title(keyword)
            if not matches:
                return responder.confirm_not_found(keyword)
            target   = matches[0]
            new_dt   = parser.extract_new_time(raw, now)
            if new_dt:
                store.update(target["id"], datetime=new_dt)
                target["datetime"] = new_dt.isoformat()
                return responder.confirm_updated(target, now)
            return f"I couldn't determine the new time for your {target['title']}."

        if action == "list":
            return responder.list_all(store.get_all(), now)

    # ── Query: today / tomorrow / upcoming / next event ───────────────────────
    if parser.is_reminder_query(user_text):
        lower = user_text.lower()
        if "tomorrow" in lower:
            tomorrow = (now + datetime.timedelta(days=1)).date()
            return responder.answer_tomorrow(store.get_for_date(tomorrow), now)
        if "next" in lower and "event" in lower:
            return responder.answer_next_event(store.get_all(), now)
        if any(k in lower for k in ("upcoming", "next 7", "this week", "week")):
            return responder.answer_upcoming(store.get_upcoming(now, days=7), now)
        # Default: today
        return responder.answer_today(store.get_for_date(now.date()), now)

    # ── Add: create a new reminder ────────────────────────────────────────────
    if parser.is_reminder_intent(user_text):
        parsed = parser.parse_reminder(user_text, now)
        if parsed and parsed.get("confidence", 0) >= 0.3:
            try:
                reminder = store.add(
                    title          = parsed["title"],
                    event_type     = parsed["event_type"],
                    event_datetime = parsed["event_datetime"],
                )
                print(f"\n📅 [Reminder stored] {reminder['title']} → {reminder['datetime']}")
                return responder.confirm_added(reminder, now)
            except Exception as e:
                print(f"\n⚠️ Could not store reminder: {e}")
        
        # Intent detected but no valid time found — ask user to clarify
        if parsed is None:
            return (
                "I'd love to set that reminder for you, Sir, but I couldn't find "
                "a specific date or time. Could you tell me when exactly?"
            )
        return ""

    return ""


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
    reminder_parser=None,
    reminder_engine=None,
    reminder_responder=None,
    reminder_store=None,
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

    # Step 3: Reminder interception (before LLM)
    if reminder_parser and reminder_store:
        reminder_response = handle_reminder_if_needed(
            user_text, reminder_parser, reminder_engine,
            reminder_responder, reminder_store, speak, memory
        )
        if reminder_response:
            print(f"\n🤖 TABS: {reminder_response}")
            speak(reminder_response)
            memory.add_turn("user",      user_text)
            memory.add_turn("assistant", reminder_response)
            session_log.append(("user",      user_text))
            session_log.append(("assistant", reminder_response))
            return True

    # Step 4: Build LLM prompt (with memory + RAG)
    messages = build_messages(user_text, rag, memory)

    # Step 5: Stream response — pipe tokens into TTS sentence-by-sentence
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

    # Step 6: Persist this turn to memory and session log
    memory.add_turn("user", user_text)
    memory.add_turn("assistant", full_response)
    session_log.append(("user", user_text))
    session_log.append(("assistant", full_response))

    return True


# ─── Text fallback mode ───────────────────────────────────────────────────────

def text_turn(
    speak,
    speak_streaming,
    llm,
    rag,
    memory,
    session_log: list,
    reminder_parser=None,
    reminder_engine=None,
    reminder_responder=None,
    reminder_store=None,
) -> bool:
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

    # Reminder interception (before LLM)
    if reminder_parser and reminder_store:
        reminder_response = handle_reminder_if_needed(
            user_text, reminder_parser, reminder_engine,
            reminder_responder, reminder_store, speak, memory
        )
        if reminder_response:
            print(f"\n🤖 TABS: {reminder_response}")
            speak(reminder_response)
            memory.add_turn("user",      user_text)
            memory.add_turn("assistant", reminder_response)
            session_log.append(("user",      user_text))
            session_log.append(("assistant", reminder_response))
            return True

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
        (
            listen_and_transcribe, speak, speak_streaming,
            llm, rag, memory,
            reminder_store, reminder_parser, reminder_engine, reminder_responder,
        ) = load_all_modules()
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    # ── Morning digest check at startup ──
    reminder_engine.check_morning_digest_now()

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
                running = text_turn(
                    speak, speak_streaming, llm, rag, memory, session_log,
                    reminder_parser, reminder_engine, reminder_responder, reminder_store,
                )
            else:
                running = voice_turn(
                    listen_and_transcribe, speak, speak_streaming, llm, rag, memory, session_log,
                    reminder_parser, reminder_engine, reminder_responder, reminder_store,
                )
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user.")
            break
        except Exception as e:
            print(f"\n⚠️ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)  # brief pause before retrying

    # ── Shutdown ──
    reminder_engine.stop()
    save_and_index_conversation(session_log, rag)

    print("\n✅ TABS shut down. Goodbye!\n")


if __name__ == "__main__":
    main()
