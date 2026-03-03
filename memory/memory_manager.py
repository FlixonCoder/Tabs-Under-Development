"""
Memory Manager — Short-term + Long-term Persistent Memory
Short-term: sliding window of recent messages (for LLM context)
Long-term: ChromaDB-backed semantic search over all past conversations
"""

import os
import uuid
import datetime
from typing import List, Dict, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_memory")
COLLECTION_NAME = "conversation_memory"
SHORT_TERM_LIMIT = 20      # max messages kept in short-term (sliding window)
LONG_TERM_TOP_K = 3        # how many past memories to inject as context


class MemoryManager:
    """
    Manages both short-term conversation context and persistent long-term memory.
    Long-term memory is stored in a separate ChromaDB collection so the agent
    can recall relevant past conversations across sessions.
    """

    def __init__(self, embedding_model: Optional[object] = None):
        print("⏳ Initializing memory manager...")

        # Reuse embedding model if provided (avoids loading it twice)
        if embedding_model is None:
            self._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            self._embeddings = embedding_model

        os.makedirs(PERSIST_DIR, exist_ok=True)
        self._long_term_db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=PERSIST_DIR,
        )

        # Short-term memory: plain list of {role, content} dicts
        self._short_term: List[Dict] = []

        count = self._long_term_db._collection.count()
        print(f"✅ Memory ready — {count} long-term memories loaded.")

    # -------------------------------------------------------------------------
    # Short-term memory
    # -------------------------------------------------------------------------

    def add_turn(self, role: str, content: str):
        """
        Add a message to both short-term and long-term memory.
        role: 'user' or 'assistant'
        """
        msg = {"role": role, "content": content}
        self._short_term.append(msg)

        # Trim sliding window
        if len(self._short_term) > SHORT_TERM_LIMIT:
            self._short_term = self._short_term[-SHORT_TERM_LIMIT:]

        # Persist to long-term store
        self._persist(role, content)

    def get_short_term(self) -> List[Dict]:
        """Return the current short-term message list (for LLM prompt)."""
        return list(self._short_term)

    def clear_short_term(self):
        """Reset short-term memory (e.g., on topic change)."""
        self._short_term = []

    # -------------------------------------------------------------------------
    # Long-term memory
    # -------------------------------------------------------------------------

    def search_long_term(self, query: str, k: int = LONG_TERM_TOP_K) -> str:
        """
        Semantic search over all past conversation turns.
        Returns a formatted string of relevant memories, or empty string.
        """
        try:
            count = self._long_term_db._collection.count()
            if count == 0:
                return ""

            results = self._long_term_db.similarity_search(query, k=k)
            if not results:
                return ""

            memories = []
            for doc in results:
                role = doc.metadata.get("role", "unknown")
                timestamp = doc.metadata.get("timestamp", "")
                memories.append(f"[{timestamp}] {role}: {doc.page_content}")

            return "\n".join(memories)
        except Exception as e:
            print(f"⚠️ Long-term memory search error: {e}")
            return ""

    def _persist(self, role: str, content: str):
        """Save a message to the persistent ChromaDB long-term store."""
        try:
            doc_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            self._long_term_db.add_texts(
                texts=[content],
                metadatas=[{"role": role, "timestamp": timestamp}],
                ids=[doc_id],
            )
        except Exception as e:
            print(f"⚠️ Memory persist error: {e}")

    def get_memory_count(self) -> int:
        """Return total number of stored memory entries."""
        return self._long_term_db._collection.count()


if __name__ == "__main__":
    mem = MemoryManager()
    mem.add_turn("user", "Who is the founder of Tesla?")
    mem.add_turn("assistant", "Elon Musk co-founded Tesla in 2003.")
    results = mem.search_long_term("Tesla founder")
    print("Memory search results:\n", results)
