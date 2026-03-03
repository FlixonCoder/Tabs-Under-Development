"""
Agentic RAG Engine — Persistent ChromaDB + Smart Routing
Indexes PDFs and TXT files (including saved conversation logs) into a
persistent vector store, and intelligently decides when retrieval is needed.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "Agentic-RAG", "data")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_rag")
COLLECTION_NAME = "rag_store"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 3

# Keywords that strongly suggest a need for document retrieval
RAG_KEYWORDS = [
    # Document / info requests
    "pdf", "document", "file", "data", "report", "brochure", "rulebook",
    "summarize", "summary", "information about", "tell me about",
    "what does", "according to", "find", "look up", "details",
    # Memory / conversation recall
    "we talked", "we discussed", "you said", "i said", "last time",
    "remember", "earlier", "previous", "before", "yesterday", "our conversation",
    "recall", "history",
]

# Confidence threshold — if keyword score is below this, skip RAG
RAG_KEYWORD_THRESHOLD = 1


class RAGEngine:
    """Persistent ChromaDB RAG with agentic routing."""

    def __init__(self):
        print("⏳ Initializing RAG engine...")

        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load or create persistent vector store
        os.makedirs(PERSIST_DIR, exist_ok=True)
        self.db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_model,
            persist_directory=PERSIST_DIR,
        )

        # Always check for new unindexed files; index if store is empty
        existing_count = self.db._collection.count()
        if existing_count == 0:
            print("📚 No documents found in store. Indexing data folder...")
            self._index_documents()
        else:
            print(f"✅ RAG store loaded — {existing_count} chunks already indexed.")

        self.retriever = self.db.as_retriever(search_kwargs={"k": TOP_K})

    def _index_documents(self):
        """Load PDFs and TXT files from data folder, chunk, and embed into ChromaDB."""
        docs = []
        data_path = os.path.abspath(DATA_FOLDER)

        if not os.path.exists(data_path):
            print(f"⚠️ Data folder not found: {data_path}")
            return

        for filename in os.listdir(data_path):
            filepath = os.path.join(data_path, filename)
            name_lower = filename.lower()
            try:
                if name_lower.endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                    loaded = loader.load()
                    docs.extend(loaded)
                    print(f"   📄 Loaded PDF: {filename} ({len(loaded)} pages)")
                elif name_lower.endswith(".txt"):
                    loader = TextLoader(filepath, encoding="utf-8")
                    loaded = loader.load()
                    docs.extend(loaded)
                    print(f"   📝 Loaded TXT: {filename}")
            except Exception as e:
                print(f"   ⚠️ Failed to load {filename}: {e}")

        if not docs:
            print("⚠️ No documents found to index.")
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        self.db.add_documents(chunks)
        print(f"✅ Indexed {len(chunks)} chunks from {len(docs)} document(s).")

    def should_use_rag(self, query: str) -> bool:
        """
        Determine if the query should use document retrieval.
        Uses keyword scoring as a fast, reliable heuristic.
        """
        q_lower = query.lower()
        score = sum(1 for kw in RAG_KEYWORDS if kw in q_lower)
        return score >= RAG_KEYWORD_THRESHOLD

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve relevant document chunks for the given query.
        Returns a formatted context string.
        """
        try:
            results = self.retriever.invoke(query)
            if not results:
                return ""
            context_pieces = [r.page_content for r in results]
            return "\n\n---\n\n".join(context_pieces)
        except Exception as e:
            print(f"⚠️ RAG retrieval error: {e}")
            return ""

    def index_file(self, filepath: str):
        """
        Hot-index a single PDF or TXT file into the RAG store at runtime.
        Used to immediately ingest conversation exports on shutdown.
        """
        name_lower = os.path.basename(filepath).lower()
        try:
            if name_lower.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif name_lower.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                print(f"⚠️ Unsupported file type: {filepath}")
                return

            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)
            self.db.add_documents(chunks)
            print(f"✅ Hot-indexed {len(chunks)} chunks from {os.path.basename(filepath)}")
        except Exception as e:
            print(f"⚠️ Failed to index {os.path.basename(filepath)}: {e}")


if __name__ == "__main__":
    engine = RAGEngine()
    query = "What did we talk about last time?"
    if engine.should_use_rag(query):
        ctx = engine.retrieve_context(query)
        print("Context:\n", ctx)
    else:
        print("No RAG needed.")
