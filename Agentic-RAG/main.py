import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# load PDFs from a folder
def load_docs(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    return docs

# Update this to where your pdfs are stored
docs = load_docs(r"C:\\Users\\Dell\\OneDrive\\Desktop\\tabs\\Learning\\Agentic-RAG\\data")

# Splitting pdfs into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 80
)
chunks = text_splitter.split_documents(docs)
print("Chunks created: ", len(chunks))

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

# saving text into chroma vector DB
db = Chroma(
    collection_name="rag_store",
    embedding_function=embedding_model
)

db.add_documents(chunks)


retriever = db.as_retriever(search_kwargs={"k":3})


# The brain
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=150
)


# The agent
def agent_controller(query):
    q = query.lower()
    if any(word in q for word in ["pdf", "document", "data", "summarize", "information", "find"]):
        return "search"
    return "direct"

# RAG
def rag_answer(query):
    action = agent_controller(query)

    if action == "search":
        print(f"🕵️ Agent decided to SEARCH document for: '{query}'")
        results = retriever.invoke(query)         
        context = "\n".join([r.page_content for r in results])
        final_prompt = f"Use this context:\n{context}\n\nAnswer:\n{query}"
    else:
        print(f"🤖 Agent decided to answer DIRECTLY: '{query}'")
        final_prompt = query

    response = llm(final_prompt)[0]["generated_text"]
    return response

exit = True
while(exit):
    x = input("Enter: ")
    if "exit" in x or "quit" in x:
        exit = False
        result = "Good Bye!!"
    else:
        result = rag_answer(x)
    print(f"AI: {result}")