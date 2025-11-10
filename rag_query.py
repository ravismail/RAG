import requests
from langchain_community.vectorstores import Chroma
from chromadb import HttpClient

# --- CONFIG ---
CHROMA_URL = "http://localhost:8000"
COLLECTION_NAME = "pdf_new-collection"
MODEL_RUNNER_URL = "http://localhost:12434/engines/v1"
EMBED_MODEL = "ai/nomic-embed-text-v1.5"
LLM_MODEL = "ai/llama3.2:latest"  # or your chosen text model name
TOP_K = 3

# --- CUSTOM EMBEDDINGS CLASS ---
class CustomRESTEmbeddings:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed_query(self, text: str):
        r = requests.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": text},
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

# --- CONNECT TO EXISTING CHROMA ---
embeddings = CustomRESTEmbeddings(base_url=MODEL_RUNNER_URL, model=EMBED_MODEL)
client = HttpClient(host="localhost", port=8000)
vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

print("✅ Connected to ChromaDB and ready for retrieval.")

# --- FUNCTION: Query Chroma + Ask LLM ---
def ask_rag(query: str):
    # Step 1: Retrieve top relevant chunks
    results = vectorstore.similarity_search(query, k=TOP_K)
    context = "\n\n".join([r.page_content for r in results])

    # Step 2: Build prompt for LLM
    prompt = f"""You are an AI assistant answering questions based on a document.
Use the following context to answer clearly and concisely:

Context:
{context}

Question: {query}

Answer:"""

    # Step 3: Send to Model Runner Completion Endpoint
    resp = requests.post(
        f"{MODEL_RUNNER_URL}/completions",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "max_tokens": 500,
        },
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["text"]
    return answer.strip()

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("\n🧠 RAG Query System Ready!")
    print("Type your question (or 'exit' to quit)\n")

    while True:
        query = input("❓ Your question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            answer = ask_rag(query)
            print("\n💬 Answer:\n", answer)
        except Exception as e:
            print("⚠️ Error:", e)
