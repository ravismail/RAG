#working one
import streamlit as st
import requests
from langchain_community.vectorstores import Chroma
from chromadb import HttpClient
from langchain import embeddings

# ==========================
# CONFIG
# ==========================
CHROMA_URL = "http://localhost:8000"
COLLECTION_NAME = "pdf_new-collection"
MODEL_RUNNER_URL = "http://localhost:12434/engines/v1"
EMBED_MODEL = "ai/nomic-embed-text-v1.5"
LLM_MODEL = "ai/llama3.2:latest"
TOP_K = 3

# ==========================
# CUSTOM EMBEDDINGS CLASS
# ==========================
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

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


# ==========================
# INITIALIZE CONNECTIONS (cache to avoid reload)
# ==========================
@st.cache_resource
def init_chroma():
    embeddings = CustomRESTEmbeddings(base_url=MODEL_RUNNER_URL, model=EMBED_MODEL)
    client = HttpClient(host="localhost", port=8000)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        client=client,
    )
    return vectorstore

vectorstore = init_chroma()

# ==========================
# QUERY FUNCTION
# ==========================
def ask_rag(query: str):
    # Step 1: Retrieve context
    results = vectorstore.similarity_search(query, k=TOP_K)

    if not results:
        return "No relevant context found.", ""

    context = "\n\n".join([r.page_content for r in results])

    # Step 2: Build prompt
    prompt = f"""You are an AI assistant answering questions based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

    # Step 3: Query LLM
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
    return answer.strip(), results


# ==========================
# STREAMLIT APP
# ==========================
st.set_page_config(page_title="📘 Local RAG Chatbot", layout="centered")

st.title("📘 Local RAG Chatbot")
st.caption("Using ChromaDB + Local LLM via Model Runner")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
query = st.chat_input("Ask a question about your documents...")

# Handle user input
if query:
    with st.spinner("Thinking..."):
        try:
            answer, sources = ask_rag(query)
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Show latest exchange
            st.chat_message("user").write(query)
            st.chat_message("assistant").write(answer)

            # Optional: Show retrieved sources
            with st.expander("🔍 Retrieved Chunks"):
                for i, r in enumerate(sources, 1):
                    st.markdown(f"**Result {i}**: {r.metadata}")
                    st.write(r.page_content)
                    st.markdown("---")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# Display previous chat history
#for msg in st.session_state.chat_history:
#    st.chat_message(msg["role"]).write(msg["content"])
