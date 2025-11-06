
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- CONFIG ---
CHROMA_DB_DIR = "./chroma_store"
MODEL_RUNNER_URL = "http://localhost:12434/engines/v1"
MODEL_ID = "ai/nomic-embed-text-v1.5"

CHROMADB_HOST = "localhost"  # Adjust if ChromaDB is on different host
CHROMADB_PORT = 8000  # Default ChromaDB port
PDF_PATH = "C:\\Users\\ravis\\Documents\\langchain\\Course\\Sampleproject\\RAG\\Terraform.pdf"  # Path to your PDF file
COLLECTION_NAME = "pdf_embeddings"
MODEL_NAME = "ai/mxbai-embed-large:latest"  # Embedding model name

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

    def embed_documents(self, texts):
        r = requests.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
        )
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]

# --- LOAD PDF ---
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# --- SPLIT CHUNKS ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# --- INIT EMBEDDINGS + VECTORSTORE ---
embeddings = CustomRESTEmbeddings(base_url=MODEL_RUNNER_URL, model=MODEL_ID)
vectorstore = Chroma(
    collection_name="pdf_embeddings",
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings
)

# --- STORE ---
vectorstore.add_documents(chunks)
print(f"✅ Stored {len(chunks)} chunks in ChromaDB!")


# --- TEST SEARCH ---
query = "What is the main topic of this PDF?"
results = vectorstore.similarity_search(query, k=2)

for r in results:
    print("\n🔍", r.page_content[:200], "...")
    print("Metadata:", r.metadata)
