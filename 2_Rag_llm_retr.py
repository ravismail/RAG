import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import HttpClient
from langchain_community.vectorstores import Chroma

# --- CONFIG ---

PDF_PATH = "C:\\Users\\ravis\\Documents\\langchain\\Course\\Sampleproject\\RAG\\Terraform.pdf"  # Path to your PDF file
CHROMA_URL = "http://localhost:8000"
COLLECTION_NAME = "pdf_new-collection"
MODEL_RUNNER_URL = "http://localhost:12434/engines/v1"
MODEL_ID = "ai/nomic-embed-text-v1.5"

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
print(f"✅ Loaded {len(docs)} pages from PDF")

# --- SPLIT CHUNKS ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"✅ Split into {len(chunks)} chunks")

# --- INIT EMBEDDINGS ---
embeddings = CustomRESTEmbeddings(base_url=MODEL_RUNNER_URL, model=MODEL_ID)

# --- CONNECT TO REMOTE CHROMA (Docker) ---
client = HttpClient(host="localhost", port=8000)
vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)
print("✅ Connected to ChromaDB running in Docker")

# --- STORE DOCUMENTS ---
vectorstore.add_documents(chunks)
print(f"✅ Stored {len(chunks)} chunks in ChromaDB!")

# --- TEST SEARCH ---
query = "What is the main topic of this PDF?"
results = vectorstore.similarity_search(query, k=2)

print("\n🔍 Top results:")
for r in results:
    print(f"- {r.page_content[:200]}...")
    print(f"  Metadata: {r.metadata}\n")
