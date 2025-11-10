# --- Ingest Script for RAG with Ollama and ChromaDB ---
# This script loads a PDF, splits it into chunks, generates embeddings using Ollama,    
# and stores them in a ChromaDB vector store running in Docker.
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from chromadb import HttpClient
from time import sleep

# --- Configuration ---
# Connect to services running in Docker exposed on localhost
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text" # Must be pulled in docker-compose
COLLECTION_NAME = "tr_rag_knowledge"
PDF_PATH = "C:\\Users\\ravis\\Documents\\langchain\\Course\\Sampleproject\\RAG\\Terraformnotes.pdf" # <-- Name of the PDF you must provide

def ingest_documents():
    """
    Loads a PDF, chunks it, generates embeddings using Ollama, and stores them in ChromaDB.
    """
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file '{PDF_PATH}' not found. Please place it in the same directory.")
        return

    print(f"--- Starting Ingestion Process for {PDF_PATH} ---")

    # 1. Load the PDF document
    print("Loading documents...")
    try:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return

    # 2. Split the documents into chunks
    print(f"Splitting {len(documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    # 3. Define Ollama Embeddings
    print(f"Initializing OllamaEmbeddings using model: {EMBEDDING_MODEL}...")
    # The OllamaEmbeddings object handles connecting to the Ollama container
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
    )

    # 4. Create or Connect to the ChromaDB vector store
    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT} and ingesting data...")
    # Chroma.from_documents connects to a remote HTTP client

  
    try:
        chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            client=chroma_client,
            # Pass connection details to Chroma client
       

        )
        print(f"✅ Connected to Chroma at {CHROMA_HOST}:{CHROMA_PORT}")
        print("\n--- SUCCESS ---")
        print(f"Documents successfully chunked and embedded into ChromaDB Collection: '{COLLECTION_NAME}'")
        print(f"Vector Store Status: {vector_store._client.heartbeat()}")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Failed to connect or ingest data to ChromaDB. Ensure Docker services are running: {e}")
        print("Waiting 10 seconds and retrying to allow services to warm up...")
        sleep(10)
        # Simple retry attempt (or you can manually run this script again)
        try:
             vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                client_kwargs={
                    "host": CHROMA_HOST,
                    "port": CHROMA_PORT,
                    "settings": Settings(
                        chroma_server_host=CHROMA_HOST,
                        chroma_server_http_port=CHROMA_PORT,
                    ),
                }
            )
             print("\n--- SUCCESS (Retry) ---")
             print(f"Documents successfully chunked and embedded into ChromaDB Collection: '{COLLECTION_NAME}'")

        except Exception as retry_e:
            print(f"Retry failed: {retry_e}")


if __name__ == "__main__":
    ingest_documents()

# Steps to run:
# 1. Start Docker services: docker-compose up -d
# 2. Place your document.pdf in this directory.
# 3. Run this script: python ingest.py
