#Working example of a Streamlit app that connects a local LLM (Ollama) with a ChromaDB vector store for a chatbot interface.
import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from chromadb import HttpClient
# -----------------------------
# CONFIG
# -----------------------------
CHROMA_URL = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"  # or llama3, etc.

# -----------------------------
# SETUP
# -----------------------------
st.set_page_config(page_title="Local LLM Chatbot", layout="wide")
st.title("🤖 Local LLM + Chroma Chatbot")

# Initialize session state for history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create embeddings + vector store client
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_URL
    )
#embeddings = OpenAIEmbeddings(
#    model="nomic-embed-text",
#    openai_api_base=OLLAMA_URL,
#    openai_api_key="none"
    #base_url=OLLAMA_URL
#)

# Connect to existing ChromaDB
client = HttpClient(host="localhost", port=8000)
vectorstore = Chroma(
    client=client,
    collection_name="tr_rag_knowledge",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Custom prompt template
prompt_template = """
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Helpful Answer:
"""
PROMPT = PromptTemplate(
    input_variables=["context", "question"], 
    template=prompt_template
)

# Setup LLM
llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# -----------------------------
# UI SECTION
# -----------------------------
with st.chat_message("assistant"):
    st.markdown("Hi there 👋, I’m your local LLM assistant connected to ChromaDB!")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain(user_input)
            answer = response["result"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
