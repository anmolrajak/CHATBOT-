import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from rag.chat import KnowledgeChatbot

# Constants
VECTORSTORE_DIR = "data/faiss_index"
OLLAMA_MODEL = "mistral"
OLLAMA_URL = os.getenv("OLLAMA_URL")

# Streamlit Page Setup
st.set_page_config(page_title="Knowledge Chatbot", layout="wide")
st.title("💬 Knowledge Chatbot")
st.markdown("Ask questions based on the uploaded documents.")

# Init session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar - Upload
st.sidebar.title("📄 Upload Documents")
uploaded_file = st.sidebar.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

# Process upload
if uploaded_file:
    file_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(file_path) if uploaded_file.name.endswith(".pdf") else TextLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
    new_embedding_dim = len(embeddings.embed_query("test"))

    try:
        if os.path.exists(VECTORSTORE_DIR):
            vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

            existing_dim = vectorstore.index.d
            if existing_dim != new_embedding_dim:
                st.sidebar.error(f"⚠️ Dimension mismatch: index={existing_dim}, new={new_embedding_dim}")
            else:
                vectorstore.add_documents(chunks)
                vectorstore.save_local(VECTORSTORE_DIR)
                st.sidebar.success("✅ Document added to knowledge base!")
        else:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(VECTORSTORE_DIR)
            st.sidebar.success("✅ Knowledge base initialized!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to update FAISS index: {e}")

# Load chatbot
if os.path.exists(VECTORSTORE_DIR):
    try:
        bot = KnowledgeChatbot(faiss_index_path=VECTORSTORE_DIR)
    except AssertionError as e:
        st.error(f"❌ FAISS index failed to load properly: {e}")
        st.stop()
else:
    st.warning("📂 Upload a document to begin using the chatbot.")
    st.stop()

# Chat form
with st.form("chat_form"):
    query = st.text_input("Your question", placeholder="Ask something...")
    submitted = st.form_submit_button("Submit")

if submitted and query:
    try:
        with st.spinner("Thinking..."):
            answer = bot.ask(query)
            st.session_state.chat_history.append(("🧑 You", query))
            st.session_state.chat_history.append(("🤖 Assistant", answer))
    except Exception as e:
        st.error(f"❌ Failed to get response: {e}")

# Chat history
if st.session_state.chat_history:
    st.markdown("---")
    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role}:** {msg}")
