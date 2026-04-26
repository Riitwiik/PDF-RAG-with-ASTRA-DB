import streamlit as st
from dotenv import load_dotenv
import os
import cassio
from PyPDF2 import PdfReader
import re
import hashlib

from langchain_community.vectorstores.cassandra import Cassandra
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------ CONFIG ------------------
st.set_page_config(page_title="PDF Q&A App", layout="wide")
st.title("📄 Chat with your PDF (Groq + AstraDB)")

# ------------------ LOAD ENV ------------------
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not ASTRA_DB_APPLICATION_TOKEN or not ASTRA_DB_ID or not GROQ_API_KEY:
    st.error("❌ Missing environment variables.")
    st.stop()

# ------------------ INIT DB ------------------
@st.cache_resource
def init_db():
    cassio.init(
        token=ASTRA_DB_APPLICATION_TOKEN,
        database_id=ASTRA_DB_ID
    )

init_db()

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    return embedding, llm

embedding, llm = load_models()

# ------------------ GLOBAL VECTOR STORE ------------------
@st.cache_resource
def get_vector_store():
    return Cassandra(
        embedding=embedding,
        table_name="pdf_documents"   
    )

vector_store = get_vector_store()

# ------------------ SESSION INIT ------------------
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # ------------------ PROCESS PDF ------------------
    if file_hash not in st.session_state.processed_files:

        with st.spinner("Processing PDF..."):

            try:
                # READ PDF
                pdfreader = PdfReader(uploaded_file)
                raw_text = ""

                for page in pdfreader.pages:
                    content = page.extract_text()
                    if content:
                        raw_text += content

                if not raw_text.strip():
                    st.error("❌ Could not extract text from PDF.")
                    st.stop()

                # CLEAN TEXT
                raw_text = re.sub(r'\s+', ' ', raw_text)

                # SPLIT TEXT
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=200
                )
                texts = splitter.split_text(raw_text)

                # METADATA (IMPORTANT)
                metadatas = [{"file_id": file_hash} for _ in texts]

                # INSERT IN BATCHES
                batch_size = 32
                for i in range(0, len(texts), batch_size):
                    vector_store.add_texts(
                        texts[i:i + batch_size],
                        metadatas=metadatas[i:i + batch_size]
                    )

                # SAVE STATE
                st.session_state.processed_files.add(file_hash)
                st.session_state.current_file = file_hash

                st.success("✅ PDF processed successfully!")

            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.stop()

    else:
        st.info("📌 PDF already processed. Ready to query!")

    # ------------------ RETRIEVER ------------------
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 6,
            "filter": {"file_id": file_hash}   
        }
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # ------------------ CHAT ------------------
    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Thinking..."):
            try:
                result = qa({"query": query})

                answer = result["result"]
                sources = result["source_documents"]

                st.session_state.chat_history.append((query, answer))

                st.subheader("💡 Answer")
                st.write(answer)

                with st.expander("📄 Sources"):
                    for doc in sources:
                        st.write(doc.page_content[:300] + "...")

            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")

    # ------------------ CHAT HISTORY ------------------
    if st.session_state.chat_history:
        st.subheader("🧠 Chat History")
        for q, a in st.session_state.chat_history[::-1]:
            st.write(f"**Q:** {q}")
            st.write(f"**A:** {a}")