# PDF-RAG-with-ASTRA-DB
📄 Chat with Your PDF (RAG App using Groq + AstraDB)

A Retrieval-Augmented Generation (RAG) based web application built with Streamlit, allowing users to upload PDFs and ask questions from their content. The app processes documents, stores embeddings in AstraDB, and retrieves relevant context to generate accurate answers using Groq LLM.

🚀 Features

📤 Upload and process PDF files

🧠 Ask questions from document content

⚡ Fast inference using Groq LLM

🗄️ Vector storage using AstraDB (Cassandra)

🔍 Context-aware answers using RAG

📚 Source document preview

🧾 Chat history tracking

🚀 Live Demo

👉 Try the app here:

https://pdf-rag-with-astra-db-87.streamlit.app/

## 📸 Screenshots

![Output 1](https://raw.githubusercontent.com/Riitwiik/PDF-RAG-with-ASTRA-DB/main/output1.png)

![Output 2](https://raw.githubusercontent.com/Riitwiik/PDF-RAG-with-ASTRA-DB/main/output2.png)

![Output 3](https://raw.githubusercontent.com/Riitwiik/PDF-RAG-with-ASTRA-DB/main/output3.png)


🏗️ Tech Stack

Frontend: Streamlit

LLM: Groq (LLaMA 3.1 8B Instant)

Embeddings: Sentence Transformers (MiniLM)

Vector DB: AstraDB (Cassandra via Cassio)

PDF Parsing: PyPDF2

Framework: LangChain

📂 Project Structure

├── app.py

├── .env

├── requirements.txt

└── README.md

