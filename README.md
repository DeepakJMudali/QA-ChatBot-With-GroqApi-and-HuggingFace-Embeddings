# RAG Document Q&A Chatbot

A lightweight Retrieval-Augmented Generation (RAG) Document Q&A Chatbot built with LangChain, GROQ (ChatGroq model), and OpenAI embeddings. This project loads PDF research papers from the `Research_papers/` directory, creates vector embeddings (FAISS), and exposes a simple Streamlit UI in `app.py` for interactive Q&A.

---

## 🚀 Project Overview

- Loads PDF files from `Research_papers/` using `PyPDFDirectoryLoader`.
- Splits documents into chunks with `RecursiveCharacterTextSplitter`.
- Creates embeddings with OpenAI (or compatible) embeddings.
- Stores vectors in FAISS for retrieval.
- Connects a `retriever` with a chain (prompt -> LLM) using `ChatGroq` LLM (GROQ).
- Hosts a simple Streamlit UI for asking questions.

---

## ✅ Features

- Simple Streamlit front-end for question answering.
- PDF batch ingestion and chunking for accurate retrieval.
- Vector index (FAISS) for fast semantic search.
- Configurable model keys via `.env` file.

---

## Prerequisites

- Python 3.10+ (recommend 3.10 or 3.11)
- Windows PowerShell (or bash) to use the setup commands below
- An OpenAI API key (if using OpenAI embeddings)
- A GROQ API key (for the ChatGroq LLM)

---

## ⚙️ Setup & Installation

1. Clone or open this repository in your local environment.

2. Create / activate a virtual environment (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

> Note: Some packages (like FAISS) can be difficult to install on Windows via pip. If you encounter issues, consider using conda or follow FAISS installation instructions for Windows or use `faiss-cpu` prebuilt wheels.

---

## 🔐 Environment Variables

Create a `.env` file in the repository root (there is a `.env` file present; if needed update it) and add the keys:

```
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

The app will load these keys using `python-dotenv`.

---

## 📁 Add Your Documents

Drop PDF files into the `Research_papers/` directory. The app loads the PDFs automatically using `PyPDFDirectoryLoader`.

---

## ▶️ Run the App (PowerShell)

```powershell
# Activate venv if not active
venv\Scripts\Activate.ps1
# Run streamlit app
streamlit run app.py
```

After running, open `http://localhost:8501` in your browser to interact with the UI.

---

## 🔧 Customization

- Change the model or model options in `app.py`: the ChatGroq instantiation uses `model_name=` and you can adjust settings or replace the LLM.
- Modify the text chunk size or overlap in `RecursiveCharacterTextSplitter` by updating `chunk_size` and `chunk_overlap` values.
- The app currently uses the first 50 documents (`st.session_state["documents"][:50]`) to limit the scale. Update this to process more docs if needed.

---

## 📌 Troubleshooting & Tips

- Missing API keys: confirm `.env` contains `GROQ_API_KEY` and `OPENAI_API_KEY`. If not, the app will throw errors when calling LLM or embeddings.
- Embedding/FAISS errors: confirm `faiss-cpu` is installed successfully. On Windows, `pip` may fail; prefer conda for those packages.
- If your PDF text extraction is imperfect, consider using `pymupdf`, `pypdf`, or another loader/cleaner to preprocess PDFs.

---

## 🧪 Example

1. Place PDFs into `Research_papers/`.
2. Run `streamlit run app.py`.
3. Type a question in the UI and click "Submit".

---

## ✍️ Contributing

Pull requests and issues are welcome. If you add features, please update `README.md` with example usage and tests where applicable.

---

## 📄 License

Use this repository per the license chosen by the project owner (no LICENSE file present by default). Add a license file if you intend to publish or share widely.

---

## Author

Repository owner: local workspace author (update if you want an author name)


If you want a different filename (e.g., `.readme` or `README.rst`) or more/less detail, tell me what to include and I will revise the file.