import streamlit as st
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()

# ----------------------------
# PAGE CONFIG + SIMPLE ELEGANT UI
# ----------------------------
st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.markdown("""
<style>

body {
    background-color: #0f1217; /* Clean dark mode */
}

/* Title */
.center-title {
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    color: #dee2e6;
    padding-bottom: 20px;
}

/* Remove ALL unwanted white boxes */
div.block-container {
    padding-top: 3rem;
}

/* Clean input style */
input[type="text"] {
    background: #1c1f26 !important;
    border: 1px solid #444 !important;
    color: white !important;
    padding: 14px !important;
    border-radius: 10px !important;
    font-size: 17px !important;
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    background-color: #4dabf7;
    color: white;
    height: 45px;
    font-size: 17px;
    font-weight: 600;
    border: none;
}
.stButton>button:hover {
    background-color: #339af0;
}

/* Answer panel */
.answer-box {
    padding: 1.5rem;
    background: #1c1f26;
    border-radius: 14px;
    border-left: 4px solid #4dabf7;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
    margin-top: 20px;
    color: #e9ecef;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# TITLE
# ----------------------------
st.markdown('<div class="center-title">🤖 RAG Document Q&A with GROQ</div>', unsafe_allow_html=True)

# ----------------------------
# CACHED RESOURCES
# ----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_documents():
    pdf_dir = os.path.join(os.path.dirname(__file__), "Research_papers")
    loader = PyPDFDirectoryLoader(pdf_dir)
    return loader.load()

@st.cache_resource
def split_documents(_documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(_documents)

@st.cache_resource
def load_vector_store():
    embeddings = load_embeddings()
    documents = load_documents()
    chunks = split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)

@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

# ----------------------------
# RAG PIPELINE
# ----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a friendly and helpful AI assistant.

Use the information from the following context to answer the user's question.
If the context contains useful information, prioritize it.
If not, answer using general knowledge but keep it accurate.

Keep your answer:
- Simple
- Clear
- Helpful
- Human-like

<context>
{context}
</context>

User Question: {input}

Answer:
""")


vector_store = load_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
llm = load_llm()

rag_chain = (
    RunnableParallel({
        "context": retriever,
        "input": RunnablePassthrough()
    })
    | prompt
    | llm
)

# ----------------------------
# INPUT UI
# ----------------------------

st.markdown('<div class="input-card">', unsafe_allow_html=True)
question = st.text_input("Ask any question")
submit = st.button("Submit")
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# ANSWER
# ----------------------------
if submit and question:
    with st.spinner("🔍 Retrieving answer..."):
        response = rag_chain.invoke(question)


    st.write("### 📘 Answer")
    st.write(response.content)
    st.markdown('</div>', unsafe_allow_html=True)
