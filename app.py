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

# Load ENV variables
load_dotenv()

# ----------------------------
# PAGE CONFIG & CUSTOM CSS
# ----------------------------
st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.markdown("""
<style>
.input-card {
    padding: 1.5rem;
    background: #f8f9fa;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}
.answer-box {
    padding: 1.5rem;
    background: white;
    border-radius: 12px;
    border: 1px solid #eaeaea;
    box-shadow: 0px 1px 4px rgba(0,0,0,0.05);
}
.center-title {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    padding-top: 10px;
    padding-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# TITLE
# ----------------------------
st.markdown('<div class="center-title">🤖 RAG Document Q&A with GROQ</div>', unsafe_allow_html=True)


# ----------------------------------------------------
# CACHED RESOURCES
# ----------------------------------------------------

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
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )


# ----------------------------------------------------
# RAG Chain
# ----------------------------------------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful and accurate AI assistant.

<context>
{context}
</context>

Question: {input}

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

# ----------------------------------------------------
# INPUT UI
# ----------------------------------------------------
st.markdown('<div class="input-card">', unsafe_allow_html=True)
question = st.text_input("Ask any question from the research papers:")
submit = st.button("Submit", type="primary")
st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------
# ANSWER OUTPUT
# ----------------------------------------------------
if submit and question:
    with st.spinner("🔍 Retrieving answer..."):
        response = rag_chain.invoke(question)

    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    st.write("### ✅ Answer")
    st.write(response.content)
    st.markdown('</div>', unsafe_allow_html=True)
