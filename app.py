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

st.title("RAG Document Q&A with GROQ + HuggingFace Embeddings")

# ----------------------------------------------------
# CACHED RESOURCES (Heavy operations cached ONCE)
# ----------------------------------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


@st.cache_resource
def load_documents():
    pdf_dir = os.path.join(os.path.dirname(__file__), "Research_papers")
    loader = PyPDFDirectoryLoader(pdf_dir)
    return loader.load()


@st.cache_resource
def split_documents(_documents):  # underscore to avoid hashing error
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(_documents)


@st.cache_resource
def load_vector_store():
    embeddings = load_embeddings()
    documents = load_documents()
    chunks = split_documents(documents)

    # FAISS works in RAM & is Cloud-friendly
    return FAISS.from_documents(chunks, embeddings)


@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        cache=False
    )


# ----------------------------------------------------
# RAG LLM + Prompt + Retrieval Chain
# ----------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are a helpful and accurate AI assistant.

<context>
{context}
</context>

Question: {input}

Answer:
""")

embeddings = load_embeddings()
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
# UI — Ask Question
# ----------------------------------------------------

question = st.text_input("Ask any question:")

if st.button("Submit") and question:
    with st.spinner("🔍 Retrieving answer..."):
        response = rag_chain.invoke(question)
    st.write("### ✅ Answer")
    st.write(response.content)
