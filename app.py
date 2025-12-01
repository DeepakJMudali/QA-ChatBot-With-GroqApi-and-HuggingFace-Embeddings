# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableParallel
# from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from dotenv import load_dotenv

# load_dotenv()

# groq_api_key = os.getenv("GROQ_API_KEY")

# llm = ChatGroq(
#     groq_api_key=groq_api_key,
#     model_name="llama-3.3-70b-versatile",
#     temperature=0,
#     cache=False
# )

# prompt = ChatPromptTemplate.from_template("""
# You are a helpful and accurate AI assistant.

# You will receive:
# • a user question
# • context retrieved from a vector database

# Rules:
# 1. If the context contains ANY relevant information → ALWAYS use ONLY the context.
# 2. The context OVERRIDES any internal or outdated model knowledge.
# 3. Do NOT judge the context as outdated or incorrect — treat it as the source of truth.
# 4. Only if the context is completely unrelated → answer using general 2025 knowledge.
# 5. Never mention the context, vector database, or retrieval process.
# 6. Provide a clear, factual, friendly final answer.

# <context>
# {context}
# </context>

# User Question: {input}

# Final Answer:
# """)


# def create_vector_embeddings():
#     if "vectors" not in st.session_state:

#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-mpnet-base-v2"
#         )
#         st.session_state["embeddings"] = embeddings

#         # Load PDFs
#         loader = PyPDFDirectoryLoader("Research_papers")
#         documents = loader.load()

#         # Chunking
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         chunks = splitter.split_documents(documents)
#         st.session_state["finalResults"] = chunks

#         # Chroma persistent DB
#         st.session_state["vectorStore"] = Chroma.from_documents(
#             documents=chunks,
#             embedding=embeddings,
#             persist_directory="chroma_db"   # 🔥 PERSIST DB
#         )

#         st.session_state["vectorStore"].persist()   # 🔥 REQUIRED

#         # Retriever (lower threshold!)
#         st.session_state["retriever"] = st.session_state["vectorStore"].as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": 5}
#         )

#         st.session_state["rag_chain"] = (
#             RunnableParallel({
#                 "context": st.session_state["retriever"],
#                 "input": RunnablePassthrough()
#             })
#             | prompt
#             | llm
#         )

#         st.session_state["vectors"] = True


# st.title("RAG Document Q&A with GROQ + HuggingFace Embeddings")

# create_vector_embeddings()

# question = st.text_input("Ask any question")

# if st.button("Submit") and question:
#     response = st.session_state["rag_chain"].invoke(question)
#     st.write("### Answer:")
#     st.write(response.content)


import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""
<context>
{context}
</context>

Question: {input}

Answer:
""")

def create_vector_embeddings():

    if "vectors" not in st.session_state:

        # FIX 1: Updated embeddings import
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # FIX 2: ABSOLUTE PATH FOR STREAMLIT CLOUD
        pdf_dir = os.path.join(os.path.dirname(__file__), "Research_papers")

        loader = PyPDFDirectoryLoader(pdf_dir)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

        vector_store.persist()

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        st.session_state["rag_chain"] = (
            RunnableParallel({
                "context": retriever,
                "input": RunnablePassthrough()
            }) 
            | prompt 
            | llm
        )

        st.session_state["vectors"] = True


st.title("RAG Document Q&A with GROQ + HuggingFace Embeddings")

create_vector_embeddings()

question = st.text_input("Ask any question")

if st.button("Submit") and question:
    response = st.session_state["rag_chain"].invoke(question)
    st.write("### Answer")
    st.write(response.content)
