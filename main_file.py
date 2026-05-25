import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load the API key from your local .env file
load_dotenv("key_filename_here")
api_key = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

#UI Configurati
st.set_page_config(page_title="Semantic Document Research", page_icon="🧠", layout="wide")
st.title("🧠 AI Research Workspace")
st.markdown("Upload enterprise datasets, technical documentation, or historical archives for semantic analysis.")

# Session State Management 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Sidebar: Configuration & Upload
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    if api_key:
        st.success("API Key successfully loaded from local .env file.")
    else:
        st.error("API Key not found! Please check your .env file.")
    
    st.markdown("---")
    st.subheader("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload files (PDF/TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    with st.expander("Advanced Vector Settings"):
        chunk_size = st.slider("Chunk Size", 200, 2000, 800)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100)

    if st.button("Process Documents", type="primary"):
        if not api_key:
            st.error("Please ensure your .env file contains a valid GOOGLE_API_KEY.")
        elif not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            os.environ["GOOGLE_API_KEY"] = api_key
            
            with st.spinner("Initializing neural embeddings and vectorizing documents..."):
                documents = []
                for uploaded_file in uploaded_files:
                    ext = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_path = temp_file.name
                    
                    try:
                        if ext.lower() == ".pdf":
                            loader = PyPDFLoader(temp_path)
                        else:
                            loader = TextLoader(temp_path)
                        documents.extend(loader.load())
                    finally:
                        os.remove(temp_path)

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                splits = text_splitter.split_documents(documents)
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings
                )
                st.session_state.vectorstore = vectorstore
                
                st.success(f"Success! Embedded {len(splits)} chunks into ChromaDB.")
                
    st.markdown("---")
    st.caption("Built with Custom Orchestration, Chroma, and Gemini 2.5")

#Main Chat Interface
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View Source Documents"):
                for i, doc in enumerate(msg["sources"]):
                    st.markdown(f"**Source {i+1}:**\n```text\n{doc.page_content}\n```")

# Chat Input
query = st.chat_input("e.g., Are the Luni classified under the Miana or Panni lineages in these administrative records?")

if query:
    if not st.session_state.vectorstore:
        st.error("Please upload and process documents first.")
    elif not api_key:
        st.error("API Key is missing.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Querying vector database and synthesizing response..."):
                os.environ["GOOGLE_API_KEY"] = api_key
                
                # 1. Initialize the LLM
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                
                # 2. Retrieve the raw documents manually
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                source_docs = retriever.invoke(query)
                
                # 3. Format the documents into a single text block
                context_text = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(source_docs)])
                
                # 4. Inject the raw context and query into the prompt template
                prompt = PromptTemplate.from_template(
                    """You are an expert research assistant. Answer the question based ONLY on the provided context.
                    If you cannot find the answer in the context, explicitly state that.
                    
                    <context>
                    {context}
                    </context>
                    
                    Question: {question}
                    Answer:"""
                )
                
                formatted_prompt = prompt.format(context=context_text, question=query)
                
                # 5. Pass the final formatted string directly to Gemini
                response = llm.invoke(formatted_prompt)
                answer = response.content
                
                # Display output
                st.markdown(answer)
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Source {i+1}:**\n```text\n{doc.page_content}\n```")
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": source_docs
                })