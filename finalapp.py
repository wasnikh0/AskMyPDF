import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

# ----------------------
# API KEY: Secure Handling
# ----------------------
# Automatically loads secrets from .streamlit/secrets.toml
if "NVIDIA_API_KEY" in st.secrets:
    os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]
else:
    st.error("❌ NVIDIA API Key not found in Streamlit secrets.")
    st.stop()

# ----------------------
# Initialize LLM
# ----------------------
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# ----------------------
# Embedding Function
# ----------------------
def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state and uploaded_files:
        st.session_state.embeddings = NVIDIAEmbeddings()
        docs = []

        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(uploaded_file.name)
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        final_documents = splitter.split_documents(docs)
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        st.session_state.documents = final_documents

# ----------------------
# Streamlit UI
# ----------------------
st.title('🧠 AskMyPdf')
uploaded_files = st.file_uploader("📎 Upload your PDF files", type="pdf", accept_multiple_files=True)

if st.button("📌 Embed Documents"):
    if uploaded_files:
        vector_embedding(uploaded_files)
        st.success("✅ Vector DB is ready using NVIDIA Embeddings")
    else:
        st.warning("Please upload at least one PDF.")

prompt_text = st.text_input("💬 Ask a question from the uploaded documents")

if prompt_text:
    if "vectors" not in st.session_state:
        st.warning("Please upload and embed your PDFs first.")
    else:
        prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the text only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        Questions: {input}
        """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt_text})
        st.write("🧠 Answer:", response['answer'])
        st.write(f"⏱️ Response Time: {round(time.process_time() - start, 2)}s")

        with st.expander("📄 Document Chunks (Context Used)"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Chunk {i+1}**")
                st.write(doc.page_content)
                st.write("---")
