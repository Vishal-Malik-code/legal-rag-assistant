"""
Streamlit application for the Legal RAG Assistant.

Features:
- Upload a legal PDF.
- Build a FAISS vector index.
"""

import tempfile

import streamlit as st

from rag_index_builder import build_index_from_pdf

FAISS_DIR = "./rag_faiss_store"

st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
    <div style="
        background: linear-gradient(135deg,#e3f2fd,#fce4ec);
        padding:25px;
        border-radius:15px;
        text-align:center;
    ">
        <h1>⚖️ Legal RAG Assistant</h1>
        <p>
            Ask questions about legal documents using
            <b>Retrieval-Augmented Generation (RAG)</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

uploaded_file = st.file_uploader(
    "📄 Upload a Legal PDF",
    type=["pdf"],
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    st.success(f"✅ Uploaded: {uploaded_file.name}")

    try:
        with st.spinner("Building FAISS index..."):
            build_index_from_pdf(
                pdf_path,
                persist_dir=FAISS_DIR,
                source_name=uploaded_file.name,
            )

        st.success("✅ Document indexed successfully!")

    except Exception as e:
        st.error(f"Failed to build index.\n\n{e}")
        st.stop()

else:
    st.info("📄 Upload a legal PDF to begin.")
