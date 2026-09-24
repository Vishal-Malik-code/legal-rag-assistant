"""
Streamlit application for the Legal RAG Assistant.

Features:
- Upload a legal PDF.
- Build a FAISS vector index.
- Retrieve relevant legal context.
- Generate answers using Groq.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from rag_index_builder import build_index_from_pdf
from tools import retrieve_legal_context

# Configuration
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-120b"
FAISS_DIR = "./rag_faiss_store"

if not API_KEY:
    st.error("GROQ_API_KEY not found. Please configure your .env file.")
    st.stop()

client = Groq(api_key=API_KEY)

# Streamlit Page Configuration
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
            <b>Retrieval-Augmented Generation (RAG)</b> powered by
            <b>Groq + FAISS + Hugging Face Embeddings</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# PDF Upload
uploaded_file = st.file_uploader(
    "📄 Upload a Legal PDF",
    type=["pdf"],
)

if uploaded_file:
    file_key = (uploaded_file.name, uploaded_file.size)

    # Streamlit reruns the script on every interaction, so only
    # build the index when a new file has been uploaded.
    if st.session_state.get("indexed_file") != file_key:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            pdf_path = temp_file.name

        try:
            with st.spinner("Building FAISS index..."):
                build_index_from_pdf(
                    pdf_path,
                    persist_dir=FAISS_DIR,
                    source_name=uploaded_file.name,
                )

            st.session_state["indexed_file"] = file_key

        except Exception as e:
            st.error(f"Failed to build index.\n\n{e}")
            st.stop()

        finally:
            os.remove(pdf_path)

    st.success(f"✅ {uploaded_file.name} indexed successfully!")

    st.divider()

    # Question Answering
    question = st.text_input("💬 Ask a legal question")

    if question:
        try:
            with st.spinner("Searching document..."):
                context, sources = retrieve_legal_context(question)

                prompt = f"""
You are an expert legal assistant.

Answer ONLY using the legal context provided below.

If the answer is not available in the context,
respond with:
"I couldn't find this information in the uploaded document."

Keep your answer concise, accurate, and professional.

Do not invent facts, clauses, dates, amounts, or obligations
that are not present in the provided legal context.

Legal Context:
{context}

Question:
{question}
"""

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )

            st.subheader("🧠 Answer")
            st.success(response.choices[0].message.content)

            st.subheader("📚 Source Documents")

            if sources:
                for source in sources:
                    st.write(f"• {source}")
            else:
                st.write("No source information available.")

            with st.expander("🔍 Retrieved Context"):
                st.write(context)

        except Exception as e:
            st.error(f"Error while generating response.\n\n{e}")

else:
    st.info("📄 Upload a legal PDF to begin.")