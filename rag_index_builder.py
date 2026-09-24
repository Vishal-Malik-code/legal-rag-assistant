"""
Builds a FAISS vector index from a legal PDF.

Workflow:
1. Extract text from the PDF.
2. Split the text into overlapping chunks.
3. Generate embeddings using Hugging Face.
4. Store the embeddings in a FAISS vector database.
"""

import os

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_PERSIST_DIR = "./rag_faiss_store"


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        Extracted text as a single string.
    """
    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


def build_index_from_pdf(
    pdf_path: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    source_name: str | None = None,
) -> None:
    """
    Build and save a FAISS index from a PDF.

    Args:
        pdf_path: Path to the PDF.
        persist_dir: Directory where the FAISS index is stored.
        source_name: Original PDF filename (used in metadata).
    """

    full_text = extract_text_from_pdf(pdf_path)

    if not full_text.strip():
        raise ValueError("The uploaded PDF does not contain extractable text.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    documents = text_splitter.create_documents(
        texts=[full_text],
        metadatas=[
            {
                "source": source_name or os.path.basename(pdf_path)
            }
        ],
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings,
    )

    os.makedirs(persist_dir, exist_ok=True)

    vector_store.save_local(persist_dir)


if __name__ == "__main__":
    build_index_from_pdf("./docs/sample_rental_agreement.pdf")