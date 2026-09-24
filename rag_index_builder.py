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


def extract_pages_from_pdf(pdf_path: str) -> list[str]:
    """
    Extract the text of every page in a PDF.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        List with one string per page (page 1 is index 0).
    """
    with fitz.open(pdf_path) as document:
        return [page.get_text() for page in document]


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

    pages = extract_pages_from_pdf(pdf_path)

    if not any(page.strip() for page in pages):
        raise ValueError("The uploaded PDF does not contain extractable text.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    source = source_name or os.path.basename(pdf_path)

    # Split page by page so every chunk keeps its page number.
    documents = text_splitter.create_documents(
        texts=pages,
        metadatas=[
            {"source": source, "page": page_number}
            for page_number in range(1, len(pages) + 1)
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