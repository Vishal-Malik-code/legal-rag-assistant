"""
Utility functions for retrieving relevant legal context from the FAISS vector store.
"""

import os
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FAISS_INDEX_PATH = "./rag_faiss_store"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once, on first use."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _load_vector_store(index_mtime: float) -> FAISS:
    """
    Load the FAISS index from disk.

    The index modification time is part of the cache key, so the store
    is reloaded automatically whenever a new PDF has been indexed.
    """
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        get_embeddings(),
        # The index is generated locally by rag_index_builder.py.
        allow_dangerous_deserialization=True,
    )


def _format_source(metadata: dict) -> str:
    """Build a readable source label, e.g. 'agreement.pdf (page 3)'."""
    source = metadata.get("source", "Unknown Source")
    page = metadata.get("page")

    return f"{source} (page {page})" if page else source


def retrieve_legal_context(
    query: str,
    top_k: int = 3,
    debug: bool = False,
    vector_store: FAISS | None = None,
):
    """
    Retrieve the most relevant document chunks for a user query.

    Args:
        query: User's legal question.
        top_k: Number of similar chunks to retrieve.
        debug: Print retrieved chunks to the console.
        vector_store: In-memory index to search. If omitted, the index
            saved on disk is used.

    Returns:
        tuple:
            context (str): Combined retrieved text.
            sources (list[str]): Unique source labels (file name and page).
    """

    if vector_store is None:
        index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
        vector_store = _load_vector_store(os.path.getmtime(index_file))

    docs = vector_store.similarity_search(
        query,
        k=top_k,
    )

    if debug:
        print("\n========== Retrieved Chunks ==========\n")

        for index, doc in enumerate(docs, start=1):
            print(f"Chunk {index}")
            print(f"Source: {_format_source(doc.metadata)}")
            print("-" * 80)
            print(doc.page_content)
            print("-" * 80)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    sources = sorted(
        {_format_source(doc.metadata) for doc in docs}
    )

    return context, sources


if __name__ == "__main__":

    sample_question = (
        "What are the key terms and conditions "
        "of the rental agreement?"
    )

    context, sources = retrieve_legal_context(
        sample_question,
        debug=True,
    )

    print("\n========== Context ==========\n")
    print(context)

    print("\n========== Sources ==========\n")

    for source in sources:
        print(source)