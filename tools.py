"""
Utility functions for retrieving relevant legal context from the FAISS vector store.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FAISS_INDEX_PATH = "./rag_faiss_store"

# Initialize embeddings once
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


def retrieve_legal_context(query: str, top_k: int = 3):
    """
    Retrieve the most relevant document chunks for a user query.

    Returns:
        tuple:
            context (str): Combined retrieved text.
            sources (list[str]): Unique source document names.
    """

    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    docs = vector_store.similarity_search(query, k=top_k)

    context = "\n\n".join(doc.page_content for doc in docs)

    sources = sorted(
        {doc.metadata.get("source", "Unknown Source") for doc in docs}
    )

    return context, sources
