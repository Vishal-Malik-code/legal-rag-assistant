"""
Command-line interface for the Legal RAG Assistant.

Workflow:
1. Retrieve relevant legal context from the FAISS index.
2. Send the retrieved context to Gemini.
3. Display the generated answer along with the source documents.
"""

import os

from dotenv import load_dotenv
from google import genai

from tools import retrieve_legal_context

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3.6-flash"

if not API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found. Please configure your .env file."
    )

client = genai.Client(api_key=API_KEY)


def ask_legal_assistant(question: str) -> tuple[str, list[str]]:
    """Retrieve relevant legal context and generate an answer using Gemini."""

    context, sources = retrieve_legal_context(question)

    prompt = f"""
You are an expert legal assistant.

Use ONLY the legal context below to answer the user's question.

Instructions:
- Do not invent or assume information.
- If the answer is not contained in the context, respond:
  "I couldn't find this information in the provided document."
- Keep the answer concise, accurate, and easy to understand.
- Summarize the information instead of copying it verbatim.

-------------------------
Legal Context
-------------------------
{context}

-------------------------
Question
-------------------------
{question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text, sources


if __name__ == "__main__":
    answer, sources = ask_legal_assistant(
        "What are the key terms of the agreement?"
    )
    print(answer)
    print("Sources:", ", ".join(sources))
