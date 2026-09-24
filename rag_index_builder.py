"""
Builds a FAISS vector index from a legal PDF.

Workflow:
1. Extract text from the PDF.
"""

import fitz  # PyMuPDF


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


if __name__ == "__main__":
    print(extract_text_from_pdf("./docs/sample_rental_agreement.pdf")[:500])
