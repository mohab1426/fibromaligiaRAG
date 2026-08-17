"""Core RAG pipeline logic for the Fibromyalgia article.

This module factors out the pipeline steps from the notebook
(`notebooks/fibromyalgia_rag_pipeline.ipynb`) so they can be reused by the
Gradio app (`src/app.py`) without duplicating code. The notebook remains the
primary, documented walkthrough; this module is the "library" version of the
same logic.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = DATA_PROCESSED_DIR / "faiss_index"
PDF_PATH = DATA_RAW_DIR / "biomedicines-12-01543.pdf"

SECTIONS = [
    "1. Introduction",
    "2. Epidemiology",
    "3. Physiopathology",
    "4. Etiopathogenesis",
    "5. Diagnosis",
    "6. Treatment",
    "7. Conclusions",
]

METADATA = {
    "title": (
        "Fibromyalgia: A Review of the Pathophysiological Mechanisms and "
        "Multidisciplinary Treatment Strategies"
    ),
    "authors": [
        "Lina Noelia Jurado-Priego",
        "Cristina Cueto-Ureña",
        "María Jesús Ramírez-Expósito",
        "José Manuel Martínez-Martos",
    ],
    "source": PDF_PATH.name,
    "journal": "Biomedicines",
    "year": 2024,
}


def extract_pdf_text(pdf_path: Path = PDF_PATH) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    raw_text = "".join(page.get_text() + "\n" for page in doc)
    doc.close()
    return raw_text


def clean_pdf_text(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"-\s*\n\s*", "", cleaned)
    cleaned = re.sub(r"Biomedicines 2024, 12, 1543\.?", "", cleaned)
    cleaned = re.sub(r"https://doi\.org/\S+", "", cleaned)
    cleaned = re.sub(r"https://www\.mdpi\.com/journal/biomedicines", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\b\d+\s+of\s+22\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_sections(clean_text: str) -> dict[str, str]:
    main_text = clean_text.split("References")[0]
    parsed_sections = {}
    for i, section in enumerate(SECTIONS):
        start = main_text.find(section)
        if start == -1:
            raise ValueError(f"Section heading not found in article text: {section!r}")
        end = main_text.find(SECTIONS[i + 1]) if i + 1 < len(SECTIONS) else len(main_text)
        parsed_sections[section] = main_text[start:end].strip()
    return parsed_sections


def build_documents(parsed_sections: dict[str, str]) -> list[Document]:
    return [
        Document(page_content=content, metadata={**METADATA, "section": section})
        for section, content in parsed_sections.items()
    ]


def chunk_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    bad_chunks = [c for c in chunks if re.search(r"\b\d+\s+of\s+22\b", c.page_content)]
    assert not bad_chunks, "Text cleaning regression: page numbers leaked into chunks"
    return chunks


def build_vectorstore(chunks: list[Document]):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, embeddings


def get_retriever(k: int = 3, force_rebuild: bool = False):
    """Load a cached FAISS index from disk if present, otherwise build it
    from the source PDF and cache it for next time."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if INDEX_DIR.exists() and not force_rebuild:
        vectorstore = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
    else:
        if not PDF_PATH.exists():
            raise FileNotFoundError(
                f"Source PDF not found at {PDF_PATH}. Place "
                "'biomedicines-12-01543.pdf' in data/raw/ first."
            )
        raw_text = extract_pdf_text()
        clean_text = clean_pdf_text(raw_text)
        parsed_sections = parse_sections(clean_text)
        documents = build_documents(parsed_sections)
        chunks = chunk_documents(documents)
        vectorstore, embeddings = build_vectorstore(chunks)
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))

    return vectorstore.as_retriever(search_kwargs={"k": k})


def evaluate_retrieval(eval_dataset: list[dict], retriever) -> float:
    relevant_count = 0
    for item in eval_dataset:
        docs = retriever.invoke(item["question"])
        retrieved_text = " ".join(d.page_content for d in docs)
        if any(kw.lower() in retrieved_text.lower() for kw in item["keywords"]):
            relevant_count += 1
    return (relevant_count / len(eval_dataset)) * 100


def generate_answer(question: str, retriever, model: str = "gpt-4o-mini") -> tuple[str, list[Document]]:
    """Answer `question` using retrieved context (RAG).

    Returns (answer_text, retrieved_docs) so callers (e.g. the Gradio app)
    can display both the answer and its supporting context.
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[Section: {d.metadata['section']}]\n{d.page_content}" for d in docs)

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        system_prompt = (
            "You are a biomedical research assistant. Answer the question "
            "using ONLY the provided context from the article. If the answer "
            "is not contained in the context, say so explicitly. Cite the "
            "section name(s) you drew on."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0,
        )
        return response.choices[0].message.content, docs

    sections_used = ", ".join(sorted({d.metadata["section"] for d in docs}))
    preview = "\n\n".join(d.page_content.strip() for d in docs)
    answer = (
        f"[Extractive fallback - no OPENAI_API_KEY set]\n"
        f"Most relevant sections: {sections_used}\n\n"
        f"Retrieved context:\n{preview}"
    )
    return answer, docs
