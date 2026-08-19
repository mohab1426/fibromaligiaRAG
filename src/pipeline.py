"""Core RAG pipeline logic for the Fibromyalgia article.

This module factors out the pipeline steps from the notebook
(`notebooks/fibromyalgia_rag_pipeline.ipynb`) so they can be reused by the
Gradio app (`src/app.py`) without duplicating code. The notebook remains the
primary, documented walkthrough; this module is the "library" version of the
same logic.

v2 - layout-aware parsing. Sections are no longer located by matching a
hardcoded list of heading strings; they are detected from PDF layout
metadata (font weight/style), which generalizes to any MDPI-style review
article using the same `N.` / `N.N.` numbering convention. The reference
list is parsed into individually addressable entries, and each chunk is
sentence-aware and citation-aware (a `[12,45]`-style marker is never split
across two chunks), carrying the list of reference numbers it cites so they
can be resolved back to the original source at answer-generation time.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = DATA_PROCESSED_DIR / "faiss_index"
REFERENCES_PATH = DATA_PROCESSED_DIR / "references.json"
PDF_PATH = DATA_RAW_DIR / "biomedicines-12-01543.pdf"

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

# ---------------------------------------------------------------------------
# Layout-aware extraction
# ---------------------------------------------------------------------------

JUNK_LINE_PATTERNS = [
    r'^Biomedicines\s+\d{4},\s*\d+,?\s*(x FOR PEER REVIEW|\d+)$',  # repeated citation banner
    r'^\d+\s+of\s+\d+$',                                             # "4 of 22" page markers
]

_PAGE_NUM_RE = re.compile(r'\b\d+\s+of\s+\d+\b')


def extract_lines(pdf_path: Path = PDF_PATH) -> list[dict]:
    """Extract text lines with font metadata (layout-aware, not plain text).

    Returns line dicts (page, text, fonts, y) with running headers/footers
    and page-number markers already filtered out.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    lines = []
    for pno, page in enumerate(doc):
        d = page.get_text("dict")
        for block in d["blocks"]:
            if block["type"] != 0:  # skip images
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                lines.append({
                    "page": pno,
                    "text": text,
                    "fonts": {s["font"] for s in spans},
                    "y": line["bbox"][1],
                })
    doc.close()
    return [l for l in lines if not any(re.match(p, l["text"]) for p in JUNK_LINE_PATTERNS)]


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

H1 = re.compile(r'^(\d{1,2})\.\s+(.+)$')
H2 = re.compile(r'^(\d{1,2}\.\d{1,2})\.\s+(.+)$')
H3 = re.compile(r'^(\d{1,2}\.\d{1,2}\.\d{1,2})\.\s+(.+)$')


def _is_bold(line: dict) -> bool:
    return any('Bold' in f for f in line["fonts"])


def _is_italic(line: dict) -> bool:
    return any('Ital' in f for f in line["fonts"])


def detect_headings(lines: list[dict]) -> list[dict]:
    """Detect numbered headings by typography (bold `N.`, italic `N.N.`)
    rather than by hardcoding the article's actual section names."""
    headings = []
    for i, b in enumerate(lines):
        t = b["text"]
        m3 = H3.match(t)
        m2 = H2.match(t) if not m3 else None
        m1 = H1.match(t) if not (m2 or m3) else None
        if m3:
            headings.append({"level": 3, "number": m3.group(1), "title": m3.group(2), "line_idx": i})
        elif m2 and _is_italic(b):
            headings.append({"level": 2, "number": m2.group(1), "title": m2.group(2), "line_idx": i})
        elif m1 and _is_bold(b):
            headings.append({"level": 1, "number": m1.group(1), "title": m1.group(2), "line_idx": i})
    return headings


# ---------------------------------------------------------------------------
# Section building
# ---------------------------------------------------------------------------

def join_lines_dehyphenated(text_lines: list[str]) -> str:
    """Rejoin words split across a line break ("muscu-" + "loskeletal" ->
    "musculoskeletal") at the only point where the original `-` +
    line-break pattern is still visible: while lines are still separate."""
    out = ""
    for line in text_lines:
        if out.endswith("-") and line and line[0].islower():
            out = out[:-1] + line          # rejoin split word, no space
        elif out:
            out = out + " " + line
        else:
            out = line
    return out


def build_sections(lines: list[dict], headings: list[dict]) -> list[dict]:
    """Slice the line stream between consecutive headings into section
    dicts carrying their number, level, title, top-level section, and text."""
    sections = []
    for i, h in enumerate(headings):
        start = h["line_idx"] + 1
        end = headings[i + 1]["line_idx"] if i + 1 < len(headings) else len(lines)
        body_lines = [l["text"] for l in lines[start:end]]
        text = re.sub(r'\s+', ' ', join_lines_dehyphenated(body_lines)).strip()
        sections.append({
            "number": h["number"],
            "level": h["level"],
            "title": h["title"],
            "top_section": h["number"].split(".")[0],
            "text": text,
        })
    return sections


# ---------------------------------------------------------------------------
# Reference-list parsing
# ---------------------------------------------------------------------------

REF_ENTRY = re.compile(r'\n(\d{1,3})\.\s+(?=[A-Za-z])')


def parse_references(pdf_path: Path = PDF_PATH) -> dict[int, str]:
    """Parse the article's reference list into individually addressable
    entries keyed by citation number, so an in-text `[N]` marker can be
    resolved back to its source at answer-generation time."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    raw_text = "".join(page.get_text() + "\n" for page in doc)
    doc.close()

    m = re.search(r'\nReferences\n', raw_text)
    if not m:
        return {}
    ref_text = raw_text[m.end():]
    ref_text = ref_text.split("Disclaimer/Publisher")[0]
    ref_text = re.sub(r'Biomedicines\s+\d{4},\s*\d+,?\s*\d+\s*\n?\d*\s*of\s*\d+\s*\n?', '', ref_text)
    ref_text = re.sub(r'\n\d+\s+of\s+\d+\n', '\n', ref_text)

    matches = list(REF_ENTRY.finditer("\n" + ref_text))
    entries: dict[int, str] = {}
    for i, mm in enumerate(matches):
        num = int(mm.group(1))
        start = mm.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(ref_text) + 1
        content = ("\n" + ref_text)[start:end]
        entries[num] = re.sub(r'\s+', ' ', content).strip()
    return entries


def _save_references(references: dict[int, str]) -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCES_PATH.write_text(
        json.dumps({str(k): v for k, v in references.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_references() -> dict[int, str]:
    """Load the cached, parsed reference list (empty dict if not built yet)."""
    if not REFERENCES_PATH.exists():
        return {}
    raw = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Citation-aware chunking
# ---------------------------------------------------------------------------

CITATION_MARKER = re.compile(r'\[([\d,\s]+)\]')


def split_sentences_keep_citations(text: str) -> list[str]:
    """Split into sentences without ever cutting inside a `[12,45]`-style
    citation marker (commas inside brackets are masked, then restored)."""
    protected = re.sub(r'\[([\d,\s\-]+)\]',
                        lambda mm: '[' + mm.group(1).replace(',', '\u00a7') + ']', text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    return [s.replace('\u00a7', ',') for s in sentences]


def extract_citation_numbers(text: str) -> list[int]:
    nums = set()
    for m in CITATION_MARKER.finditer(text):
        for n in m.group(1).split(','):
            n = n.strip()
            if n.isdigit():
                nums.add(int(n))
    return sorted(nums)


def chunk_section(section: dict, max_chars: int = 900) -> list[dict]:
    """Build sentence-by-sentence chunks up to a soft character cap, so a
    sentence - and therefore a citation - is never split across two chunks.
    Each chunk carries the reference numbers it actually cites."""
    sentences = split_sentences_keep_citations(section["text"])
    chunks, current = [], ""
    for sent in sentences:
        if current and len(current) + len(sent) + 1 > max_chars:
            chunks.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}".strip()
    if current:
        chunks.append(current.strip())
    return [
        {
            "text": c,
            "section_number": section["number"],
            "section_title": section["title"],
            "top_section": section["top_section"],
            "level": section["level"],
            "cited_refs": extract_citation_numbers(c),
        }
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def build_documents(sections: list[dict]) -> list[Document]:
    """Chunk every section and wrap each chunk as a langchain `Document`
    carrying both the article-level metadata and the chunk's section/
    citation metadata."""
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                **METADATA,
                "section": f"{chunk['section_number']}. {chunk['section_title']}",
                "section_number": chunk["section_number"],
                "section_title": chunk["section_title"],
                "top_section": chunk["top_section"],
                "level": chunk["level"],
                "cited_refs": chunk["cited_refs"],
            },
        )
        for section in sections
        for chunk in chunk_section(section)
    ]

    bad = [d for d in documents if _PAGE_NUM_RE.search(d.page_content)]
    assert not bad, "Text extraction regression: page-number artifacts leaked into chunks"
    return documents


def build_vectorstore(documents: list[Document]):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore, embeddings


def get_retriever(k: int = 3, force_rebuild: bool = False):
    """Load a cached FAISS index from disk if present, otherwise build it
    from the source PDF and cache it (plus the parsed reference list) for
    next time."""
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
        lines = extract_lines()
        headings = detect_headings(lines)
        sections = build_sections(lines, headings)
        references = parse_references()
        documents = build_documents(sections)
        vectorstore, embeddings = build_vectorstore(documents)
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))
        _save_references(references)

    return vectorstore.as_retriever(search_kwargs={"k": k})


def evaluate_retrieval(eval_dataset: list[dict], retriever) -> float:
    relevant_count = 0
    for item in eval_dataset:
        docs = retriever.invoke(item["question"])
        retrieved_text = " ".join(d.page_content for d in docs)
        if any(kw.lower() in retrieved_text.lower() for kw in item["keywords"]):
            relevant_count += 1
    return (relevant_count / len(eval_dataset)) * 100


def generate_answer(
    question: str,
    retriever,
    model: str = "gpt-4o-mini",
    references: Optional[dict[int, str]] = None,
) -> tuple[str, list[Document]]:
    """Answer `question` using retrieved context (RAG).

    Returns (answer_text, retrieved_docs) so callers (e.g. the Gradio app)
    can display both the answer and its supporting context. If `references`
    (the parsed reference list, see `load_references()`) is provided, any
    citation numbers carried by the retrieved chunks are resolved back to
    their full reference text and appended to the answer.
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
        answer = response.choices[0].message.content
    else:
        sections_used = ", ".join(sorted({d.metadata["section"] for d in docs}))
        preview = "\n\n".join(d.page_content.strip() for d in docs)
        answer = (
            f"[Extractive fallback - no OPENAI_API_KEY set]\n"
            f"Most relevant sections: {sections_used}\n\n"
            f"Retrieved context:\n{preview}"
        )

    if references:
        cited_nums = sorted({n for d in docs for n in d.metadata.get("cited_refs", [])})
        citation_lines = [f"[{n}] {references[n]}" for n in cited_nums if n in references]
        if citation_lines:
            answer = f"{answer}\n\nReferences cited in the retrieved passages:\n" + "\n".join(citation_lines)

    return answer, docs
