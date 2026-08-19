"""Core RAG pipeline logic for the Fibromyalgia article.

This module factors out the pipeline steps from the notebook
(`notebooks/fibromyalgia_rag_pipeline.ipynb`) so they can be reused by the
Gradio app (`src/app.py`) without duplicating code. The notebook remains the
primary, documented walkthrough; this module is the "library" version of the
same logic.

Parsing is layout-aware: headings are detected from PDF font metadata
(bold/italic + numbering pattern) rather than matched against a fixed list of
section-title strings, so the pipeline picks up subsections (e.g. "6.1.
Pharmacological Treatment") in addition to the 7 top-level sections, and
keeps working if a heading's wording changes slightly. Chunking is
token-based (tiktoken) and grouped per subsection, which keeps each chunk's
page number(s) accurate via a character-offset -> page map instead of a
string search.

Each chunk also carries the in-text citation numbers (`[12,45]`-style
markers) it contains (`cited_refs`), which `generate_answer()` can resolve
back to their full reference text via the parsed, cached reference list
(`parse_references()` / `load_references()`).
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
from collections import defaultdict
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

# Embedding backend selection (see `_build_embeddings` below). Defaults to
# the local sentence-transformer model so the pipeline runs with zero
# configuration, exactly as before.
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "huggingface").lower()
OPENROUTER_EMBEDDING_MODEL = os.environ.get(
    "OPENROUTER_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free"
)

# ---------------------------------------------------------------------------
# Parsing (layout-aware: headings detected from font metadata, not a fixed
# list of section-title strings)
# ---------------------------------------------------------------------------

JUNK_LINE_PATTERNS = [
    r"^Biomedicines\s+\d{4},\s*\d+,?\s*(x FOR PEER REVIEW|\d+)$",  # repeated citation banner
    r"^\d+\s+of\s+\d+$",  # "4 of 22" page markers
]

H1 = re.compile(r"^(\d{1,2})\.\s+(.+)$")
H2 = re.compile(r"^(\d{1,2}\.\d{1,2})\.\s+(.+)$")
H3 = re.compile(r"^(\d{1,2}\.\d{1,2}\.\d{1,2})\.\s+(.+)$")


def extract_lines(pdf_path: Path = PDF_PATH) -> list[dict]:
    """Extract text lines with font metadata (layout-aware, not plain text)."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    lines = []
    block_id = 0
    for pno, page in enumerate(doc):
        d = page.get_text("dict")
        for block in d["blocks"]:
            if block["type"] != 0:  # skip images
                continue
            block_id += 1
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                lines.append(
                    {
                        "page": pno + 1,
                        "block_id": block_id,
                        "text": text,
                        "fonts": {s["font"] for s in spans},
                        "y": line["bbox"][1],
                    }
                )
    doc.close()
    return [l for l in lines if not any(re.match(p, l["text"]) for p in JUNK_LINE_PATTERNS)]


def _is_bold(line: dict) -> bool:
    return any("Bold" in f for f in line["fonts"])


def _is_italic(line: dict) -> bool:
    return any("Ital" in f for f in line["fonts"])


def detect_headings(lines: list[dict]) -> list[dict]:
    """Identify H1/H2/H3 headings from numbering pattern + bold/italic formatting."""
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


def join_lines_dehyphenated(text_lines: list[dict]) -> str:
    out = ""
    for line_dict in text_lines:
        line = line_dict["text"]
        if out.endswith("-") and line and line[0].islower():
            out = out[:-1] + line  # rejoin split word, no space
        elif out:
            out = out + " " + line
        else:
            out = line
    return out


def build_sections(lines: list[dict], headings: list[dict]) -> list[dict]:
    """Slice the line stream between consecutive headings into paragraph-level
    elements, each carrying page number, section, and subsection metadata."""
    elements = []

    current_h1 = "Unknown Section"
    current_sub = "Unknown Subsection"

    for i, h in enumerate(headings):
        if h["level"] == 1:
            current_h1 = f"{h['number']} {h['title']}"
            current_sub = current_h1
        else:
            current_sub = f"{h['number']} {h['title']}"

        start = h["line_idx"] + 1
        end = headings[i + 1]["line_idx"] if i + 1 < len(headings) else len(lines)

        current_block_lines: list[dict] = []
        current_block_id = None

        for line in lines[start:end]:
            if current_block_id is None:
                current_block_id = line["block_id"]

            if line["block_id"] != current_block_id:
                if current_block_lines:
                    text = re.sub(r"\s+", " ", join_lines_dehyphenated(current_block_lines)).strip()
                    if text:
                        elements.append(
                            {
                                "source": PDF_PATH.name,
                                "page_number": current_block_lines[0]["page"],
                                "section": current_h1,
                                "subsection": current_sub,
                                "text": text,
                            }
                        )
                current_block_lines = [line]
                current_block_id = line["block_id"]
            else:
                current_block_lines.append(line)

        if current_block_lines:
            text = re.sub(r"\s+", " ", join_lines_dehyphenated(current_block_lines)).strip()
            if text:
                elements.append(
                    {
                        "source": PDF_PATH.name,
                        "page_number": current_block_lines[0]["page"],
                        "section": current_h1,
                        "subsection": current_sub,
                        "text": text,
                    }
                )

    return elements


REF_ENTRY = re.compile(r"\n(\d{1,3})\.\s+(?=[A-Za-z])")


def parse_references(pdf_path: Path = PDF_PATH) -> dict[int, str]:
    """Parse the reference list into individually addressable, numbered
    entries so an in-text marker like `[12,45]` can be resolved back to its
    source. Not required by chunking/retrieval below; kept as a standalone
    utility since it's independently useful (e.g. for citation lookups)."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    raw_text = "".join(page.get_text() + "\n" for page in doc)
    doc.close()

    m = re.search(r"\nReferences\n", raw_text)
    if not m:
        return {}
    ref_text = raw_text[m.end():]
    ref_text = ref_text.split("Disclaimer/Publisher")[0]
    ref_text = re.sub(
        r"Biomedicines\s+\d{4},\s*\d+,?\s*\d+\s*\n?\d*\s*of\s*\d+\s*\n?", "", ref_text
    )
    ref_text = re.sub(r"\n\d+\s+of\s+\d+\n", "\n", ref_text)

    matches = list(REF_ENTRY.finditer("\n" + ref_text))
    entries: dict[int, str] = {}
    for i, mm in enumerate(matches):
        num = int(mm.group(1))
        start = mm.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(ref_text) + 1
        content = ("\n" + ref_text)[start:end]
        entries[num] = re.sub(r"\s+", " ", content).strip()
    return entries


def _save_references(references: dict[int, str]) -> None:
    """Cache the parsed reference list to disk alongside the FAISS index so
    it can be reloaded (via `load_references()`) without re-parsing the PDF."""
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
# Citation markers
# ---------------------------------------------------------------------------

CITATION_MARKER = re.compile(r"\[([\d,\s]+)\]")


def extract_citation_numbers(text: str) -> list[int]:
    """Return the sorted, de-duplicated reference numbers cited in `text`
    via `[N]` / `[N,M]`-style markers."""
    nums = set()
    for m in CITATION_MARKER.finditer(text):
        for n in m.group(1).split(","):
            n = n.strip()
            if n.isdigit():
                nums.add(int(n))
    return sorted(nums)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_element_text(text: str) -> str:
    cleaned = text
    # Re-join words hyphenated across a line break (safety net; parsing already does this)
    cleaned = re.sub(r"-\s*\n\s*", "", cleaned)
    # Remove the repeated journal header/footer boilerplate
    cleaned = re.sub(r"Biomedicines\s+2024,\s*12,\s*1543\.?", "", cleaned)
    # Remove DOI / journal URL boilerplate (article banner, not reference-list DOIs)
    cleaned = re.sub(r"https://doi\.org/10\.3390/biomedicines\d+", "", cleaned)
    cleaned = re.sub(r"https://www\.mdpi\.com/journal/biomedicines", "", cleaned)
    # Remove leftover "N of 22" page markers (extraction noise)
    cleaned = re.sub(r"\b\d+\s+of\s+22\b", "", cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_elements(elements: list[dict]) -> list[dict]:
    cleaned = []
    for el in elements:
        text = clean_element_text(el["text"])
        if text:  # drop elements that were pure boilerplate and are now empty
            cleaned.append({**el, "text": text})
    return cleaned


# ---------------------------------------------------------------------------
# Chunking (token-based, grouped per subsection, page-number-accurate)
# ---------------------------------------------------------------------------

_encoder = None


def token_len(text: str) -> int:
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
    return len(_encoder.encode(text))


def chunk_elements(elements: list[dict], chunk_size: int = 350, chunk_overlap: int = 60) -> list[dict]:
    """Group cleaned elements by (section, subsection) and split each group
    into token-sized chunks, tracking the source page number(s) of every
    chunk via a character-offset -> page map (instead of a text search)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    subsections_map: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for el in elements:
        key = (el["section"], el["subsection"])
        subsections_map[key].append(el)

    # Separators demote commas so we don't inappropriately split scientific sentences early.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    )

    final_chunks: list[dict] = []
    section_counters: dict[str, int] = {}

    for (section, subsection), sub_elements in subsections_map.items():
        combined_text = ""
        page_map: list[tuple[int, int]] = []  # (char_start_index, page_number)

        for el in sub_elements:
            page_map.append((len(combined_text), el["page_number"]))
            combined_text += el["text"] + "\n\n"

        chunk_texts = text_splitter.split_text(combined_text)

        search_start = 0
        section_counters[section] = section_counters.get(section, 0)

        for chunk_text in chunk_texts:
            section_counters[section] += 1

            chunk_start_idx = combined_text.find(chunk_text, search_start)
            if chunk_start_idx != -1:
                search_start = chunk_start_idx + 1
            else:
                chunk_start_idx = search_start  # fallback

            chunk_end_idx = chunk_start_idx + len(chunk_text)

            offsets = [m[0] for m in page_map]
            mapping_start_idx = max(0, bisect.bisect_right(offsets, chunk_start_idx) - 1)
            mapping_end_idx = max(0, bisect.bisect_right(offsets, chunk_end_idx) - 1)

            page_numbers = sorted(
                {page_map[m_idx][1] for m_idx in range(mapping_start_idx, mapping_end_idx + 1)}
            )

            source = sub_elements[0]["source"]
            chunk_index = section_counters[section]
            hash_input = f"{source}_{section}_{subsection}_{chunk_index}"
            chunk_id = hashlib.md5(hash_input.encode("utf-8")).hexdigest()[:12]

            final_chunks.append(
                {
                    "source": source,
                    "page_numbers": page_numbers,
                    "section": section,
                    "subsection": subsection,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "n_tokens": token_len(chunk_text),
                    "n_chars": len(chunk_text),
                    "text": chunk_text,
                    "cited_refs": extract_citation_numbers(chunk_text),
                }
            )

    return final_chunks


def is_meaningful_chunk(chunk: dict) -> bool:
    if chunk["n_tokens"] < 5:
        return False
    # Require at least some alphabetical characters to avoid dropping just numbers/punctuation noise
    if not re.search(r"[a-zA-Z]{3,}", chunk["text"]):
        return False
    return True


def filter_meaningful_chunks(chunks: list[dict]) -> list[dict]:
    return [c for c in chunks if is_meaningful_chunk(c)]


def build_documents(chunks: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=chunk["text"],
            metadata={
                **METADATA,
                "section": chunk["section"],
                "subsection": chunk["subsection"],
                "page_numbers": chunk["page_numbers"],
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "n_tokens": chunk["n_tokens"],
                "n_chars": chunk["n_chars"],
                "cited_refs": chunk["cited_refs"],
            },
        )
        for chunk in chunks
    ]


def parse_and_chunk_pdf(pdf_path: Path = PDF_PATH) -> list[Document]:
    """Run the full parsing -> cleaning -> chunking pipeline end to end and
    return LangChain `Document`s ready for embedding."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Source PDF not found at {pdf_path}. Place "
            "'biomedicines-12-01543.pdf' in data/raw/ first."
        )
    lines = extract_lines(pdf_path)
    headings = detect_headings(lines)
    elements = build_sections(lines, headings)
    elements = clean_elements(elements)
    chunks = chunk_elements(elements)
    chunks = filter_meaningful_chunks(chunks)
    return build_documents(chunks)


# ---------------------------------------------------------------------------
# Embedding & Indexing
# ---------------------------------------------------------------------------


def _build_embeddings():
    """Return the configured embedding backend.

    Defaults to a local sentence-transformer model (`all-MiniLM-L6-v2`, via
    `EMBEDDING_BACKEND=huggingface`, the default) so the pipeline runs with
    no API key. Set `EMBEDDING_BACKEND=openrouter` and `OPENROUTER_API_KEY`
    to use OpenRouter's embeddings API instead. The key is only ever read
    from the environment -- never hardcoded or committed.
    """
    if EMBEDDING_BACKEND == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_BACKEND=openrouter requires the OPENROUTER_API_KEY "
                "environment variable to be set."
            )
        from langchain_core.embeddings import Embeddings
        from openai import OpenAI

        class OpenRouterEmbeddings(Embeddings):
            def __init__(self, model: str, key: str):
                self._model = model
                self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                response = self._client.embeddings.create(
                    model=self._model, input=texts, encoding_format="float"
                )
                return [item.embedding for item in response.data]

            def embed_query(self, text: str) -> list[float]:
                response = self._client.embeddings.create(
                    model=self._model, input=text, encoding_format="float"
                )
                return response.data[0].embedding

        return OpenRouterEmbeddings(OPENROUTER_EMBEDDING_MODEL, api_key)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vectorstore(documents: list[Document]):
    from langchain_community.vectorstores import FAISS

    embeddings = _build_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore, embeddings


def get_retriever(k: int = 3, force_rebuild: bool = False):
    """Load a cached FAISS index from disk if present, otherwise build it
    from the source PDF and cache it for next time."""
    from langchain_community.vectorstores import FAISS

    embeddings = _build_embeddings()

    if INDEX_DIR.exists() and not force_rebuild:
        vectorstore = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
    else:
        documents = parse_and_chunk_pdf()
        vectorstore, embeddings = build_vectorstore(documents)
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))
        _save_references(parse_references())

    return vectorstore.as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# Retrieval Evaluation
# ---------------------------------------------------------------------------


def evaluate_retrieval(eval_dataset: list[dict], retriever) -> float:
    relevant_count = 0
    for item in eval_dataset:
        docs = retriever.invoke(item["question"])
        retrieved_text = " ".join(d.page_content for d in docs)
        if any(kw.lower() in retrieved_text.lower() for kw in item["keywords"]):
            relevant_count += 1
    return (relevant_count / len(eval_dataset)) * 100


# ---------------------------------------------------------------------------
# Answer Generation (RAG)
# ---------------------------------------------------------------------------


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
    citation numbers carried by the retrieved chunks (`cited_refs`) are
    resolved back to their full reference text and appended to the answer.
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