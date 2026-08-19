# Fibromyalgia RAG Pipeline

A small Retrieval-Augmented Generation (RAG) pipeline built over a single
biomedical review article: *"Fibromyalgia: A Review of the Pathophysiological
Mechanisms and Multidisciplinary Treatment Strategies"* (Jurado-Priego,
Cueto-Ureña, Ramírez-Expósito & Martínez-Martos, 2024, *Biomedicines* 12(7),
1543, [doi:10.3390/biomedicines12071543](https://doi.org/10.3390/biomedicines12071543)).

## Overview

Given the source PDF, the pipeline:

1. Extracts text lines with font metadata from every page (PyMuPDF,
   layout-aware `"dict"` extraction).
2. Detects numbered H1/H2/H3 headings from their formatting (bold/italic)
   and numbering pattern, then slices the line stream between headings into
   paragraph-level elements carrying page/section/subsection metadata.
3. Parses the reference list into individually addressable, numbered
   entries (available for citation lookups).
4. Cleans each element's text — de-hyphenates words broken across line
   breaks, strips the repeated journal header/footer, DOI, and journal URL
   boilerplate.
5. Groups elements by `(section, subsection)` and chunks each group with a
   token-based length function (`tiktoken`), recovering each chunk's page
   number(s) via a character-offset → page map. Low-signal chunks (too
   short, or no meaningful alphabetic content) are filtered out.
6. Embeds the chunks and indexes them in a FAISS vector store — via a local
   sentence-transformer model by default, or OpenRouter's embeddings API if
   configured (see **Embedding backend** below).
7. Retrieves the top-K chunks for a question and scores retrieval quality
   with a small Precision@K keyword benchmark.
8. Generates a grounded answer from the retrieved chunks — via the OpenAI API
   if a key is configured, or via a template-based extractive fallback
   otherwise, so the notebook runs end-to-end without any secret.

## Objectives

- Turn a single dense biomedical review article into a queryable knowledge
  base.
- Demonstrate a complete, minimal RAG pipeline (extraction → cleaning →
  chunking → indexing → retrieval → generation) that is reproducible on any
  machine.

## Dataset

- **Source**: one open-access PDF, `data/raw/biomedicines-12-01543.pdf`
  (10 pages, ~2.4 MB), licensed CC BY 4.0.
- No tabular dataset or labels are involved — the "dataset" is the article
  text itself, chunked for retrieval.
- The PDF is small enough to commit directly to the repository; no
  `.gitignore` exclusion was necessary for it (see [`data/raw/`](data/raw)).

## Methodology

See [`notebooks/fibromyalgia_rag_pipeline.ipynb`](notebooks/fibromyalgia_rag_pipeline.ipynb)
for the full, executable pipeline, organized into the sections listed under
**Overview** above. The notebook runs top to bottom without any
machine-specific paths.

## Technologies

- Python 3
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF text extraction
- [LangChain](https://python.langchain.com/) (`langchain-core`,
  `langchain-text-splitters`, `langchain-community`, `langchain-huggingface`)
  — document/chunking abstractions and vector store integration
- [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) — chunk
  embeddings (default backend)
- [tiktoken](https://github.com/openai/tiktoken) — token-based chunk sizing
- [FAISS](https://faiss.ai/) (`faiss-cpu`) — similarity search index
- [OpenAI API](https://platform.openai.com/) (optional) — grounded answer
  generation, and (optional, via OpenRouter) an alternate embeddings backend

## Project Structure

```text
project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── fibromyalgia_rag_pipeline.ipynb
│
├── src/
│   ├── pipeline.py         # shared pipeline logic (used by app.py)
│   └── app.py              # Gradio web interface
│
├── data/
│   ├── raw/
│   │   └── biomedicines-12-01543.pdf
│   └── processed/          # generated at runtime (FAISS index); git-ignored
│
├── outputs/                # reserved for exported answers/reports, if any
│
└── docs/
    └── modifications.md    # audit findings and full change log
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Ensure `data/raw/biomedicines-12-01543.pdf` is present (already included
   in this repository).
2. Launch Jupyter and run the notebook top to bottom:

   ```bash
   jupyter notebook notebooks/fibromyalgia_rag_pipeline.ipynb
   ```

3. **Optional** — to get full LLM-generated answers instead of the
   extractive fallback, set an OpenAI API key before launching Jupyter:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

   No key is required to run the notebook; without one, Section 9 falls back
   to a template-based answer built directly from the retrieved chunks.

### Embedding backend

By default, chunks are embedded locally with `sentence-transformers`
(`all-MiniLM-L6-v2`) — no API key needed. To use OpenRouter's embeddings API
instead:

```bash
export EMBEDDING_BACKEND=openrouter
export OPENROUTER_API_KEY="sk-or-..."   # your own key — never commit this
```

The key is only ever read from the environment; it is never hardcoded or
committed anywhere in this repository. Switching backends changes the vector
dimensionality, so delete `data/processed/faiss_index/` (or pass
`force_rebuild=True` to `get_retriever()`) before rebuilding the index after
a backend change.

### Web interface (Gradio)

A simple chat-style web UI is available in `src/app.py`, built on top of the
same pipeline logic used by the notebook (`src/pipeline.py`):

```bash
python src/app.py
```

- The first launch builds the FAISS index from the PDF and caches it in
  `data/processed/faiss_index/` (git-ignored); later launches load the
  cached index and start instantly.
- Open the local URL Gradio prints (typically `http://127.0.0.1:7860`).
- Type a question, hit **Ask**, and you'll get the generated answer plus the
  retrieved source passages underneath, labeled by article section.
- Works with or without `OPENAI_API_KEY` set, exactly like the notebook's
  generation step (Section 8) — the app header shows which mode is active.

To share the interface temporarily (e.g. for a demo), pass `share=True`:

```python
demo.launch(share=True)  # edit the last line of src/app.py
```

## Results

- The article parses cleanly into its top-level sections and numbered
  subsections (detected from PDF font metadata) with no leftover PDF
  artifacts (hyphenation, running headers, page numbers).
- Chunking (token-based, ~350 tokens with 60-token overlap, grouped per
  subsection) produces chunks with **zero** leftover `"N of 22"`
  page-number artifacts (verified with a regression assertion in the
  notebook); the exact chunk count depends on the live tokenizer/model run,
  so it is not hard-coded here.
- A 3-question Precision@K retrieval benchmark is included as a smoke test;
  the exact score is printed when the notebook is run and depends on the
  embedding model's live output, so it is not hard-coded here.
- The RAG loop is complete end-to-end: retrieval feeds into an answer
  generation step, which was missing in the original notebook (see
  [`docs/modifications.md`](docs/modifications.md)).

## Conclusion

This project implements a minimal but complete RAG pipeline over a single
biomedical review article. It is best understood as a proof-of-concept for
the pipeline mechanics — PDF → clean text → sections → chunks → embeddings →
retrieval → grounded generation — rather than a large-scale retrieval
benchmark.

## Limitations

- The corpus is a single article; the retrieval evaluation is a 3-question
  smoke test, not a statistically powered benchmark.
- Section parsing matches the exact numbered heading text used in this
  article and would need adapting for articles with a different heading
  scheme.
- Full LLM-based generation requires an OpenAI API key (not included, and
  never committed to this repository). Without one, generation degrades to
  an extractive summary of the retrieved context.
- No automated evaluation (e.g. faithfulness or answer-relevance scoring) is
  performed on the generation step itself — only on retrieval.
