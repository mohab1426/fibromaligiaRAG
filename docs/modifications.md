# Modifications Report

## 1. Original Project

The uploaded notebook (`Biomedicines__1_.ipynb`, 30 cells) implemented the
first two-thirds of a Retrieval-Augmented Generation (RAG) pipeline over a
single biomedical review article (`biomedicines-12-01543.pdf`):

- PDF text extraction with PyMuPDF.
- Regex-based text cleaning (de-hyphenation, header/footer/URL removal).
- Splitting the article into 7 named sections.
- Chunking each section with `RecursiveCharacterTextSplitter`.
- Embedding chunks with `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) and
  indexing them in a FAISS vector store.
- A retriever with a small Precision@K keyword-based evaluation function.
- A single example query against the retriever.

It ran in a notebook environment (likely Google Colab, given `%pip install`
style calls and a hard-coded relative PDF path) and had no README,
`requirements.txt`, `.gitignore`, or repository structure.

## 2. Problems Found

| # | Issue | Type |
|---|-------|------|
| 1 | Cell 4 duplicates the cleaning logic in cell 2 (same regex pipeline, re-run into the same `clean_text` variable) with no comment explaining why. | Dead/duplicate code |
| 2 | Cell 28 installs `openai` but the package is never imported or used anywhere; cell 29 is empty. The pipeline stops after retrieval and never generates an answer, so the "R" in RAG has no "G". | Incomplete implementation |
| 3 | `pdf_path = "biomedicines-12-01543.pdf"` is a bare relative filename with no directory structure — works only if the PDF happens to sit next to the notebook. | Portability |
| 4 | No `requirements.txt`, `.gitignore`, README, or folder structure — not GitHub-ready as-is. | Project hygiene |
| 5 | The page-number regex `r'Biomedicines 2024, 12, 1543\.?\s*\d* of 22'` (cell 2) never matches, because PyMuPDF extracts the header and the "`N of 22`" page marker as separate lines; only after whitespace is collapsed (cell 4's `\b\d+\s+of\s+22\b`, applied *after* collapsing) does the page-number pattern actually get removed. Cell 2's version is therefore silently ineffective and only cell 4's corrected version works. | Bug (silently non-functional regex) |
| 6 | `evaluate_retrieval`'s definition (cell 24) and its first real call with `eval_dataset` (cell 25) were separated from the retriever construction by two throwaway "preview" cells (25 also mixes a preview query with the evaluation call), making the notebook's flow hard to follow. | Code organization |
| 7 | No verification that section parsing actually found each heading (`main_text.find(section)` returning `-1` would silently produce an empty/garbage section) — no assertion guarded this. | Missing error handling |
| 8 | Repeated, near-identical debug print cells (cells 18, 20, 21, 23 all reprint chunk previews with minor variations) — leftover exploration code that adds no value in a final version. | Dead code |

There is **no classical ML model** in this project (no train/test split, no
labels, no leakage risk in the traditional sense) — it is a retrieval
pipeline, so the ML-specific checks in the audit checklist (data leakage,
class imbalance, etc.) do not apply here. The closest analogue, "evaluation
correctness," was checked instead (see below).

## 3. Changes Made

| File | What changed | Why |
|------|--------------|-----|
| `notebooks/fibromyalgia_rag_pipeline.ipynb` | Rebuilt from scratch, preserving every original processing step and its logic, but: removed the duplicate cleaning cell (kept the corrected version whose page-number regex actually works, and documented why); removed redundant debug-print cells; added markdown section headers (Setup, Data Loading, Cleaning, Section Parsing, Chunking, Embedding & Indexing, Retrieval Evaluation, Answer Generation, Results, Conclusion, Limitations); added a regression assertion that no page-number artifacts survive chunking; added the missing generation step (Section 8, see below); made the PDF path project-relative instead of a bare filename. | Completeness, correctness, portability, GitHub-readiness |
| `requirements.txt` | Created, listing the actual libraries used (`pymupdf`, `langchain-core`, `langchain-text-splitters`, `langchain-community`, `langchain-huggingface`, `faiss-cpu`, `sentence-transformers`, `openai`, `nbformat`). | Reproducible installs |
| `.gitignore` | Created (standard Python/Jupyter ignores, plus the generated FAISS index directory and explicit secret-file patterns). | Repo hygiene |
| `README.md` | Created, describing the project, dataset, methodology, install/usage instructions, results, limitations. | Documentation |
| `docs/modifications.md` | This file. | Change history |
| `data/raw/biomedicines-12-01543.pdf` | Copied into the repository structure (small, open-access, CC BY 4.0 — safe to commit). | Data placement |

## 4. New Files

- `requirements.txt`
- `.gitignore`
- `README.md`
- `docs/modifications.md`
- `notebooks/fibromyalgia_rag_pipeline.ipynb` (replaces the uploaded
  notebook; see below)

## 5. Removed Files

- The original `Biomedicines__1_.ipynb` is not carried over as-is; it is
  superseded by `notebooks/fibromyalgia_rag_pipeline.ipynb`, which preserves
  all of its working logic. No functionality was dropped — only duplicate
  cleaning code and repetitive debug-print cells were removed (see Problems
  6 and 8 above).

## 6. RAG-Pipeline-Specific Changes

- **Generation step added (Section 8 of the notebook).** This is the most
  substantive addition. The original notebook installed `openai` and then
  never used it — the pipeline had no way to actually answer a question, only
  to retrieve passages. The new `generate_answer()` function:
  - Uses the OpenAI Chat Completions API to produce a grounded answer from
    the retrieved chunks when `OPENAI_API_KEY` is set in the environment.
  - Falls back to a template-based extractive answer (listing the relevant
    sections and quoting the retrieved chunks) when no key is set, so the
    notebook remains fully runnable — including in CI or by anyone without
    an OpenAI account — without needing any secret.
  - No API key is hard-coded or committed anywhere in this repository.
- **Evaluation preserved as-is.** The Precision@K keyword-matching
  `evaluate_retrieval` function was correct in the original and is
  unchanged in logic; it was only moved next to its supporting retriever
  code and given its own clearly labeled section.
- **Cleaning regex bug fixed.** As noted in Problem 5, cell 2's page-number
  regex never matched; cell 4's corrected version (matching after whitespace
  collapse) was kept as the single source of truth, and a regression
  assertion (`assert len(bad_chunks) == 0`) was added directly after chunking
  so this class of bug cannot silently reappear.

## 7. Testing

Testing was performed in a sandboxed Linux environment with restricted
network egress (only PyPI-family domains are reachable; `huggingface.co` is
not on the allowlist). Given that constraint, testing was done in two parts:

1. **Full pipeline logic, without live embeddings** — extraction, cleaning,
   section parsing, chunking, and the FAISS/retriever/`evaluate_retrieval`
   code path were executed end-to-end against the real PDF using a
   deterministic offline `HashingVectorizer`-based embedding stand-in (only
   for testing purposes; the deliverable notebook still uses the real
   `HuggingFaceEmbeddings`/`all-MiniLM-L6-v2`). Results:
   - Extracted 100,572 raw characters from the 10-page PDF; 99,464 after
     cleaning.
   - All 7 sections found and correctly delimited (e.g. `3. Physiopathology`
     → 14,240 chars, `6. Treatment` → 15,911 chars).
   - Chunking produced 70 chunks; **0** contained leftover `"N of 22"`
     page-number artifacts.
   - FAISS index built successfully over all 70 chunks; `evaluate_retrieval`
     ran without error and returned a plausible score (66.7% on the 3-item
     benchmark with the offline stand-in embedder — not representative of
     the real model's expected higher accuracy, since a hashing vectorizer
     is far weaker than a trained sentence-transformer).
   - `generate_answer()`'s no-API-key fallback path was unit-tested directly
     and returns a well-formed, section-labeled extractive answer.
2. **Not tested in this environment**: the real `all-MiniLM-L6-v2` embedding
   download (blocked by network egress restrictions to `huggingface.co` in
   this sandbox) and the live OpenAI generation path (requires a paid API
   key, which is intentionally not provided or stored anywhere in this
   project). Both code paths are standard, well-established library calls
   (`langchain_huggingface.HuggingFaceEmbeddings`, `openai.OpenAI(...).chat.completions.create`)
   and are expected to work unmodified once run with normal internet access
   and, optionally, an API key — but you should run the notebook yourself
   once to confirm on your machine.

## 7b. Follow-up: Gradio Web Interface

After the initial audit/delivery, a Gradio-based web UI was added on request:

- `src/pipeline.py` — the notebook's pipeline steps (extraction, cleaning,
  section parsing, chunking, indexing, retrieval, generation) were
  refactored into reusable functions so the notebook and the web app share
  one implementation instead of duplicating logic. `get_retriever()` also
  adds FAISS index caching to disk (`data/processed/faiss_index/`) so the
  app doesn't rebuild embeddings on every launch.
- `src/app.py` — a `gradio.Blocks` app with a question box, an "Ask" button,
  an answer box, and a markdown panel listing the retrieved source passages
  (labeled by article section). It reuses `generate_answer()` from
  `pipeline.py`, so it inherits the same OpenAI/extractive-fallback
  behavior as the notebook — no API key is required to run it.
- `requirements.txt` — added `gradio>=4.0`.
- `README.md` — added a "Web interface (Gradio)" usage section.

Testing: `src/pipeline.py` was exercised end-to-end for extraction, cleaning,
section parsing, and chunking against the real PDF (70 chunks, correct
sections) in this sandbox. Both `src/pipeline.py` and `src/app.py` were
syntax-checked (`ast.parse`). The embedding/indexing step and the live
Gradio server were not run end-to-end here due to the same
`huggingface.co` network restriction noted in Section 7 — run
`python src/app.py` yourself once to confirm on your machine.

## 8. Remaining Limitations

- The real embedding + OpenAI generation paths could not be executed in this
  sandboxed audit environment due to network restrictions; see Testing
  above. Please run the notebook once end-to-end on a machine with normal
  internet access before relying on it.
- No answer-quality (faithfulness/relevance) metric is implemented for the
  generation step — only retrieval quality is scored.
- The retrieval benchmark is intentionally small (3 questions) and should be
  treated as a smoke test, not a statistically meaningful evaluation.
