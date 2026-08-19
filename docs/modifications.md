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

## 7c. Follow-up: Layout-aware parsing, subsection chunking, optional OpenRouter embeddings

A second team notebook (`fibromyalgia_.ipynb`, 33 cells) prototyped a more
detailed parsing/chunking approach and an alternate embeddings backend. It
was **not** added to the repository — its logic was integrated into
`src/pipeline.py` and the notebook was regenerated to match. Changes:

- **Parsing replaced with layout-aware heading detection.** `extract_lines()`
  now reads PyMuPDF's `"dict"` output (text + font metadata) instead of
  plain text. `detect_headings()` identifies H1/H2/H3 headings from a
  numbering-pattern regex combined with bold/italic formatting, and
  `build_sections()` slices the line stream between headings into
  paragraph-level elements carrying page/section/**subsection** metadata.
  This replaces the previous `parse_sections()`, which matched a fixed list
  of 7 section-title strings and had no subsection granularity.
- **Reference-list parsing added.** `parse_references()` parses the
  bibliography into individually addressable, numbered entries. It's a
  standalone utility (not wired into chunking/retrieval), preserved because
  it was part of the notebook's already-working parsing logic.
- **Chunking replaced with token-based, subsection-grouped chunking.**
  `chunk_elements()` groups cleaned elements by `(section, subsection)`,
  splits each group with `RecursiveCharacterTextSplitter` using a
  `tiktoken`-based token-length function (previously character-based), and
  recovers each chunk's page number(s) via a character-offset → page map
  instead of a text search. `filter_meaningful_chunks()` drops low-signal
  chunks (fewer than 5 tokens, or no meaningful alphabetic content).
- **OpenRouter embeddings added as an optional backend.** `_build_embeddings()`
  dispatches on the `EMBEDDING_BACKEND` environment variable: `huggingface`
  (default, unchanged, no key required) or `openrouter`, which requires
  `OPENROUTER_API_KEY` to be set. `src/app.py` now shows the active backend
  alongside the existing generation-mode indicator.
- **Retrieval evaluation unchanged.** The notebook's `evaluate_retrieval()`
  logic and 3-question eval set were already identical to the existing
  implementation; nothing to integrate there.
- **Answer generation unchanged.** The team notebook had no generation step
  (it stopped after retrieval, the same gap noted in Problem 2 above); the
  existing `generate_answer()` was kept as-is.

**Security note:** the uploaded notebook contained a live, hardcoded
OpenRouter API key in its embeddings cell. It was **not** carried into the
repository in any form — `OPENROUTER_API_KEY` is read from the environment
only. The exposed key should be revoked/rotated at
https://openrouter.ai/keys.

Files touched: `src/pipeline.py` (rewritten), `src/app.py` (one line added
to display the active embedding backend), `notebooks/fibromyalgia_rag_pipeline.ipynb`
(regenerated to mirror the new pipeline steps), `requirements.txt` (added
`tiktoken`), `README.md` (pipeline description, embedding-backend docs,
Results section updated to not hard-code a chunk count that changed).

Testing: this sandbox has no network access at all (not even PyPI), so none
of the third-party libraries (`fitz`, `langchain_*`, `tiktoken`, `openai`,
`faiss`) could be installed or exercised directly. The new parsing/chunking
logic (`detect_headings`, `build_sections`, `clean_elements`,
`chunk_elements`, `filter_meaningful_chunks`, `parse_references`'s regex
core, `is_meaningful_chunk`) was instead verified against synthetic
line/text fixtures using lightweight stand-ins for `tiktoken` and
`RecursiveCharacterTextSplitter`, and the `_build_embeddings()` backend
dispatch was verified against a stubbed `openai` client — confirming
correct heading levels, section/subsection propagation, page-number
recovery, chunk_id generation, the meaningful-chunk filter, and that the
`OPENROUTER_API_KEY` is read only from the environment (a source-level
check also confirmed no literal key string is present in `pipeline.py`).
All edited/created files were syntax-checked (`py_compile` for `.py`
files; the notebook's JSON structure and each code cell's `ast.parse`
were checked — the `%pip install` magic cell is expected to fail plain
Python parsing, same as the pre-existing notebook). The real PDF parsing,
embedding, indexing, and OpenAI/OpenRouter API calls were **not** run
end-to-end in this sandbox; run the notebook or `src/app.py` yourself once
on a machine with normal internet access to confirm.

## 8. Remaining Limitations

- The real embedding + OpenAI generation paths could not be executed in this
  sandboxed audit environment due to network restrictions; see Testing
  above. Please run the notebook once end-to-end on a machine with normal
  internet access before relying on it.
- No answer-quality (faithfulness/relevance) metric is implemented for the
  generation step — only retrieval quality is scored.
- The retrieval benchmark is intentionally small (3 questions) and should be
  treated as a smoke test, not a statistically meaningful evaluation.
- The heading-detection regexes assume the numbering/formatting conventions
  of this specific article (bold `N.` for H1, italic `N.N.` for H2, plain
  `N.N.N.` on its own line for H3) and would need adapting for articles with
  a different heading scheme.
- Switching `EMBEDDING_BACKEND` changes the vector dimensionality; the
  cached FAISS index at `data/processed/faiss_index/` must be rebuilt (it is
  git-ignored, so this is just a local cache concern, not a repo issue).
- Root-level `pipeline.py`/`app.py` duplicates: see "7d" below — an earlier
  pass in this doc claimed they'd been removed, but the merge in "9" found
  they were still present in both branches; they are now actually deleted.

## 7d. Follow-up: removed redundant root-level duplicates

`pipeline.py` and `app.py` at the repository root were traced and confirmed
unused:

- Neither file is imported by anything else in the repo (`src/app.py`
  inserts `src/` at the front of `sys.path` and imports `pipeline` from
  there, i.e. `src/pipeline.py` — never the root copy).
- Neither is referenced by the README's project structure, installation, or
  usage instructions (`python src/app.py`, notebook imports from
  `src/pipeline.py`).
- `pipeline.py` was a stale pre-integration snapshot of `src/pipeline.py`
  (missing section 7c's and section 9's changes); `app.py` differed from
  `src/app.py` only in `demo.launch(share=True)` vs. `demo.launch()`.
- A repo-wide grep for `pipeline.py`, `app.py`, and `import pipeline` found
  no other call sites.

They were provably dead, unreferenced code, so they are deleted as part of
this merge (see "9" below) rather than kept as a diverging, unmaintained
second copy of the pipeline. `src/` is the project's one implementation
going forward.

## 9. v2: Layout-Aware Parsing + Reference Resolution (merge of the `S1` branch)

Two parallel follow-ups were developed independently on separate branches
and needed reconciling in this PR:

- **`main` (7c above)**: layout-aware, font-metadata-based heading detection;
  token-based (`tiktoken`), subsection-grouped chunking with an accurate
  character-offset → page map; boilerplate/DOI/URL cleanup; an optional
  OpenRouter embeddings backend. It did **not** wire the already-existing
  `parse_references()` utility into chunking or answer generation.
- **`S1`**: an independently-developed v2 that also added layout-aware
  parsing, plus a citation-resolution feature the `main` line was missing —
  each chunk tracks the reference numbers (`cited_refs`) it cites, and
  `generate_answer()` resolves them back to full citation text via a
  cached `data/processed/references.json`. Its chunker was
  sentence-based/character-capped rather than token-based, and it did not
  have the boilerplate cleanup or OpenRouter backend.

Note that `src/app.py`'s Gradio-facing code (`references = load_references()`,
`generate_answer(question, retriever, references=references)`, and the
sources panel's `cited_refs` display) was **already relying on the
citation-resolution feature** — it just hadn't been re-added to `src/pipeline.py`
after `main` diverged, which would have been a hard runtime error
(`NameError: load_references`) had it been merged as-is by favoring `main`
outright.

**Resolution**: `src/pipeline.py` keeps `main`'s more capable parsing/
chunking/embedding pipeline, with `S1`'s citation-resolution feature added
back on top:

- `extract_citation_numbers()` (from `S1`) is applied to each token-based
  chunk's text in `chunk_elements()`, populating a `cited_refs` field per
  chunk; `build_documents()` copies it into each `Document`'s metadata.
- `_save_references()` / `load_references()` (from `S1`) cache the parsed
  reference list to `data/processed/references.json`; `get_retriever()`
  now parses and caches references whenever it (re)builds the FAISS index.
- `generate_answer()` gained back the optional `references` parameter and
  appends resolved citation text to the answer, in both the OpenAI and
  extractive-fallback paths.
- `src/app.py`'s import line now pulls in both `EMBEDDING_BACKEND` (`main`)
  and `load_references` (`S1`); the rest of the file was already written
  against this combined API and needed no changes.
- `notebooks/fibromyalgia_rag_pipeline.ipynb` (which already just calls into
  `src/pipeline.py`'s functions rather than duplicating logic) needed only
  small additions: caching the reference list after building the index, and
  passing `references=references` into the Section 9 `generate_answer()`
  call, plus an updated Results section.
- `README.md` — Overview and Results merged to describe the combined
  pipeline; a Limitations bullet was added noting that, unlike `S1`'s
  sentence-based chunker, `main`'s token-based chunker does not guarantee a
  `[N,M]` citation marker can never fall on a chunk boundary, so citation
  resolution is best-effort.
- Root-level `pipeline.py`/`app.py` — actually deleted this time (see "7d").

### Not carried over from `S1`
- The sentence-based/character-capped chunker (`split_sentences_keep_citations`,
  `chunk_section`) was not kept — `main`'s token-based, subsection-grouped
  chunker with page-number recovery and boilerplate cleanup is the more
  complete implementation, and citation tracking was ported onto it instead
  of the other way around.

### Testing
- `src/pipeline.py` and `src/app.py` were syntax-checked (`py_compile`).
- The merge was reproduced and resolved from the actual `main`/`S1` branch
  tips (not just the PR's diff view) to confirm the true content of each
  side before deciding how to combine them.
- **Not tested in this environment**: the real PDF extraction, embedding,
  FAISS, and OpenAI/OpenRouter API calls (no network egress to PyPI/model
  hosts in this sandbox for those packages). Run
  `jupyter notebook notebooks/fibromyalgia_rag_pipeline.ipynb` or
  `python src/app.py` on a machine with normal internet access to confirm
  end-to-end, including that a fresh index build produces a
  `data/processed/references.json` and that citation resolution shows up in
  answers.

### Remaining limitations
- Citation resolution is best-effort: `main`'s token-based chunker can in
  principle split a `[N,M]` marker across a chunk boundary, unlike `S1`'s
  sentence-based chunker (see "Not carried over from `S1`" above).
- Heading detection still assumes the specific bold/italic typographic
  convention observed in this article.
- `generate_answer()`'s citation resolution only surfaces references cited
  by the *retrieved* chunks — it does not verify that the LLM's generated
  answer actually discusses those citations correctly; this is the same
  "no faithfulness metric" limitation noted in Section 8 above, now
  extended to citations.