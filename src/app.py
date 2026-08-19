"""Gradio interface for the Fibromyalgia RAG pipeline.

Run with:
    python src/app.py

The first launch builds the FAISS index from the source PDF (may take a
minute); subsequent launches load the cached index from
data/processed/faiss_index/.

Set OPENAI_API_KEY as an environment variable before launching to get full
LLM-generated answers. Without it, the app still works and falls back to an
extractive answer built from the retrieved passages.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from pipeline import EMBEDDING_BACKEND, generate_answer, get_retriever, load_references

print("Loading / building the retrieval index (first run may take a minute)...")
retriever = get_retriever(k=3)
references = load_references()
print("Ready.")


def answer_question(question: str):
    if not question or not question.strip():
        return "من فضلك اكتب سؤال أولاً.", ""

    answer, docs = generate_answer(question, retriever, references=references)

    def _format_source(i: int, d) -> str:
        header = f"**[{i}] Section: {d.metadata['section']}**"
        cited = d.metadata.get("cited_refs")
        if cited:
            header += f"  \nCites: {', '.join(f'[{n}]' for n in cited)}"
        return f"{header}\n\n{d.page_content}"

    sources_md = "\n\n".join(_format_source(i, d) for i, d in enumerate(docs, 1))
    return answer, sources_md


EXAMPLE_QUESTIONS = [
    "What are the FDA-approved drugs for fibromyalgia?",
    "What is fibromyalgia characterized by?",
    "What non-pharmacological treatments are discussed for fibromyalgia?",
    "What diagnostic criteria are used for fibromyalgia?",
]

with gr.Blocks(title="Fibromyalgia RAG") as demo:
    generation_mode = (
        "OpenAI (API key set)" if os.environ.get("OPENAI_API_KEY")
        else "Extractive fallback (no OPENAI_API_KEY set)"
    )
    gr.Markdown(
        "# 🩺 Fibromyalgia Article Q&A (RAG)\n"
        "Ask a question about the fibromyalgia review article and get a "
        "grounded answer with the retrieved source passages.\n\n"
        f"**Generation mode:** {generation_mode}  \n"
        f"**Embedding backend:** {EMBEDDING_BACKEND}"
    )

    with gr.Row():
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. What are the FDA-approved drugs for fibromyalgia?",
            lines=2,
        )

    ask_btn = gr.Button("Ask", variant="primary")

    answer_box = gr.Textbox(label="Answer", lines=8)
    sources_box = gr.Markdown(label="Retrieved sources")

    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question_box)

    ask_btn.click(fn=answer_question, inputs=question_box, outputs=[answer_box, sources_box])
    question_box.submit(fn=answer_question, inputs=question_box, outputs=[answer_box, sources_box])


if __name__ == "__main__":
    demo.launch()