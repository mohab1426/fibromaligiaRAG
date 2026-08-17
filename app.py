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
from pipeline import generate_answer, get_retriever

print("Loading / building the retrieval index (first run may take a minute)...")
retriever = get_retriever(k=3)
print("Ready.")


def answer_question(question: str):
    if not question or not question.strip():
        return "من فضلك اكتب سؤال أولاً.", ""

    answer, docs = generate_answer(question, retriever)

    sources_md = "\n\n".join(
        f"**[{i}] Section: {d.metadata['section']}**\n\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )
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
        f"**Generation mode:** {generation_mode}"
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
    demo.launch(share=True)
