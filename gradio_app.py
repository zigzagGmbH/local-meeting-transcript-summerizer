#!/usr/bin/env python3
"""
Gradio front-end for the Local Meeting Transcript Summarizer.

Thin skin over ``main.py``'s pipeline. See ``contexts/gradio_app.md`` for the
full spec, decisions, and milestone breakdown.

Run:
    uv run python gradio_app.py

Then open http://localhost:7860 in a browser.

Status: M2 — UI skeleton only. No upload, no pipeline, no MCP yet. Settings
fields are rendered but not wired to any handlers.
"""

from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv


# ─── Defaults ─────────────────────────────────────────────────────────────

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_EDITOR_MODEL = "gemma4:26b"
DEFAULT_EXTRACTOR_MODEL = "qwen3.5:27b"

SERVER_PORT = 7860
SERVER_HOST = "0.0.0.0"  # reachable on LAN / inside Docker; revisit at M9


# ─── Dark-mode forcing ────────────────────────────────────────────────────
# Gradio 6 respects the `?__theme=dark` URL param, but sending the user to
# the raw URL is awkward. Injecting a one-line JS snippet on page load adds
# the `dark` class to <body>, which the Monochrome theme then honors.
FORCE_DARK_MODE_JS = """
() => {
    if (!document.body.classList.contains('dark')) {
        document.body.classList.add('dark');
    }
}
"""


# ─── UI construction ──────────────────────────────────────────────────────


def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks app (UI shell only; no functionality yet)."""
    theme = gr.themes.Monochrome()  # squarish by default (radius_none)

    with gr.Blocks(title="Local Meeting Summarizer", theme=theme) as demo:
        with gr.Sidebar():
            gr.Markdown("### Settings")

            gr.Textbox(
                label="Ollama host",
                value=DEFAULT_OLLAMA_HOST,
                placeholder="http://<host>:<port>",
                info="Where your Ollama server is reachable.",
            )
            gr.Textbox(
                label="Editor model (steps 2 & 5)",
                value=DEFAULT_EDITOR_MODEL,
                info="Used for cleanup + final formatting.",
            )
            gr.Textbox(
                label="Extractor model (step 4)",
                value=DEFAULT_EXTRACTOR_MODEL,
                info="Used for information extraction.",
            )
            gr.Button("Test connection", variant="secondary")

        with gr.Column():
            gr.Markdown("# Local Meeting Summarizer")
            gr.Markdown(
                "Upload a transcript to begin. Supports `.rtf` from Moonshine "
                "and `.md` from the local transcriber."
            )
            gr.Markdown(
                "_(Upload component comes online in milestone M4. This is the "
                "UI skeleton — M2.)_"
            )

        gr.Markdown(
            "<sub>Each tab runs independently — multiple tabs = multiple "
            "queue slots. Tab close ejects any models this session loaded.</sub>"
        )

        # Force dark-mode class on body at every page load.
        demo.load(fn=None, inputs=None, outputs=None, js=FORCE_DARK_MODE_JS)

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()
    demo = build_demo()
    demo.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
