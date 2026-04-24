#!/usr/bin/env python3
"""
Markdown → PDF conversion.

Post-pipeline, opt-in export step. Converts the final meeting-minutes
markdown into a PDF using the ``markdown-pdf`` (vb64) library. Pure
Python; no system libraries required. Runs inside the CPU-only Docker
image unchanged.

Used by:
  * ``main.py`` when the CLI is invoked with ``--pdf``.
  * ``app.py`` when the Web UI user clicks the "PDF" radio option.

Not exposed via the MCP tool (``summarize_transcript``). The MCP return
type stays ``str`` (markdown) for v1 — see ``contexts/pdf_export.md``
rationale note D5.
"""

from __future__ import annotations

from pathlib import Path

from markdown_pdf import MarkdownPdf, Section


def md_to_pdf(
    md_source: str | Path,
    out_path: Path,
    css: str | None = None,
) -> Path:
    """Convert markdown to PDF.

    Args:
        md_source: Either markdown text (str) or a path to a ``.md``
            file (Path). Path inputs are read with UTF-8 encoding.
        out_path: Where to write the PDF. Parent directory must
            already exist.
        css: Optional CSS to apply via ``markdown-pdf``'s ``user_css``
            hook. None = library defaults (system serif, plain table
            borders). Reserved for future branding work; unused in v1.

    Returns:
        ``out_path`` (same object as the input argument, for chaining).

    Raises:
        ValueError: if the markdown source is empty or whitespace only.
        OSError: if the output file cannot be written (bad path,
            permission denied, disk full, etc.). Propagated from the
            underlying ``markdown-pdf`` call.
    """
    # Resolve source → text.
    if isinstance(md_source, Path):
        md_text = md_source.read_text(encoding="utf-8")
    else:
        md_text = md_source

    if not md_text or not md_text.strip():
        raise ValueError("Cannot convert empty markdown to PDF.")

    # toc_level=2 keeps the generated TOC to H1+H2 headings (our
    # summaries have H1 title + H2 section heads + H3 sub-sections;
    # deeper levels clutter the TOC).
    # optimize=True compresses the output (~20-30% smaller files,
    # negligible CPU cost for documents this size).
    pdf = MarkdownPdf(toc_level=2, optimize=True)

    if css is not None:
        pdf.add_section(Section(md_text), user_css=css)
    else:
        pdf.add_section(Section(md_text))

    pdf.save(str(out_path))
    return out_path
