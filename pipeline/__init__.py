"""meeting-summarizer-pipeline — local-first meeting transcript extraction and formatting."""

__version__ = "0.1.0"


def announce(n: int, total: int, verb: str, model: str | None = None) -> None:
    """Print a consistent step banner to stdout.

    Shared by the CLI orchestrator (main.py) and the Gradio front-end
    (app.py) so terminal output looks the same whether you run the
    pipeline directly or through the web UI. The CLI has five steps
    including step-1 ingestion; the web UI folds step 1 into its
    file-upload handler and prints four. Pass the appropriate ``total``.

    Example:
        announce(2, 5, "Cleaning transcript", "gemma4:26b")
        # → "\\n[2/5] Cleaning transcript using gemma4:26b..."
    """
    suffix = f" using {model}" if model else ""
    print(f"\n[{n}/{total}] {verb}{suffix}...")
