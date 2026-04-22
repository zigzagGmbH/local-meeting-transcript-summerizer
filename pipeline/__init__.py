"""meeting-summarizer-pipeline — local-first meeting transcript extraction and formatting."""

from typing import Iterable

__version__ = "0.1.0"

# Shared banner width so the start/done separators match across paths.
_BANNER_WIDTH = 40


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


def announce_start(label: str, source: str) -> None:
    """Print a run-start banner. Matches ``announce_done`` visually.

    Used at the top of a pipeline run to mark where one invocation
    begins in a long-running service's log. Intentionally includes a
    leading blank line so it visually separates from any prior run's
    tail output.

    Example:
        announce_start("MCP summarize_transcript", "meeting.rtf")
        # → "\\n🔨 MCP summarize_transcript: meeting.rtf"
        # → "----------------------------------------"
    """
    print(f"\n🔨 {label}: {source}")
    print("-" * _BANNER_WIDTH)


def announce_done(char_count: int, destination: str) -> None:
    """Print a run-done banner. Pairs with ``announce_start``.

    The ``destination`` clause lets callers distinguish which entry
    point finished — e.g. ``"Returning to MCP client"`` for the MCP
    path, ``"Rendered in Web UI"`` for the Gradio path.

    Example:
        announce_done(3702, "Returning to MCP client")
        # → "----------------------------------------"
        # → "✅ Summary ready (3,702 chars). Returning to MCP client."
    """
    print("-" * _BANNER_WIDTH)
    print(f"✅ Summary ready ({char_count:,} chars). {destination}.")


def announce_unload(host: str, models: Iterable[str]) -> None:
    """Print the VRAM-eject header + one bullet line per model.

    ``host`` is accepted so callers pass whatever host was actually
    used for this run (matches the pattern from ``summarize_transcript``
    where the host may be a per-call override), even though the
    header itself doesn't echo it — we keep that for potential future
    use / debugging log augmentation. The function does NOT perform
    the unload — callers use ``unload_model`` themselves and then call
    ``announce_unload_result`` per model to report success or failure.

    This split keeps ``pipeline/__init__.py`` free of HTTP concerns
    while letting both entry points emit identical log lines.

    De-duplicates models while preserving first-seen order so passing
    a ``set`` still produces stable output. Emits nothing if no
    non-empty models are passed.

    Example:
        announce_unload("http://ollama:11434", {"gemma4:26b"})
        # → ""
        # → "Ejecting models from VRAM..."
    """
    _ = host  # accepted for symmetry with caller state; unused today
    seen = {m for m in models if m}
    if not seen:
        return
    print("\nEjecting models from VRAM...")


def announce_unload_result(model: str, ok: bool, error: str | None = None) -> None:
    """Print one per-model unload result line.

    Called once per model AFTER ``announce_unload`` (the header) and
    AFTER the caller has tried ``unload_model``. Output matches the
    old hand-rolled format in ``summarize_transcript``'s finally block
    so nothing in existing log-parsing tools needs to change.

    Example:
        announce_unload_result("gemma4:26b", ok=True)
        # → "  - Successfully unloaded: gemma4:26b"

        announce_unload_result("gemma4:26b", ok=False, error="timeout")
        # → "  - Note: could not confirm unload for gemma4:26b: timeout"
    """
    if ok:
        print(f"  - Successfully unloaded: {model}")
    else:
        suffix = f": {error}" if error else ""
        print(f"  - Note: could not confirm unload for {model}{suffix}")
