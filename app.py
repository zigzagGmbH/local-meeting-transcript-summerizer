#!/usr/bin/env python3
"""
Gradio front-end for the Local Meeting Transcript Summarizer.

Thin skin over ``main.py``'s pipeline. See ``contexts/gradio_app.md`` for the
full spec, decisions, and milestone breakdown.

Run:
    uv run app.py

Then open http://localhost:7860 in a browser.

Status: M6 complete · M6.5 in progress (theme integration + raw/render toggle
+ sidebar compaction pass on top of L1+M1+M2).
"""

from __future__ import annotations

import atexit
import contextlib
import io
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import gradio as gr
import httpx
from dotenv import load_dotenv

from pipeline.step1_convert import convert
from pipeline.step2_cleanup import clean_transcript
from pipeline.step3_mapping import apply_speaker_mapping, detect_generic_speakers
from pipeline.step4_extraction import extract_information
from pipeline.step5_formatter import format_summary


# Load .env at module import so OLLAMA_HOST (and any other env-driven
# defaults below) are populated BEFORE the constants block evaluates.
# main() hard-fails if OLLAMA_HOST is still missing, matching main.py.
load_dotenv()


# ─── Defaults ─────────────────────────────────────────────────────────────

# No silent fallback to localhost — that hides misconfiguration. If
# OLLAMA_HOST isn't set in .env or the shell environment, this is "" and
# main() will refuse to launch.
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "")
DEFAULT_EDITOR_MODEL = "gemma4:26b"
DEFAULT_EXTRACTOR_MODEL = "gemma4:26b"

SERVER_PORT = 7860
SERVER_HOST = "0.0.0.0"  # reachable on LAN / inside Docker; revisit at M9

# HTTP timeout for Ollama probes. Short because these are reachability checks,
# not model-generation calls (which happen inside the step functions with
# their own implicit timeouts via ollama.Client).
OLLAMA_PROBE_TIMEOUT = 3.0

# Upload cap. Enforced at demo.launch(max_file_size=...); gr.File itself has
# no per-component size kwarg (issue #7825).
UPLOAD_MAX_SIZE = "10mb"

# Stable intermediate-file stem used by steps 2→5. Keeps the
# `input_md.stem.replace("_cleaned"/"_named"/"_extracted", …)` chain in the
# pipeline modules from producing ugly filenames when the upload's original
# stem happens to contain those substrings (M6.5 S3 fix). The user-visible
# download name uses ``uploaded_stem`` verbatim, so this is invisible from
# the outside — it only affects files inside the session tempdir.
STABLE_STEM = "transcript"

# Log panel height in lines (M6.5 L1). 12 lines ≈ one Ollama step's worth of
# prints visible at once. Past 12, the Textbox scrolls and autoscroll keeps
# the latest line in view unless the user manually scrolls up.
LOG_PANEL_LINES = 12

# Fixed height of the transcript preview AND the final summary render boxes
# in pixels. gr.Markdown(height=N, max_height=N) + container=True gives a
# theme-bordered box that scrolls internally past N pixels. Same height on
# both so the rendered vs raw toggle doesn't make the page jump.
PREVIEW_HEIGHT = 400
SUMMARY_HEIGHT = 500

# Raw summary textbox rows when "Raw" view is selected. Roughly matches the
# SUMMARY_HEIGHT pixel target — Gradio Textbox sizes by line count rather
# than CSS height, so there's no perfect correspondence; 20 lines is close
# enough that the radio toggle doesn't produce visible jumps.
SUMMARY_RAW_LINES = 20


# ─── Process-level tracking for shutdown hooks ────────────────────────────

# Every (host, model) pair ever touched by any session during this process's
# lifetime. Populated inside ``run_pipeline_generator``. Walked by the atexit
# and SIGTERM handlers to guarantee models are ejected on normal shutdown
# and `docker stop` (SIGTERM). Not bulletproof against `kill -9` — see spec
# risks table.
_ALL_MODELS_EVER_LOADED: set[tuple[str, str]] = set()


# ─── Custom CSS (M6.5 L1 + M1 + theme integration) ───────────────────────

# Injected via demo.launch(css=...). Two scopes:
#
#   .log-panel textarea — monospace + smaller font for the CLI-style log
#   Textbox. elem_classes=["log-panel"] selects the wrapper; the actual
#   <textarea> is a descendant.
#
#   .progress-* — styles for the custom gr.HTML progress bar.
#
# Color strategy: theme-aware with hardcoded fallbacks. ``--color-accent``
# is Gradio's canonical accent-color var (populated from the theme's
# ``primary`` palette). Nymbo_Theme sets this to #11ba88 (green). The
# fallback #11ba88 matches Nymbo explicitly in case the var isn't set for
# a given theme. ``--font-mono`` pulls the theme's mono font stack; Nymbo
# provides JetBrains Mono here.
#
# Error + cancelled fills stay hardcoded (red #ef4444, amber #f59e0b)
# because theme accent palettes don't have semantically correct slots for
# these states — using accent for everything would collapse the "something
# is wrong" signal into the normal success hue.
CUSTOM_CSS = """
.log-panel textarea {
    font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, 'SF Mono', 'Cascadia Code', Menlo, monospace) !important;
    font-size: 0.82em !important;
    line-height: 1.45 !important;
}

.progress-block {
    padding: 8px 2px 4px 2px;
}
.progress-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, Menlo, monospace);
    font-size: 0.9em;
    margin-bottom: 6px;
    color: var(--body-text-color, inherit);
}
.progress-pct {
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}
.progress-track {
    background: var(--block-border-color, rgba(128, 128, 128, 0.22));
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.35s ease, background 0.25s ease;
}

/* Small LED indicator for the connection status dot in the sidebar. The
   actual character is a colored emoji (🟢/🔴/⚪) — this just ensures
   vertical alignment with the adjacent refresh button. */
#conn-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1em;
    min-width: 28px;
}

/* Keep the compact refresh button aligned with the textbox it sits next
   to. Gradio's row baseline defaults can drift when a textbox has a
   label and the button doesn't. */
#test-btn {
    align-self: flex-end;
}
"""


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

# JS fired by the Copy button. Reads the raw markdown source from the
# final_summary_source Textbox's <textarea> rather than the rendered
# markdown's innerText, so the clipboard gets `# Heading` / `**bold**` /
# `| table |` syntax intact. See build_demo() for the matching Textbox.
# (M6.5 S1)
COPY_SUMMARY_JS = """
() => {
    const host = document.getElementById('final-summary-source');
    if (!host) {
        console.warn('final-summary-source element not found');
        return;
    }
    const ta = host.querySelector('textarea');
    if (!ta) {
        console.warn('final-summary-source textarea not found');
        return;
    }
    navigator.clipboard.writeText(ta.value || '');
}
"""


# ─── Progress-bar HTML helper (M6.5 M1) ──────────────────────────────────

def _progress_html(phase: str, pct: int, state: str = "running") -> str:
    """Render the progress bar's HTML for injection into a gr.HTML component.

    ``state`` picks the fill color and matches the four UX states:
      running   → theme accent (Nymbo: green)
      success   → theme accent (same green; icon + text differentiates)
      error     → red (hardcoded — no safe theme slot for destructive state)
      cancelled → amber (hardcoded — no safe theme slot for paused state)

    Inline ``background:`` uses ``var(--color-accent, #11ba88)`` so the bar
    tracks whatever theme is loaded; the #11ba88 fallback matches Nymbo
    explicitly for themes that don't populate the var. Modern browsers
    resolve CSS vars inside inline style attributes, which is what makes
    this work without a separate CSS class per state.

    ``pct`` is clamped to [0, 100]. Non-int inputs are coerced via int().
    The phase text is HTML-escaped because exception messages surfaced to
    the bar could contain `<` / `>`.
    """
    colors = {
        "running":   "var(--color-accent, #11ba88)",
        "success":   "var(--color-accent, #11ba88)",
        "error":     "#ef4444",
        "cancelled": "#f59e0b",
    }
    color = colors.get(state, colors["running"])
    try:
        pct_int = max(0, min(100, int(pct)))
    except (TypeError, ValueError):
        pct_int = 0
    safe_phase = (
        phase.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        '<div class="progress-block">'
        '<div class="progress-header">'
        f'<span class="progress-phase">{safe_phase}</span>'
        f'<span class="progress-pct">{pct_int}%</span>'
        '</div>'
        '<div class="progress-track">'
        f'<div class="progress-fill" style="width: {pct_int}%; background: {color};"></div>'
        '</div>'
        '</div>'
    )


# ─── Ollama helpers (pure I/O; no Gradio imports) ────────────────────────

def test_ollama_connection(host: str) -> tuple[bool, str]:
    """Ping ``GET {host}/api/tags`` with a short timeout.

    Returns (success, user-facing message). Best-effort; never raises.
    """
    if not host or not host.strip():
        return False, "Host is empty."
    url = f"{host.rstrip('/')}/api/tags"
    try:
        r = httpx.get(url, timeout=OLLAMA_PROBE_TIMEOUT)
        r.raise_for_status()
        return True, f"✓ Connected to {host}"
    except httpx.ConnectError:
        return False, f"✗ Cannot reach {host} (connection refused)"
    except httpx.TimeoutException:
        return False, f"✗ Cannot reach {host} (timed out after {OLLAMA_PROBE_TIMEOUT}s)"
    except httpx.HTTPStatusError as e:
        return False, f"✗ {host} responded with HTTP {e.response.status_code}"
    except Exception as e:  # pragma: no cover — catch-all for exotic failures
        return False, f"✗ {host}: {type(e).__name__}: {e}"


def list_available_models(host: str) -> list[str]:
    """Return list of model names currently pulled on Ollama.

    Returns [] on any connection failure. Never raises.
    """
    if not host or not host.strip():
        return []
    url = f"{host.rstrip('/')}/api/tags"
    try:
        r = httpx.get(url, timeout=OLLAMA_PROBE_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def validate_model_available(host: str, model: str) -> bool:
    """True iff ``model`` is in Ollama's pulled list. Does NOT attempt pull."""
    if not model or not host:
        return False
    return model in list_available_models(host)


def unload_model(host: str, model: str) -> None:
    """Send ``keep_alive=0`` to Ollama to evict a model.

    Best-effort: swallows all exceptions. Called from the pipeline's
    finally block, the session-cleanup hook, and the process-exit handlers.
    """
    if not host or not model:
        return
    url = f"{host.rstrip('/')}/api/generate"
    try:
        httpx.post(
            url,
            json={"model": model, "keep_alive": 0, "prompt": ""},
            timeout=OLLAMA_PROBE_TIMEOUT,
        )
    except Exception:
        pass


def preflight_check(
    host: str, editor_model: str, extractor_model: str
) -> tuple[bool, str]:
    """Gate function for Run.

    Verifies Ollama is reachable AND both required models are pulled.
    Returns (ready, user-facing message).
    """
    ok, msg = test_ollama_connection(host)
    if not ok:
        return False, msg
    models = set(list_available_models(host))
    missing = [m for m in (editor_model, extractor_model) if m and m not in models]
    if missing:
        return False, f"✗ Models not pulled on {host}: {', '.join(missing)}"
    return True, "✓ Ready"


# ─── Process-level shutdown handlers ─────────────────────────────────────

def _global_cleanup_loaded_models() -> None:
    """Walk the module-level set and unload every (host, model) pair.

    Invoked by ``atexit`` on normal interpreter exit and by the SIGTERM
    handler below. Idempotent — calling ``unload_model`` on an already-
    ejected model is a harmless HTTP roundtrip that Ollama no-ops.
    """
    for host, model in list(_ALL_MODELS_EVER_LOADED):
        unload_model(host, model)


def _sigterm_handler(signum: int, frame: Any) -> None:
    """SIGTERM → eject tracked models, then let the default exit path run.

    Docker's ``docker stop`` sends SIGTERM then waits ~10s before SIGKILL,
    so we have time to best-effort unload everything loaded on ziggie by
    this process's sessions before the container dies.

    NOTE: We do NOT install a SIGINT (Ctrl-C) handler. Gradio installs its
    own for graceful dev-loop shutdown; overriding it breaks the dev UX.
    ``atexit`` covers the Ctrl-C path regardless — SIGINT raises
    KeyboardInterrupt, interpreter exits, ``atexit`` fires.
    """
    _global_cleanup_loaded_models()
    # Re-raise with the default disposition so Gradio/uvicorn can tear down
    # their sockets cleanly.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _install_process_hooks() -> None:
    """Wire ``atexit`` + SIGTERM. Called once from ``main()``.

    Signal handlers must be installed from the main thread; calling this
    inside ``main()`` (before ``demo.launch()``) satisfies that.
    """
    atexit.register(_global_cleanup_loaded_models)
    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (ValueError, OSError):
        # Non-main-thread invocation (shouldn't happen from main()) or a
        # platform that doesn't support this signal — log and continue.
        print(
            "Warning: could not install SIGTERM handler; "
            "model cleanup on container stop will be best-effort.",
            file=sys.stderr,
        )


# ─── Session state factory ───────────────────────────────────────────────

def init_session_state() -> dict[str, Any]:
    """Factory for ``gr.State``. Returns a fresh dict per session.

    See ``contexts/gradio_app.md`` for the full shape. Brief reminder:
    ``log_text`` / ``progress_pct`` / ``progress_phase`` were added in
    M6.5 L1+M1 so ``on_stop`` can reconstruct the cancellation view at
    the exact point the user hit Stop (on_stop only receives state_val,
    no step-local context — the generator publishes its progress to
    state on every yield for this reason).

    Passed as ``gr.State(init_session_state())`` — the call produces a
    dict which Gradio then deepcopies per session. (Passing the callable
    itself, ``gr.State(init_session_state)``, does NOT work in Gradio 6.x:
    the function object is handed through to handlers verbatim instead of
    being invoked.)
    """
    return {
        "tempdir_path": None,
        "canonical_md": None,
        "uploaded_stem": None,
        "final_summary_path": None,
        "models_used": set(),
        "ollama_host": DEFAULT_OLLAMA_HOST,
        "run_in_progress": False,
        "log_text": "",
        "progress_pct": 0,
        "progress_phase": "",
    }


def _ensure_tempdir(state_val: dict[str, Any]) -> Path:
    """Lazily create the session tempdir on first upload. Returns the Path."""
    if not state_val.get("tempdir_path"):
        state_val["tempdir_path"] = tempfile.mkdtemp(prefix="meeting_summarizer_")
    return Path(state_val["tempdir_path"])


def cleanup_session(state_val: dict[str, Any] | None = None) -> None:
    """Per-session cleanup. Fired by ``gr.State``'s ``delete_callback``.

    Gradio 6 deletes a session's ``gr.State`` ~60 minutes after the user's
    browser disconnects. Ejects the session's touched models and removes
    its tempdir. Best-effort: all exceptions swallowed.

    The optional ``state_val`` arg silences a spurious Gradio 6
    ``check_function_inputs_match`` warning at State creation time. See
    spec changelog (M6 fix-up) for the full rationale.
    """
    if not isinstance(state_val, dict):
        return
    host = state_val.get("ollama_host", "")
    for model in list(state_val.get("models_used", ())):
        unload_model(host, model)
    tempdir_path = state_val.get("tempdir_path")
    if tempdir_path:
        shutil.rmtree(tempdir_path, ignore_errors=True)


# ─── UI event handlers ───────────────────────────────────────────────────

def _banner_update_for_host(host: str) -> dict:
    """Build a ``gr.update`` for the unreachable-Ollama banner."""
    ok, _ = test_ollama_connection(host)
    if ok:
        return gr.update(value="", visible=False)
    return gr.update(
        value=(
            f"⚠ Cannot reach Ollama at `{host}`. "
            "Update the host in the sidebar and click the **🔄** button."
        ),
        visible=True,
    )


def _connection_indicator(host: str) -> str:
    """LED indicator markdown for the sidebar.

    Returns a single emoji:
      🟢 — reachable
      🔴 — unreachable
      ⚪ — empty host (not tested)

    Called from ``on_startup``, ``on_host_change``, ``on_test_connection``.
    Each returns its own probe; if this ends up showing performance lag on
    slow networks we can cache per-host for a few seconds, but for now the
    OLLAMA_PROBE_TIMEOUT caps worst-case latency at 3s.
    """
    if not host or not host.strip():
        return "⚪"
    ok, _ = test_ollama_connection(host)
    return "🟢" if ok else "🔴"


def _model_indicator(host: str, model: str) -> str:
    """Return small markdown indicating model status. Empty string when no
    model typed; '—' when we can't tell (host unreachable or no models pulled)
    so the user doesn't see a false ✗.
    """
    if not model:
        return ""
    if not host:
        return "—"
    models = list_available_models(host)
    if not models:
        return "— _(host unreachable or no models pulled)_"
    if model in models:
        return "✓ available"
    return "✗ not pulled"


def on_startup(state_val: dict[str, Any]) -> tuple[dict, str, str, str]:
    """``demo.load`` handler. Checks reachability + validates default models.

    Returns a 4-tuple: (banner, editor_status, extractor_status,
    connection_indicator). Grew from 3 → 4 at M6.5 when the standalone
    ``Test connection`` button + status markdown were collapsed into a
    compact refresh button + LED indicator next to the host Textbox.
    """
    host = state_val.get("ollama_host", DEFAULT_OLLAMA_HOST)
    banner = _banner_update_for_host(host)
    editor_ind = _model_indicator(host, DEFAULT_EDITOR_MODEL)
    extractor_ind = _model_indicator(host, DEFAULT_EXTRACTOR_MODEL)
    conn_ind = _connection_indicator(host)
    return banner, editor_ind, extractor_ind, conn_ind


def on_host_change(
    host: str, editor: str, extractor: str, state_val: dict[str, Any]
) -> tuple[dict, dict, str, str, str]:
    """When host textbox changes: update session state, refresh banner,
    re-validate both model fields, and refresh the LED indicator.

    5-tuple return. On every keystroke in the host textbox we probe the
    server, which fires three HTTP requests (banner, model indicator ×2,
    LED indicator all independently call test_ollama_connection /
    list_available_models). Not ideal for rapid typing but acceptable
    given OLLAMA_PROBE_TIMEOUT=3s and the rarity of editing the host URL
    during normal use. If typing lag becomes annoying, debounce with a
    Gradio Timer or collapse into a single probe.
    """
    state_val["ollama_host"] = host
    banner = _banner_update_for_host(host)
    editor_ind = _model_indicator(host, editor)
    extractor_ind = _model_indicator(host, extractor)
    conn_ind = _connection_indicator(host)
    return state_val, banner, editor_ind, extractor_ind, conn_ind


def on_test_connection(host: str) -> tuple[str, dict]:
    """Compact refresh button handler.

    Returns (connection_indicator, banner). Replaces the old 'Test
    connection' text button + separate status markdown; now the button
    just updates the LED and the banner in one click.
    """
    ok, _ = test_ollama_connection(host)
    conn_ind = "🟢" if ok else "🔴"
    if ok:
        banner = gr.update(value="", visible=False)
    else:
        banner = gr.update(
            value=(
                f"⚠ Cannot reach Ollama at `{host}`. "
                "Check the host URL and try again."
            ),
            visible=True,
        )
    return conn_ind, banner


def on_view_mode_change(mode: str) -> tuple[dict, dict]:
    """Radio toggle between rendered markdown and raw markdown source.

    Returns (final_summary_md update, final_summary_source update). Only
    toggles visibility — values are preserved across toggles because
    Gradio doesn't reset component value on visibility change.

    CRITICAL: ``final_summary_source`` must never be flipped to
    ``visible=False`` or the Copy button's JS loses the <textarea> it
    reads from. "Rendered" mode uses ``visible="hidden"`` (CSS-hidden but
    in DOM); "Raw" mode uses ``visible=True``. Both keep the DOM element
    present. See COPY_SUMMARY_JS + the Textbox declaration in build_demo.
    """
    if mode == "Raw":
        return (
            gr.update(visible=False),      # final_summary_md hidden
            gr.update(visible=True),        # final_summary_source visible (still in DOM)
        )
    # "Rendered" (default)
    return (
        gr.update(visible=True),            # final_summary_md visible
        gr.update(visible="hidden"),        # final_summary_source hidden but still in DOM
    )


def on_stop(state_val: dict[str, Any]) -> tuple:
    """Stop-button click handler.

    Returns a 12-tuple matching the Run button's output layout:

        1. console_group visibility      (kept visible so user sees the log)
        2. progress_html value           (rebuilt with 'cancelled' state + last pct)
        3. log_panel value               (stored log + appended cancel marker)
        4. summary_section visibility    (hidden — nothing to show)
        5. view_mode update              (reset to "Rendered", hidden)
        6. final_summary_md update       (cleared + hidden)
        7. final_summary_source update   (value-only; stays visible="hidden")
        8. copy_btn visibility           (hidden)
        9. download_btn update           (cleared + hidden)
       10. run_btn interactive           (re-enabled)
       11. session_state                 (updated log_text)
       12. stop_btn visibility           (hidden)

    Grew from 10 → 12 at M6.5 second pass (added summary_section and
    view_mode for the rendered/raw toggle + panel-wrapper visibility).

    Non-streaming Ollama (spec F3): clicking Stop while a step is mid-call
    does not interrupt the HTTP request. The generator cancels when the
    current step returns (1–3 min on big models). The UI returns to idle
    immediately; the backend finishes catching up in the background. The
    appended log line tells the user to expect this delay.
    """
    log_text = state_val.get("log_text", "")
    if log_text and not log_text.endswith("\n"):
        log_text += "\n"
    log_text += (
        "\n⏸ Cancel requested. Current step will finish before models "
        "unload (non-streaming Ollama trade-off; usually 1–3 min).\n"
    )
    state_val["log_text"] = log_text

    pct = state_val.get("progress_pct", 0)
    phase = state_val.get("progress_phase", "")
    header = f"⏸ Cancelled during {phase}" if phase else "⏸ Cancelled by user"

    return (
        gr.update(visible=True),                                   # console_group
        gr.update(value=_progress_html(header, pct, "cancelled")),# progress_html
        gr.update(value=log_text),                                 # log_panel
        gr.update(visible=False),                                  # summary_section
        gr.update(value="Rendered", visible=False),                # view_mode (reset + hide)
        gr.update(value="", visible=False),                        # final_summary_md
        gr.update(value=""),                                       # final_summary_source (stays "hidden")
        gr.update(visible=False),                                  # copy_btn
        gr.update(value=None, visible=False),                      # download_btn
        gr.update(interactive=True),                               # run_btn
        state_val,                                                 # session_state
        gr.update(visible=False),                                  # stop_btn
    )


# Empty updates for the nine run-output components. Used by on_file_upload
# to clear stale results when a new file is uploaded or the upload is cleared.
#
# Grew 5 → 7 (M6.5 L1+M1+M2) → 9 (M6.5 panel + toggle pass) as the run
# output surface area grew. Order MUST match the event-listener outputs
# list in build_demo.
_RUN_OUTPUT_RESET = (
    gr.update(visible=False),                # console_group
    gr.update(value=""),                     # progress_html
    gr.update(value=""),                     # log_panel
    gr.update(visible=False),                # summary_section (whole panel hidden)
    gr.update(value="Rendered", visible=False),  # view_mode (reset + hide)
    gr.update(value="", visible=False),      # final_summary_md
    gr.update(value=""),                     # final_summary_source (stays "hidden")
    gr.update(visible=False),                # copy_btn
    gr.update(value=None, visible=False),    # download_btn
)


def on_file_upload(
    uploaded_file: str | None,
    state_val: dict[str, Any],
) -> tuple:
    """Handle file upload / clear.

    Returns a 16-tuple in this order:
        preview_md, error_md, run_btn, detected_speakers_state,
        speaker_map_state, session_state, meta_md,
        + 9 entries from _RUN_OUTPUT_RESET (console panel, summary panel,
          view_mode, summary + copy + download — all cleared/hidden).

    Grew from 12 → 14 (M6.5 L1+M1+M2) → 16 (panel + toggle pass).

    Any exception from step1 — including ``SystemExit`` raised on format-
    detection failures — is caught explicitly. A bare ``except Exception``
    would miss ``SystemExit`` (it's a ``BaseException`` subclass) and kill
    the worker.
    """
    # User cleared the file — reset everything to the pre-upload state.
    if uploaded_file is None:
        state_val["canonical_md"] = None
        state_val["uploaded_stem"] = None
        state_val["final_summary_path"] = None
        state_val["log_text"] = ""
        state_val["progress_pct"] = 0
        state_val["progress_phase"] = ""
        return (
            gr.update(value="", visible=False),   # preview
            gr.update(value="", visible=False),   # error
            gr.update(interactive=False),         # run button
            [],                                   # detected_speakers_state
            {},                                   # speaker_map_state
            state_val,                            # session_state
            gr.update(value="", visible=False),   # meta line
            *_RUN_OUTPUT_RESET,                   # clear stale run results
        )

    src = Path(uploaded_file)

    try:
        tempdir = _ensure_tempdir(state_val)
        raw_dir = tempdir / "raw_files"
        raw_dir.mkdir(parents=True, exist_ok=True)
        # Copy upload into session-owned tempdir so step1 writes its outputs
        # alongside and everything for this session lives under one root.
        dst = raw_dir / src.name
        shutil.copy(src, dst)
        _, md_path = convert(dst, raw_dir)
    except BaseException as e:
        err_type = type(e).__name__
        err_msg = str(e) or err_type
        state_val["canonical_md"] = None
        state_val["uploaded_stem"] = None
        state_val["final_summary_path"] = None
        state_val["log_text"] = ""
        state_val["progress_pct"] = 0
        state_val["progress_phase"] = ""
        return (
            gr.update(value="", visible=False),
            gr.update(
                value=f"❌ Could not ingest `{src.name}`: {err_msg}",
                visible=True,
            ),
            gr.update(interactive=False),
            [],
            {},
            state_val,
            gr.update(value="", visible=False),
            *_RUN_OUTPUT_RESET,
        )

    canonical_md = md_path.read_text(encoding="utf-8")
    state_val["canonical_md"] = str(md_path)
    state_val["uploaded_stem"] = src.stem
    state_val["final_summary_path"] = None  # prior runs' output is now stale
    state_val["log_text"] = ""
    state_val["progress_pct"] = 0
    state_val["progress_phase"] = ""

    speakers = detect_generic_speakers(canonical_md)
    n_turns = sum(
        1 for line in canonical_md.splitlines() if line.startswith("**")
    )
    meta_bits = [f"{n_turns} turns ingested"]
    if speakers:
        meta_bits.append(f"{len(speakers)} generic speakers detected")
    else:
        meta_bits.append("no generic speakers — Run enabled")
    meta_line = " · ".join(meta_bits)

    return (
        gr.update(value=canonical_md, visible=True),
        gr.update(value="", visible=False),
        gr.update(interactive=True),
        speakers,
        {},  # reset speaker_map for the new upload
        state_val,
        gr.update(value=f"<sub>{meta_line}</sub>", visible=True),
        *_RUN_OUTPUT_RESET,
    )


# ─── Pipeline orchestration ──────────────────────────────────────────────

def run_pipeline_generator(
    state_val: dict[str, Any],
    editor_model: str,
    extractor_model: str,
    ollama_host: str,
    speaker_map: dict[str, str],
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple]:
    """Run steps 2 → 5 on the session's ingested transcript.

    Generator: yields a 12-tuple at each phase boundary matching the Run
    button's outputs list:

        1. console_group       — visibility
        2. progress_html       — custom bar HTML (phase + % + state color)
        3. log_panel           — accumulated stdout from the step modules
        4. summary_section     — visibility of the whole summary panel
        5. view_mode           — radio reset on success, hidden otherwise
        6. final_summary_md    — rendered summary (visible on success)
        7. final_summary_source — raw source (value always set; visibility
                                   via view_mode toggle — see on_view_mode_change)
        8. copy_btn
        9. download_btn
       10. run_btn
       11. session_state
       12. stop_btn

    Step-to-model mapping (M6.5 polish pass):
      Step 2 (clean_transcript)   → editor_model
      Step 3 (apply_speaker_mapping) → pure, no model
      Step 4 (extract_information) → extractor_model
      Step 5 (format_summary)      → extractor_model  (was editor_model in M5)

    The step-5 reassignment matches the new sidebar labels where the
    extractor model is the one doing "information extraction + final
    formatting." If both textboxes point at the same model (default), this
    is a no-op; if users set different models, step 5 now uses the extractor.

    Stdout capture (L1):
    Each Ollama-touching step is wrapped in
    ``contextlib.redirect_stdout(io.StringIO())`` inside try/finally so
    partial output is still captured on exception. Ollama calls are
    non-streaming (F3), so prints from a step appear in the log panel as
    a burst when that step returns. Accepted trade-off — true streaming
    would require threads + a queue.

    Known limitation: ``redirect_stdout`` patches ``sys.stdout`` globally.
    Concurrent runs in the same process would interleave captures.
    Single-user usage is unaffected; revisit at M7+ if multi-user becomes
    a real scenario.

    Progress bar granularity: step-grain only — 0 / 25 / 50 / 75 / 100 %.
    No mid-step smoothing since Ollama response time is opaque.

    Pre-flight failures yield once and return WITHOUT entering the
    try/finally — so no models are touched. The error message is surfaced
    in the log panel itself (consistent with run-time log panel UX).

    ``GeneratorExit`` (from Stop's ``cancels=``) reaches ``finally``,
    ejecting models. The cancellation view is rendered by ``on_stop``
    which reads ``progress_pct`` + ``log_text`` from state.
    """

    # ── Pre-flight: no transcript ────────────────────────────────────────
    if not state_val.get("canonical_md"):
        yield (
            gr.update(visible=True),                                     # console_group
            gr.update(value=_progress_html("Cannot run", 0, "error")),  # progress_html
            gr.update(value="❌ No transcript ingested. Upload a file first."),  # log_panel
            gr.update(visible=False),                                    # summary_section
            gr.update(value="Rendered", visible=False),                  # view_mode
            gr.update(value="", visible=False),                          # final_summary_md
            gr.update(value=""),                                         # final_summary_source
            gr.update(visible=False),                                    # copy_btn
            gr.update(value=None, visible=False),                        # download_btn
            gr.update(interactive=True),                                 # run_btn
            state_val,
            gr.update(visible=False),                                    # stop_btn
        )
        return

    # ── Pre-flight: bad models / unreachable host ────────────────────────
    ok, msg = preflight_check(ollama_host, editor_model, extractor_model)
    if not ok:
        yield (
            gr.update(visible=True),
            gr.update(value=_progress_html("Pre-flight failed", 0, "error")),
            gr.update(value=f"❌ {msg}"),
            gr.update(visible=False),
            gr.update(value="Rendered", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=True),
            state_val,
            gr.update(visible=False),
        )
        return

    tempdir = Path(state_val["tempdir_path"])
    canonical_md_path = Path(state_val["canonical_md"])

    state_val["models_used"].update({editor_model, extractor_model})
    _ALL_MODELS_EVER_LOADED.add((ollama_host, editor_model))
    _ALL_MODELS_EVER_LOADED.add((ollama_host, extractor_model))
    state_val["run_in_progress"] = True

    state_val["log_text"] = ""
    state_val["progress_pct"] = 0
    state_val["progress_phase"] = ""

    try:
        # ── Step 1/4: clean_transcript (editor_model) ────────────────────
        state_val["progress_pct"] = 0
        state_val["progress_phase"] = "Step 1/4 · Cleaning transcript"
        yield (
            gr.update(visible=True),
            gr.update(value=_progress_html(state_val["progress_phase"] + "…", 0, "running")),
            gr.update(value=""),                   # log empty before step 2 runs
            gr.update(visible=False),              # summary_section hidden
            gr.update(value="Rendered", visible=False),  # view_mode hidden
            gr.update(value="", visible=False),    # final_summary_md hidden
            gr.update(value=""),                   # final_summary_source cleared (stays "hidden")
            gr.update(visible=False),              # copy_btn hidden
            gr.update(value=None, visible=False),  # download_btn hidden
            gr.update(interactive=False),          # run_btn disabled during run
            state_val,
            gr.update(visible=True),               # stop_btn visible during run
        )

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                cleaned_path = clean_transcript(
                    canonical_md_path,
                    tempdir / "cleaned",
                    editor_model,
                    ollama_host,
                )
        finally:
            state_val["log_text"] += buf.getvalue()

        # S3 fix: rename step 2's output to a stable stem before chaining.
        stable_cleaned_path = cleaned_path.parent / f"{STABLE_STEM}_cleaned.md"
        if cleaned_path != stable_cleaned_path:
            cleaned_path.rename(stable_cleaned_path)
            cleaned_path = stable_cleaned_path

        # ── Step 2/4: apply_speaker_mapping (pure, no model) ─────────────
        state_val["progress_pct"] = 25
        state_val["progress_phase"] = "Step 2/4 · Applying speaker names"
        yield (
            gr.update(visible=True),
            gr.update(value=_progress_html(state_val["progress_phase"] + "…", 25, "running")),
            gr.update(value=state_val["log_text"]),
            gr.update(visible=False),
            gr.update(value="Rendered", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=False),
            state_val,
            gr.update(visible=True),
        )
        cleaned_text = cleaned_path.read_text(encoding="utf-8")
        named_text = apply_speaker_mapping(cleaned_text, speaker_map or {})
        named_dir = tempdir / "named"
        named_dir.mkdir(parents=True, exist_ok=True)
        named_path = named_dir / f"{STABLE_STEM}_named.md"
        named_path.write_text(named_text, encoding="utf-8")

        # ── Step 3/4: extract_information (extractor_model) ──────────────
        state_val["progress_pct"] = 50
        state_val["progress_phase"] = "Step 3/4 · Extracting information"
        yield (
            gr.update(visible=True),
            gr.update(value=_progress_html(state_val["progress_phase"] + "…", 50, "running")),
            gr.update(value=state_val["log_text"]),
            gr.update(visible=False),
            gr.update(value="Rendered", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=False),
            state_val,
            gr.update(visible=True),
        )
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                extracted_path = extract_information(
                    named_path,
                    tempdir / "extracted",
                    extractor_model,
                    ollama_host,
                )
        finally:
            state_val["log_text"] += buf.getvalue()

        # ── Step 4/4: format_summary (extractor_model — M6.5 polish) ─────
        # Previously used editor_model. Re-assignment matches the updated
        # sidebar labels where the extractor handles both extraction AND
        # final formatting. Effect for the default config (editor ==
        # extractor) is zero; for users with distinct models, step 5 now
        # stays on the extractor rather than re-loading the editor.
        state_val["progress_pct"] = 75
        state_val["progress_phase"] = "Step 4/4 · Formatting summary"
        yield (
            gr.update(visible=True),
            gr.update(value=_progress_html(state_val["progress_phase"] + "…", 75, "running")),
            gr.update(value=state_val["log_text"]),
            gr.update(visible=False),
            gr.update(value="Rendered", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=False),
            state_val,
            gr.update(visible=True),
        )
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                final_path = format_summary(
                    extracted_path,
                    tempdir / "final",
                    extractor_model,          # CHANGED from editor_model
                    ollama_host,
                )
        finally:
            state_val["log_text"] += buf.getvalue()

        # Rename the summary file to something user-friendly for download,
        # based on the original upload's stem.
        stem = state_val.get("uploaded_stem") or final_path.stem
        download_dir = tempdir / "download"
        download_dir.mkdir(parents=True, exist_ok=True)
        download_path = download_dir / f"{stem}_summary.md"
        shutil.copy(final_path, download_path)
        state_val["final_summary_path"] = str(download_path)

        final_content = download_path.read_text(encoding="utf-8")

        # ── Terminal success ─────────────────────────────────────────────
        # Summary section becomes visible. view_mode radio shows + resets
        # to "Rendered". final_summary_md filled + visible. 
        # final_summary_source filled (value only — visibility stays
        # "hidden" until the user toggles the radio to "Raw"). Copy +
        # Download revealed.
        state_val["progress_pct"] = 100
        state_val["progress_phase"] = "Done"
        yield (
            gr.update(visible=True),                             # console_group
            gr.update(value=_progress_html("✅ Done", 100, "success")),
            gr.update(value=state_val["log_text"]),
            gr.update(visible=True),                             # summary_section shown
            gr.update(value="Rendered", visible=True),           # view_mode shown + reset
            gr.update(value=final_content, visible=True),       # final_summary_md
            gr.update(value=final_content),                      # final_summary_source (stays "hidden")
            gr.update(visible=True),                             # copy_btn
            gr.update(value=str(download_path), visible=True),   # download_btn
            gr.update(interactive=True),                         # run_btn re-enabled
            state_val,
            gr.update(visible=False),                            # stop_btn hidden
        )

    except (Exception, SystemExit) as e:
        err_type = type(e).__name__
        err_msg = str(e) or err_type
        state_val["log_text"] += f"\n❌ {err_type}: {err_msg}\n"
        pct = state_val.get("progress_pct", 0)
        phase = state_val.get("progress_phase", "Pipeline")
        header = f"❌ Failed at {phase}" if phase else "❌ Failed"
        yield (
            gr.update(visible=True),
            gr.update(value=_progress_html(header, pct, "error")),
            gr.update(value=state_val["log_text"]),
            gr.update(visible=False),                            # summary_section hidden on error
            gr.update(value="Rendered", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=True),
            state_val,
            gr.update(visible=False),
        )

    finally:
        state_val["run_in_progress"] = False
        for m in list(state_val.get("models_used", ())):
            unload_model(ollama_host, m)


# ─── UI construction ──────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks app."""
    # NOTE: ``css=CUSTOM_CSS`` is NOT passed here. In Gradio 6.0 the ``css``
    # kwarg was moved from the Blocks constructor to ``demo.launch()``.
    # Passing it here emits a UserWarning and is ignored.
    with gr.Blocks(title="Local Meeting Summarizer") as demo:
        session_state = gr.State(
            value=init_session_state(),
            delete_callback=cleanup_session,
        )
        detected_speakers_state = gr.State([])
        speaker_map_state = gr.State({})

        # ── Sidebar ──────────────────────────────────────────────────────
        with gr.Sidebar():
            gr.Markdown("### Settings")

            # Ollama host row: textbox + compact 🔄 button + LED indicator.
            # Inspired by OpenWebUI's connection edit modal (user screenshot).
            # Label is rendered separately above the row so the button and
            # indicator align with the textbox baseline rather than its
            # label. The textbox has show_label=False but keeps its
            # placeholder. Replaces the old "Test connection" button + a
            # separate status markdown that used to sit below the models.
            gr.Markdown("**Ollama host**")
            with gr.Row():
                ollama_host = gr.Textbox(
                    value=DEFAULT_OLLAMA_HOST,
                    placeholder="http://<host>:<port>",
                    show_label=False,
                    container=False,
                    scale=5,
                )
                test_btn = gr.Button(
                    "🔄",
                    scale=0,
                    min_width=40,
                    variant="secondary",
                    elem_id="test-btn",
                )
                connection_indicator = gr.Markdown(
                    "⚪",
                    elem_id="conn-indicator",
                )

            # Step-to-model labels corrected (M6.5 polish):
            #   Editor = step 2 (cleanup only)
            #   Extractor = steps 4 + 5 (extraction + final formatting)
            # run_pipeline_generator's step-5 call now also uses
            # extractor_model to match.
            editor_model = gr.Textbox(
                label="Editor model (step 2)",
                value=DEFAULT_EDITOR_MODEL,
                info="Used for cleanup.",
            )
            editor_status = gr.Markdown("", elem_id="editor-status")

            extractor_model = gr.Textbox(
                label="Extractor model (steps 4 & 5)",
                value=DEFAULT_EXTRACTOR_MODEL,
                info="Used for information extraction + final formatting.",
            )
            extractor_status = gr.Markdown("", elem_id="extractor-status")

        # ── Main column ──────────────────────────────────────────────────
        with gr.Column():
            # Startup banner: shown only when Ollama is unreachable.
            banner = gr.Markdown("", visible=False, elem_id="ollama-banner")

            gr.Markdown("# Local Meeting Summarizer")
            gr.Markdown(
                "Upload an exported meeting transcript (`.rtf`) from "
                "[moonshine-notetaker](https://note-taker.moonshine.ai/) or "
                "from the zz's local transcriber (`.md`). Files above 10 MB "
                "are rejected; ~2.5h of speech is the practical ceiling "
                "(LLM context window)."
            )

            # S2: multi-tab note in "Good to know" accordion. Spec default
            # is open=False; user prefers open=True and that preference is
            # preserved here.
            with gr.Accordion("Good to know", open=True):
                gr.Markdown(
                    "Each tab runs independently — multiple tabs = multiple "
                    "queue slots. When you close the tab, the models your "
                    "session loaded and its temp files are cleaned up "
                    "automatically (within about an hour)."
                )

            # ── Upload ────────────────────────────────────────────────────
            upload = gr.File(
                label="Transcript",
                file_types=[".rtf", ".md"],
                file_count="single",
                elem_id="transcript-upload",
            )

            # Inline error message (e.g. "Could not detect transcript format").
            error_md = gr.Markdown("", visible=False, elem_id="ingest-error")

            # One-line status under the upload — turn count + speaker summary.
            meta_md = gr.Markdown("", visible=False, elem_id="ingest-meta")

            # ── Transcript preview (M6.5 panel wrapper) ──────────────────
            # container=True + padding=True gives the Markdown the theme's
            # panel border + internal padding, matching what the user
            # asked for ("border stroke only if the theme component
            # provides"). height=max_height=PREVIEW_HEIGHT gives internal
            # scrolling past that height.
            preview_md = gr.Markdown(
                "",
                label="Preview",
                height=PREVIEW_HEIGHT,
                max_height=PREVIEW_HEIGHT,
                visible=False,
                container=True,
                padding=True,
                elem_id="transcript-preview",
            )

            # ── Dynamic speaker form ─────────────────────────────────────
            @gr.render(inputs=[detected_speakers_state])
            def render_speaker_form(speakers: list[str]):
                if not speakers:
                    return
                gr.Markdown(
                    "### Speaker names\n"
                    "Enter a real name for each detected speaker, or leave a "
                    "field blank to keep the original `Speaker N` label."
                )
                for tag in speakers:
                    tb = gr.Textbox(
                        label=tag,
                        placeholder="Leave blank to keep original label",
                    )

                    def _make_updater(captured_tag: str):
                        def _update(new_val: str, current_map: dict[str, str]):
                            new_map = dict(current_map or {})
                            if new_val and new_val.strip():
                                new_map[captured_tag] = new_val.strip()
                            else:
                                new_map.pop(captured_tag, None)
                            return new_map
                        return _update

                    tb.change(
                        _make_updater(tag),
                        inputs=[tb, speaker_map_state],
                        outputs=[speaker_map_state],
                    )

            # Run + Stop share a row.
            with gr.Row():
                run_btn = gr.Button(
                    "Run",
                    variant="primary",
                    interactive=False,
                    elem_id="run-btn",
                )
                stop_btn = gr.Button(
                    "Stop",
                    variant="stop",
                    visible=False,
                    elem_id="stop-btn",
                )

            # ── Console panel: progress bar + log (M6.5 L1+M1+M2) ────────
            with gr.Group(visible=False) as console_group:
                progress_html = gr.HTML(value="", elem_id="progress-html")
                log_panel = gr.Textbox(
                    value="",
                    label="",
                    show_label=False,
                    lines=LOG_PANEL_LINES,
                    max_lines=LOG_PANEL_LINES,
                    autoscroll=True,
                    interactive=False,
                    elem_id="log-panel",
                    elem_classes=["log-panel"],
                    placeholder="",
                )

            # ── Results section (separate panel further down) ────────────
            # Wrapped in gr.Column(variant="panel") for the theme-provided
            # border stroke. visible=False on the column hides the entire
            # section (including its border) pre-run and on error/cancel,
            # so we don't show an empty bordered box waiting for content.
            with gr.Column(variant="panel", visible=False) as summary_section:
                # Radio toggle between rendered markdown and raw source.
                # show_label=False + container=False keep it tight
                # horizontally. horizontal orientation comes from Gradio's
                # default for 2-option radios. Value drives
                # on_view_mode_change which flips visibility of the two
                # summary components below.
                view_mode = gr.Radio(
                    choices=["Rendered", "Raw"],
                    value="Rendered",
                    show_label=False,
                    container=False,
                    interactive=True,
                    elem_id="view-mode",
                )

                # Rendered markdown. container=True + padding=True +
                # height=max_height=SUMMARY_HEIGHT gives the theme's
                # bordered panel with internal scrolling past the fixed
                # height (user asks #3).
                final_summary_md = gr.Markdown(
                    "",
                    label="Meeting summary",
                    height=SUMMARY_HEIGHT,
                    max_height=SUMMARY_HEIGHT,
                    container=True,
                    padding=True,
                    visible=False,
                    elem_id="final-summary",
                )

                # Raw markdown source — same component that powers the
                # Copy button via COPY_SUMMARY_JS. DOM-persistent via
                # visible="hidden" so the <textarea> is always available
                # for the JS lookup even when the user has "Rendered"
                # selected. Switching the radio to "Raw" flips it to
                # visible=True (still in DOM). MUST NEVER go to
                # visible=False — that would remove the element and break
                # Copy. See COPY_SUMMARY_JS comment.
                #
                # lines=max_lines=SUMMARY_RAW_LINES keeps the Raw box
                # roughly the same vertical size as the Rendered box so
                # the radio toggle doesn't jump the layout.
                final_summary_source = gr.Textbox(
                    value="",
                    visible="hidden",
                    elem_id="final-summary-source",
                    interactive=True,
                    show_label=False,
                    lines=SUMMARY_RAW_LINES,
                    max_lines=SUMMARY_RAW_LINES,
                )

                with gr.Row():
                    copy_btn = gr.Button(
                        "Copy",
                        variant="secondary",
                        visible=False,
                        elem_id="copy-btn",
                    )
                    download_btn = gr.DownloadButton(
                        "Download",
                        visible=False,
                        elem_id="download-btn",
                    )

        # ── Event wiring ──────────────────────────────────────────────────

        demo.load(
            on_startup,
            inputs=[session_state],
            outputs=[banner, editor_status, extractor_status, connection_indicator],
        )
        demo.load(fn=None, inputs=None, outputs=None, js=FORCE_DARK_MODE_JS)

        ollama_host.change(
            on_host_change,
            inputs=[ollama_host, editor_model, extractor_model, session_state],
            outputs=[session_state, banner, editor_status, extractor_status, connection_indicator],
        )

        editor_model.change(
            _model_indicator,
            inputs=[ollama_host, editor_model],
            outputs=[editor_status],
        )
        extractor_model.change(
            _model_indicator,
            inputs=[ollama_host, extractor_model],
            outputs=[extractor_status],
        )

        # Compact refresh button in the host row — replaces the old
        # standalone "Test connection" button + separate status line.
        test_btn.click(
            on_test_connection,
            inputs=[ollama_host],
            outputs=[connection_indicator, banner],
        )

        # Radio toggle — rendered vs raw. Only toggles visibility; values
        # are set by the run_pipeline_generator's success yield and
        # preserved across visibility changes.
        view_mode.change(
            on_view_mode_change,
            inputs=[view_mode],
            outputs=[final_summary_md, final_summary_source],
        )

        upload.change(
            on_file_upload,
            inputs=[upload, session_state],
            outputs=[
                preview_md,
                error_md,
                run_btn,
                detected_speakers_state,
                speaker_map_state,
                session_state,
                meta_md,
                console_group,
                progress_html,
                log_panel,
                summary_section,
                view_mode,
                final_summary_md,
                final_summary_source,
                copy_btn,
                download_btn,
            ],
        )

        run_event = run_btn.click(
            run_pipeline_generator,
            inputs=[
                session_state,
                editor_model,
                extractor_model,
                ollama_host,
                speaker_map_state,
            ],
            outputs=[
                console_group,
                progress_html,
                log_panel,
                summary_section,
                view_mode,
                final_summary_md,
                final_summary_source,
                copy_btn,
                download_btn,
                run_btn,
                session_state,
                stop_btn,
            ],
        )

        stop_btn.click(
            on_stop,
            inputs=[session_state],
            outputs=[
                console_group,
                progress_html,
                log_panel,
                summary_section,
                view_mode,
                final_summary_md,
                final_summary_source,
                copy_btn,
                download_btn,
                run_btn,
                session_state,
                stop_btn,
            ],
            cancels=[run_event],
        )

        copy_btn.click(
            None,
            inputs=None,
            outputs=None,
            js=COPY_SUMMARY_JS,
        )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────

def main() -> None:
    if not DEFAULT_OLLAMA_HOST:
        print(
            "Error: OLLAMA_HOST is missing. Please define it in a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    _install_process_hooks()

    demo = build_demo()
    demo.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        inbrowser=True,
        theme=gr.Theme.from_hub("Nymbo/Nymbo_Theme"),
        css=CUSTOM_CSS,  # Gradio 6 moved `css` from Blocks() to launch()
        max_file_size=UPLOAD_MAX_SIZE,
    )


if __name__ == "__main__":
    main()
