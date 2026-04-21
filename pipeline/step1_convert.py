#!/usr/bin/env python3
"""
Step 1: Ingest + Normalize a meeting transcript.

Dispatches by file suffix:
  - .rtf  ->  Moonshine.ai transcript (via striprtf)
  - .md   ->  Our local transcriber's export (M2, not yet implemented)

Both paths emit the same canonical markdown shape:

    # Transcript: <source_filename>

    **Speaker N:** merged text for this turn...

    **Speaker M:** merged text for this turn...

Plus a JSON sidecar with turn stats.

Usage:
    uv run python -m pipeline.step1_convert input.rtf [--out-dir .]
    uv run python -m pipeline.step1_convert input.md  [--out-dir .]   # M2+

Dependencies:
    striprtf  (pure-python, no native deps)

Output:
    <stem>.json  — structured turns, suitable for programmatic use
    <stem>.md    — human-readable, optimized for LLM summarization input
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from striprtf.striprtf import rtf_to_text

SPEAKER_RE = re.compile(r"^\s*(Speaker\s+\d+)\s*:\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Shared parsing/emission helpers (format-agnostic)
# ---------------------------------------------------------------------------


def parse_turns(plain_text: str) -> list[dict]:
    """Group consecutive lines under each `Speaker N:` header into one turn."""
    turns: list[dict] = []
    current_speaker: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_speaker and buffer:
            text = " ".join(s.strip() for s in buffer if s.strip())
            if text:
                turns.append(
                    {
                        "index": len(turns),
                        "speaker": current_speaker,
                        "text": text,
                    }
                )

    for raw in plain_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = SPEAKER_RE.match(line)
        if m:
            flush()
            current_speaker = m.group(1).title()  # normalize "speaker 1" -> "Speaker 1"
            buffer = []
        else:
            buffer.append(line)
    flush()
    return turns


def build_markdown(turns: list[dict], source_name: str) -> str:
    lines = [f"# Transcript: {source_name}", ""]
    for t in turns:
        lines.append(f"**{t['speaker']}:** {t['text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_json(turns: list[dict], source_name: str) -> dict:
    speaker_counts = Counter(t["speaker"] for t in turns)
    word_count = sum(len(t["text"].split()) for t in turns)
    return {
        "source": source_name,
        "n_turns": len(turns),
        "speakers": sorted(
            speaker_counts.keys(),
            key=lambda s: int(s.split()[-1]) if s.split()[-1].isdigit() else 0,
        ),
        "turns_per_speaker": dict(speaker_counts),
        "word_count": word_count,
        "turns": turns,
    }


def _write_outputs(
    turns: list[dict], source_name: str, out_dir: Path, stem: str
) -> tuple[Path, Path]:
    """Shared writer: emit <stem>.json + <stem>.md into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    json_path.write_text(
        json.dumps(build_json(turns, source_name), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(build_markdown(turns, source_name), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# Format-specific ingesters
# ---------------------------------------------------------------------------


def _ingest_moonshine_rtf(rtf_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Parse a Moonshine.ai RTF transcript into canonical markdown + JSON."""
    rtf_text = rtf_path.read_text(encoding="utf-8", errors="replace")
    plain = rtf_to_text(rtf_text, errors="ignore")
    turns = parse_turns(plain)
    if not turns:
        raise SystemExit(
            f"No speaker turns found in {rtf_path}. Is this really a moonshine RTF?"
        )
    return _write_outputs(turns, rtf_path.name, out_dir, rtf_path.stem)


def _ingest_markdown(md_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Parse our local transcriber's .md export into canonical markdown + JSON.

    Stubbed until M2. See contexts/multi_format_ingest.md.
    """
    raise NotImplementedError(
        f"Markdown ingest not yet implemented (milestone M2). "
        f"Received: {md_path}. "
        f"For now, only .rtf inputs are supported."
    )


# ---------------------------------------------------------------------------
# Public entry point (dispatch by suffix)
# ---------------------------------------------------------------------------


def convert(input_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Ingest a transcript file and emit canonical markdown + JSON sidecar.

    Dispatches by file suffix:
      .rtf  ->  Moonshine RTF ingester
      .md   ->  Local transcriber markdown ingester (M2)

    Returns (json_path, md_path).
    """
    suffix = input_path.suffix.lower()
    if suffix == ".rtf":
        return _ingest_moonshine_rtf(input_path, out_dir)
    if suffix == ".md":
        return _ingest_markdown(input_path, out_dir)
    raise SystemExit(
        f"Unsupported input format: '{suffix}'. Expected .rtf or .md."
    )


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def main() -> None:
    doc_lines = (__doc__ or "").splitlines()
    ap = argparse.ArgumentParser(
        description=doc_lines[1] if len(doc_lines) > 1 else ""
    )
    ap.add_argument(
        "input_file",
        type=Path,
        help="Input transcript (.rtf from Moonshine, or .md from our transcriber)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: cwd)",
    )
    args = ap.parse_args()

    json_path, md_path = convert(args.input_file, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
