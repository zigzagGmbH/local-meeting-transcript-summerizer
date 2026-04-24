#!/usr/bin/env python3
"""
Master Orchestrator for the Meeting Summarization Pipeline.
Processes a meeting transcript (.rtf from Moonshine or .md from our transcriber)
through cleanup, mapping, extraction, and formatting.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from ollama import Client

# Import the core functions from our pipeline modules
from pipeline import announce
from pipeline.pdf_export import md_to_pdf
from pipeline.step1_convert import convert
from pipeline.step2_cleanup import clean_transcript
from pipeline.step3_mapping import map_speakers
from pipeline.step4_extraction import extract_information
from pipeline.step5_formatter import format_summary


def main():
    # Load environment variables
    load_dotenv()

    # Strictly enforce the presence of OLLAMA_HOST
    ollama_host = os.environ.get("OLLAMA_HOST")
    if not ollama_host:
        print("Error: OLLAMA_HOST is missing. Please define it in a .env file.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="End-to-end Local LLM Meeting Summarizer."
    )

    # Required Input
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the transcript (.rtf from Moonshine, or .md from our transcriber)",
    )

    # Global/Path Options
    parser.add_argument(
        "--out-dir", type=Path, default=Path("./output"), help="Base output directory"
    )

    # FIX: Use the .env variable as the default here!
    parser.add_argument(
        "--host",
        type=str,
        default=ollama_host,
        help="Ollama host URL",
    )
    
    # Model Selection Options
    parser.add_argument(
        "--editor-model",
        type=str,
        default="gemma4:26b",
        help="Model used for Cleanup (Step 2) and Formatting (Step 5)",
    )
    parser.add_argument(
        "--extractor-model",
        type=str,
        default="gemma4:26b",
        help="Model used for Information Extraction (Step 4)",
    )

    # Opt-in PDF export. Default off so scripts built against the
    # pre-M11 CLI see zero behavioural change. When passed, the
    # markdown-pdf conversion runs AFTER the pipeline's success
    # signal — a conversion failure warns but does not flip the
    # exit code.
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write a PDF alongside the markdown summary. Off by default.",
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    suffix = args.input_file.suffix.lower()
    if suffix not in {".rtf", ".md"}:
        print(f"Error: Unsupported input format '{suffix}'. Expected .rtf or .md.")
        sys.exit(1)

    print(f"\n Starting Pipeline for: {args.input_file.name}")
    print("-" * 40)

    # --- Setup Directories ---
    dir_raw = args.out_dir / "raw_files"
    dir_cleaned = args.out_dir / "cleaned_files"
    dir_named = args.out_dir / "named_files"
    dir_extracted = args.out_dir / "extracted_files"
    dir_final = args.out_dir / "final_summaries"

    # --- Execute Pipeline ---
    try:
        # Step 1: Ingest & normalize (RTF or MD)
        announce(1, 5, "Ingesting transcript to Markdown")
        _, md_path = convert(args.input_file, dir_raw)

        # Step 2: Cleanup (Uses Editor Model)
        announce(2, 5, "Cleaning transcript", args.editor_model)
        cleaned_path = clean_transcript(
            md_path, dir_cleaned, args.editor_model, args.host
        )

        # Step 3: Human-in-the-Loop Mapping
        announce(3, 5, "Human-in-the-loop Speaker Mapping")
        named_path = map_speakers(cleaned_path, dir_named)

        # Step 4: Extraction (Uses Extractor Model)
        announce(4, 5, "Extracting intelligence", args.extractor_model)
        extracted_path = extract_information(
            named_path, dir_extracted, args.extractor_model, args.host
        )

        # Step 5: Formatting (Uses Extractor Model — aligned with app.py's
        # M6.5 round 2 change; the extractor model produces the final
        # structured summary, not the editor model.)
        announce(5, 5, "Formatting final summary", args.extractor_model)
        final_path = format_summary(
            extracted_path, dir_final, args.extractor_model, args.host
        )

        print("-" * 40)
        print(f"✅ Pipeline Complete! Final summary saved to:\n{final_path.absolute()}")

        # Opt-in PDF export (M11). Runs AFTER the success signal so
        # the markdown is always the primary output; PDF is a bonus.
        # Wrapped in its own try/except so a conversion failure logs
        # a warning but does not flip the whole-pipeline exit code.
        if args.pdf:
            pdf_path = final_path.with_suffix(".pdf")
            try:
                md_to_pdf(final_path, pdf_path)
                print(f"📄 PDF written to: {pdf_path.absolute()}")
            except Exception as e:
                print(f"⚠️  PDF generation failed (markdown is fine): {e}")

        print("\nEjecting models from VRAM...")
        client = Client(host=ollama_host)
        
        # Use a set to avoid trying to unload the same model twice if editor == extractor
        models_used = {args.editor_model, args.extractor_model}
        for m in models_used:
            try:
                # Sending keep_alive=0 frees the model from memory
                client.generate(model=m, keep_alive=0)
                print(f"  - Successfully unloaded: {m}")
            except Exception as e:
                print(f"  - Note: Could not confirm unload for {m}: {e}")
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
