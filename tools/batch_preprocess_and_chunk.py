import json
from pathlib import Path

from experiments.chunking.experiment_semantic_boundary_chunking import (
    run_experiment as run_chunking,
)
from experiments.pdf_parsing.experiment_magic_pdf import run_magic_pdf_experiment


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAG_DATA_DIR = PROJECT_DIR / "data/raw_pdfs"
PARSER_OUTPUT_DIR = PROJECT_DIR / "artifacts/pdf_parsing"
CHUNK_OUTPUT_DIR = PROJECT_DIR / "artifacts/chunking"
MAGIC_PDF_CONFIG = Path.home() / "magic-pdf.json"

MAX_TOKENS = 500
MIN_TOKENS = 150
TOKENIZER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def parser_status_path(pdf_path: Path):
    return PARSER_OUTPUT_DIR / pdf_path.stem / "magic_pdf" / "status.json"


def cleaned_text_path(pdf_path: Path):
    return PARSER_OUTPUT_DIR / pdf_path.stem / "magic_pdf" / "cleaned_text.md"


def chunk_summary_path(pdf_path: Path):
    return (
        CHUNK_OUTPUT_DIR
        / pdf_path.stem
        / f"cleaned_text_structure_semantic_boundary_{MAX_TOKENS}_min_{MIN_TOKENS}"
        / "summary.json"
    )


def is_parser_success(pdf_path: Path):
    status_path = parser_status_path(pdf_path)
    cleaned_path = cleaned_text_path(pdf_path)
    if not status_path.exists() or not cleaned_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return status.get("status") == "success" and cleaned_path.stat().st_size > 0


def is_chunking_done(pdf_path: Path):
    summary_path = chunk_summary_path(pdf_path)
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return summary.get("chunk_count", 0) > 0


def main():
    pdf_paths = sorted(RAG_DATA_DIR.glob("*.pdf"))
    results = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        print(f"\n=== [{index}/{len(pdf_paths)}] {pdf_path.name} ===", flush=True)

        parser_result = {"status": "skipped_existing_success"}
        if not is_parser_success(pdf_path):
            print("[Preprocess] Running Magic-PDF...", flush=True)
            parser_result = run_magic_pdf_experiment(
                pdf_path=pdf_path,
                output_root=PARSER_OUTPUT_DIR,
                preferred_command="magic-pdf",
                backend="pipeline",
                method="auto",
                lang="ch",
                config_path=MAGIC_PDF_CONFIG,
                rebuild_only=False,
            )
            print(
                f"[Preprocess] {parser_result.get('status')} "
                f"({parser_result.get('seconds')}s)",
                flush=True,
            )
        else:
            print("[Preprocess] Skip existing success.", flush=True)

        chunk_result = {"status": "skipped_no_cleaned_text"}
        if cleaned_text_path(pdf_path).exists():
            if is_chunking_done(pdf_path):
                print("[Chunking] Skip existing chunks.", flush=True)
                chunk_result = {"status": "skipped_existing_chunks"}
            else:
                print("[Chunking] Running semantic boundary chunking...", flush=True)
                chunk_result = run_chunking(
                    input_path=cleaned_text_path(pdf_path),
                    output_dir=CHUNK_OUTPUT_DIR,
                    max_tokens=MAX_TOKENS,
                    min_tokens=MIN_TOKENS,
                    tokenizer_name=TOKENIZER_NAME,
                )
                summary = chunk_result["summary"]
                print(
                    "[Chunking] success "
                    f"chunks={summary['chunk_count']} "
                    f"avg_tokens={summary['avg_chunk_tokens']}",
                    flush=True,
                )
        else:
            print("[Chunking] Skip because cleaned_text.md is missing.", flush=True)

        results.append(
            {
                "pdf": str(pdf_path),
                "preprocess": parser_result,
                "chunking": chunk_result,
            }
        )

    output_path = CHUNK_OUTPUT_DIR / "batch_preprocess_and_chunk_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[Done] Batch summary: {output_path}", flush=True)


if __name__ == "__main__":
    main()
