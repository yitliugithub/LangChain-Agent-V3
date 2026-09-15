import argparse
import json
import re
from pathlib import Path

from experiments.pdf_parsing.experiment_magic_pdf import (
    DEFAULT_MAGIC_PDF_CONFIG,
    run_magic_pdf_experiment,
)
from rag_config import NORMALIZED_DATA_DIR, PARSER_OUTPUT_DIR, RAG_DATA_DIR


def parser_directory(pdf_path: Path):
    return PARSER_OUTPUT_DIR / pdf_path.stem / "magic_pdf"


def parser_status_path(pdf_path: Path):
    return parser_directory(pdf_path) / "status.json"


def parser_cleaned_path(pdf_path: Path):
    return parser_directory(pdf_path) / "cleaned_text.md"


def normalized_path(pdf_path: Path):
    return NORMALIZED_DATA_DIR / f"{pdf_path.stem}.md"


def has_successful_existing_parse(pdf_path: Path):
    status_path = parser_status_path(pdf_path)
    cleaned_path = parser_cleaned_path(pdf_path)
    if not status_path.exists() or not cleaned_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return status.get("status") == "success" and cleaned_path.stat().st_size > 0


def normalize_cleaned_markdown(text: str):
    """Apply the same Markdown normalization used by the frozen experiments."""
    lines = []
    image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        stripped = line.strip()
        if stripped == "# Cleaned Magic-PDF / MinerU Text":
            continue
        if stripped.startswith("Source: "):
            continue
        if stripped.startswith("Raw output directory: "):
            continue
        if stripped.startswith("## Source file: "):
            continue
        line = image_pattern.sub(r"[Image: \1]", line)
        lines.append(line.rstrip())

    normalized = "\n".join(lines).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized + "\n" if normalized else ""


def sync_normalized_markdown(pdf_path: Path):
    source = parser_cleaned_path(pdf_path)
    if not source.exists():
        raise FileNotFoundError(f"Magic-PDF cleaned output not found: {source}")

    normalized = normalize_cleaned_markdown(source.read_text(encoding="utf-8"))
    if not normalized.strip():
        raise ValueError(f"Magic-PDF cleaned output is empty: {source}")

    target = normalized_path(pdf_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(normalized, encoding="utf-8")
    return target


def process_pdf(pdf_path: Path, force=False, rebuild_cleaned_only=False):
    pdf_path = pdf_path.expanduser().resolve()
    reused = has_successful_existing_parse(pdf_path) and not force

    if reused:
        print(f"[Magic-PDF] 复用已有成功结果：{pdf_path.name}")
        status = json.loads(parser_status_path(pdf_path).read_text(encoding="utf-8"))
    else:
        print(f"[Magic-PDF] 正在解析：{pdf_path.name}")
        status = run_magic_pdf_experiment(
            pdf_path=pdf_path,
            output_root=PARSER_OUTPUT_DIR,
            preferred_command="",
            backend="pipeline",
            method="auto",
            lang="ch",
            config_path=DEFAULT_MAGIC_PDF_CONFIG,
            rebuild_only=rebuild_cleaned_only,
        )
        if status.get("status") != "success":
            raise RuntimeError(
                f"Magic-PDF failed for {pdf_path.name}: {status.get('error', status)}"
            )

    output = sync_normalized_markdown(pdf_path)
    print(f"[Normalized] 已保存：{output}")
    return {
        "pdf": str(pdf_path),
        "parser_reused": reused,
        "parser_status": status.get("status"),
        "normalized_markdown": str(output),
    }


def preprocess_all_pdfs(force=False, rebuild_cleaned_only=False):
    if not RAG_DATA_DIR.exists():
        raise FileNotFoundError(f"找不到目录：{RAG_DATA_DIR}")

    pdf_files = sorted(RAG_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"{RAG_DATA_DIR} 中没有PDF文件")

    NORMALIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    expected_outputs = {normalized_path(path).resolve() for path in pdf_files}
    for old_file in NORMALIZED_DATA_DIR.glob("*.md"):
        if old_file.resolve() not in expected_outputs:
            old_file.unlink()

    results = []
    failures = []
    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n=== [{index}/{len(pdf_files)}] {pdf_path.name} ===")
        try:
            results.append(process_pdf(pdf_path, force, rebuild_cleaned_only))
        except Exception as exc:
            failures.append({"pdf": str(pdf_path), "error": str(exc)})
            print(f"[Failed] {exc}")

    summary = {
        "parser": "Magic-PDF / MinerU local CLI",
        "pdf_count": len(pdf_files),
        "success_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
    summary_path = NORMALIZED_DATA_DIR / "preprocess_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} PDF(s) failed; see {summary_path} for details"
        )
    print(f"\n全部预处理完成：{summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to normalized Markdown with Magic-PDF."
    )
    parser.add_argument("--pdf", help="Only process one PDF path.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run Magic-PDF again even when a successful result exists.",
    )
    parser.add_argument(
        "--rebuild-cleaned-only",
        action="store_true",
        help="Rebuild cleaned Markdown from existing Magic-PDF raw output.",
    )
    args = parser.parse_args()

    if args.pdf:
        result = process_pdf(
            Path(args.pdf),
            force=args.force,
            rebuild_cleaned_only=args.rebuild_cleaned_only,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        preprocess_all_pdfs(
            force=args.force,
            rebuild_cleaned_only=args.rebuild_cleaned_only,
        )


if __name__ == "__main__":
    main()
