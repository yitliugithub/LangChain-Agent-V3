import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_OUTPUT_DIR = "pdf_parser_experiments"
TOOL_NAME = "magic_pdf"


def now_seconds():
    return time.perf_counter()


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_command(preferred_command: str):
    if preferred_command:
        command_path = shutil.which(preferred_command)
        if not command_path:
            raise FileNotFoundError(
                f"Command not found: {preferred_command}. "
                "Please install MinerU/Magic-PDF in the active environment first."
            )
        return preferred_command

    for command in ("mineru", "magic-pdf"):
        if shutil.which(command):
            return command

    raise FileNotFoundError(
        "Neither 'mineru' nor 'magic-pdf' was found in PATH. "
        "Install MinerU/Magic-PDF first, then run this script again."
    )


def build_command(
    command: str,
    pdf_path: Path,
    raw_output_dir: Path,
    backend: str,
    method: str,
    lang: str,
):
    if command == "mineru":
        cmd = [
            command,
            "-p",
            str(pdf_path),
            "-o",
            str(raw_output_dir),
        ]
        if backend:
            cmd.extend(["-b", backend])
        return cmd

    if command == "magic-pdf":
        # magic-pdf 1.x uses the unified CLI style below. It writes parser
        # artifacts directly under the output directory.
        cmd = [
            command,
            "-p",
            str(pdf_path),
            "-o",
            str(raw_output_dir),
        ]
        if method:
            cmd.extend(["-m", method])
        if lang:
            cmd.extend(["-l", lang])
        return cmd

    raise ValueError(f"Unsupported command: {command}")


def collect_markdown_files(output_dir: Path):
    ignored_dirs = {"__MACOSX", ".git", "__pycache__"}
    markdown_files = []

    for path in output_dir.rglob("*.md"):
        if any(part in ignored_dirs for part in path.parts):
            continue
        markdown_files.append(path)

    return sorted(markdown_files)


def clean_markdown_for_text_rag(markdown_text: str):
    text = markdown_text

    text = re.sub(
        r'<div[^>]*>\s*<img\s+[^>]*src="([^"]+)"[^>]*>\s*</div>',
        r"\n[Image omitted: \1]\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'<img\s+[^>]*src="([^"]+)"[^>]*>',
        r"\n[Image omitted: \1]\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_cleaned_text(pdf_path: Path, raw_output_dir: Path, cleaned_output_path: Path):
    markdown_files = collect_markdown_files(raw_output_dir)
    parts = [
        "# Cleaned Magic-PDF / MinerU Text\n",
        f"Source: {pdf_path.name}\n",
        f"Raw output directory: {raw_output_dir}\n",
    ]

    for markdown_file in markdown_files:
        relative_path = markdown_file.relative_to(raw_output_dir)
        raw_text = markdown_file.read_text(encoding="utf-8", errors="replace")
        cleaned_text = clean_markdown_for_text_rag(raw_text)
        if not cleaned_text:
            continue

        parts.append(f"\n\n## Source file: {relative_path}\n")
        parts.append(cleaned_text)

    write_text(cleaned_output_path, "\n".join(parts).strip() + "\n")
    return markdown_files


def run_magic_pdf_experiment(
    pdf_path: Path,
    output_root: Path,
    preferred_command: str,
    backend: str,
    method: str,
    lang: str,
):
    start = now_seconds()
    pdf_path = pdf_path.expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input must be a PDF file: {pdf_path}")

    experiment_dir = output_root / pdf_path.stem / TOOL_NAME
    raw_output_dir = experiment_dir / "raw_output"
    cleaned_output_path = experiment_dir / "cleaned_text.md"
    log_path = experiment_dir / "run.log"
    status_path = experiment_dir / "status.json"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "tool": TOOL_NAME,
        "status": "running",
        "input": str(pdf_path),
        "raw_output_dir": str(raw_output_dir),
        "cleaned_text": str(cleaned_output_path),
    }
    write_json(status_path, status)

    try:
        command = resolve_command(preferred_command)
        cmd = build_command(
            command,
            pdf_path,
            raw_output_dir,
            backend,
            method,
            lang,
        )

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("Command:\n")
            log_file.write(" ".join(cmd) + "\n\n")
            log_file.flush()

            completed = subprocess.run(
                cmd,
                cwd=str(Path.cwd()),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        markdown_files = build_cleaned_text(
            pdf_path,
            raw_output_dir,
            cleaned_output_path,
        )

        status = {
            "tool": TOOL_NAME,
            "status": "success" if completed.returncode == 0 else "failed",
            "seconds": round(now_seconds() - start, 2),
            "command": command,
            "backend": backend,
            "method": method,
            "lang": lang,
            "returncode": completed.returncode,
            "input": str(pdf_path),
            "raw_output_dir": str(raw_output_dir),
            "cleaned_text": str(cleaned_output_path),
            "log": str(log_path),
            "markdown_files_found": [str(path) for path in markdown_files],
        }
        if completed.returncode != 0:
            status["error"] = "MinerU/Magic-PDF command returned a non-zero exit code."
    except Exception as exc:
        status = {
            "tool": TOOL_NAME,
            "status": "failed",
            "seconds": round(now_seconds() - start, 2),
            "input": str(pdf_path),
            "raw_output_dir": str(raw_output_dir),
            "cleaned_text": str(cleaned_output_path),
            "log": str(log_path),
            "error": str(exc),
        }

    write_json(status_path, status)
    return status


def main():
    parser = argparse.ArgumentParser(
        description="Run a local Magic-PDF/MinerU PDF parsing experiment."
    )
    parser.add_argument(
        "pdf",
        help="PDF file to parse, for example rag_data/2024年KOL发展年报.pdf",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for parser experiment outputs.",
    )
    parser.add_argument(
        "--command",
        default="",
        choices=["", "mineru", "magic-pdf"],
        help="Parser CLI command to use. Default: auto-detect mineru, then magic-pdf.",
    )
    parser.add_argument(
        "--backend",
        default="pipeline",
        help=(
            "MinerU backend. Use 'pipeline' for CPU/local compatibility. "
            "Ignored by older magic-pdf command."
        ),
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "ocr", "txt"],
        help=(
            "magic-pdf parsing method. auto chooses between text extraction "
            "and OCR; ocr forces OCR; txt is faster for text-based PDFs."
        ),
    )
    parser.add_argument(
        "--lang",
        default="ch",
        help="OCR language hint for magic-pdf, for example 'ch' for Chinese.",
    )

    args = parser.parse_args()
    status = run_magic_pdf_experiment(
        pdf_path=Path(args.pdf),
        output_root=Path(args.output_dir),
        preferred_command=args.command,
        backend=args.backend,
        method=args.method,
        lang=args.lang,
    )

    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
