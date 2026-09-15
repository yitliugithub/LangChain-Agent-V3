import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "artifacts/pdf_parsing"
TOOL_NAME = "magic_pdf"
DEFAULT_MAGIC_PDF_CONFIG = Path.home() / "magic-pdf.json"


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


def load_magic_pdf_config(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(
            f"Official Magic-PDF config not found: {config_path}. "
            "Run the official download_models_hf.py script first."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    models_dir = config.get("models-dir")
    layoutreader_dir = config.get("layoutreader-model-dir")
    missing_paths = [
        path
        for path in (models_dir, layoutreader_dir)
        if path and not Path(path).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Magic-PDF config exists, but these configured model paths "
            f"do not exist: {missing_paths}"
        )

    return config


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

    # Magic-PDF often keeps recognized tables as HTML. Convert tables before
    # stripping tags so RAG receives row/column structure instead of flat text.
    text = convert_html_tables_to_markdown(text)

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
    text = clean_pdf_parse_noise(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pdf_parse_noise(text: str):
    """Remove parser artifacts while keeping readable Markdown content.

    Some OCR/PDF parsers emit invisible control characters from decorative
    bullets, page markers, or embedded fonts. These characters hurt retrieval
    but carry no useful text meaning for the current text-only RAG prototype.
    """
    # Keep normal whitespace, but remove non-printable control characters.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # OCR sometimes turns visual bullets into solid blocks. Normalize them
    # into a plain Markdown-friendly bullet instead of deleting list structure.
    text = re.sub(r"^[ \t]*[■□]{1,}[ \t]*", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"[■□]{2,}", " ", text)

    # Remove empty headings left behind after control-character cleanup.
    text = re.sub(r"(?m)^#{1,6}\s*$", "", text)
    text = re.sub(r"(?m)^(#{1,6})\s+", r"\1 ", text)
    return text


def convert_html_tables_to_markdown(text: str):
    if "<table" not in text.lower():
        return text

    try:
        from bs4 import BeautifulSoup
    except Exception:
        return re.sub(
            r"<table\b.*?</table>",
            html_table_to_markdown_fallback,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

    soup = BeautifulSoup(text, "html.parser")
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            row = [
                normalize_table_cell(cell.get_text(" ", strip=True))
                for cell in cells
            ]
            if any(row):
                rows.append(row)

        table.replace_with("\n" + rows_to_markdown_table(rows) + "\n")

    return str(soup)


def html_table_to_markdown_fallback(match):
    table_html = match.group(0)
    row_htmls = re.findall(
        r"<tr\b.*?</tr>",
        table_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    rows = []
    for row_html in row_htmls:
        cell_htmls = re.findall(
            r"<t[dh]\b.*?</t[dh]>",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        row = [
            normalize_table_cell(re.sub(r"<[^>]+>", " ", cell_html))
            for cell_html in cell_htmls
        ]
        if any(row):
            rows.append(row)

    return "\n" + rows_to_markdown_table(rows) + "\n"


def normalize_table_cell(value: str):
    value = clean_pdf_parse_noise(value)
    value = re.sub(r"\s+", " ", value)
    value = value.replace("|", "\\|")
    return value.strip()


def rows_to_markdown_table(rows):
    if not rows:
        return ""

    max_columns = max(len(row) for row in rows)
    normalized_rows = [
        row + [""] * (max_columns - len(row))
        for row in rows
    ]
    header = normalized_rows[0]
    body = normalized_rows[1:]

    markdown_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * max_columns) + " |",
    ]
    for row in body:
        markdown_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(markdown_lines)


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
    config_path: Path,
    rebuild_only: bool,
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
    config_snapshot_path = experiment_dir / "magic-pdf.config.snapshot.json"

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
        if rebuild_only:
            markdown_files = build_cleaned_text(
                pdf_path,
                raw_output_dir,
                cleaned_output_path,
            )
            status = {
                "tool": TOOL_NAME,
                "status": "success" if markdown_files else "failed",
                "seconds": round(now_seconds() - start, 2),
                "mode": "rebuild_only",
                "input": str(pdf_path),
                "raw_output_dir": str(raw_output_dir),
                "cleaned_text": str(cleaned_output_path),
                "markdown_files_found": [str(path) for path in markdown_files],
            }
            if not markdown_files:
                status["error"] = "No Markdown output was found under raw_output."
            write_json(status_path, status)
            return status

        command = resolve_command(preferred_command)
        cmd = build_command(
            command,
            pdf_path,
            raw_output_dir,
            backend,
            method,
            lang,
        )
        env = None
        config = None
        if command == "magic-pdf":
            config = load_magic_pdf_config(config_path)
            write_json(config_snapshot_path, config)
            env = {
                **os.environ,
                "MINERU_TOOLS_CONFIG_JSON": str(config_path.resolve()),
            }

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("Command:\n")
            log_file.write(" ".join(cmd) + "\n\n")
            if command == "magic-pdf":
                log_file.write(f"Config: {config_path.resolve()}\n\n")
                log_file.write(f"Config snapshot: {config_snapshot_path.resolve()}\n\n")
            log_file.flush()

            completed = subprocess.run(
                cmd,
                cwd=str(Path.cwd()),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                env=env,
            )

        markdown_files = build_cleaned_text(
            pdf_path,
            raw_output_dir,
            cleaned_output_path,
        )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        has_runtime_error = "Traceback" in log_text or "ERROR" in log_text
        has_markdown = bool(markdown_files)
        ok = completed.returncode == 0 and has_markdown and not has_runtime_error

        status = {
            "tool": TOOL_NAME,
            "status": "success" if ok else "failed",
            "seconds": round(now_seconds() - start, 2),
            "command": command,
            "backend": backend,
            "method": method,
            "lang": lang,
            "config": str(config_path.resolve()) if command == "magic-pdf" else "",
            "config_snapshot": str(config_snapshot_path) if command == "magic-pdf" else "",
            "returncode": completed.returncode,
            "input": str(pdf_path),
            "raw_output_dir": str(raw_output_dir),
            "cleaned_text": str(cleaned_output_path),
            "log": str(log_path),
            "markdown_files_found": [str(path) for path in markdown_files],
        }
        if completed.returncode != 0:
            status["error"] = "MinerU/Magic-PDF command returned a non-zero exit code."
        elif has_runtime_error:
            status["error"] = "MinerU/Magic-PDF wrote an ERROR or Traceback in run.log."
        elif not has_markdown:
            status["error"] = "No Markdown output was found under raw_output."
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
        help="PDF file to parse, for example data/raw_pdfs/2024年KOL发展年报.pdf",
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
    parser.add_argument(
        "--config",
        default=str(DEFAULT_MAGIC_PDF_CONFIG),
        help=(
            "Official Magic-PDF config path generated by download_models_hf.py. "
            "Default: ~/magic-pdf.json"
        ),
    )
    parser.add_argument(
        "--rebuild-cleaned-only",
        action="store_true",
        help="Rebuild cleaned_text.md from existing raw_output without running Magic-PDF.",
    )

    args = parser.parse_args()
    status = run_magic_pdf_experiment(
        pdf_path=Path(args.pdf),
        output_root=Path(args.output_dir),
        preferred_command=args.command,
        backend=args.backend,
        method=args.method,
        lang=args.lang,
        config_path=Path(args.config).expanduser(),
        rebuild_only=args.rebuild_cleaned_only,
    )

    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
