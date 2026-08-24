import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


DEFAULT_INPUT_DIR = "rag_data"
DEFAULT_OUTPUT_DIR = "pdf_parser_experiments"


def now_seconds():
    return time.perf_counter()


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_status(output_dir: Path, tool_name: str, status: dict):
    status_path = output_dir / tool_name / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_pdfs(input_path: Path):
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]

    if input_path.is_dir():
        return sorted(input_path.glob("*.pdf"))

    raise FileNotFoundError(f"PDF file or directory not found: {input_path}")


def parse_with_pymupdf4llm(pdf_path: Path, output_dir: Path):
    tool_name = "pymupdf4llm"
    start = now_seconds()

    try:
        import pymupdf4llm

        markdown = pymupdf4llm.to_markdown(str(pdf_path))
        write_text(output_dir / tool_name / "output.md", markdown)
        status = {
            "tool": tool_name,
            "status": "success",
            "seconds": round(now_seconds() - start, 2),
            "output": str(output_dir / tool_name / "output.md"),
        }
    except Exception as exc:
        status = {
            "tool": tool_name,
            "status": "failed",
            "seconds": round(now_seconds() - start, 2),
            "error": str(exc),
        }

    save_status(output_dir, tool_name, status)
    return status


def parse_with_mineru(pdf_path: Path, output_dir: Path):
    tool_name = "mineru"
    start = now_seconds()
    mineru_cmd = shutil.which("mineru")

    if not mineru_cmd:
        status = {
            "tool": tool_name,
            "status": "skipped",
            "seconds": 0,
            "error": "MinerU CLI not found. Install MinerU first, then ensure the `mineru` command is on PATH.",
        }
        save_status(output_dir, tool_name, status)
        return status

    mineru_output_dir = output_dir / tool_name
    mineru_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        mineru_cmd,
        "-p",
        str(pdf_path),
        "-o",
        str(mineru_output_dir),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )

        markdown_files = sorted(mineru_output_dir.rglob("*.md"))
        status = {
            "tool": tool_name,
            "status": "success" if result.returncode == 0 else "failed",
            "seconds": round(now_seconds() - start, 2),
            "command": " ".join(cmd),
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-3000:],
            "markdown_outputs": [str(path) for path in markdown_files],
        }
    except Exception as exc:
        status = {
            "tool": tool_name,
            "status": "failed",
            "seconds": round(now_seconds() - start, 2),
            "command": " ".join(cmd),
            "error": str(exc),
        }

    save_status(output_dir, tool_name, status)
    return status


def _pdf_to_page_images(pdf_path: Path, image_dir: Path, dpi_scale: float = 2.0):
    import fitz

    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    document = fitz.open(str(pdf_path))

    try:
        matrix = fitz.Matrix(dpi_scale, dpi_scale)
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = image_dir / f"page_{page_index + 1:04d}.png"
            pixmap.save(str(image_path))
            image_paths.append(image_path)
    finally:
        document.close()

    return image_paths


def _ppstructure_result_to_markdown(page_number: int, result):
    lines = [f"\n\n## Page {page_number}\n"]

    for block_index, block in enumerate(result, start=1):
        block_type = block.get("type", "unknown")
        text = ""

        if isinstance(block.get("res"), list):
            text_parts = []
            for item in block["res"]:
                if isinstance(item, dict):
                    text_parts.append(str(item.get("text", "")).strip())
            text = "\n".join(part for part in text_parts if part)
        elif isinstance(block.get("res"), dict):
            text = str(block["res"].get("html", "") or block["res"].get("text", "")).strip()
        elif block.get("text"):
            text = str(block.get("text")).strip()

        if text:
            lines.append(f"\n### Block {block_index}: {block_type}\n")
            lines.append(text)

    return "\n".join(lines)


def parse_with_pp_structure(pdf_path: Path, output_dir: Path):
    tool_name = "pp_structure"
    start = now_seconds()

    try:
        from paddleocr import PPStructure
    except Exception as exc:
        status = {
            "tool": tool_name,
            "status": "skipped",
            "seconds": 0,
            "error": f"PaddleOCR PPStructure import failed: {exc}",
        }
        save_status(output_dir, tool_name, status)
        return status

    pp_output_dir = output_dir / tool_name
    pp_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2

        markdown_parts = [f"# PP-Structure Output\n\nSource: {pdf_path.name}\n"]
        raw_results = []

        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = _pdf_to_page_images(pdf_path, Path(temp_dir))
            engine = PPStructure(show_log=True)

            for page_number, image_path in enumerate(image_paths, start=1):
                image = cv2.imread(str(image_path))
                result = engine(image)
                serializable_result = []

                for block in result:
                    clean_block = dict(block)
                    clean_block.pop("img", None)
                    serializable_result.append(clean_block)

                raw_results.append(
                    {
                        "page": page_number,
                        "result": serializable_result,
                    }
                )
                markdown_parts.append(
                    _ppstructure_result_to_markdown(page_number, serializable_result)
                )

        write_text(pp_output_dir / "output.md", "\n".join(markdown_parts))
        (pp_output_dir / "raw_result.json").write_text(
            json.dumps(raw_results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        status = {
            "tool": tool_name,
            "status": "success",
            "seconds": round(now_seconds() - start, 2),
            "output": str(pp_output_dir / "output.md"),
            "raw_output": str(pp_output_dir / "raw_result.json"),
        }
    except Exception as exc:
        status = {
            "tool": tool_name,
            "status": "failed",
            "seconds": round(now_seconds() - start, 2),
            "error": str(exc),
        }

    save_status(output_dir, tool_name, status)
    return status


def write_review_template(pdf_output_dir: Path, pdf_name: str, statuses: list[dict]):
    template = f"""# PDF Parser Experiment Review

PDF: {pdf_name}

## Tool Status

| Tool | Status | Seconds | Notes |
| --- | --- | ---: | --- |
"""

    for status in statuses:
        note = status.get("error") or status.get("output") or ""
        template += (
            f"| {status['tool']} | {status['status']} | "
            f"{status.get('seconds', 0)} | {note} |\n"
        )

    template += """
## Manual Quality Checklist

Score each item from 1 to 5.

| Criterion | pymupdf4llm | MinerU | PP-Structure | Notes |
| --- | ---: | ---: | ---: | --- |
| Reading order is correct |  |  |  |  |
| Double-column layout is handled |  |  |  |  |
| Headings are preserved |  |  |  |  |
| Paragraph blank lines are preserved |  |  |  |  |
| Tables are readable |  |  |  |  |
| Figures/captions are represented |  |  |  |  |
| No obvious header/footer noise |  |  |  |  |
| Output is suitable for chunking |  |  |  |  |

## Decision

Best parser for this PDF:

Reason:

Follow-up action:
"""

    write_text(pdf_output_dir / "review_checklist.md", template)


def run_experiment(pdf_path: Path, output_root: Path, tools: list[str]):
    pdf_output_dir = output_root / pdf_path.stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    statuses = []
    if "pymupdf4llm" in tools:
        statuses.append(parse_with_pymupdf4llm(pdf_path, pdf_output_dir))
    if "mineru" in tools:
        statuses.append(parse_with_mineru(pdf_path, pdf_output_dir))
    if "pp_structure" in tools:
        statuses.append(parse_with_pp_structure(pdf_path, pdf_output_dir))

    write_review_template(pdf_output_dir, pdf_path.name, statuses)
    return statuses


def main():
    parser = argparse.ArgumentParser(
        description="Compare PDF parsers before RAG chunking."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help="A PDF file path or a directory containing PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=["pymupdf4llm", "mineru", "pp_structure"],
        choices=["pymupdf4llm", "mineru", "pp_structure"],
        help="Parsers to run.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_dir)
    pdfs = collect_pdfs(input_path)

    if not pdfs:
        print(f"No PDF files found in: {input_path}")
        return

    print(f"Found {len(pdfs)} PDF file(s).")
    print(f"Output directory: {output_root}")

    for pdf_path in pdfs:
        print(f"\n=== Running parser experiment for {pdf_path} ===")
        statuses = run_experiment(pdf_path, output_root, args.tools)
        for status in statuses:
            print(
                f"- {status['tool']}: {status['status']} "
                f"({status.get('seconds', 0)}s)"
            )

    print("\nDone. Open each review_checklist.md and score the parser outputs manually.")


if __name__ == "__main__":
    main()
