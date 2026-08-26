import argparse
import json
import re
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


def parse_page_range(page_range: str, total_pages: int):
    if not page_range:
        return list(range(total_pages))

    selected_pages = set()
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            for page in range(start, end + 1):
                if 1 <= page <= total_pages:
                    selected_pages.add(page - 1)
        else:
            page = int(part)
            if 1 <= page <= total_pages:
                selected_pages.add(page - 1)

    return sorted(selected_pages)


def _pdf_to_page_images(
    pdf_path: Path,
    image_dir: Path,
    dpi_scale: float = 2.0,
    page_range: str = "",
):
    import fitz

    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    document = fitz.open(str(pdf_path))

    try:
        matrix = fitz.Matrix(dpi_scale, dpi_scale)
        page_indexes = parse_page_range(page_range, document.page_count)
        for page_index in page_indexes:
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = image_dir / f"page_{page_index + 1:04d}.png"
            pixmap.save(str(image_path))
            image_paths.append((page_index + 1, image_path))
    finally:
        document.close()

    return image_paths


def create_pp_structure_engine(options: dict):
    import paddleocr

    mode = options.get("mode", "full")
    show_log = options.get("show_log", True)

    if hasattr(paddleocr, "PPStructureV3"):
        return (
            "v3",
            paddleocr.PPStructureV3(
                use_table_recognition=mode in {"full", "table_only"},
                use_formula_recognition=False,
                use_chart_recognition=False,
                use_region_detection=False,
                lang="ch",
            ),
        )

    if not hasattr(paddleocr, "PPStructure"):
        raise ImportError(
            "Neither PPStructure nor PPStructureV3 is available in paddleocr."
        )

    PPStructure = paddleocr.PPStructure
    if mode == "layout_only":
        return (
            "legacy",
            PPStructure(
                show_log=show_log,
                recovery=False,
                return_ocr_result_in_table=False,
            ),
        )
    if mode == "table_only":
        return (
            "legacy",
            PPStructure(
                show_log=show_log,
                layout=False,
                table=True,
                ocr=True,
                recovery=False,
            ),
        )

    return (
        "legacy",
        PPStructure(
            show_log=show_log,
            recovery=False,
            return_ocr_result_in_table=True,
        ),
    )


def block_bbox(block):
    bbox = block.get("bbox") or block.get("bounding_box")
    if bbox is None:
        return []
    return [float(value) for value in bbox]


def sort_blocks_for_reading(blocks):
    def sort_key(block):
        bbox = block_bbox(block)
        if len(bbox) >= 4:
            x1, y1, _, _ = bbox
            return (round(y1 / 20) * 20, x1)
        return (10**9, 10**9)

    return sorted(blocks, key=sort_key)


def _ppstructure_result_to_markdown(page_number: int, result):
    lines = [f"\n\n## Page {page_number}\n"]

    for block_index, block in enumerate(result, start=1):
        block_type = block.get("type", "unknown")
        bbox = block_bbox(block)
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
            if bbox:
                lines.append(f"bbox: {bbox}\n")
            lines.append(text)

    return "\n".join(lines)


def _v3_result_to_serializable(result):
    if isinstance(result, dict):
        return result

    try:
        return dict(result)
    except Exception:
        pass

    json_attr = getattr(result, "json", None)
    if isinstance(json_attr, dict):
        return json_attr

    return {"repr": repr(result)}


def _v3_result_to_markdown(page_number: int, results):
    lines = [f"\n\n## Page {page_number}\n"]

    for result_index, result in enumerate(results, start=1):
        markdown = getattr(result, "markdown", None)
        if isinstance(markdown, dict):
            markdown_text = markdown.get("markdown_texts", "")
            if markdown_text:
                lines.append(f"\n### Result {result_index}: markdown\n")
                lines.append(markdown_text)
                continue

        serializable = _v3_result_to_serializable(result)
        markdown_text = serializable.get("markdown_texts", "")
        if markdown_text:
            lines.append(f"\n### Result {result_index}: markdown\n")
            lines.append(markdown_text)
        else:
            lines.append(f"\n### Result {result_index}: raw\n")
            lines.append("```json")
            lines.append(json.dumps(serializable, ensure_ascii=False, indent=2, default=str))
            lines.append("```")

    return "\n".join(lines)


def save_v3_raw_markdown(result, raw_markdown_dir: Path, page_number: int):
    if not hasattr(result, "save_to_markdown"):
        return ""

    page_dir = raw_markdown_dir / f"page_{page_number:04d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    result.save_to_markdown(str(page_dir), pretty=True)
    markdown_files = sorted(page_dir.rglob("*.md"))
    if markdown_files:
        return str(markdown_files[0])
    return ""


def find_saved_page_markdown(raw_markdown_dir: Path, page_number: int):
    page_dir = raw_markdown_dir / f"page_{page_number:04d}"
    markdown_files = sorted(page_dir.rglob("*.md"))
    if markdown_files:
        return markdown_files[0]
    return None


def rebuild_cleaned_text_from_raw_pages(pp_output_dir: Path, pdf_name: str, options: dict):
    raw_markdown_dir = pp_output_dir / "raw_markdown"
    cleaned_parts = [
        "# Cleaned PP-Structure Text\n",
        f"Source: {pdf_name}\n",
        f"Mode: {options.get('mode', 'full')}\n",
        f"DPI scale: {options.get('dpi_scale', 2.0)}\n",
        f"Page range: {options.get('page_range', '') or 'all'}\n",
    ]

    for page_md in sorted(raw_markdown_dir.glob("page_*/page_*.md")):
        page_num = page_md.parent.name.split("_")[-1].lstrip("0") or "0"
        cleaned_parts.append(f"\n\n## Page {page_num}\n")
        cleaned_parts.append(
            clean_markdown_for_text_rag(page_md.read_text(encoding="utf-8"))
        )

    write_text(pp_output_dir / "cleaned_text.md", "\n".join(cleaned_parts))


def extract_v3_markdown_text(result):
    markdown = getattr(result, "markdown", None)
    if isinstance(markdown, dict):
        return markdown.get("markdown_texts", "")

    serializable = _v3_result_to_serializable(result)
    return serializable.get("markdown_texts", "")


def clean_markdown_for_text_rag(markdown_text: str):
    text = markdown_text

    # Convert HTML tables before stripping other HTML tags. Otherwise table
    # rows/cells collapse into one unreadable line and become poor RAG evidence.
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
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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

        table_markdown = rows_to_markdown_table(rows)
        table.replace_with("\n" + table_markdown + "\n")

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
    separator = ["---"] * max_columns

    markdown_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        markdown_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(markdown_lines)


def collect_v3_media_metadata(serializable_result, page_number: int):
    metadata = []

    for item in serializable_result.get("imgs_in_doc", []):
        metadata.append(
            {
                "page": page_number,
                "type": item.get("label", "image"),
                "path": item.get("path", ""),
                "coordinate": item.get("coordinate", []),
                "score": item.get("score"),
            }
        )

    for block in serializable_result.get("parsing_res_list", []):
        if not isinstance(block, dict):
            try:
                block = dict(block)
            except Exception:
                block = {"repr": repr(block)}

        label = block.get("label") or block.get("block_label")
        if label in {"table", "chart", "image"}:
            metadata.append(
                {
                    "page": page_number,
                    "type": label,
                    "bbox": block.get("bbox", []),
                    "content": block.get("content", ""),
                }
            )

    return metadata


def parse_with_pp_structure(pdf_path: Path, output_dir: Path, options: dict):
    tool_name = "pp_structure"
    start = now_seconds()

    try:
        import paddleocr
        if not (hasattr(paddleocr, "PPStructure") or hasattr(paddleocr, "PPStructureV3")):
            raise ImportError(
                "paddleocr is installed, but PPStructure/PPStructureV3 is not exported."
            )
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

        pp_output_dir = output_dir / tool_name
        pp_output_dir.mkdir(parents=True, exist_ok=True)
        raw_markdown_dir = pp_output_dir / "raw_markdown"
        metadata_dir = pp_output_dir / "metadata"
        raw_markdown_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        markdown_parts = [
            "# PP-Structure Output\n",
            f"Source: {pdf_path.name}\n",
            f"Mode: {options.get('mode', 'full')}\n",
            f"DPI scale: {options.get('dpi_scale', 2.0)}\n",
            f"Page range: {options.get('page_range', '') or 'all'}\n",
        ]
        raw_results = []
        media_metadata = []
        api_version = "unknown"

        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = _pdf_to_page_images(
                pdf_path,
                Path(temp_dir),
                dpi_scale=options.get("dpi_scale", 2.0),
                page_range=options.get("page_range", ""),
            )
            engine = None

            for page_number, image_path in image_paths:
                saved_page_md = find_saved_page_markdown(raw_markdown_dir, page_number)
                if saved_page_md:
                    print(
                        "[PP-Structure] skipping existing page "
                        f"{page_number}: {saved_page_md}",
                        flush=True,
                    )
                    markdown_parts.append(f"\n\n## Page {page_number}\n")
                    markdown_parts.append(saved_page_md.read_text(encoding="utf-8"))
                    raw_results.append(
                        {
                            "page": page_number,
                            "api_version": "cached",
                            "raw_markdown_outputs": [str(saved_page_md)],
                            "result": [],
                        }
                    )
                    rebuild_cleaned_text_from_raw_pages(pp_output_dir, pdf_path.name, options)
                    continue

                print(
                    f"[PP-Structure] parsing page {page_number}: {image_path.name}",
                    flush=True,
                )
                if engine is None:
                    api_version, engine = create_pp_structure_engine(options)

                if api_version == "v3":
                    page_results = list(
                        engine.predict(
                            str(image_path),
                            use_table_recognition=options.get("mode", "full")
                            in {"full", "table_only"},
                            use_formula_recognition=False,
                            use_chart_recognition=False,
                            use_region_detection=False,
                        )
                    )
                    serializable_result = [
                        _v3_result_to_serializable(result)
                        for result in page_results
                    ]
                    raw_markdown_outputs = []
                    page_text_parts = []

                    for result, serializable in zip(page_results, serializable_result):
                        raw_markdown_output = save_v3_raw_markdown(
                            result,
                            raw_markdown_dir,
                            page_number,
                        )
                        if raw_markdown_output:
                            raw_markdown_outputs.append(raw_markdown_output)

                        markdown_text = extract_v3_markdown_text(result)
                        if markdown_text:
                            page_text_parts.append(markdown_text)

                        media_metadata.extend(
                            collect_v3_media_metadata(serializable, page_number)
                        )

                    raw_results.append(
                        {
                            "page": page_number,
                            "api_version": api_version,
                            "raw_markdown_outputs": raw_markdown_outputs,
                            "result": serializable_result,
                        }
                    )
                    markdown_parts.append(
                        _v3_result_to_markdown(page_number, page_results)
                    )
                    if page_text_parts:
                        rebuild_cleaned_text_from_raw_pages(
                            pp_output_dir,
                            pdf_path.name,
                            options,
                        )
                else:
                    image = cv2.imread(str(image_path))
                    result = engine(image)
                    serializable_result = []

                    for block in result:
                        clean_block = dict(block)
                        clean_block.pop("img", None)
                        serializable_result.append(clean_block)

                    if options.get("sort_blocks", True):
                        serializable_result = sort_blocks_for_reading(serializable_result)

                    raw_results.append(
                        {
                            "page": page_number,
                            "api_version": api_version,
                            "result": serializable_result,
                        }
                    )
                    markdown_parts.append(
                        _ppstructure_result_to_markdown(page_number, serializable_result)
                    )
                    page_dir = raw_markdown_dir / f"page_{page_number:04d}"
                    write_text(
                        page_dir / f"page_{page_number:04d}.md",
                        _ppstructure_result_to_markdown(page_number, serializable_result),
                    )
                    rebuild_cleaned_text_from_raw_pages(
                        pp_output_dir,
                        pdf_path.name,
                        options,
                    )

        write_text(pp_output_dir / "output.md", "\n".join(markdown_parts))
        rebuild_cleaned_text_from_raw_pages(pp_output_dir, pdf_path.name, options)
        (pp_output_dir / "raw_result.json").write_text(
            json.dumps(raw_results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (metadata_dir / "image_table_metadata.json").write_text(
            json.dumps(media_metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        status = {
            "tool": tool_name,
            "status": "success",
            "seconds": round(now_seconds() - start, 2),
            "api_version": api_version,
            "mode": options.get("mode", "full"),
            "dpi_scale": options.get("dpi_scale", 2.0),
            "page_range": options.get("page_range", ""),
            "output": str(pp_output_dir / "output.md"),
            "cleaned_text": str(pp_output_dir / "cleaned_text.md"),
            "raw_output": str(pp_output_dir / "raw_result.json"),
            "raw_markdown_dir": str(raw_markdown_dir),
            "media_metadata": str(metadata_dir / "image_table_metadata.json"),
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

| Criterion | pymupdf4llm | PP-Structure | Notes |
| --- | ---: | ---: | --- |
| Reading order is correct |  |  |  |
| Double-column layout is handled |  |  |  |
| Headings are preserved |  |  |  |
| Paragraph blank lines are preserved |  |  |  |
| Tables are readable |  |  |  |
| Figures/captions are represented |  |  |  |
| No obvious header/footer noise |  |  |  |
| Output is suitable for chunking |  |  |  |

## Decision

Best parser for this PDF:

Reason:

Follow-up action:
"""

    write_text(pdf_output_dir / "review_checklist.md", template)


def run_experiment(
    pdf_path: Path,
    output_root: Path,
    tools: list[str],
    pp_options: dict,
):
    pdf_output_dir = output_root / pdf_path.stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    statuses = []
    if "pymupdf4llm" in tools:
        statuses.append(parse_with_pymupdf4llm(pdf_path, pdf_output_dir))
    if "pp_structure" in tools:
        statuses.append(parse_with_pp_structure(pdf_path, pdf_output_dir, pp_options))

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
        default=["pymupdf4llm", "pp_structure"],
        choices=["pymupdf4llm", "pp_structure"],
        help="Parsers to run.",
    )
    parser.add_argument(
        "--pp-mode",
        default="full",
        choices=["full", "layout_only", "table_only"],
        help="PP-Structure mode for local document parsing.",
    )
    parser.add_argument(
        "--pp-dpi-scale",
        type=float,
        default=2.0,
        help="PDF page render scale before PP-Structure OCR/layout parsing.",
    )
    parser.add_argument(
        "--pp-page-range",
        default="",
        help='Optional PP-Structure page range, for example "1-3" or "2,4-6".',
    )
    parser.add_argument(
        "--pp-no-sort",
        action="store_true",
        help="Keep PP-Structure original block order instead of bbox sorting.",
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

    pp_options = {
        "mode": args.pp_mode,
        "dpi_scale": args.pp_dpi_scale,
        "page_range": args.pp_page_range,
        "sort_blocks": not args.pp_no_sort,
        "show_log": True,
    }

    for pdf_path in pdfs:
        print(f"\n=== Running parser experiment for {pdf_path} ===")
        statuses = run_experiment(
            pdf_path,
            output_root,
            args.tools,
            pp_options,
        )
        for status in statuses:
            print(
                f"- {status['tool']}: {status['status']} "
                f"({status.get('seconds', 0)}s)"
            )

    print("\nDone. Open each review_checklist.md and score the parser outputs manually.")


if __name__ == "__main__":
    main()
