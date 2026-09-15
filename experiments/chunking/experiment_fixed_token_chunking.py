import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT = (
    "artifacts/pdf_parsing/2025年品牌营销趋势报告/"
    "magic_pdf/cleaned_text.md"
)
DEFAULT_OUTPUT_DIR = "artifacts/chunking"
DEFAULT_CHUNK_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 80


def load_text(path: Path):
    return path.read_text(encoding="utf-8")


def normalize_markdown_for_fixed_baseline(text: str, keep_markdown_tables: bool):
    """Remove Markdown noise before fixed-size chunking, without changing meaning."""
    cleaned_lines = []
    image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

    for line in text.splitlines():
        stripped = line.strip()

        # These lines describe the preprocessing experiment, not report content.
        if stripped in {"# Cleaned Magic-PDF / MinerU Text"}:
            continue
        if stripped.startswith("Source: "):
            continue
        if stripped.startswith("Raw output directory: "):
            continue
        if stripped.startswith("## Source file: "):
            continue

        # Keep heading words as semantic hints, but remove Markdown syntax.
        line = re.sub(r"^#{1,6}\s+", "", line)

        # Keep image references as lightweight metadata instead of raw Markdown.
        line = image_pattern.sub(r"[Image: \1]", line)

        # Convert Markdown table rows to readable text for text-only RAG chunks.
        # Keeping tables is useful when reviewing whether retrieval preserves
        # structured evidence.
        if (
            not keep_markdown_tables
            and line.strip().startswith("|")
            and line.strip().endswith("|")
        ):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            is_separator = all(
                re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
                for cell in cells
                if cell
            )
            if is_separator:
                continue
            line = "；".join(cell for cell in cells if cell)

        cleaned_lines.append(line.rstrip())

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip() + "\n"


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_tokenizer(model_name: str = ""):
    if not model_name:
        return None

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # We only use the tokenizer for counting/slicing, not model inference.
        tokenizer.model_max_length = 10**9
        return tokenizer
    except Exception as exc:
        print(f"[Warning] tokenizer load failed: {exc}")
        print("[Warning] fallback to regex tokenization.")
        return None


def regex_tokenize(text: str):
    # Fallback tokenizer for experiments only. It keeps CJK characters,
    # Latin/number sequences, punctuation, and whitespace as separate units.
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\S\r\n]+|\n|[^\w\s]", text)


def regex_detokenize(tokens):
    return "".join(tokens)


def encode_text(text: str, tokenizer):
    if tokenizer is None:
        return regex_tokenize(text)
    return tokenizer.encode(text, add_special_tokens=False)


def decode_tokens(tokens, tokenizer):
    if tokenizer is None:
        return regex_detokenize(tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def create_fixed_token_chunks(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
    tokenizer=None,
):
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be greater than 0.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be greater than or equal to 0.")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens.")

    tokens = encode_text(text, tokenizer)
    step = chunk_tokens - overlap_tokens
    chunks = []
    start = 0
    index = 0

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_token_ids = tokens[start:end]
        chunk_text = decode_tokens(chunk_token_ids, tokenizer).strip()

        if chunk_text:
            chunks.append(
                {
                    "chunk_index": index,
                    "start_token": start,
                    "end_token": end,
                    "token_count": len(chunk_token_ids),
                    "text": chunk_text,
                }
            )
            index += 1

        if end == len(tokens):
            break
        start += step

    return chunks, len(tokens)


def summarize_chunks(
    chunks,
    total_tokens,
    chunk_tokens,
    overlap_tokens,
    input_path,
    preprocessing,
):
    token_counts = [chunk["token_count"] for chunk in chunks]
    text_lengths = [len(chunk["text"]) for chunk in chunks]

    return {
        "input": str(input_path),
        "method": "fixed_token_with_overlap",
        "preprocessing": preprocessing,
        "chunk_tokens": chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "total_tokens": total_tokens,
        "chunk_count": len(chunks),
        "min_chunk_tokens": min(token_counts) if token_counts else 0,
        "max_chunk_tokens": max(token_counts) if token_counts else 0,
        "avg_chunk_tokens": round(sum(token_counts) / len(token_counts), 2)
        if token_counts
        else 0,
        "min_chars": min(text_lengths) if text_lengths else 0,
        "max_chars": max(text_lengths) if text_lengths else 0,
        "avg_chars": round(sum(text_lengths) / len(text_lengths), 2)
        if text_lengths
        else 0,
    }


def write_chunks_jsonl(path: Path, chunks, source_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            row = {
                "source": source_name,
                "chunk_method": "fixed_token_with_overlap",
                **chunk,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review_markdown(path: Path, chunks, summary):
    preview_parts = [
        "# Fixed-Size Token Chunking Experiment\n",
        "## Summary\n",
        f"- Input: `{summary['input']}`",
        f"- Preprocessing: `{summary['preprocessing']}`",
        f"- Chunk tokens: `{summary['chunk_tokens']}`",
        f"- Overlap tokens: `{summary['overlap_tokens']}`",
        f"- Total tokens: `{summary['total_tokens']}`",
        f"- Chunk count: `{summary['chunk_count']}`",
        f"- Avg chunk tokens: `{summary['avg_chunk_tokens']}`",
        "\n## Preview\n",
    ]

    for chunk in chunks[:8]:
        preview_parts.append(
            f"### Chunk {chunk['chunk_index']} "
            f"({chunk['start_token']}-{chunk['end_token']}, "
            f"{chunk['token_count']} tokens)\n"
        )
        preview = chunk["text"][:900]
        preview_parts.append(preview)
        preview_parts.append("\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(preview_parts), encoding="utf-8")


def build_output_paths(
    output_dir: Path,
    input_path: Path,
    chunk_tokens,
    overlap_tokens,
    clean_markdown_syntax,
    keep_markdown_tables,
):
    experiment_name = f"{input_path.stem}_fixed_{chunk_tokens}_overlap_{overlap_tokens}"
    if clean_markdown_syntax:
        experiment_name += "_cleaned"
    if clean_markdown_syntax and keep_markdown_tables:
        experiment_name += "_keep_tables"
    experiment_dir = output_dir / input_path.parent.parent.name / experiment_name
    return {
        "dir": experiment_dir,
        "chunks": experiment_dir / "chunks.jsonl",
        "summary": experiment_dir / "summary.json",
        "review": experiment_dir / "review.md",
    }


def run_experiment(
    input_path: Path,
    output_dir: Path,
    chunk_tokens,
    overlap_tokens,
    tokenizer_name,
    clean_markdown_syntax,
    keep_markdown_tables,
):
    text = load_text(input_path)
    preprocessing = "clean_markdown_syntax" if clean_markdown_syntax else "none"
    if clean_markdown_syntax:
        text = normalize_markdown_for_fixed_baseline(
            text,
            keep_markdown_tables=keep_markdown_tables,
        )
    if clean_markdown_syntax and keep_markdown_tables:
        preprocessing = "clean_markdown_syntax_keep_markdown_tables"

    tokenizer = get_tokenizer(tokenizer_name)
    chunks, total_tokens = create_fixed_token_chunks(
        text=text,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer=tokenizer,
    )
    summary = summarize_chunks(
        chunks=chunks,
        total_tokens=total_tokens,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        input_path=input_path,
        preprocessing=preprocessing,
    )
    paths = build_output_paths(
        output_dir,
        input_path,
        chunk_tokens,
        overlap_tokens,
        clean_markdown_syntax,
        keep_markdown_tables,
    )

    write_chunks_jsonl(paths["chunks"], chunks, source_name=input_path.name)
    write_json(paths["summary"], summary)
    write_review_markdown(paths["review"], chunks, summary)

    return {
        "summary": summary,
        "outputs": {key: str(value) for key, value in paths.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-size token chunking + overlap experiment."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help="Input cleaned_text.md file.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for chunking experiment files.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help="Maximum tokens per chunk.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=DEFAULT_OVERLAP_TOKENS,
        help="Overlapping tokens between neighboring chunks.",
    )
    parser.add_argument(
        "--tokenizer",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help=(
            "Tokenizer used for token counting. Empty string uses regex fallback."
        ),
    )
    parser.add_argument(
        "--clean-markdown-syntax",
        action="store_true",
        help=(
            "Remove Markdown syntax and experiment metadata before fixed-size "
            "chunking. The original cleaned_text.md is not modified."
        ),
    )
    parser.add_argument(
        "--keep-markdown-tables",
        action="store_true",
        help=(
            "When --clean-markdown-syntax is enabled, keep Markdown table rows "
            "instead of converting them to plain text."
        ),
    )

    args = parser.parse_args()
    result = run_experiment(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        tokenizer_name=args.tokenizer,
        clean_markdown_syntax=args.clean_markdown_syntax,
        keep_markdown_tables=args.keep_markdown_tables,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
