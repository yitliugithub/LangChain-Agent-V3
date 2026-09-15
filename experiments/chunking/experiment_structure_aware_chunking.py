import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT = (
    "artifacts/pdf_parsing/2025年品牌营销趋势报告/"
    "magic_pdf/cleaned_text.md"
)
DEFAULT_OUTPUT_DIR = "artifacts/chunking"
DEFAULT_MAX_TOKENS = 400
DEFAULT_MIN_TOKENS = 120


def load_text(path: Path):
    return path.read_text(encoding="utf-8")


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
        # This experiment uses the tokenizer only for counting and slicing.
        tokenizer.model_max_length = 10**9
        return tokenizer
    except Exception as exc:
        print(f"[Warning] tokenizer load failed: {exc}")
        print("[Warning] fallback to regex tokenization.")
        return None


def regex_tokenize(text: str):
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


def count_tokens(text: str, tokenizer):
    return len(encode_text(text, tokenizer))


def clean_experiment_metadata(text: str):
    """Keep document structure, but remove parser experiment metadata."""
    cleaned_lines = []
    image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

    for line in text.splitlines():
        stripped = line.strip()

        if stripped in {"# Cleaned Magic-PDF / MinerU Text"}:
            continue
        if stripped.startswith("Source: "):
            continue
        if stripped.startswith("Raw output directory: "):
            continue
        if stripped.startswith("## Source file: "):
            continue

        line = image_pattern.sub(r"[Image: \1]", line)
        cleaned_lines.append(line.rstrip())

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip() + "\n"


def heading_text(line: str):
    match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if not match:
        return None
    return match.group(2).strip()


def is_table_line(line: str):
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def split_document_into_sections(text: str):
    sections = []
    current_heading = "Document"
    current_lines = []

    for line in text.splitlines():
        heading = heading_text(line)
        if heading:
            if current_lines:
                sections.append(
                    {
                        "section": current_heading,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
            current_heading = heading
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "section": current_heading,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return [section for section in sections if section["text"]]


def split_section_into_blocks(section_text: str):
    blocks = []
    paragraph_lines = []
    table_lines = []

    def flush_paragraph():
        if paragraph_lines:
            text = "\n".join(paragraph_lines).strip()
            if text:
                blocks.append({"type": "paragraph", "text": text})
            paragraph_lines.clear()

    def flush_table():
        if table_lines:
            text = "\n".join(table_lines).strip()
            if text:
                blocks.append({"type": "table", "text": text})
            table_lines.clear()

    for line in section_text.splitlines():
        if is_table_line(line):
            flush_paragraph()
            table_lines.append(line)
            continue

        flush_table()
        if not line.strip():
            flush_paragraph()
        else:
            paragraph_lines.append(line)

    flush_paragraph()
    flush_table()
    return blocks


def split_sentences(text: str):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？.!?])\s*", text)
        if sentence.strip()
    ]


def token_fallback_split(text: str, max_tokens: int, tokenizer):
    tokens = encode_text(text, tokenizer)
    chunks = []

    for start in range(0, len(tokens), max_tokens):
        chunk_text = decode_tokens(tokens[start:start + max_tokens], tokenizer).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def split_oversized_block(block, max_tokens: int, tokenizer):
    token_count = count_tokens(block["text"], tokenizer)
    if token_count <= max_tokens:
        return [block]

    if block["type"] == "table":
        lines = block["text"].splitlines()
        header = lines[:2] if len(lines) >= 2 else []
        rows = lines[2:] if len(lines) >= 2 else lines
        split_blocks = []
        current_rows = []

        for row in rows:
            candidate_lines = header + current_rows + [row]
            candidate = "\n".join(candidate_lines)
            if count_tokens(candidate, tokenizer) <= max_tokens:
                current_rows.append(row)
                continue

            if current_rows:
                split_blocks.append(
                    {"type": "table_split", "text": "\n".join(header + current_rows)}
                )
                current_rows = [row]
            else:
                split_blocks.extend(
                    {
                        "type": "table_token_fallback",
                        "text": part,
                    }
                    for part in token_fallback_split(row, max_tokens, tokenizer)
                )

        if current_rows:
            split_blocks.append(
                {"type": "table_split", "text": "\n".join(header + current_rows)}
            )
        return split_blocks

    sentences = split_sentences(block["text"])
    if len(sentences) > 1:
        return pack_units_as_blocks(sentences, "sentence_group", max_tokens, tokenizer)

    return [
        {"type": "token_fallback", "text": part}
        for part in token_fallback_split(block["text"], max_tokens, tokenizer)
    ]


def pack_units_as_blocks(units, block_type: str, max_tokens: int, tokenizer):
    blocks = []
    current = ""

    for unit in units:
        candidate = f"{current}{unit}" if current else unit
        if count_tokens(candidate, tokenizer) <= max_tokens:
            current = candidate
            continue

        if current:
            blocks.append({"type": block_type, "text": current})

        if count_tokens(unit, tokenizer) > max_tokens:
            blocks.extend(
                {"type": "token_fallback", "text": part}
                for part in token_fallback_split(unit, max_tokens, tokenizer)
            )
            current = ""
        else:
            current = unit

    if current:
        blocks.append({"type": block_type, "text": current})

    return blocks


def chunk_prefix(section: str):
    return f"Section: {section}\n\n"


def create_structure_aware_chunks(text: str, max_tokens: int, min_tokens: int, tokenizer):
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0.")
    if min_tokens < 0:
        raise ValueError("min_tokens must be greater than or equal to 0.")
    if min_tokens >= max_tokens:
        raise ValueError("min_tokens must be smaller than max_tokens.")

    sections = split_document_into_sections(text)
    chunks = []
    chunk_index = 0

    for section in sections:
        section_name = section["section"]
        prefix = chunk_prefix(section_name)
        prefix_tokens = count_tokens(prefix, tokenizer)
        content_max_tokens = max(1, max_tokens - prefix_tokens)
        raw_blocks = split_section_into_blocks(section["text"])
        blocks = []
        for block in raw_blocks:
            blocks.extend(split_oversized_block(block, content_max_tokens, tokenizer))

        current_blocks = []
        current_types = []

        def flush_current():
            nonlocal chunk_index
            if not current_blocks:
                return
            content = "\n\n".join(current_blocks).strip()
            chunk_text = f"{prefix}{content}".strip()
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "section": section_name,
                    "token_count": count_tokens(chunk_text, tokenizer),
                    "block_types": sorted(set(current_types)),
                    "text": chunk_text,
                }
            )
            chunk_index += 1
            current_blocks.clear()
            current_types.clear()

        for block in blocks:
            candidate_blocks = current_blocks + [block["text"]]
            candidate_text = f"{prefix}{chr(10).join(candidate_blocks)}"

            if count_tokens(candidate_text, tokenizer) <= max_tokens:
                current_blocks.append(block["text"])
                current_types.append(block["type"])
                continue

            flush_current()
            current_blocks.append(block["text"])
            current_types.append(block["type"])

        flush_current()

    return merge_small_neighbor_chunks(chunks, max_tokens, min_tokens, tokenizer)


def merge_small_neighbor_chunks(chunks, max_tokens: int, min_tokens: int, tokenizer):
    if not chunks:
        return []

    merged = []
    for chunk in chunks:
        if (
            merged
            and chunk["section"] == merged[-1]["section"]
            and chunk["token_count"] < min_tokens
            and "table" not in chunk["block_types"]
        ):
            candidate_text = merged[-1]["text"] + "\n\n" + chunk["text"].split("\n\n", 1)[-1]
            candidate_tokens = count_tokens(candidate_text, tokenizer)
            if candidate_tokens <= max_tokens:
                merged[-1]["text"] = candidate_text
                merged[-1]["token_count"] = candidate_tokens
                merged[-1]["block_types"] = sorted(
                    set(merged[-1]["block_types"] + chunk["block_types"])
                )
                continue
        merged.append(chunk)

    for index, chunk in enumerate(merged):
        chunk["chunk_index"] = index

    return merged


def summarize_chunks(chunks, input_path: Path, max_tokens: int, min_tokens: int):
    token_counts = [chunk["token_count"] for chunk in chunks]
    table_chunks = [
        chunk for chunk in chunks
        if any("table" in block_type for block_type in chunk["block_types"])
    ]

    return {
        "input": str(input_path),
        "method": "structure_aware_heading_paragraph_table_sentence_token",
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "chunk_count": len(chunks),
        "table_chunk_count": len(table_chunks),
        "min_chunk_tokens": min(token_counts) if token_counts else 0,
        "max_chunk_tokens": max(token_counts) if token_counts else 0,
        "avg_chunk_tokens": round(sum(token_counts) / len(token_counts), 2)
        if token_counts
        else 0,
    }


def build_output_paths(output_dir: Path, input_path: Path, max_tokens, min_tokens):
    experiment_name = f"{input_path.stem}_structure_aware_{max_tokens}_min_{min_tokens}"
    experiment_dir = output_dir / input_path.parent.parent.name / experiment_name
    return {
        "dir": experiment_dir,
        "chunks": experiment_dir / "chunks.jsonl",
        "summary": experiment_dir / "summary.json",
        "review": experiment_dir / "review.md",
    }


def write_chunks_jsonl(path: Path, chunks, source_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            row = {
                "source": source_name,
                "chunk_method": "structure_aware",
                **chunk,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review_markdown(path: Path, chunks, summary):
    preview_parts = [
        "# Structure-Aware Chunking Experiment\n",
        "## Summary\n",
        f"- Input: `{summary['input']}`",
        f"- Method: `{summary['method']}`",
        f"- Max tokens: `{summary['max_tokens']}`",
        f"- Min tokens: `{summary['min_tokens']}`",
        f"- Chunk count: `{summary['chunk_count']}`",
        f"- Table chunk count: `{summary['table_chunk_count']}`",
        f"- Avg chunk tokens: `{summary['avg_chunk_tokens']}`",
        "\n## Preview\n",
    ]

    for chunk in chunks[:12]:
        preview_parts.append(
            f"### Chunk {chunk['chunk_index']} "
            f"({chunk['token_count']} tokens, "
            f"section: {chunk['section']}, "
            f"blocks: {', '.join(chunk['block_types'])})\n"
        )
        preview_parts.append(chunk["text"][:1200])
        preview_parts.append("\n")

    table_examples = [
        chunk for chunk in chunks
        if any("table" in block_type for block_type in chunk["block_types"])
    ][:5]
    if table_examples:
        preview_parts.append("\n## Table Chunk Examples\n")
        for chunk in table_examples:
            preview_parts.append(
                f"### Chunk {chunk['chunk_index']} "
                f"({chunk['token_count']} tokens, section: {chunk['section']})\n"
            )
            preview_parts.append(chunk["text"][:1400])
            preview_parts.append("\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(preview_parts), encoding="utf-8")


def run_experiment(input_path: Path, output_dir: Path, max_tokens, min_tokens, tokenizer_name):
    raw_text = load_text(input_path)
    text = clean_experiment_metadata(raw_text)
    tokenizer = get_tokenizer(tokenizer_name)
    chunks = create_structure_aware_chunks(
        text=text,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        tokenizer=tokenizer,
    )
    summary = summarize_chunks(
        chunks=chunks,
        input_path=input_path,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
    )
    paths = build_output_paths(output_dir, input_path, max_tokens, min_tokens)

    write_chunks_jsonl(paths["chunks"], chunks, source_name=input_path.name)
    write_json(paths["summary"], summary)
    write_review_markdown(paths["review"], chunks, summary)

    return {
        "summary": summary,
        "outputs": {key: str(value) for key, value in paths.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Structure-aware token chunking experiment."
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
        help="Output directory for experiment files.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens per chunk.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help="Small chunk merge threshold.",
    )
    parser.add_argument(
        "--tokenizer",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Tokenizer used for token counting. Empty string uses regex fallback.",
    )

    args = parser.parse_args()
    result = run_experiment(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        tokenizer_name=args.tokenizer,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
