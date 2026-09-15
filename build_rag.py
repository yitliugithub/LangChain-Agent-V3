import hashlib
import json
import re
from pathlib import Path

import numpy as np

from rag_config import (
    CHROMA_DIR,
    CHUNK_METHOD,
    CHUNK_TOKENIZER_MODEL,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_PASSAGE_PREFIX,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    NORMALIZED_DATA_DIR,
)
from rag_models import E5Embedder


def get_chunk_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHUNK_TOKENIZER_MODEL)
    tokenizer.model_max_length = 10**9
    return tokenizer


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def token_fallback_split(text, max_tokens, tokenizer):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(token_ids), max_tokens):
        part = tokenizer.decode(
            token_ids[start:start + max_tokens],
            skip_special_tokens=True,
        ).strip()
        if part:
            chunks.append(part)
    return chunks


def heading_text(line):
    match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    return match.group(2).strip() if match else None


def split_by_headings(text):
    sections = []
    current_heading = "Document"
    current_lines = []

    for line in text.splitlines():
        heading = heading_text(line)
        if heading:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({"section": current_heading, "text": content})
            current_heading = heading
            current_lines = []
        else:
            current_lines.append(line.rstrip())

    content = "\n".join(current_lines).strip()
    if content:
        sections.append({"section": current_heading, "text": content})
    return sections


def is_table_line(line):
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def split_section_into_blocks(text):
    blocks = []
    paragraph_lines = []
    table_lines = []

    def flush_paragraph():
        if paragraph_lines:
            value = "\n".join(paragraph_lines).strip()
            if value:
                blocks.append({"type": "paragraph", "text": value})
            paragraph_lines.clear()

    def flush_table():
        if table_lines:
            value = "\n".join(table_lines).strip()
            if value:
                blocks.append({"type": "table", "text": value})
            table_lines.clear()

    for line in text.splitlines():
        if is_table_line(line):
            flush_paragraph()
            table_lines.append(line)
        else:
            flush_table()
            if line.strip():
                paragraph_lines.append(line)
            else:
                flush_paragraph()

    flush_paragraph()
    flush_table()
    return blocks


def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？.!?])\s*", text)
        if sentence.strip()
    ]


def split_oversized_table(text, max_tokens, tokenizer):
    lines = text.splitlines()
    header = lines[:2] if len(lines) >= 2 else []
    rows = lines[2:] if len(lines) >= 2 else lines
    chunks = []
    current_rows = []

    for row in rows:
        candidate = "\n".join(header + current_rows + [row])
        if count_tokens(candidate, tokenizer) <= max_tokens:
            current_rows.append(row)
            continue

        if current_rows:
            chunks.append(
                {"type": "table_split", "text": "\n".join(header + current_rows)}
            )
            current_rows = [row]
        else:
            chunks.extend(
                {"type": "table_token_fallback", "text": part}
                for part in token_fallback_split(row, max_tokens, tokenizer)
            )

    if current_rows:
        chunks.append(
            {"type": "table_split", "text": "\n".join(header + current_rows)}
        )
    return chunks


def sentence_distances(sentences, embedder):
    if len(sentences) < 2:
        return []

    embeddings = embedder.encode(
        [EMBEDDING_PASSAGE_PREFIX + sentence for sentence in sentences]
    )
    similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    return [float(1 - similarity) for similarity in similarities]


def split_index_by_semantic_distance(
    sentences,
    distances,
    start,
    end,
    min_tokens,
    tokenizer,
):
    candidates = []
    for split_index in range(start + 1, end):
        left_text = "".join(sentences[start:split_index])
        right_text = "".join(sentences[split_index:end])
        if count_tokens(left_text, tokenizer) < min_tokens:
            continue
        if right_text and count_tokens(right_text, tokenizer) < min_tokens:
            continue
        candidates.append((split_index, distances[split_index - 1]))

    if candidates:
        return max(candidates, key=lambda item: item[1])[0]

    fallback_candidates = []
    for split_index in range(start + 1, end):
        left_text = "".join(sentences[start:split_index])
        if count_tokens(left_text, tokenizer) >= min_tokens:
            fallback_candidates.append((split_index, distances[split_index - 1]))

    if fallback_candidates:
        return max(fallback_candidates, key=lambda item: item[1])[0]
    return end - 1


def semantic_sentence_split(
    sentences,
    max_tokens,
    min_tokens,
    tokenizer,
    embedder,
):
    distances = sentence_distances(sentences, embedder)
    units = []
    start = 0
    end = 1

    while end <= len(sentences):
        candidate = "".join(sentences[start:end])
        if count_tokens(candidate, tokenizer) <= max_tokens:
            end += 1
            continue

        split_index = split_index_by_semantic_distance(
            sentences,
            distances,
            start,
            end - 1,
            min_tokens,
            tokenizer,
        )
        text = "".join(sentences[start:split_index]).strip()
        if text:
            units.append({"type": "semantic_sentence_group", "text": text})
        start = split_index
        end = start + 1

    remainder = "".join(sentences[start:]).strip()
    if remainder:
        units.append({"type": "semantic_sentence_group", "text": remainder})
    return units


def split_oversized_unit(
    unit,
    max_tokens,
    min_tokens,
    tokenizer,
    embedder,
):
    if unit["type"] == "table":
        return split_oversized_table(unit["text"], max_tokens, tokenizer)

    sentences = split_sentences(unit["text"])
    if len(sentences) > 1:
        return semantic_sentence_split(
            sentences,
            max_tokens,
            min_tokens,
            tokenizer,
            embedder,
        )
    return [
        {"type": "token_fallback", "text": part}
        for part in token_fallback_split(unit["text"], max_tokens, tokenizer)
    ]


def create_units(section_text, max_tokens, min_tokens, tokenizer, embedder):
    raw_units = []
    for block in split_section_into_blocks(section_text):
        if block["type"] == "table":
            raw_units.append(block)
            continue

        sentences = split_sentences(block["text"])
        if len(sentences) > 1:
            raw_units.extend(
                {"type": "sentence", "text": sentence}
                for sentence in sentences
            )
        else:
            raw_units.append(block)

    units = []
    for unit in raw_units:
        if count_tokens(unit["text"], tokenizer) <= max_tokens:
            units.append(unit)
        else:
            units.extend(
                split_oversized_unit(
                    unit,
                    max_tokens,
                    min_tokens,
                    tokenizer,
                    embedder,
                )
            )
    return units


def unit_distances(units, embedder):
    distances = [None] * max(0, len(units) - 1)
    pairs = []
    positions = []
    for index in range(len(units) - 1):
        if units[index]["type"] == "table" or units[index + 1]["type"] == "table":
            continue
        pairs.append(
            (
                EMBEDDING_PASSAGE_PREFIX + units[index]["text"],
                EMBEDDING_PASSAGE_PREFIX + units[index + 1]["text"],
            )
        )
        positions.append(index)

    if not pairs:
        return distances

    flattened = [text for pair in pairs for text in pair]
    embeddings = embedder.encode(flattened)
    for pair_number, position in enumerate(positions):
        left = embeddings[pair_number * 2]
        right = embeddings[pair_number * 2 + 1]
        distances[position] = 1 - float(np.sum(left * right))
    return distances


def select_semantic_split(units, distances, start, end, min_tokens, tokenizer):
    candidates = []
    for split_index in range(start + 1, end):
        distance = distances[split_index - 1]
        if distance is None:
            continue
        left = "\n\n".join(unit["text"] for unit in units[start:split_index])
        right = "\n\n".join(unit["text"] for unit in units[split_index:end])
        if count_tokens(left, tokenizer) < min_tokens:
            continue
        if right and count_tokens(right, tokenizer) < min_tokens:
            continue
        candidates.append((split_index, distance))

    if candidates:
        return max(candidates, key=lambda item: item[1])[0]

    for split_index in range(end - 1, start, -1):
        left = "\n\n".join(unit["text"] for unit in units[start:split_index])
        if count_tokens(left, tokenizer) >= min_tokens:
            return split_index
    return max(start + 1, end - 1)


def make_chunk(section, units, boundary_distance, index, tokenizer):
    content = "\n\n".join(unit["text"] for unit in units).strip()
    text = f"Section: {section}\n\n{content}".strip()
    return {
        "chunk_index": index,
        "section": section,
        "token_count": count_tokens(text, tokenizer),
        "block_types": sorted({unit["type"] for unit in units}),
        "semantic_boundary_distance": boundary_distance,
        "text": text,
    }


def create_chunks_from_markdown(markdown_text, tokenizer, embedder):
    chunks = []
    for section in split_by_headings(markdown_text):
        name = section["section"]
        prefix_tokens = count_tokens(f"Section: {name}\n\n", tokenizer)
        content_limit = max(1, MAX_CHUNK_TOKENS - prefix_tokens)
        units = create_units(
            section["text"],
            content_limit,
            MIN_CHUNK_TOKENS,
            tokenizer,
            embedder,
        )
        if not units:
            continue
        distances = unit_distances(units, embedder)
        start = 0
        end = 1

        while end <= len(units):
            candidate_units = units[start:end]
            candidate = make_chunk(name, candidate_units, None, 0, tokenizer)
            if candidate["token_count"] <= MAX_CHUNK_TOKENS:
                end += 1
                continue

            split_index = select_semantic_split(
                units,
                distances,
                start,
                end - 1,
                MIN_CHUNK_TOKENS,
                tokenizer,
            )
            boundary = distances[split_index - 1]
            chunks.append(
                make_chunk(
                    name,
                    units[start:split_index],
                    boundary,
                    len(chunks),
                    tokenizer,
                )
            )
            start = split_index
            end = start + 1

        if start < len(units):
            chunks.append(
                make_chunk(name, units[start:], None, len(chunks), tokenizer)
            )

    return chunks


def create_chunk_id(source, section, index, text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    raw = f"{source}|{section}|{index}|{digest}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def recreate_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = {
        item.name if hasattr(item, "name") else str(item)
        for item in client.list_collections()
    }
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "chunk_method": CHUNK_METHOD,
        },
    )


def build_knowledge_base():
    markdown_files = sorted(NORMALIZED_DATA_DIR.glob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(
            f"{NORMALIZED_DATA_DIR} 中没有Markdown；请先运行 python preprocess_pdf.py"
        )

    print(f"找到 {len(markdown_files)} 个标准化Markdown文件。")
    print(f"Chunking: max={MAX_CHUNK_TOKENS}, min={MIN_CHUNK_TOKENS}")
    print(f"Embedding: {EMBEDDING_MODEL}")
    tokenizer = get_chunk_tokenizer()
    embedder = E5Embedder(EMBEDDING_MODEL)

    records = []
    try:
        for file_path in markdown_files:
            print(f"[Chunking] {file_path.name}")
            markdown = file_path.read_text(encoding="utf-8")
            chunks = create_chunks_from_markdown(markdown, tokenizer, embedder)
            print(f"  -> {len(chunks)} chunks")
            for chunk in chunks:
                records.append(
                    {
                        **chunk,
                        "source": file_path.stem,
                        "source_file": file_path.name,
                    }
                )

        texts = [EMBEDDING_PASSAGE_PREFIX + record["text"] for record in records]
        print(f"[Embedding] 正在编码 {len(texts)} 个Chunks...")
        embeddings = embedder.encode(texts)
    finally:
        embedder.close()

    collection = recreate_collection()
    batch_size = 128
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        batch_embeddings = embeddings[start:start + batch_size]
        collection.add(
            ids=[
                create_chunk_id(
                    item["source"],
                    item["section"],
                    item["chunk_index"],
                    item["text"],
                )
                for item in batch
            ],
            documents=[item["text"] for item in batch],
            embeddings=[embedding.tolist() for embedding in batch_embeddings],
            metadatas=[
                {
                    "source": item["source"],
                    "source_file": item["source_file"],
                    "section": item["section"],
                    "chunk_index": item["chunk_index"],
                    "token_count": item["token_count"],
                    "chunk_method": CHUNK_METHOD,
                    "block_types": ",".join(item["block_types"]),
                }
                for item in batch
            ],
        )

    summary = {
        "documents": len(markdown_files),
        "chunks": len(records),
        "collection": COLLECTION_NAME,
        "chunk_method": CHUNK_METHOD,
        "max_chunk_tokens": MAX_CHUNK_TOKENS,
        "min_chunk_tokens": MIN_CHUNK_TOKENS,
        "chunk_tokenizer": CHUNK_TOKENIZER_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    (CHROMA_DIR / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    build_knowledge_base()
