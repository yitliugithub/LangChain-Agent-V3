import argparse
import json
import re
from pathlib import Path

import numpy as np

from experiments.retrieval.experiment_retrieval_evaluation import TransformersMeanPoolingEmbedder, embed_texts
from experiments.chunking.experiment_structure_aware_chunking import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    chunk_prefix,
    clean_experiment_metadata,
    count_tokens,
    get_tokenizer,
    load_text,
    split_document_into_sections,
    split_section_into_blocks,
    split_sentences,
    split_oversized_block,
    summarize_chunks,
    write_chunks_jsonl,
    write_json,
    write_review_markdown,
)


DEFAULT_MAX_TOKENS = 500
DEFAULT_MIN_TOKENS = 150
DEFAULT_BOUNDARY_THRESHOLD = "mean"
SEMANTIC_MODEL_NAME = "intfloat/multilingual-e5-base"
SEMANTIC_PASSAGE_PREFIX = "passage: "


def sentence_distances(sentences, semantic_model):
    if len(sentences) < 2:
        return []

    embeddings = embed_texts(
        semantic_model,
        [SEMANTIC_PASSAGE_PREFIX + sentence for sentence in sentences],
    )
    similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    distances = 1 - similarities
    return [float(distance) for distance in distances]


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


def semantic_sentence_split(sentences, max_tokens, min_tokens, tokenizer, semantic_model):
    if not sentences:
        return []

    distances = sentence_distances(sentences, semantic_model)
    chunks = []
    start = 0
    end = 1

    while end <= len(sentences):
        candidate = "".join(sentences[start:end])
        if count_tokens(candidate, tokenizer) <= max_tokens:
            end += 1
            continue

        split_index = split_index_by_semantic_distance(
            sentences=sentences,
            distances=distances,
            start=start,
            end=end - 1,
            min_tokens=min_tokens,
            tokenizer=tokenizer,
        )
        chunk_text = "".join(sentences[start:split_index]).strip()
        if chunk_text:
            chunks.append(
                {
                    "type": "semantic_sentence_group",
                    "text": chunk_text,
                    "semantic_boundary_distance": distances[split_index - 1]
                    if split_index - 1 < len(distances)
                    else None,
                }
            )
        start = split_index
        end = start + 1

    remainder = "".join(sentences[start:]).strip()
    if remainder:
        chunks.append(
            {
                "type": "semantic_sentence_group",
                "text": remainder,
                "semantic_boundary_distance": None,
            }
        )
    return chunks


def split_oversized_block_semantic(
    block,
    max_tokens,
    min_tokens,
    tokenizer,
    semantic_model,
):
    token_count = count_tokens(block["text"], tokenizer)
    if token_count <= max_tokens:
        return [block]

    if block["type"] == "table":
        return split_oversized_block(block, max_tokens, tokenizer)

    sentences = split_sentences(block["text"])
    if len(sentences) > 1:
        return semantic_sentence_split(
            sentences=sentences,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            tokenizer=tokenizer,
            semantic_model=semantic_model,
        )

    return split_oversized_block(block, max_tokens, tokenizer)


def section_to_semantic_units(section_text):
    units = []
    for block in split_section_into_blocks(section_text):
        if block["type"] == "table":
            units.append({"type": "table", "text": block["text"]})
            continue

        sentences = split_sentences(block["text"])
        if len(sentences) > 1:
            units.extend(
                {"type": "sentence", "text": sentence}
                for sentence in sentences
            )
        else:
            units.append({"type": "paragraph", "text": block["text"]})
    return units


def unit_distances(units, semantic_model):
    distance_values = [None] * max(0, len(units) - 1)
    comparable_pairs = []
    pair_positions = []

    for index in range(len(units) - 1):
        if units[index]["type"] == "table" or units[index + 1]["type"] == "table":
            continue
        comparable_pairs.append(
            (
                SEMANTIC_PASSAGE_PREFIX + units[index]["text"],
                SEMANTIC_PASSAGE_PREFIX + units[index + 1]["text"],
            )
        )
        pair_positions.append(index)

    if not comparable_pairs:
        return distance_values

    flattened = []
    for left, right in comparable_pairs:
        flattened.extend([left, right])
    embeddings = embed_texts(semantic_model, flattened)

    for pair_number, position in enumerate(pair_positions):
        left_embedding = embeddings[pair_number * 2]
        right_embedding = embeddings[pair_number * 2 + 1]
        similarity = float(np.sum(left_embedding * right_embedding))
        distance_values[position] = 1 - similarity

    return distance_values


def split_index_for_unit_window(
    units,
    distances,
    start,
    end,
    min_tokens,
    tokenizer,
):
    candidates = []
    for split_index in range(start + 1, end):
        distance = distances[split_index - 1]
        if distance is None:
            continue
        left_text = "\n\n".join(unit["text"] for unit in units[start:split_index])
        right_text = "\n\n".join(unit["text"] for unit in units[split_index:end])
        if count_tokens(left_text, tokenizer) < min_tokens:
            continue
        if right_text and count_tokens(right_text, tokenizer) < min_tokens:
            continue
        candidates.append((split_index, distance))

    if candidates:
        return max(candidates, key=lambda item: item[1])[0]

    for split_index in range(end - 1, start, -1):
        left_text = "\n\n".join(unit["text"] for unit in units[start:split_index])
        if count_tokens(left_text, tokenizer) >= min_tokens:
            return split_index

    return max(start + 1, end - 1)


def create_semantic_boundary_chunks(
    text,
    max_tokens,
    min_tokens,
    tokenizer,
    semantic_model,
):
    sections = split_document_into_sections(text)
    chunks = []
    chunk_index = 0
    semantic_boundary_count = 0

    for section in sections:
        section_name = section["section"]
        prefix = chunk_prefix(section_name)
        prefix_tokens = count_tokens(prefix, tokenizer)
        content_max_tokens = max(1, max_tokens - prefix_tokens)
        raw_units = section_to_semantic_units(section["text"])
        units = []
        for unit in raw_units:
            if count_tokens(unit["text"], tokenizer) <= content_max_tokens:
                units.append(unit)
                continue
            split_blocks = split_oversized_block_semantic(
                block=unit,
                max_tokens=content_max_tokens,
                min_tokens=min_tokens,
                tokenizer=tokenizer,
                semantic_model=semantic_model,
            )
            units.extend(
                {"type": split_block["type"], "text": split_block["text"]}
                for split_block in split_blocks
            )

        distances = unit_distances(units, semantic_model)
        start = 0
        end = 1

        while end <= len(units):
            content = "\n\n".join(unit["text"] for unit in units[start:end])
            candidate_text = f"{prefix}{content}"
            if count_tokens(candidate_text, tokenizer) <= max_tokens:
                end += 1
                continue

            split_index = split_index_for_unit_window(
                units=units,
                distances=distances,
                start=start,
                end=end - 1,
                min_tokens=min_tokens,
                tokenizer=tokenizer,
            )
            boundary_distance = (
                distances[split_index - 1]
                if split_index - 1 < len(distances)
                else None
            )
            if boundary_distance is not None:
                semantic_boundary_count += 1

            chunk_units = units[start:split_index]
            content = "\n\n".join(unit["text"] for unit in chunk_units).strip()
            chunk_text = f"{prefix}{content}".strip()
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "section": section_name,
                    "token_count": count_tokens(chunk_text, tokenizer),
                    "block_types": sorted(set(unit["type"] for unit in chunk_units)),
                    "semantic_boundary_distances": [boundary_distance]
                    if boundary_distance is not None
                    else [],
                    "text": chunk_text,
                }
            )
            chunk_index += 1
            start = split_index
            end = start + 1

        if start < len(units):
            chunk_units = units[start:]
            content = "\n\n".join(unit["text"] for unit in chunk_units).strip()
            chunk_text = f"{prefix}{content}".strip()
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "section": section_name,
                    "token_count": count_tokens(chunk_text, tokenizer),
                    "block_types": sorted(set(unit["type"] for unit in chunk_units)),
                    "semantic_boundary_distances": [],
                    "text": chunk_text,
                }
            )
            chunk_index += 1

    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index

    return chunks, semantic_boundary_count


def build_output_paths(output_dir: Path, input_path: Path, max_tokens, min_tokens):
    experiment_name = (
        f"{input_path.stem}_structure_semantic_boundary_{max_tokens}_min_{min_tokens}"
    )
    experiment_dir = output_dir / input_path.parent.parent.name / experiment_name
    return {
        "dir": experiment_dir,
        "chunks": experiment_dir / "chunks.jsonl",
        "summary": experiment_dir / "summary.json",
        "review": experiment_dir / "review.md",
    }


def add_theme_boundary_summary(chunks):
    section_transitions = 0
    near_limit_same_section_transitions = 0

    for previous, current in zip(chunks, chunks[1:]):
        if previous["section"] != current["section"]:
            continue
        section_transitions += 1
        if previous["token_count"] >= DEFAULT_MAX_TOKENS * 0.85:
            near_limit_same_section_transitions += 1

    return {
        "same_section_chunk_transitions": section_transitions,
        "near_limit_same_section_transitions": near_limit_same_section_transitions,
    }


def run_experiment(input_path, output_dir, max_tokens, min_tokens, tokenizer_name):
    raw_text = load_text(input_path)
    text = clean_experiment_metadata(raw_text)
    tokenizer = get_tokenizer(tokenizer_name)
    semantic_model = TransformersMeanPoolingEmbedder(SEMANTIC_MODEL_NAME)
    chunks, semantic_boundary_count = create_semantic_boundary_chunks(
        text=text,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        tokenizer=tokenizer,
        semantic_model=semantic_model,
    )
    summary = summarize_chunks(
        chunks=chunks,
        input_path=input_path,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
    )
    summary["method"] = "structure_aware_plus_semantic_sentence_boundary"
    summary["semantic_model"] = SEMANTIC_MODEL_NAME
    summary["semantic_boundary_count"] = semantic_boundary_count
    summary.update(add_theme_boundary_summary(chunks))

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
        description="Structure-aware chunking with semantic sentence boundaries."
    )
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument(
        "--tokenizer",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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
