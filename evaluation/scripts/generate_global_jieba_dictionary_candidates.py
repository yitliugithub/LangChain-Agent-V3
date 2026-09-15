import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import jieba


BASE_DIR = Path(__file__).resolve().parents[2]
CHUNK_ROOT = BASE_DIR / "artifacts/chunking"
CHUNK_DIR_NAME = "cleaned_text_structure_semantic_boundary_500_min_150"
OUTPUT_DIR = BASE_DIR / "resources" / "jieba_dictionary_candidates"

MAX_CANDIDATES = 300
MIN_TERM_LENGTH = 2
MAX_TERM_LENGTH = 8
MIN_FREQUENCY = 4

STOP_TERMS = {
    "一个",
    "一些",
    "一种",
    "以及",
    "因此",
    "但是",
    "同时",
    "其中",
    "进行",
    "通过",
    "对于",
    "由于",
    "如果",
    "需要",
    "可以",
    "更加",
    "可能",
    "不同",
    "整体",
    "主要",
    "相关",
    "方面",
    "情况",
    "问题",
    "方式",
    "能力",
    "目前",
    "未来",
    "当前",
    "成为",
    "已经",
    "仍然",
    "进一步",
    "越来越",
    "发展",
    "分析",
    "数据",
    "报告",
    "来源",
    "页码",
}

BAD_BOUNDARY_CHARS = set(
    "的一了是在和与及或而被把对从为于中上下来去着过更最也都将可会能有无各其该此这那"
)

ASCII_STOP_TERMS = {
    "section",
    "image",
    "images",
    "jpg",
    "jpeg",
    "png",
    "http",
    "https",
    "www",
    "com",
    "pdf",
    "page",
    "source",
    "copyright",
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def load_corpus():
    documents = []
    chunks = []
    pattern = f"*/{CHUNK_DIR_NAME}/chunks.jsonl"
    for path in sorted(CHUNK_ROOT.glob(pattern)):
        document = path.parent.parent.name
        documents.append(document)
        for chunk in load_jsonl(path):
            chunks.append({**chunk, "document": document})
    return documents, chunks


def clean_text(text):
    text = re.sub(r"\[Image:\s*[^\]]+\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"^Section:\s*", "", text, flags=re.MULTILINE)
    return text


def chinese_sequences(text):
    return re.findall(r"[\u4e00-\u9fff]+", clean_text(text))


def entropy(counter):
    total = sum(counter.values())
    if total == 0 or len(counter) <= 1:
        return 0.0
    value = 0.0
    for count in counter.values():
        probability = count / total
        value -= probability * math.log2(probability)
    return value


def collect_ngram_statistics(chunks):
    counts = Counter()
    left_neighbors = defaultdict(Counter)
    right_neighbors = defaultdict(Counter)
    document_hits = defaultdict(set)
    total_characters = 0

    for chunk in chunks:
        document = chunk["document"]
        for sequence in chinese_sequences(chunk.get("text", "")):
            total_characters += len(sequence)
            sequence_length = len(sequence)
            for start in range(sequence_length):
                max_size = min(MAX_TERM_LENGTH, sequence_length - start)
                for size in range(1, max_size + 1):
                    term = sequence[start : start + size]
                    counts[term] += 1
                    if size >= MIN_TERM_LENGTH:
                        left = sequence[start - 1] if start > 0 else "<B>"
                        end = start + size
                        right = sequence[end] if end < sequence_length else "<E>"
                        left_neighbors[term][left] += 1
                        right_neighbors[term][right] += 1
                        document_hits[term].add(document)

    return counts, left_neighbors, right_neighbors, document_hits, total_characters


def minimum_pmi(term, counts, total_characters):
    term_count = counts[term]
    values = []
    for split in range(1, len(term)):
        left = term[:split]
        right = term[split:]
        denominator = counts[left] * counts[right]
        if denominator:
            values.append(math.log2((term_count * total_characters) / denominator))
    return min(values) if values else 0.0


def normalize_metadata_phrase(text):
    text = re.sub(r"\d{4}年?", " ", text)
    text = re.sub(r"\.(pdf|md)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\*_#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def metadata_phrases(documents, chunks):
    terms = defaultdict(set)
    term_documents = defaultdict(set)

    values = [(document, "document_title", document) for document in documents]
    unique_sections = {
        (chunk.get("section", "").strip(), chunk["document"])
        for chunk in chunks
        if chunk.get("section", "").strip()
    }
    values.extend(
        (section, "section", document) for section, document in unique_sections
    )

    for raw_value, source, document in values:
        value = normalize_metadata_phrase(raw_value)
        for part in re.split(r"[，。！？；：、（）()《》“”‘’/|\s]+", value):
            part = part.strip()
            if re.fullmatch(r"[\u4e00-\u9fff]{2,12}", part):
                terms[part].add(source)
                term_documents[part].add(document)
            for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", part):
                if 2 <= len(sequence) <= 12:
                    terms[sequence].add(source)
                    term_documents[sequence].add(document)

    return terms, term_documents


def collect_ascii_terms(chunks):
    counts = Counter()
    documents = defaultdict(set)
    for chunk in chunks:
        text = clean_text(chunk.get("text", ""))
        for term in re.findall(r"(?<![A-Za-z])[A-Za-z][A-Za-z0-9]{1,11}", text):
            normalized = term.upper()
            if normalized.lower() in ASCII_STOP_TERMS:
                continue
            counts[normalized] += 1
            documents[normalized].add(chunk["document"])
    return counts, documents


def is_valid_chinese_term(term, frequency, pmi, boundary_entropy, sources):
    if term in STOP_TERMS or len(set(term)) == 1:
        return False
    if term[0] in BAD_BOUNDARY_CHARS or term[-1] in BAD_BOUNDARY_CHARS:
        return False
    if not (MIN_TERM_LENGTH <= len(term) <= MAX_TERM_LENGTH):
        return False

    from_metadata = bool(sources & {"document_title", "section"})
    if from_metadata:
        return frequency >= 1
    if frequency < MIN_FREQUENCY:
        return False
    if len(term) == 2:
        return frequency >= 8 and pmi >= 2.5 and boundary_entropy >= 0.35
    return pmi >= 3.0 and boundary_entropy >= 0.15


def score_candidate(frequency, pmi, boundary_entropy, sources, document_count):
    score = math.log2(frequency + 1) * 5
    score += max(pmi, 0) * 1.5
    score += boundary_entropy * 2
    score += min(document_count, 5) * 1.5
    if "document_title" in sources:
        score += 24
    if "section" in sources:
        score += 16
    if "ascii_domain_term" in sources:
        score += 10
    return score


def remove_redundant_fragments(candidates):
    by_term = {candidate["term"]: candidate for candidate in candidates}
    terms_by_length = sorted(by_term, key=len, reverse=True)
    retained = []

    for term in terms_by_length:
        candidate = by_term[term]
        is_fragment = False
        for longer in retained:
            longer_term = longer["term"]
            if len(longer_term) - len(term) > 2 or term not in longer_term:
                continue
            if longer["frequency"] >= candidate["frequency"] * 0.85:
                is_fragment = True
                break
        if not is_fragment:
            retained.append(candidate)
    return retained


def keep_terms_jieba_splits(candidates):
    retained = []
    for candidate in candidates:
        term = candidate["term"]
        default_tokens = [token for token in jieba.cut(term, HMM=False) if token.strip()]
        if default_tokens == [term]:
            continue
        retained.append(
            {
                **candidate,
                "default_jieba_tokens": default_tokens,
            }
        )
    return retained


def build_candidates(documents, chunks):
    (
        counts,
        left_neighbors,
        right_neighbors,
        ngram_documents,
        total_characters,
    ) = collect_ngram_statistics(chunks)
    metadata, metadata_documents = metadata_phrases(documents, chunks)
    ascii_counts, ascii_documents = collect_ascii_terms(chunks)

    candidate_terms = {
        term
        for term, frequency in counts.items()
        if MIN_TERM_LENGTH <= len(term) <= MAX_TERM_LENGTH
        and frequency >= MIN_FREQUENCY
    }
    candidate_terms.update(
        term for term in metadata if MIN_TERM_LENGTH <= len(term) <= MAX_TERM_LENGTH
    )

    candidates = []
    for term in candidate_terms:
        frequency = counts.get(term, 0)
        pmi = minimum_pmi(term, counts, total_characters) if frequency else 0.0
        left_entropy = entropy(left_neighbors.get(term, Counter()))
        right_entropy = entropy(right_neighbors.get(term, Counter()))
        boundary_entropy = min(left_entropy, right_entropy)
        sources = set(metadata.get(term, set()))
        if frequency >= MIN_FREQUENCY:
            sources.add("corpus_phrase")
        document_names = set(ngram_documents.get(term, set()))
        document_names.update(metadata_documents.get(term, set()))

        if not is_valid_chinese_term(
            term,
            frequency,
            pmi,
            boundary_entropy,
            sources,
        ):
            continue

        candidates.append(
            {
                "term": term,
                "score": round(
                    score_candidate(
                        frequency,
                        pmi,
                        boundary_entropy,
                        sources,
                        len(document_names),
                    ),
                    3,
                ),
                "frequency": frequency,
                "minimum_pmi": round(pmi, 3),
                "boundary_entropy": round(boundary_entropy, 3),
                "document_count": len(document_names),
                "sources": sorted(sources),
                "example_documents": sorted(document_names)[:3],
            }
        )

    for term, frequency in ascii_counts.items():
        if frequency < 3:
            continue
        document_names = ascii_documents[term]
        sources = {"ascii_domain_term"}
        candidates.append(
            {
                "term": term,
                "score": round(
                    score_candidate(
                        frequency,
                        0.0,
                        0.0,
                        sources,
                        len(document_names),
                    ),
                    3,
                ),
                "frequency": frequency,
                "minimum_pmi": None,
                "boundary_entropy": None,
                "document_count": len(document_names),
                "sources": sorted(sources),
                "example_documents": sorted(document_names)[:3],
            }
        )

    unique = {}
    for candidate in candidates:
        current = unique.get(candidate["term"])
        if current is None or candidate["score"] > current["score"]:
            unique[candidate["term"]] = candidate

    # Remove incomplete substrings before the jieba split check. The longer,
    # complete term may already exist in jieba and therefore disappear later.
    useful_candidates = remove_redundant_fragments(unique.values())
    useful_candidates = keep_terms_jieba_splits(useful_candidates)
    ranked = sorted(
        useful_candidates,
        key=lambda item: (
            item["score"],
            item["document_count"],
            item["frequency"],
            len(item["term"]),
        ),
        reverse=True,
    )
    return ranked[:MAX_CANDIDATES], total_characters


def write_outputs(candidates, documents, chunks, total_characters):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_json = OUTPUT_DIR / "candidate_terms.json"
    candidate_txt = OUTPUT_DIR / "candidate_terms.txt"
    review_md = OUTPUT_DIR / "candidate_terms_review.md"
    summary_json = OUTPUT_DIR / "summary.json"

    candidate_json.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    candidate_txt.write_text(
        "\n".join(item["term"] for item in candidates) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Global Jieba Dictionary Candidates\n",
        "只把确认是完整专有词、平台名、行业名、指标名或固定业务概念的条目标记为 `[yes]`。",
        "删除或不选择泛词、完整句子、OCR 乱码、过长标题和只在某个上下文中偶然拼接的片段。",
        "候选生成只使用知识库文档，没有读取评测问题、答案、keywords 或 gold labels。",
        "",
        "只列出会被默认 jieba 拆开的词；`Default split` 展示不加载自定义词典时的切分结果。",
        "",
        "| Keep | Term | Default split | Score | Freq | Docs | PMI | Entropy | Sources |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in candidates:
        pmi = "-" if item["minimum_pmi"] is None else f"{item['minimum_pmi']:.2f}"
        entropy_value = item["boundary_entropy"]
        entropy_text = "-" if entropy_value is None else f"{entropy_value:.2f}"
        lines.append(
            "| [ ] | {term} | {default_split} | {score:.2f} | {frequency} | {documents} | "
            "{pmi} | {entropy} | {sources} |".format(
                term=item["term"].replace("|", "/"),
                default_split=" / ".join(item["default_jieba_tokens"]),
                score=item["score"],
                frequency=item["frequency"],
                documents=item["document_count"],
                pmi=pmi,
                entropy=entropy_text,
                sources=", ".join(item["sources"]),
            )
        )
    review_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "text_characters": total_characters,
        "candidate_count": len(candidates),
        "max_candidates": MAX_CANDIDATES,
        "candidate_sources": [
            "document titles",
            "chunk section metadata",
            "corpus phrase frequency",
            "PMI",
            "left/right boundary entropy",
            "repeated ASCII domain terms",
            "default jieba split check",
        ],
        "evaluation_data_used": False,
        "review_file": str(review_md),
        "candidate_json": str(candidate_json),
        "candidate_txt": str(candidate_txt),
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    documents, chunks = load_corpus()
    if not documents or not chunks:
        raise RuntimeError("No baseline chunks found.")
    candidates, total_characters = build_candidates(documents, chunks)
    summary = write_outputs(candidates, documents, chunks, total_characters)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
