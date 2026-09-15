import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.retrieval.experiment_retrieval_evaluation import (
    QUESTIONS,
    TransformersMeanPoolingEmbedder,
    embed_texts,
    keyword_coverage,
    load_chunks,
    preview_text,
    write_json,
    write_jsonl,
)


CHUNKS_PATH = Path(
    "artifacts/chunking/2025年品牌营销趋势报告/"
    "cleaned_text_structure_semantic_boundary_500_min_150/chunks.jsonl"
)
OUTPUT_DIR = Path("artifacts/chunking/2025年品牌营销趋势报告/hybrid_retrieval_eval")

DENSE_MODEL_NAME = "intfloat/multilingual-e5-base"
DENSE_QUERY_PREFIX = "query: "
DENSE_PASSAGE_PREFIX = "passage: "

DENSE_CANDIDATE_TOP_K = 10
BM25_CANDIDATE_TOP_K = 5
FINAL_TOP_K = 3
RRF_K = 60
JIEBA_CUSTOM_DICT_PATH = Path(
    "artifacts/chunking/2025年品牌营销趋势报告/"
    "bm25_jieba_terms/final_custom_dict.txt"
)


def tokenize_for_bm25(text: str):
    # Dependency-free tokenizer for Chinese/English prototype evaluation.
    # Chinese characters are kept as single-character terms; English/numbers
    # are kept as word-like terms so percentages such as 25% can be matched.
    return re.findall(r"\d+(?:\.\d+)?%?|[A-Za-z]+|[\u4e00-\u9fff]", text.lower())


def tokenize_for_bm25_jieba(text: str, custom_dict_path: Path = JIEBA_CUSTOM_DICT_PATH):
    import jieba

    if custom_dict_path.exists() and not getattr(
        tokenize_for_bm25_jieba,
        "_custom_dict_loaded",
        False,
    ):
        for line in custom_dict_path.read_text(encoding="utf-8").splitlines():
            term = line.strip()
            if term:
                jieba.add_word(term, freq=200000)
        tokenize_for_bm25_jieba._custom_dict_loaded = True

    tokens = []
    pattern = re.compile(r"\d+(?:\.\d+)?%?|[A-Za-z]+|[\u4e00-\u9fff]+")
    for match in pattern.finditer(text.lower()):
        piece = match.group(0)
        if re.fullmatch(r"\d+(?:\.\d+)?%?|[A-Za-z]+", piece):
            tokens.append(piece)
        else:
            tokens.extend(token.strip() for token in jieba.cut(piece) if token.strip())
    return tokens


class SimpleBM25:
    def __init__(self, documents, k1=1.5, b=0.75, tokenizer=tokenize_for_bm25):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.tokenized_docs = [self.tokenizer(document) for document in documents]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths
            else 0
        )
        self.term_frequencies = [Counter(tokens) for tokens in self.tokenized_docs]
        self.doc_frequencies = Counter()

        for term_frequency in self.term_frequencies:
            for term in term_frequency:
                self.doc_frequencies[term] += 1

        self.doc_count = len(documents)

    def idf(self, term):
        doc_frequency = self.doc_frequencies.get(term, 0)
        return math.log(1 + (self.doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5))

    def score(self, query: str):
        query_terms = self.tokenizer(query)
        scores = np.zeros(self.doc_count, dtype=np.float32)

        for index, term_frequency in enumerate(self.term_frequencies):
            doc_length = self.doc_lengths[index]
            score = 0.0

            for term in query_terms:
                frequency = term_frequency.get(term, 0)
                if frequency == 0:
                    continue

                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / self.avg_doc_length
                )
                score += self.idf(term) * frequency * (self.k1 + 1) / denominator

            scores[index] = score

        return scores


def rank_scores(scores, top_k):
    indices = np.argsort(scores)[::-1][:top_k]
    return [(int(index), float(scores[int(index)])) for index in indices]


def rank_map(ranked_items):
    return {chunk_index: rank for rank, (chunk_index, _score) in enumerate(ranked_items, start=1)}


def rrf_rank(dense_ranked, bm25_ranked):
    dense_ranks = rank_map(dense_ranked)
    bm25_ranks = rank_map(bm25_ranked)
    candidate_indices = set(dense_ranks) | set(bm25_ranks)
    ranked = []

    for chunk_index in candidate_indices:
        score = 0.0
        if chunk_index in dense_ranks:
            score += 1 / (RRF_K + dense_ranks[chunk_index])
        if chunk_index in bm25_ranks:
            score += 1 / (RRF_K + bm25_ranks[chunk_index])
        ranked.append((chunk_index, score))

    return sorted(ranked, key=lambda item: item[1], reverse=True)


def build_result_row(question, retrieval_method, chunks, ranked_items):
    top_items = ranked_items[:FINAL_TOP_K]
    top_results = []
    combined_top_text = ""

    for rank, (chunk_index, score) in enumerate(top_items, start=1):
        chunk = chunks[chunk_index]
        combined_top_text += "\n" + chunk["text"]
        top_results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_index": chunk.get("chunk_index"),
                "section": chunk.get("section", ""),
                "token_count": chunk.get("token_count"),
                "block_types": chunk.get("block_types", []),
                "keyword_coverage": keyword_coverage(
                    chunk["text"],
                    question["expected_keywords"],
                ),
                "preview": preview_text(chunk["text"]),
            }
        )

    top3_coverage = keyword_coverage(combined_top_text, question["expected_keywords"])
    expected_section_hit = any(
        question["section"] in result["section"]
        or question["section"] in result["preview"]
        for result in top_results
    )

    return {
        "question_id": question["id"],
        "question": question["question"],
        "question_type": question["type"],
        "retrieval_method": retrieval_method,
        "top1_score": top_results[0]["score"],
        "top1_section": top_results[0]["section"],
        "top1_chunk_index": top_results[0]["chunk_index"],
        "top1_keyword_coverage": top_results[0]["keyword_coverage"],
        "top3_keyword_coverage": top3_coverage,
        "expected_section_hit_top3": expected_section_hit,
        "top_results": top_results,
    }


def evaluate_retrieval_methods(chunks):
    chunk_texts = [chunk["text"] for chunk in chunks]
    dense_model = TransformersMeanPoolingEmbedder(DENSE_MODEL_NAME)
    passage_texts = [DENSE_PASSAGE_PREFIX + text for text in chunk_texts]
    chunk_embeddings = embed_texts(dense_model, passage_texts)
    bm25 = SimpleBM25(chunk_texts)
    rows = []

    for question in QUESTIONS:
        query_embedding = embed_texts(
            dense_model,
            [DENSE_QUERY_PREFIX + question["question"]],
        )[0]
        dense_scores = np.dot(chunk_embeddings, query_embedding)
        bm25_scores = bm25.score(question["question"])

        dense_ranked = rank_scores(dense_scores, DENSE_CANDIDATE_TOP_K)
        bm25_ranked = rank_scores(bm25_scores, BM25_CANDIDATE_TOP_K)
        hybrid_ranked = rrf_rank(dense_ranked, bm25_ranked)

        rows.append(
            build_result_row(
                question,
                f"dense_only_top{DENSE_CANDIDATE_TOP_K}",
                chunks,
                dense_ranked,
            )
        )
        rows.append(
            build_result_row(
                question,
                f"bm25_only_top{BM25_CANDIDATE_TOP_K}",
                chunks,
                bm25_ranked,
            )
        )
        rows.append(
            build_result_row(
                question,
                f"dense_top{DENSE_CANDIDATE_TOP_K}_bm25_top{BM25_CANDIDATE_TOP_K}_rrf",
                chunks,
                hybrid_ranked,
            )
        )

    return rows


def write_review(path: Path, rows):
    lines = [
        "# Hybrid Retrieval Evaluation\n",
        "- Chunking: `structure-aware + semantic sentence boundary 500/150`",
        f"- Dense embedding model: `{DENSE_MODEL_NAME}`",
        f"- Dense candidates: `top {DENSE_CANDIDATE_TOP_K}`",
        f"- BM25 candidates: `top {BM25_CANDIDATE_TOP_K}`",
        f"- Final returned evidence: `top {FINAL_TOP_K}`",
        f"- Hybrid ranking: `RRF`, k=`{RRF_K}`",
        "- Hybrid candidate pool is the deduplicated union of dense top10 and BM25 top5.",
        "\n## Evaluation Questions\n",
    ]

    for question in QUESTIONS:
        lines.extend(
            [
                f"### {question['id']} {question['question']}",
                f"- Type: `{question['type']}`",
                f"- Expected section: `{question['section']}`",
                f"- Expected keywords: `{', '.join(question['expected_keywords'])}`",
                f"- Expected answer: {question['expected_answer']}",
                "",
            ]
        )

    lines.extend(
        [
            "\n## Summary Table\n",
            "| Question | Retrieval | Top-1 section | Top-1 keywords | Top-3 keywords | Section hit top-3 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in rows:
        top1 = row["top1_keyword_coverage"]
        top3 = row["top3_keyword_coverage"]
        lines.append(
            "| {qid} | {method} | {section} | {top1_found}/{top1_total} | "
            "{top3_found}/{top3_total} | {section_hit} |".format(
                qid=row["question_id"],
                method=row["retrieval_method"],
                section=row["top1_section"].replace("|", "/") or "-",
                top1_found=len(top1["found"]),
                top1_total=len(top1["found"]) + len(top1["missing"]),
                top3_found=len(top3["found"]),
                top3_total=len(top3["found"]) + len(top3["missing"]),
                section_hit="yes" if row["expected_section_hit_top3"] else "no",
            )
        )

    lines.append("\n## Question Details\n")
    for question in QUESTIONS:
        lines.append(f"### {question['id']} {question['question']}\n")
        for row in [item for item in rows if item["question_id"] == question["id"]]:
            lines.append(f"#### {row['retrieval_method']}\n")
            for result in row["top_results"]:
                coverage = result["keyword_coverage"]
                lines.append(
                    "- Rank {rank}: score={score:.4f}, chunk={chunk}, "
                    "section=`{section}`, keywords={found}/{total}".format(
                        rank=result["rank"],
                        score=result["score"],
                        chunk=result["chunk_index"],
                        section=result["section"],
                        found=len(coverage["found"]),
                        total=len(coverage["found"]) + len(coverage["missing"]),
                    )
                )
                lines.append(f"  Preview: {result['preview']}")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    chunks = load_chunks(CHUNKS_PATH)
    rows = evaluate_retrieval_methods(chunks)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "questions.json", QUESTIONS)
    write_jsonl(OUTPUT_DIR / "hybrid_retrieval_results.jsonl", rows)
    write_review(OUTPUT_DIR / "review.md", rows)

    print(
        json.dumps(
            {
                "chunking": "structure_semantic_boundary_500_150",
                "dense_model": DENSE_MODEL_NAME,
                "questions": len(QUESTIONS),
                "retrieval_methods": 3,
                "dense_top_k": DENSE_CANDIDATE_TOP_K,
                "bm25_top_k": BM25_CANDIDATE_TOP_K,
                "final_top_k": FINAL_TOP_K,
                "results": len(rows),
                "outputs": {
                    "questions": str(OUTPUT_DIR / "questions.json"),
                    "results": str(OUTPUT_DIR / "hybrid_retrieval_results.jsonl"),
                    "review": str(OUTPUT_DIR / "review.md"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
