import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.retrieval.experiment_hybrid_retrieval import (
    BM25_CANDIDATE_TOP_K,
    CHUNKS_PATH,
    DENSE_CANDIDATE_TOP_K,
    DENSE_MODEL_NAME,
    DENSE_PASSAGE_PREFIX,
    DENSE_QUERY_PREFIX,
    FINAL_TOP_K,
    RRF_K,
    SimpleBM25,
    build_result_row,
    rank_scores,
    rrf_rank,
    tokenize_for_bm25,
    tokenize_for_bm25_jieba,
)
from experiments.retrieval.experiment_llm_judge_precision import (
    MODEL_NAME as JUDGE_MODEL_NAME,
    SLEEP_SECONDS,
    judge_one,
    load_deepseek_config,
)
from experiments.retrieval.experiment_retrieval_evaluation import (
    QUESTIONS,
    TransformersMeanPoolingEmbedder,
    embed_texts,
    load_chunks,
    write_json,
    write_jsonl,
)


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    BASE_DIR
    / "artifacts/chunking/2025年品牌营销趋势报告/bm25_tokenizer_eval"
)


BM25_TOKENIZERS = {
    "rule_char_tokenizer": tokenize_for_bm25,
    "jieba_custom_dict_tokenizer": tokenize_for_bm25_jieba,
}


def top3_keyword_ratio(row):
    coverage = row["top3_keyword_coverage"]
    total = len(coverage["found"]) + len(coverage["missing"])
    if total == 0:
        return 0.0
    return len(coverage["found"]) / total


def evaluate_tokenizer(tokenizer_name, tokenizer, chunks, chunk_embeddings, dense_model):
    chunk_texts = [chunk["text"] for chunk in chunks]
    bm25_start = time.perf_counter()
    bm25 = SimpleBM25(chunk_texts, tokenizer=tokenizer)
    bm25_build_seconds = time.perf_counter() - bm25_start
    rows = []

    for question in QUESTIONS:
        started = time.perf_counter()
        query_embedding = embed_texts(
            dense_model,
            [DENSE_QUERY_PREFIX + question["question"]],
        )[0]
        dense_scores = np.dot(chunk_embeddings, query_embedding)
        bm25_scores = bm25.score(question["question"])
        dense_ranked = rank_scores(dense_scores, DENSE_CANDIDATE_TOP_K)
        bm25_ranked = rank_scores(bm25_scores, BM25_CANDIDATE_TOP_K)
        hybrid_ranked = rrf_rank(dense_ranked, bm25_ranked)
        elapsed = time.perf_counter() - started

        for method, ranked in [
            (f"{tokenizer_name}_bm25_only", bm25_ranked),
            (f"{tokenizer_name}_hybrid_rrf", hybrid_ranked),
        ]:
            row = build_result_row(
                question=question,
                retrieval_method=method,
                chunks=chunks,
                ranked_items=ranked[:FINAL_TOP_K],
            )
            row["bm25_tokenizer"] = tokenizer_name
            row["top3_keyword_ratio"] = round(top3_keyword_ratio(row), 4)
            row["bm25_build_seconds"] = round(bm25_build_seconds, 4)
            row["retrieval_seconds"] = round(elapsed, 4)
            row["candidate_pool_size"] = len(hybrid_ranked)
            rows.append(row)

    return rows


def build_unique_judge_rows(rows, chunks):
    unique = {}
    for row in rows:
        for result in row["top_results"]:
            key = (
                row["retrieval_method"],
                row["question_id"],
                int(result["chunk_index"]),
            )
            if key in unique:
                continue
            chunk = chunks[int(result["chunk_index"])]
            unique[key] = {
                "retrieval_method": row["retrieval_method"],
                "question_id": row["question_id"],
                "question": row["question"],
                "rank": result["rank"],
                "chunk_index": int(result["chunk_index"]),
                "section": chunk.get("section", ""),
                "evidence": chunk["text"],
            }
    return list(unique.values())


def add_judge_labels(rows, judged_rows):
    labels = {
        (
            row["retrieval_method"],
            row["question_id"],
            int(row["chunk_index"]),
        ): row
        for row in judged_rows
    }
    enriched = []
    for row in rows:
        top_labels = []
        for result in row["top_results"]:
            key = (
                row["retrieval_method"],
                row["question_id"],
                int(result["chunk_index"]),
            )
            judged = labels[key]
            result["llm_label"] = judged["llm_label"]
            result["llm_reason"] = judged["llm_reason"]
            top_labels.append(judged["llm_label"])
        row["llm_labels_top3"] = top_labels
        row["llm_precision_at_3"] = round(
            sum(1 for label in top_labels if label in (1, 2)) / len(top_labels),
            4,
        )
        row["top1_direct_label2"] = top_labels[0] == 2
        row["top1_relevant_label1_or_2"] = top_labels[0] in (1, 2)
        enriched.append(row)
    return enriched


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["retrieval_method"]].append(row)

    summaries = []
    for method, method_rows in sorted(grouped.items()):
        recall = sum(row["top3_keyword_ratio"] for row in method_rows) / len(method_rows)
        hit_rate = (
            sum(1 for row in method_rows if row["expected_section_hit_top3"])
            / len(method_rows)
        )
        precision = (
            sum(row["llm_precision_at_3"] for row in method_rows) / len(method_rows)
        )
        top1_direct = (
            sum(1 for row in method_rows if row["top1_direct_label2"])
            / len(method_rows)
        )
        top1_relevant = (
            sum(1 for row in method_rows if row["top1_relevant_label1_or_2"])
            / len(method_rows)
        )
        avg_seconds = (
            sum(row["retrieval_seconds"] for row in method_rows) / len(method_rows)
        )
        avg_pool_size = (
            sum(row["candidate_pool_size"] for row in method_rows)
            / len(method_rows)
        )
        combined_score = (
            recall * 0.50
            + hit_rate * 0.20
            + precision * 0.25
            - (avg_pool_size / 40) * 0.05
        )
        summaries.append(
            {
                "retrieval_method": method,
                "bm25_tokenizer": method_rows[0]["bm25_tokenizer"],
                "recall_at_3": round(recall, 4),
                "hit_rate_at_3": round(hit_rate, 4),
                "llm_precision_at_3": round(precision, 4),
                "top1_direct_accuracy_label2": round(top1_direct, 4),
                "top1_relevance_accuracy_label1_or_2": round(top1_relevant, 4),
                "avg_candidate_pool_size": round(avg_pool_size, 2),
                "avg_retrieval_seconds": round(avg_seconds, 4),
                "combined_score": round(combined_score, 4),
            }
        )

    return sorted(summaries, key=lambda item: item["combined_score"], reverse=True)


def write_review(path, summaries):
    lines = [
        "# BM25 Tokenizer Comparison\n",
        "- Chunking: `Structure-aware + semantic sentence boundary 500/150`",
        f"- Dense: `{DENSE_MODEL_NAME}`, top{DENSE_CANDIDATE_TOP_K}",
        f"- BM25 top{BM25_CANDIDATE_TOP_K}",
        f"- Hybrid: `RRF_K={RRF_K}`, Final top{FINAL_TOP_K}",
        f"- LLM-as-judge: `{JUDGE_MODEL_NAME}`",
        "- Precision@3: judge label `1` or `2` counts as relevant.",
        "",
        "## Summary\n",
        "| Method | Tokenizer | Recall@3 | HitRate@3 | LLM Precision@3 | Top-1 direct | Top-1 relevant | Avg seconds | Combined |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {method} | {tokenizer} | {recall:.3f} | {hit:.3f} | {precision:.3f} | "
            "{top1_direct:.3f} | {top1_relevant:.3f} | {seconds:.3f} | {score:.3f} |".format(
                method=item["retrieval_method"],
                tokenizer=item["bm25_tokenizer"],
                recall=item["recall_at_3"],
                hit=item["hit_rate_at_3"],
                precision=item["llm_precision_at_3"],
                top1_direct=item["top1_direct_accuracy_label2"],
                top1_relevant=item["top1_relevance_accuracy_label1_or_2"],
                seconds=item["avg_retrieval_seconds"],
                score=item["combined_score"],
            )
        )

    lines.extend(
        [
            "",
            "## Current Conclusion\n",
            f"- Best method in this run: `{summaries[0]['retrieval_method']}`.",
            "- If jieba improves precision without lowering recall, it can replace the rule tokenizer.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_visualization(path, summaries):
    rows = ""
    for item in summaries:
        rows += (
            "<tr>"
            f"<td>{item['retrieval_method']}</td>"
            f"<td>{item['bm25_tokenizer']}</td>"
            f"<td>{item['recall_at_3']:.3f}</td>"
            f"<td>{item['hit_rate_at_3']:.3f}</td>"
            f"<td>{item['llm_precision_at_3']:.3f}</td>"
            f"<td>{item['top1_direct_accuracy_label2']:.3f}</td>"
            f"<td>{item['avg_retrieval_seconds']:.3f}</td>"
            f"<td>{item['combined_score']:.3f}</td>"
            "</tr>\n"
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>BM25 Tokenizer Comparison</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #1f2937;
      background: #f8fafc;
    }}
    .note {{
      max-width: 980px;
      line-height: 1.6;
      background: white;
      border: 1px solid #e5e7eb;
      padding: 16px 18px;
      border-radius: 8px;
      margin-bottom: 22px;
    }}
    table {{
      border-collapse: collapse;
      background: white;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
      width: 100%;
      max-width: 1180px;
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 12px 14px;
      text-align: center;
    }}
    th {{ background: #eef2f7; }}
    td:first-child, th:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>BM25 Tokenizer Comparison</h1>
  <div class="note">
    <p><strong>Rule tokenizer:</strong> 中文按单字切分，英文和数字整体保留。</p>
    <p><strong>Jieba tokenizer:</strong> 中文按词切分，并加载自定义营销领域词典。</p>
    <p><strong>Precision@3:</strong> LLM-as-judge 标注为 1 或 2 都算 relevant。</p>
  </div>
  <table>
    <tr>
      <th>Method</th>
      <th>Tokenizer</th>
      <th>Recall@3</th>
      <th>HitRate@3</th>
      <th>LLM Precision@3</th>
      <th>Top-1 direct</th>
      <th>Avg seconds</th>
      <th>Combined</th>
    </tr>
    {rows}
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main():
    chunks = load_chunks(CHUNKS_PATH)
    chunk_texts = [chunk["text"] for chunk in chunks]
    dense_model = TransformersMeanPoolingEmbedder(DENSE_MODEL_NAME)
    chunk_embeddings = embed_texts(
        dense_model,
        [DENSE_PASSAGE_PREFIX + text for text in chunk_texts],
    )

    all_rows = []
    for tokenizer_name, tokenizer in BM25_TOKENIZERS.items():
        print(f"[BM25] evaluating {tokenizer_name}")
        all_rows.extend(
            evaluate_tokenizer(
                tokenizer_name,
                tokenizer,
                chunks,
                chunk_embeddings,
                dense_model,
            )
        )

    judge_rows = build_unique_judge_rows(all_rows, chunks)
    api_key, chat_url = load_deepseek_config()
    judged_rows = []
    for index, row in enumerate(judge_rows, start=1):
        print(
            f"[Judge] {index}/{len(judge_rows)} "
            f"{row['retrieval_method']} {row['question_id']} chunk {row['chunk_index']}"
        )
        judge_result = judge_one(api_key, chat_url, row)
        judged_rows.append({**row, **judge_result})
        time.sleep(SLEEP_SECONDS)

    enriched_rows = add_judge_labels(all_rows, judged_rows)
    summaries = summarize(enriched_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "bm25_tokenizer_results.jsonl", enriched_rows)
    write_jsonl(OUTPUT_DIR / "bm25_tokenizer_judge_results.jsonl", judged_rows)
    write_json(OUTPUT_DIR / "bm25_tokenizer_summary.json", summaries)
    write_review(OUTPUT_DIR / "review.md", summaries)
    write_visualization(OUTPUT_DIR / "visualization.html", summaries)

    print(
        json.dumps(
            {
                "summary": summaries,
                "outputs": {
                    "review": str(OUTPUT_DIR / "review.md"),
                    "visualization": str(OUTPUT_DIR / "visualization.html"),
                    "summary": str(OUTPUT_DIR / "bm25_tokenizer_summary.json"),
                    "results": str(OUTPUT_DIR / "bm25_tokenizer_results.jsonl"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
