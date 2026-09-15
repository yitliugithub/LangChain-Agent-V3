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
    / "artifacts/chunking/2025年品牌营销趋势报告/reranker_eval"
)

RERANKER_MODEL_NAMES = [
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3",
]
MAX_PAIR_LENGTH = 512
BATCH_SIZE = 4


def top3_keyword_ratio(row):
    coverage = row["top3_keyword_coverage"]
    total = len(coverage["found"]) + len(coverage["missing"])
    if total == 0:
        return 0.0
    return len(coverage["found"]) / total


class CrossEncoderReranker:
    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def score_pairs(self, question, candidate_texts):
        scores = []
        with self.torch.no_grad():
            for start in range(0, len(candidate_texts), BATCH_SIZE):
                batch_texts = candidate_texts[start:start + BATCH_SIZE]
                encoded = self.tokenizer(
                    [question] * len(batch_texts),
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_PAIR_LENGTH,
                    return_tensors="pt",
                )
                output = self.model(**encoded)
                logits = output.logits
                if logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1)
                else:
                    batch_scores = logits[:, -1]
                scores.extend(float(score) for score in batch_scores.cpu().tolist())
        return scores


def build_hybrid_candidates(chunks):
    chunk_texts = [chunk["text"] for chunk in chunks]
    dense_model = TransformersMeanPoolingEmbedder(DENSE_MODEL_NAME)
    passage_texts = [DENSE_PASSAGE_PREFIX + text for text in chunk_texts]
    dense_start = time.perf_counter()
    chunk_embeddings = embed_texts(dense_model, passage_texts)
    dense_index_seconds = time.perf_counter() - dense_start
    bm25 = SimpleBM25(chunk_texts)

    rows = []
    candidate_rows = []
    retrieval_times = []

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
        retrieval_times.append(elapsed)

        baseline_row = build_result_row(
            question=question,
            retrieval_method="baseline_hybrid_rrf",
            chunks=chunks,
            ranked_items=hybrid_ranked[:FINAL_TOP_K],
        )
        baseline_row["top3_keyword_ratio"] = round(top3_keyword_ratio(baseline_row), 4)
        baseline_row["candidate_pool_size"] = len(hybrid_ranked)
        baseline_row["retrieval_seconds"] = round(elapsed, 4)
        rows.append(baseline_row)

        candidate_rows.append(
            {
                "question": question,
                "hybrid_ranked": hybrid_ranked,
                "retrieval_seconds": elapsed,
            }
        )

    return {
        "rows": rows,
        "candidate_rows": candidate_rows,
        "dense_index_seconds": dense_index_seconds,
        "avg_retrieval_seconds": sum(retrieval_times) / len(retrieval_times),
    }


def method_name_for_reranker(model_name):
    safe_name = model_name.split("/")[-1].replace("-", "_")
    return f"hybrid_plus_{safe_name}"


def rerank_candidates(chunks, candidate_rows, reranker_model_name):
    reranker = CrossEncoderReranker(reranker_model_name)
    rows = []
    rerank_times = []
    method_name = method_name_for_reranker(reranker_model_name)

    for candidate_row in candidate_rows:
        question = candidate_row["question"]
        hybrid_ranked = candidate_row["hybrid_ranked"]
        candidate_indices = [index for index, _score in hybrid_ranked]
        candidate_texts = [chunks[index]["text"] for index in candidate_indices]

        started = time.perf_counter()
        reranker_scores = reranker.score_pairs(question["question"], candidate_texts)
        reranked = sorted(
            zip(candidate_indices, reranker_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        rerank_seconds = time.perf_counter() - started
        rerank_times.append(rerank_seconds)

        row = build_result_row(
            question=question,
            retrieval_method=method_name,
            chunks=chunks,
            ranked_items=reranked[:FINAL_TOP_K],
        )
        row["reranker_model"] = reranker_model_name
        row["top3_keyword_ratio"] = round(top3_keyword_ratio(row), 4)
        row["candidate_pool_size"] = len(candidate_indices)
        row["retrieval_seconds"] = round(candidate_row["retrieval_seconds"], 4)
        row["rerank_seconds"] = round(rerank_seconds, 4)
        row["total_seconds"] = round(
            candidate_row["retrieval_seconds"] + rerank_seconds,
            4,
        )
        rows.append(row)

    return {
        "rows": rows,
        "avg_rerank_seconds": sum(rerank_times) / len(rerank_times),
        "reranker_model": reranker_model_name,
    }


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
    for method, method_rows in grouped.items():
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
        avg_total_seconds = (
            sum(
                row.get("total_seconds", row["retrieval_seconds"])
                for row in method_rows
            )
            / len(method_rows)
        )
        avg_rerank_seconds = (
            sum(row.get("rerank_seconds", 0) for row in method_rows)
            / len(method_rows)
        )
        avg_pool_size = (
            sum(row["candidate_pool_size"] for row in method_rows) / len(method_rows)
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
                "recall_at_3": round(recall, 4),
                "hit_rate_at_3": round(hit_rate, 4),
                "llm_precision_at_3": round(precision, 4),
                "top1_direct_accuracy_label2": round(top1_direct, 4),
                "top1_relevance_accuracy_label1_or_2": round(top1_relevant, 4),
                "avg_candidate_pool_size": round(avg_pool_size, 2),
                "avg_rerank_seconds": round(avg_rerank_seconds, 4),
                "avg_total_seconds": round(avg_total_seconds, 4),
                "combined_score": round(combined_score, 4),
            }
        )
    return sorted(summaries, key=lambda item: item["combined_score"], reverse=True)


def write_review(path, summaries):
    lines = [
        "# Reranker Evaluation\n",
        "- Chunking: `Structure-aware + semantic sentence boundary 500/150`",
        f"- Baseline: `Dense top{DENSE_CANDIDATE_TOP_K} + BM25 top{BM25_CANDIDATE_TOP_K} + RRF_K={RRF_K} + Final top{FINAL_TOP_K}`",
        f"- Rerankers: `{', '.join(RERANKER_MODEL_NAMES)}` over the deduplicated hybrid candidate pool",
        f"- LLM-as-judge: `{JUDGE_MODEL_NAME}`",
        "- Precision@3: judge label `1` or `2` counts as relevant.",
        "- Top-1 direct accuracy: rank-1 judge label is `2`.",
        "",
        "## Summary\n",
        "| Method | Recall@3 | HitRate@3 | LLM Precision@3 | Top-1 direct | Top-1 relevant | Avg total seconds | Avg rerank seconds | Combined |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {method} | {recall:.3f} | {hit:.3f} | {precision:.3f} | "
            "{top1_direct:.3f} | {top1_relevant:.3f} | {total:.3f} | "
            "{rerank:.3f} | {score:.3f} |".format(
                method=item["retrieval_method"],
                recall=item["recall_at_3"],
                hit=item["hit_rate_at_3"],
                precision=item["llm_precision_at_3"],
                top1_direct=item["top1_direct_accuracy_label2"],
                top1_relevant=item["top1_relevance_accuracy_label1_or_2"],
                total=item["avg_total_seconds"],
                rerank=item["avg_rerank_seconds"],
                score=item["combined_score"],
            )
        )

    best = summaries[0]
    lines.extend(
        [
            "",
            "## Current Conclusion\n",
            f"- Better overall method in this run: `{best['retrieval_method']}`.",
            "- Reranker should be kept only if it improves relevance enough to justify extra latency.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_visualization(path, summaries):
    rows = ""
    for item in summaries:
        rows += (
            "<tr>"
            f"<td>{item['retrieval_method']}</td>"
            f"<td>{item['recall_at_3']:.3f}</td>"
            f"<td>{item['hit_rate_at_3']:.3f}</td>"
            f"<td>{item['llm_precision_at_3']:.3f}</td>"
            f"<td>{item['top1_direct_accuracy_label2']:.3f}</td>"
            f"<td>{item['top1_relevance_accuracy_label1_or_2']:.3f}</td>"
            f"<td>{item['avg_total_seconds']:.3f}</td>"
            f"<td>{item['avg_rerank_seconds']:.3f}</td>"
            f"<td>{item['combined_score']:.3f}</td>"
            "</tr>\n"
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Reranker Evaluation</title>
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
  <h1>Reranker Evaluation</h1>
  <div class="note">
    <p><strong>Baseline:</strong> Dense top{DENSE_CANDIDATE_TOP_K} + BM25 top{BM25_CANDIDATE_TOP_K} + RRF_K={RRF_K} + Final top{FINAL_TOP_K}。</p>
    <p><strong>Reranker:</strong> 对同一个 hybrid candidate pool 分别使用 {', '.join(RERANKER_MODEL_NAMES)} 重新排序，再取 Final top{FINAL_TOP_K}。</p>
    <p><strong>Precision@3:</strong> LLM-as-judge 标注为 1 或 2 都算 relevant。</p>
  </div>
  <table>
    <tr>
      <th>Method</th>
      <th>Recall@3</th>
      <th>HitRate@3</th>
      <th>LLM Precision@3</th>
      <th>Top-1 direct</th>
      <th>Top-1 relevant</th>
      <th>Avg total seconds</th>
      <th>Avg rerank seconds</th>
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
    candidate_data = build_hybrid_candidates(chunks)
    baseline_rows = candidate_data["rows"]
    reranker_runs = []
    reranker_rows = []
    for reranker_model_name in RERANKER_MODEL_NAMES:
        reranker_data = rerank_candidates(
            chunks,
            candidate_data["candidate_rows"],
            reranker_model_name,
        )
        reranker_runs.append(reranker_data)
        reranker_rows.extend(reranker_data["rows"])
    all_rows = baseline_rows + reranker_rows

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
    write_jsonl(OUTPUT_DIR / "reranker_results.jsonl", enriched_rows)
    write_jsonl(OUTPUT_DIR / "reranker_judge_results.jsonl", judged_rows)
    write_json(
        OUTPUT_DIR / "reranker_summary.json",
        {
            "dense_index_seconds": round(candidate_data["dense_index_seconds"], 4),
            "avg_baseline_retrieval_seconds": round(
                candidate_data["avg_retrieval_seconds"],
                4,
            ),
            "reranker_runs": [
                {
                    "reranker_model": run["reranker_model"],
                    "avg_rerank_seconds": round(run["avg_rerank_seconds"], 4),
                }
                for run in reranker_runs
            ],
            "summary": summaries,
        },
    )
    write_review(OUTPUT_DIR / "review.md", summaries)
    write_visualization(OUTPUT_DIR / "visualization.html", summaries)

    print(
        json.dumps(
            {
                "judge_items": len(judged_rows),
                "dense_index_seconds": round(candidate_data["dense_index_seconds"], 4),
                "reranker_models": RERANKER_MODEL_NAMES,
                "summary": summaries,
                "outputs": {
                    "review": str(OUTPUT_DIR / "review.md"),
                    "visualization": str(OUTPUT_DIR / "visualization.html"),
                    "summary": str(OUTPUT_DIR / "reranker_summary.json"),
                    "results": str(OUTPUT_DIR / "reranker_results.jsonl"),
                    "judge_results": str(OUTPUT_DIR / "reranker_judge_results.jsonl"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
