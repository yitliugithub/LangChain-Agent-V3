import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.retrieval.experiment_hybrid_retrieval import (
    DENSE_MODEL_NAME,
    DENSE_PASSAGE_PREFIX,
    DENSE_QUERY_PREFIX,
    RRF_K,
    SimpleBM25,
    build_result_row,
    rank_scores,
    rrf_rank,
)
from experiments.retrieval.experiment_llm_judge_precision import (
    CHUNKS_PATH as DEFAULT_A_CHUNKS_PATH,
    MODEL_NAME as JUDGE_MODEL_NAME,
    SLEEP_SECONDS,
    attach_full_evidence,
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
A_CHUNKS_PATH = DEFAULT_A_CHUNKS_PATH
B_CHUNKS_PATH = (
    BASE_DIR
    / "artifacts/chunking/2025年品牌营销趋势报告/"
    "cleaned_text_structure_semantic_boundary_500_min_150/chunks.jsonl"
)
A_SUMMARY_PATH = A_CHUNKS_PATH.parent / "summary.json"
B_SUMMARY_PATH = B_CHUNKS_PATH.parent / "summary.json"
OUTPUT_DIR = (
    BASE_DIR
    / "artifacts/chunking/2025年品牌营销趋势报告/semantic_ab_eval"
)

DENSE_CANDIDATE_TOP_K = 10
BM25_CANDIDATE_TOP_K = 5
FINAL_TOP_K = 3


def top3_keyword_ratio(row):
    coverage = row["top3_keyword_coverage"]
    total = len(coverage["found"]) + len(coverage["missing"])
    if total == 0:
        return 0.0
    return len(coverage["found"]) / total


def load_summary(path):
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_boundary_metrics(chunks, max_tokens):
    same_section_transitions = 0
    near_limit_same_section_transitions = 0

    for previous, current in zip(chunks, chunks[1:]):
        if previous["section"] != current["section"]:
            continue
        same_section_transitions += 1
        if previous["token_count"] >= max_tokens * 0.85:
            near_limit_same_section_transitions += 1

    return {
        "same_section_chunk_transitions": same_section_transitions,
        "near_limit_same_section_transitions": near_limit_same_section_transitions,
    }


def retrieve_for_method(method_name, chunks, dense_model):
    chunk_texts = [chunk["text"] for chunk in chunks]
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
        row = build_result_row(
            question=question,
            retrieval_method=f"{method_name}_hybrid_rerank_none",
            chunks=chunks,
            ranked_items=hybrid_ranked[:FINAL_TOP_K],
        )
        row["chunking_method"] = method_name
        row["top3_keyword_ratio"] = round(top3_keyword_ratio(row), 4)
        row["candidate_pool_size"] = len(
            {index for index, _score in dense_ranked}
            | {index for index, _score in bm25_ranked}
        )
        rows.append(row)

    return rows


def full_chunk_lookup(method_name, chunks):
    return {
        (method_name, int(chunk["chunk_index"])): {
            "text": chunk["text"],
            "section": chunk.get("section", ""),
        }
        for chunk in chunks
    }


def build_unique_judge_rows(rows, chunk_lookup):
    unique = {}
    for row in rows:
        method_name = row["chunking_method"]
        for result in row["top_results"]:
            key = (
                method_name,
                row["question_id"],
                int(result["chunk_index"]),
            )
            if key in unique:
                continue
            chunk = chunk_lookup[(method_name, int(result["chunk_index"]))]
            unique[key] = {
                "chunking_method": method_name,
                "question_id": row["question_id"],
                "question": row["question"],
                "rank": result["rank"],
                "chunk_index": int(result["chunk_index"]),
                "section": chunk["section"],
                "evidence": chunk["text"],
            }
    return list(unique.values())


def add_llm_precision(rows, judged_rows):
    labels = {
        (
            row["chunking_method"],
            row["question_id"],
            int(row["chunk_index"]),
        ): row
        for row in judged_rows
    }
    enriched = []
    for row in rows:
        method_name = row["chunking_method"]
        top_labels = []
        for result in row["top_results"]:
            key = (
                method_name,
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
        enriched.append(row)
    return enriched


def summarize(rows, chunk_summaries, boundary_summaries):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["chunking_method"]].append(row)

    summaries = []
    for method_name, method_rows in grouped.items():
        recall = sum(row["top3_keyword_ratio"] for row in method_rows) / len(method_rows)
        hit_rate = (
            sum(1 for row in method_rows if row["expected_section_hit_top3"])
            / len(method_rows)
        )
        precision = (
            sum(row["llm_precision_at_3"] for row in method_rows) / len(method_rows)
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

        chunk_summary = chunk_summaries[method_name]
        boundary_summary = boundary_summaries[method_name]
        summaries.append(
            {
                "chunking_method": method_name,
                "chunk_count": chunk_summary["chunk_count"],
                "avg_chunk_tokens": chunk_summary["avg_chunk_tokens"],
                "table_chunk_count": chunk_summary["table_chunk_count"],
                "semantic_boundary_count": chunk_summary.get(
                    "semantic_boundary_count",
                    0,
                ),
                **boundary_summary,
                "recall_at_3": round(recall, 4),
                "hit_rate_at_3": round(hit_rate, 4),
                "llm_precision_at_3": round(precision, 4),
                "avg_candidate_pool_size": round(avg_pool_size, 2),
                "combined_score": round(combined_score, 4),
            }
        )

    return sorted(summaries, key=lambda item: item["combined_score"], reverse=True)


def write_review(path, summaries):
    lines = [
        "# Semantic Boundary A/B Evaluation\n",
        "- Experiment A: `structure-aware token chunking 500/150`",
        "- Experiment B: `structure-aware + semantic sentence boundary 500/150`",
        f"- Retrieval: Dense top{DENSE_CANDIDATE_TOP_K} + BM25 top{BM25_CANDIDATE_TOP_K} + RRF_K={RRF_K} + Final top{FINAL_TOP_K}",
        f"- LLM-as-judge model: `{JUDGE_MODEL_NAME}`",
        "- Precision@3 definition: label `1` or `2` counts as relevant.",
        "",
        "## Summary\n",
        "| Method | Chunks | Avg tokens | Semantic boundaries | Near-limit same-section transitions | Recall@3 | HitRate@3 | LLM Precision@3 | Combined |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {method} | {chunks} | {avg:.2f} | {semantic} | {near_limit} | "
            "{recall:.3f} | {hit:.3f} | {precision:.3f} | {score:.3f} |".format(
                method=item["chunking_method"],
                chunks=item["chunk_count"],
                avg=item["avg_chunk_tokens"],
                semantic=item["semantic_boundary_count"],
                near_limit=item["near_limit_same_section_transitions"],
                recall=item["recall_at_3"],
                hit=item["hit_rate_at_3"],
                precision=item["llm_precision_at_3"],
                score=item["combined_score"],
            )
        )

    best = summaries[0]
    lines.extend(
        [
            "",
            "## Current Conclusion\n",
            f"- Better overall method in this experiment: `{best['chunking_method']}`.",
            "- Near-limit same-section transitions are used as a proxy for hard token-limit cuts inside the same section.",
            "- This is a proxy metric, so final choice should still consider the retrieved evidence examples.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_visualization(path, summaries):
    rows = ""
    for item in summaries:
        rows += (
            "<tr>"
            f"<td>{item['chunking_method']}</td>"
            f"<td>{item['chunk_count']}</td>"
            f"<td>{item['avg_chunk_tokens']:.2f}</td>"
            f"<td>{item['semantic_boundary_count']}</td>"
            f"<td>{item['near_limit_same_section_transitions']}</td>"
            f"<td>{item['recall_at_3']:.3f}</td>"
            f"<td>{item['hit_rate_at_3']:.3f}</td>"
            f"<td>{item['llm_precision_at_3']:.3f}</td>"
            f"<td>{item['combined_score']:.3f}</td>"
            "</tr>\n"
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Semantic Boundary A/B Evaluation</title>
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
  <h1>Semantic Boundary A/B Evaluation</h1>
  <div class="note">
    <p><strong>A:</strong> structure-aware token chunking 500/150。</p>
    <p><strong>B:</strong> structure-aware + semantic sentence boundary。B 在同一个 section 内先计算相邻句子 embedding distance，并优先在语义距离较大的地方切分。</p>
    <p><strong>检索固定：</strong>Dense top10 + BM25 top5 + RRF_K={RRF_K} + Final top3。</p>
  </div>
  <table>
    <tr>
      <th>Method</th>
      <th>Chunk count</th>
      <th>Avg chunk tokens</th>
      <th>Semantic boundaries</th>
      <th>Near-limit same-section transitions</th>
      <th>Recall@3</th>
      <th>HitRate@3</th>
      <th>LLM Precision@3</th>
      <th>Combined score</th>
    </tr>
    {rows}
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main():
    a_chunks = load_chunks(A_CHUNKS_PATH)
    b_chunks = load_chunks(B_CHUNKS_PATH)
    chunk_summaries = {
        "A_structure_aware_500_150": load_summary(A_SUMMARY_PATH),
        "B_structure_semantic_boundary_500_150": load_summary(B_SUMMARY_PATH),
    }
    boundary_summaries = {
        "A_structure_aware_500_150": chunk_boundary_metrics(a_chunks, 500),
        "B_structure_semantic_boundary_500_150": chunk_boundary_metrics(b_chunks, 500),
    }

    dense_model = TransformersMeanPoolingEmbedder(DENSE_MODEL_NAME)
    rows = []
    rows.extend(
        retrieve_for_method("A_structure_aware_500_150", a_chunks, dense_model)
    )
    rows.extend(
        retrieve_for_method(
            "B_structure_semantic_boundary_500_150",
            b_chunks,
            dense_model,
        )
    )

    chunk_lookup = {}
    chunk_lookup.update(full_chunk_lookup("A_structure_aware_500_150", a_chunks))
    chunk_lookup.update(
        full_chunk_lookup("B_structure_semantic_boundary_500_150", b_chunks)
    )
    judge_rows = build_unique_judge_rows(rows, chunk_lookup)

    api_key, chat_url = load_deepseek_config()
    judged_rows = []
    for index, row in enumerate(judge_rows, start=1):
        print(
            f"[Judge] {index}/{len(judge_rows)} "
            f"{row['chunking_method']} {row['question_id']} chunk {row['chunk_index']}"
        )
        judge_result = judge_one(api_key, chat_url, row)
        judged_rows.append({**row, **judge_result})
        time.sleep(SLEEP_SECONDS)

    enriched_rows = add_llm_precision(rows, judged_rows)
    summaries = summarize(enriched_rows, chunk_summaries, boundary_summaries)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "semantic_ab_results.jsonl", enriched_rows)
    write_jsonl(OUTPUT_DIR / "semantic_ab_judge_results.jsonl", judged_rows)
    write_json(OUTPUT_DIR / "semantic_ab_summary.json", summaries)
    write_review(OUTPUT_DIR / "review.md", summaries)
    write_visualization(OUTPUT_DIR / "visualization.html", summaries)

    print(
        json.dumps(
            {
                "judge_items": len(judged_rows),
                "summary": summaries,
                "outputs": {
                    "review": str(OUTPUT_DIR / "review.md"),
                    "visualization": str(OUTPUT_DIR / "visualization.html"),
                    "summary": str(OUTPUT_DIR / "semantic_ab_summary.json"),
                    "results": str(OUTPUT_DIR / "semantic_ab_results.jsonl"),
                    "judge_results": str(OUTPUT_DIR / "semantic_ab_judge_results.jsonl"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
