import json
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
from experiments.retrieval.experiment_retrieval_evaluation import (
    QUESTIONS,
    TransformersMeanPoolingEmbedder,
    embed_texts,
    load_chunks,
    write_json,
    write_jsonl,
)


CHUNKS_PATH = Path(
    "artifacts/chunking/2025年品牌营销趋势报告/"
    "cleaned_text_structure_semantic_boundary_500_min_150/chunks.jsonl"
)
OUTPUT_DIR = Path(
    "artifacts/chunking/2025年品牌营销趋势报告/hybrid_grid_search_eval"
)

DENSE_TOP_K_VALUES = [5, 10, 20]
BM25_TOP_K_VALUES = [5, 10, 20]
FINAL_TOP_K = 3
MANUAL_LOOSE_PRECISION_BY_COMBO = {
    (10, 5): 0.7917,
}


def top3_keyword_ratio(row):
    coverage = row["top3_keyword_coverage"]
    total = len(coverage["found"]) + len(coverage["missing"])
    if total == 0:
        return 0.0
    return len(coverage["found"]) / total


def evaluate_grid(chunks):
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

        for dense_top_k in DENSE_TOP_K_VALUES:
            dense_ranked = rank_scores(dense_scores, dense_top_k)
            for bm25_top_k in BM25_TOP_K_VALUES:
                bm25_ranked = rank_scores(bm25_scores, bm25_top_k)
                hybrid_ranked = rrf_rank(dense_ranked, bm25_ranked)
                method = f"dense{dense_top_k}_bm25{bm25_top_k}_rrf"

                row = build_result_row(
                    question,
                    method,
                    chunks,
                    hybrid_ranked[:FINAL_TOP_K],
                )
                row["dense_candidate_top_k"] = dense_top_k
                row["bm25_candidate_top_k"] = bm25_top_k
                row["final_top_k"] = FINAL_TOP_K
                row["rrf_k"] = RRF_K
                row["candidate_pool_size"] = len(
                    {index for index, _score in dense_ranked}
                    | {index for index, _score in bm25_ranked}
                )
                row["top3_keyword_ratio"] = round(top3_keyword_ratio(row), 4)
                rows.append(row)

    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["dense_candidate_top_k"], row["bm25_candidate_top_k"])
        grouped[key].append(row)

    summaries = []
    for (dense_top_k, bm25_top_k), combo_rows in sorted(grouped.items()):
        keyword_avg = sum(row["top3_keyword_ratio"] for row in combo_rows) / len(combo_rows)
        section_hits = sum(1 for row in combo_rows if row["expected_section_hit_top3"])
        section_rate = section_hits / len(combo_rows)
        avg_pool_size = sum(row["candidate_pool_size"] for row in combo_rows) / len(combo_rows)
        perfect_keyword_questions = sum(
            1 for row in combo_rows if row["top3_keyword_ratio"] >= 1.0
        )
        manual_precision = MANUAL_LOOSE_PRECISION_BY_COMBO.get(
            (dense_top_k, bm25_top_k)
        )

        # Prototype comparison score with human Precision@3 when available.
        # Precision@3 requires manual or judge labeling, so only annotated
        # combinations should use this score as final evidence.
        combined_score = None
        if manual_precision is not None:
            combined_score = (
                keyword_avg * 0.50
                + section_rate * 0.20
                + manual_precision * 0.25
                - (avg_pool_size / 40) * 0.05
            )

        summaries.append(
            {
                "dense_candidate_top_k": dense_top_k,
                "bm25_candidate_top_k": bm25_top_k,
                "question_count": len(combo_rows),
                "avg_top3_keyword_recall": round(keyword_avg, 4),
                "section_hit_rate_top3": round(section_rate, 4),
                "section_hits_top3": section_hits,
                "perfect_keyword_questions": perfect_keyword_questions,
                "avg_candidate_pool_size": round(avg_pool_size, 2),
                "manual_loose_precision_at_3": round(manual_precision, 4)
                if manual_precision is not None
                else None,
                "combined_score_with_precision": round(combined_score, 4)
                if combined_score is not None
                else None,
            }
        )

    return sorted(
        summaries,
        key=lambda item: (
            item["combined_score_with_precision"]
            if item["combined_score_with_precision"] is not None
            else -1,
            item["avg_top3_keyword_recall"],
            item["section_hit_rate_top3"],
            -item["avg_candidate_pool_size"],
        ),
        reverse=True,
    )


def metric_for_combo(summaries, dense_top_k, bm25_top_k, metric):
    for item in summaries:
        if (
            item["dense_candidate_top_k"] == dense_top_k
            and item["bm25_candidate_top_k"] == bm25_top_k
        ):
            return item[metric]
    return None


def color_for_value(value, min_value, max_value):
    if max_value == min_value:
        ratio = 1.0
    else:
        ratio = (value - min_value) / (max_value - min_value)
    red = int(255 - 108 * ratio)
    green = int(245 - 71 * (1 - ratio))
    blue = int(235 - 145 * ratio)
    return f"rgb({red}, {green}, {blue})"


def heatmap_table(summaries, metric, title, formatter):
    values = [item[metric] for item in summaries if item[metric] is not None]
    if not values:
        return ""
    min_value = min(values)
    max_value = max(values)
    lines = [
        f"<h2>{title}</h2>",
        "<table>",
        "<tr><th>Dense top_k \\ BM25 top_k</th><th>5</th><th>10</th><th>20</th></tr>",
    ]
    for dense_top_k in DENSE_TOP_K_VALUES:
        cells = [f"<th>{dense_top_k}</th>"]
        for bm25_top_k in BM25_TOP_K_VALUES:
            value = metric_for_combo(summaries, dense_top_k, bm25_top_k, metric)
            if value is None:
                cells.append('<td style="background:#f3f4f6">not annotated</td>')
            else:
                color = color_for_value(value, min_value, max_value)
                cells.append(
                    f'<td style="background:{color}">{formatter(value)}</td>'
                )
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def write_visualization(path: Path, summaries):
    best = summaries[0]
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Hybrid Retrieval Grid Search</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #1f2937;
      background: #f8fafc;
    }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin-top: 28px; }}
    .note {{
      max-width: 920px;
      line-height: 1.6;
      background: white;
      border: 1px solid #e5e7eb;
      padding: 16px 18px;
      border-radius: 8px;
    }}
    table {{
      border-collapse: collapse;
      margin-top: 12px;
      background: white;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 12px 16px;
      text-align: center;
      min-width: 110px;
    }}
    th {{ background: #eef2f7; }}
    .rank-table td, .rank-table th {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Hybrid Retrieval Grid Search</h1>
  <div class="note">
    <p><strong>固定参数：</strong>chunking = structure-aware + semantic sentence boundary 500/150，dense model = {DENSE_MODEL_NAME}，RRF_K = {RRF_K}，final top_k = {FINAL_TOP_K}。</p>
    <p><strong>已标注组合：</strong>Dense top {best["dense_candidate_top_k"]} + BM25 top {best["bm25_candidate_top_k"]}。新的 combined score 使用 Recall@3、Hit Rate@3、人工宽松 Precision@3 和候选池惩罚。</p>
    <p><strong>注意：</strong>Precision@3 需要人工或 LLM-as-judge 标注。目前只有 Dense top10 + BM25 top5 有人工 Precision@3，其他组合暂不参与带 Precision 的最终排序。</p>
  </div>

  {heatmap_table(summaries, "avg_top3_keyword_recall", "Average Top-3 Keyword Recall", lambda value: f"{value:.3f}")}

  {heatmap_table(summaries, "section_hit_rate_top3", "Top-3 Section Hit Rate", lambda value: f"{value:.3f}")}

  {heatmap_table(summaries, "manual_loose_precision_at_3", "Manual Loose Precision@3", lambda value: f"{value:.3f}")}

  {heatmap_table(summaries, "combined_score_with_precision", "Combined Score With Precision", lambda value: f"{value:.3f}")}

  <h2>Ranking</h2>
  <table class="rank-table">
    <tr>
      <th>Rank</th>
      <th>Dense top_k</th>
      <th>BM25 top_k</th>
      <th>Avg keyword recall</th>
      <th>Section hit</th>
      <th>Manual P@3</th>
      <th>Perfect keyword Qs</th>
      <th>Avg pool size</th>
      <th>Combined score with precision</th>
    </tr>
"""
    for rank, item in enumerate(summaries, start=1):
        html += (
            "    <tr>"
            f"<td>{rank}</td>"
            f"<td>{item['dense_candidate_top_k']}</td>"
            f"<td>{item['bm25_candidate_top_k']}</td>"
            f"<td>{item['avg_top3_keyword_recall']:.3f}</td>"
            f"<td>{item['section_hit_rate_top3']:.3f}</td>"
            f"<td>{item['manual_loose_precision_at_3'] if item['manual_loose_precision_at_3'] is not None else 'not annotated'}</td>"
            f"<td>{item['perfect_keyword_questions']}/{item['question_count']}</td>"
            f"<td>{item['avg_candidate_pool_size']:.2f}</td>"
            f"<td>{item['combined_score_with_precision'] if item['combined_score_with_precision'] is not None else 'not annotated'}</td>"
            "</tr>\n"
        )
    html += """  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_review(path: Path, summaries):
    lines = [
        "# Hybrid Retrieval Grid Search\n",
        f"- Chunking: `structure-aware + semantic sentence boundary 500/150`",
        f"- Dense model: `{DENSE_MODEL_NAME}`",
        f"- RRF_K: `{RRF_K}`",
        f"- Final top_k: `{FINAL_TOP_K}`",
        "- Tested dense candidate top_k: `5 / 10 / 20`",
        "- Tested BM25 candidate top_k: `5 / 10 / 20`",
        "- New score formula: `Recall@3 * 0.50 + HitRate@3 * 0.20 + Precision@3 * 0.25 - CandidatePoolPenalty * 0.05`",
        "- Precision@3 source: human loose Precision@3. Only `Dense top10 + BM25 top5` has been manually annotated so far.",
        "",
        "## Ranking\n",
        "| Rank | Dense top_k | BM25 top_k | Avg Top-3 keyword recall | Top-3 section hit | Manual loose P@3 | Perfect keyword Qs | Avg candidate pool | Combined score with precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for rank, item in enumerate(summaries, start=1):
        lines.append(
            "| {rank} | {dense} | {bm25} | {keyword:.3f} | {section:.3f} | "
            "{precision} | {perfect}/{questions} | {pool:.2f} | {score} |".format(
                rank=rank,
                dense=item["dense_candidate_top_k"],
                bm25=item["bm25_candidate_top_k"],
                keyword=item["avg_top3_keyword_recall"],
                section=item["section_hit_rate_top3"],
                precision=(
                    f"{item['manual_loose_precision_at_3']:.3f}"
                    if item["manual_loose_precision_at_3"] is not None
                    else "not annotated"
                ),
                perfect=item["perfect_keyword_questions"],
                questions=item["question_count"],
                pool=item["avg_candidate_pool_size"],
                score=(
                    f"{item['combined_score_with_precision']:.3f}"
                    if item["combined_score_with_precision"] is not None
                    else "not annotated"
                ),
            )
        )

    best = summaries[0]
    lines.extend(
        [
            "",
            "## Current Best Choice\n",
            (
                f"- Dense top {best['dense_candidate_top_k']} + "
                f"BM25 top {best['bm25_candidate_top_k']}"
            ),
            "- This is currently the only combination with human Precision@3 annotation.",
            "- To compare all 9 combinations fairly by Precision@3, each combination needs the same manual or LLM-as-judge labeling process.",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    chunks = load_chunks(CHUNKS_PATH)
    rows = evaluate_grid(chunks)
    summaries = summarize(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "grid_summary.json", summaries)
    write_jsonl(OUTPUT_DIR / "grid_results.jsonl", rows)
    write_review(OUTPUT_DIR / "review.md", summaries)
    write_visualization(OUTPUT_DIR / "visualization.html", summaries)

    print(
        json.dumps(
            {
                "questions": len(QUESTIONS),
                "combinations": len(summaries),
                "best": summaries[0],
                "outputs": {
                    "summary": str(OUTPUT_DIR / "grid_summary.json"),
                    "results": str(OUTPUT_DIR / "grid_results.jsonl"),
                    "review": str(OUTPUT_DIR / "review.md"),
                    "visualization": str(OUTPUT_DIR / "visualization.html"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
