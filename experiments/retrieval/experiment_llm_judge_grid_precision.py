import json
import time
from collections import defaultdict
from pathlib import Path

from experiments.retrieval.experiment_hybrid_grid_search import OUTPUT_DIR
from experiments.retrieval.experiment_llm_judge_precision import (
    CHUNKS_PATH,
    MODEL_NAME,
    SLEEP_SECONDS,
    attach_full_evidence,
    judge_one,
    load_deepseek_config,
    load_full_chunks,
    precision,
)


GRID_RESULTS_PATH = OUTPUT_DIR / "grid_results.jsonl"
GRID_SUMMARY_PATH = OUTPUT_DIR / "grid_summary.json"
OUTPUT_JSONL = OUTPUT_DIR / "llm_judge_grid_precision_results.jsonl"
OUTPUT_SUMMARY = OUTPUT_DIR / "llm_judge_grid_precision_summary.json"
OUTPUT_REVIEW = OUTPUT_DIR / "llm_judge_grid_precision_review.md"


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_unique_judge_rows(grid_rows):
    unique = {}
    for row in grid_rows:
        for top_result in row["top_results"]:
            key = (row["question_id"], top_result["chunk_index"])
            if key in unique:
                continue
            unique[key] = {
                "question_id": row["question_id"],
                "question": row["question"],
                "rank": top_result["rank"],
                "chunk_index": top_result["chunk_index"],
                "section": top_result["section"],
            }
    return list(unique.values())


def add_precision_to_grid_rows(grid_rows, labels_by_key):
    enriched_rows = []
    for row in grid_rows:
        labels = []
        for top_result in row["top_results"]:
            key = (row["question_id"], top_result["chunk_index"])
            label_info = labels_by_key[key]
            top_result["llm_label"] = label_info["llm_label"]
            top_result["llm_reason"] = label_info["llm_reason"]
            labels.append(label_info["llm_label"])

        row["llm_labels_top3"] = labels
        row["llm_precision_at_3_loose"] = round(precision(labels, strict=False), 4)
        row["llm_precision_at_3_strict"] = round(precision(labels, strict=True), 4)
        enriched_rows.append(row)
    return enriched_rows


def summarize_by_combo(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["dense_candidate_top_k"], row["bm25_candidate_top_k"])
        grouped[key].append(row)

    summaries = []
    for (dense_top_k, bm25_top_k), combo_rows in sorted(grouped.items()):
        recall = sum(row["top3_keyword_ratio"] for row in combo_rows) / len(combo_rows)
        hit_rate = (
            sum(1 for row in combo_rows if row["expected_section_hit_top3"])
            / len(combo_rows)
        )
        precision_loose = (
            sum(row["llm_precision_at_3_loose"] for row in combo_rows)
            / len(combo_rows)
        )
        precision_strict = (
            sum(row["llm_precision_at_3_strict"] for row in combo_rows)
            / len(combo_rows)
        )
        avg_pool_size = (
            sum(row["candidate_pool_size"] for row in combo_rows)
            / len(combo_rows)
        )
        perfect_keyword_questions = sum(
            1 for row in combo_rows if row["top3_keyword_ratio"] >= 1.0
        )

        combined_score = (
            recall * 0.50
            + hit_rate * 0.20
            + precision_loose * 0.25
            - (avg_pool_size / 40) * 0.05
        )

        summaries.append(
            {
                "dense_candidate_top_k": dense_top_k,
                "bm25_candidate_top_k": bm25_top_k,
                "question_count": len(combo_rows),
                "avg_top3_keyword_recall": round(recall, 4),
                "section_hit_rate_top3": round(hit_rate, 4),
                "llm_loose_precision_at_3": round(precision_loose, 4),
                "llm_strict_precision_at_3": round(precision_strict, 4),
                "perfect_keyword_questions": perfect_keyword_questions,
                "avg_candidate_pool_size": round(avg_pool_size, 2),
                "combined_score_with_llm_precision": round(combined_score, 4),
            }
        )

    return sorted(
        summaries,
        key=lambda item: (
            item["combined_score_with_llm_precision"],
            item["llm_loose_precision_at_3"],
            item["avg_top3_keyword_recall"],
            item["section_hit_rate_top3"],
            -item["avg_candidate_pool_size"],
        ),
        reverse=True,
    )


def update_grid_summary_with_llm_precision(summary_rows):
    GRID_SUMMARY_PATH.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_review(path: Path, summaries, judged_unique_count):
    lines = [
        "# LLM-as-Judge Precision Grid Search\n",
        f"- Judge model: `{MODEL_NAME}`",
        "- Relevant definition for Precision@3: `label 1 or 2 counts as relevant`",
        "- Score formula: `Recall@3 * 0.50 + HitRate@3 * 0.20 + LLM Precision@3 * 0.25 - CandidatePoolPenalty * 0.05`",
        f"- Unique `(question, chunk)` items judged: `{judged_unique_count}`",
        "",
        "## Ranking\n",
        "| Rank | Dense top_k | BM25 top_k | Recall@3 | HitRate@3 | LLM loose P@3 | LLM strict P@3 | Avg pool | Combined score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(summaries, start=1):
        lines.append(
            "| {rank} | {dense} | {bm25} | {recall:.3f} | {hit:.3f} | "
            "{loose:.3f} | {strict:.3f} | {pool:.2f} | {score:.3f} |".format(
                rank=rank,
                dense=item["dense_candidate_top_k"],
                bm25=item["bm25_candidate_top_k"],
                recall=item["avg_top3_keyword_recall"],
                hit=item["section_hit_rate_top3"],
                loose=item["llm_loose_precision_at_3"],
                strict=item["llm_strict_precision_at_3"],
                pool=item["avg_candidate_pool_size"],
                score=item["combined_score_with_llm_precision"],
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
            "- This choice is based on LLM-as-judge Precision@3 across all 9 combinations.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    api_key, chat_url = load_deepseek_config()
    grid_rows = read_jsonl(GRID_RESULTS_PATH)
    chunks_by_index = load_full_chunks(CHUNKS_PATH)
    judge_rows = attach_full_evidence(
        build_unique_judge_rows(grid_rows),
        chunks_by_index,
    )

    judged_rows = []
    for index, row in enumerate(judge_rows, start=1):
        print(
            f"[Judge] {index}/{len(judge_rows)} "
            f"{row['question_id']} chunk {row['chunk_index']}"
        )
        judge_result = judge_one(api_key, chat_url, row)
        judged_rows.append({**row, **judge_result})
        time.sleep(SLEEP_SECONDS)

    labels_by_key = {
        (row["question_id"], row["chunk_index"]): row
        for row in judged_rows
    }
    enriched_grid_rows = add_precision_to_grid_rows(grid_rows, labels_by_key)
    summaries = summarize_by_combo(enriched_grid_rows)

    write_jsonl(OUTPUT_JSONL, enriched_grid_rows)
    OUTPUT_SUMMARY.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    update_grid_summary_with_llm_precision(summaries)
    write_review(OUTPUT_REVIEW, summaries, len(judged_rows))

    print(
        json.dumps(
            {
                "judged_unique_items": len(judged_rows),
                "combinations": len(summaries),
                "best": summaries[0],
                "outputs": {
                    "grid_summary": str(GRID_SUMMARY_PATH),
                    "llm_grid_summary": str(OUTPUT_SUMMARY),
                    "llm_grid_results": str(OUTPUT_JSONL),
                    "review": str(OUTPUT_REVIEW),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
