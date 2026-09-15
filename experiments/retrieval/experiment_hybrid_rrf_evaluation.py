import html
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.retrieval.experiment_bm25_only_evaluation import (
    BM25Index,
    CUSTOM_DICTIONARY,
    JiebaTokenizer,
)
from experiments.retrieval.experiment_dense_only_evaluation import (
    EMBEDDING_MODEL,
    EVALUATION_FILE,
    E5Embedder,
    QUERY_PREFIX,
    judge_evidence,
    load_corpus,
    load_deepseek_config,
    load_jsonl,
    load_or_create_corpus_embeddings,
    metric_block,
    pct,
    preview,
    write_jsonl,
)


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "evaluation/results/hybrid_rrf"
RETRIEVAL_RESULTS = OUTPUT_DIR / "retrieval_results.jsonl"
JUDGE_RESULTS = OUTPUT_DIR / "judge_results.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"
REVIEW_FILE = OUTPUT_DIR / "review.md"
VISUALIZATION_FILE = OUTPUT_DIR / "visualization.html"

DENSE_SUMMARY = BASE_DIR / "evaluation/results/dense_only/summary.json"
BM25_SUMMARY = BASE_DIR / "evaluation/results/bm25_only/summary.json"
DENSE_JUDGE_CACHE = BASE_DIR / "evaluation/results/dense_only/judge_results.jsonl"
BM25_JUDGE_CACHE = BASE_DIR / "evaluation/results/bm25_only/judge_results.jsonl"

DENSE_TOP_K = 10
BM25_TOP_K = 5
FINAL_TOP_K = 3
RRF_K = 60
JUDGE_SLEEP_SECONDS = 0.25


def top_indices(scores, limit):
    return [int(index) for index in np.argsort(-scores, kind="stable")[:limit]]


def reciprocal_rank_fusion(dense_indices, bm25_indices):
    dense_ranks = {index: rank for rank, index in enumerate(dense_indices, start=1)}
    bm25_ranks = {index: rank for rank, index in enumerate(bm25_indices, start=1)}
    candidates = set(dense_ranks) | set(bm25_ranks)
    ranked = []
    for index in candidates:
        score = 0.0
        if index in dense_ranks:
            score += 1 / (RRF_K + dense_ranks[index])
        if index in bm25_ranks:
            score += 1 / (RRF_K + bm25_ranks[index])
        best_rank = min(
            dense_ranks.get(index, 10**9),
            bm25_ranks.get(index, 10**9),
        )
        ranked.append(
            {
                "chunk_position": index,
                "rrf_score": score,
                "dense_rank": dense_ranks.get(index),
                "bm25_rank": bm25_ranks.get(index),
                "best_source_rank": best_rank,
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            -item["rrf_score"],
            item["best_source_rank"],
            item["chunk_position"],
        ),
    )


def run_retrieval(embedder, embeddings, bm25, chunks, questions):
    rows = []
    timings = []
    for position, question in enumerate(questions, start=1):
        started = time.perf_counter()
        query_embedding = embedder.encode([QUERY_PREFIX + question["question"]])[0]
        dense_scores = np.dot(embeddings, query_embedding)
        bm25_scores, query_tokens = bm25.score(question["question"])
        dense_indices = top_indices(dense_scores, DENSE_TOP_K)
        bm25_indices = top_indices(bm25_scores, BM25_TOP_K)
        fused = reciprocal_rank_fusion(dense_indices, bm25_indices)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)

        gold_rank = None
        results = []
        query_token_set = set(query_tokens)
        for rank, fused_item in enumerate(fused[:FINAL_TOP_K], start=1):
            chunk_position = fused_item["chunk_position"]
            chunk = chunks[chunk_position]
            if chunk["global_chunk_id"] == question["relevant_chunk_id"]:
                gold_rank = rank
            sources = []
            if fused_item["dense_rank"] is not None:
                sources.append("dense")
            if fused_item["bm25_rank"] is not None:
                sources.append("bm25")
            results.append(
                {
                    "rank": rank,
                    "rrf_score": round(fused_item["rrf_score"], 8),
                    "dense_rank": fused_item["dense_rank"],
                    "bm25_rank": fused_item["bm25_rank"],
                    "candidate_sources": sources,
                    "dense_cosine_similarity": round(
                        float(dense_scores[chunk_position]),
                        6,
                    ),
                    "bm25_score": round(float(bm25_scores[chunk_position]), 6),
                    "matched_query_tokens": sorted(
                        query_token_set
                        & set(bm25.tokenized_documents[chunk_position])
                    ),
                    "global_chunk_id": chunk["global_chunk_id"],
                    "document": chunk["document"],
                    "chunk_index": chunk["chunk_index"],
                    "section": chunk.get("section", ""),
                    "token_count": chunk.get("token_count"),
                    "evidence": chunk["text"],
                    "preview": preview(chunk["text"]),
                }
            )

        rows.append(
            {
                "evaluation_id": question["evaluation_id"],
                "question": question["question"],
                "standard_answer": question["standard_answer"],
                "question_type": question["question_type"],
                "difficulty": question["difficulty"],
                "query_tokens": query_tokens,
                "gold_chunk_id": question["relevant_chunk_id"],
                "gold_document": question["source_document"],
                "gold_chunk_index": question["source_chunk_index"],
                "gold_rank_top3": gold_rank,
                "gold_hit_at_3": gold_rank is not None,
                "reciprocal_rank_at_3": round(1 / gold_rank, 6) if gold_rank else 0.0,
                "dense_candidate_count": len(dense_indices),
                "bm25_candidate_count": len(bm25_indices),
                "fused_candidate_count": len(fused),
                "retrieval_seconds": round(elapsed, 6),
                "top_results": results,
            }
        )
        print(
            f"[Retrieve] {position}/{len(questions)} {question['evaluation_id']} "
            f"gold_rank={gold_rank} pool={len(fused)}",
            flush=True,
        )
    return rows, sum(timings) / len(timings)


def load_judge_cache(path):
    if not path.exists():
        return {}
    return {
        (row["evaluation_id"], row["global_chunk_id"]): row
        for row in load_jsonl(path)
    }


def merge_shared_caches(*caches):
    merged = {}
    conflicts = []
    for cache_name, cache in caches:
        for key, row in cache.items():
            if key in merged and merged[key]["label"] != row["label"]:
                conflicts.append(
                    {
                        "key": key,
                        "first_label": merged[key]["label"],
                        "second_label": row["label"],
                        "second_cache": cache_name,
                    }
                )
                continue
            merged[key] = row
    return merged, conflicts


def run_judge(rows):
    own_cache = load_judge_cache(JUDGE_RESULTS)
    shared_cache, conflicts = merge_shared_caches(
        ("dense_only", load_judge_cache(DENSE_JUDGE_CACHE)),
        ("bm25_only", load_judge_cache(BM25_JUDGE_CACHE)),
    )
    api_key, chat_url = load_deepseek_config()
    reused = 0
    newly_judged = 0
    current = 0
    total = len(rows) * FINAL_TOP_K

    for row in rows:
        for result in row["top_results"]:
            current += 1
            key = (row["evaluation_id"], result["global_chunk_id"])
            if key in own_cache:
                print(f"[Judge] {current}/{total} skip own cache", flush=True)
                continue
            if key in shared_cache:
                source = shared_cache[key]
                judged = {
                    "evaluation_id": row["evaluation_id"],
                    "global_chunk_id": result["global_chunk_id"],
                    "rank": result["rank"],
                    "label": source["label"],
                    "reason": source["reason"],
                    "cache_source": "dense_or_bm25",
                }
                reused += 1
                print(f"[Judge] {current}/{total} reuse shared cache", flush=True)
            else:
                print(
                    f"[Judge] {current}/{total} {row['evaluation_id']} "
                    f"rank={result['rank']}",
                    flush=True,
                )
                judge_input = {
                    "question": row["question"],
                    "standard_answer": row["standard_answer"],
                    "section": result["section"],
                    "evidence": result["evidence"],
                }
                last_error = None
                for attempt in range(1, 4):
                    try:
                        label, reason = judge_evidence(api_key, chat_url, judge_input)
                        judged = {
                            "evaluation_id": row["evaluation_id"],
                            "global_chunk_id": result["global_chunk_id"],
                            "rank": result["rank"],
                            "label": label,
                            "reason": reason,
                            "cache_source": "hybrid_rrf_new",
                        }
                        newly_judged += 1
                        break
                    except Exception as exc:
                        last_error = exc
                        print(f"  attempt {attempt} failed: {exc}", flush=True)
                        time.sleep(1.5 * attempt)
                else:
                    raise RuntimeError(f"Judge failed for {key}: {last_error}")
                time.sleep(JUDGE_SLEEP_SECONDS)
            with JUDGE_RESULTS.open("a", encoding="utf-8") as file:
                file.write(json.dumps(judged, ensure_ascii=False) + "\n")
            own_cache[key] = judged
    return own_cache, reused, newly_judged, conflicts


def add_judge_results(rows, cache):
    for row in rows:
        labels = []
        for result in row["top_results"]:
            judged = cache[(row["evaluation_id"], result["global_chunk_id"])]
            result["llm_label"] = judged["label"]
            result["llm_reason"] = judged["reason"]
            labels.append(judged["label"])
        row["llm_labels_top3"] = labels
        row["llm_precision_at_3_loose"] = round(
            sum(label in (1, 2) for label in labels) / FINAL_TOP_K,
            6,
        )
        row["llm_precision_at_3_strict"] = round(
            sum(label == 2 for label in labels) / FINAL_TOP_K,
            6,
        )
        row["llm_top1_direct"] = labels[0] == 2
        row["llm_top1_relevant"] = labels[0] in (1, 2)
    return rows


def summarize(rows, average_seconds, dense_cache_used, dense_index_seconds, bm25_seconds, reused, new, conflicts):
    by_type = defaultdict(list)
    by_document = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)
        by_document[row["gold_document"]].append(row)
    dense = json.loads(DENSE_SUMMARY.read_text(encoding="utf-8"))
    bm25 = json.loads(BM25_SUMMARY.read_text(encoding="utf-8"))
    hybrid_overall = metric_block(rows)
    comparison = {
        "dense_only": dense["overall"],
        "bm25_only": bm25["overall"],
        "hybrid_rrf": hybrid_overall,
    }
    return {
        "experiment": "hybrid_rrf_top3",
        "configuration": {
            "dense_model": EMBEDDING_MODEL,
            "dense_top_k": DENSE_TOP_K,
            "bm25_top_k": BM25_TOP_K,
            "rrf_k": RRF_K,
            "final_top_k": FINAL_TOP_K,
            "jieba_dictionary": str(CUSTOM_DICTIONARY),
        },
        "corpus_chunks": 1179,
        "dense_embedding_cache_used": dense_cache_used,
        "dense_corpus_embedding_seconds_this_run": round(dense_index_seconds, 6),
        "bm25_index_seconds": round(bm25_seconds, 6),
        "average_retrieval_seconds": round(average_seconds, 6),
        "overall": hybrid_overall,
        "by_question_type": {
            key: metric_block(value) for key, value in sorted(by_type.items())
        },
        "by_source_document": {
            key: metric_block(value) for key, value in sorted(by_document.items())
        },
        "comparison": comparison,
        "llm_judge": {
            "reused_from_previous_experiments": reused,
            "newly_judged": new,
            "shared_cache_label_conflicts": conflicts,
        },
    }


def write_review(summary, rows):
    comparison = summary["comparison"]
    lines = [
        "# Hybrid RRF Retrieval Evaluation\n",
        f"- Dense: top `{DENSE_TOP_K}` using `{EMBEDDING_MODEL}`",
        f"- BM25: top `{BM25_TOP_K}` using jieba custom dictionary",
        f"- Fusion: `RRF_K={RRF_K}`",
        f"- Final: top `{FINAL_TOP_K}`",
        "",
        "## Three-method Comparison\n",
        "| Method | Gold Hit@3 | MRR@3 | Gold Top-1 | LLM Precision@3 | LLM Top-1 Direct |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, metrics in comparison.items():
        lines.append(
            f"| {method} | {pct(metrics['gold_chunk_hit_rate_at_3'])} | "
            f"{metrics['mrr_at_3']:.3f} | {pct(metrics['gold_top1_accuracy'])} | "
            f"{pct(metrics['llm_precision_at_3_loose'])} | "
            f"{pct(metrics['llm_top1_direct_accuracy'])} |"
        )
    lines.extend(["", "## Exact Gold Misses\n"])
    misses = [row for row in rows if not row["gold_hit_at_3"]]
    if not misses:
        lines.append("- None")
    for row in misses:
        lines.extend(
            [
                f"### {row['evaluation_id']} {row['question']}",
                f"- Gold: `{row['gold_chunk_id']}`",
                f"- LLM labels: `{row['llm_labels_top3']}`",
                "",
            ]
        )
    REVIEW_FILE.write_text("\n".join(lines), encoding="utf-8")


def method_label(method):
    return {
        "dense_only": "Dense-only",
        "bm25_only": "BM25-only",
        "hybrid_rrf": "Hybrid RRF",
    }[method]


def write_visualization(summary, rows):
    metrics = [
        ("Gold Hit@3", "gold_chunk_hit_rate_at_3"),
        ("MRR@3", "mrr_at_3"),
        ("Gold Top-1", "gold_top1_accuracy"),
        ("LLM Precision@3", "llm_precision_at_3_loose"),
        ("LLM Top-1 Direct", "llm_top1_direct_accuracy"),
    ]
    comparison_rows = "".join(
        "<tr>"
        f"<td>{method_label(method)}</td>"
        + "".join(
            f"<td>{value[key]:.3f}</td>" for _label, key in metrics
        )
        + "</tr>"
        for method, value in summary["comparison"].items()
    )
    bars = "".join(
        build_metric_group(label, key, summary["comparison"])
        for label, key in metrics
    )
    type_rows = "".join(
        "<tr>"
        f"<td>{html.escape(kind)}</td><td>{value['questions']}</td>"
        f"<td>{pct(value['gold_chunk_hit_rate_at_3'])}</td>"
        f"<td>{value['mrr_at_3']:.3f}</td>"
        f"<td>{pct(value['llm_precision_at_3_loose'])}</td>"
        f"<td>{pct(value['llm_top1_direct_accuracy'])}</td></tr>"
        for kind, value in summary["by_question_type"].items()
    )
    question_rows = "".join(build_question_row(row) for row in rows)
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hybrid RRF Evaluation</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:#172033;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}header{{background:#263b63;color:white;padding:34px max(28px,calc((100vw - 1220px)/2))}}header h1{{margin:0 0 8px;font-size:30px}}header p{{margin:0;color:#d9e2f2}}main{{max-width:1220px;margin:auto;padding:26px 28px 60px}}h2{{margin:32px 0 14px;font-size:21px}}.panel{{background:white;border:1px solid #dfe3e8;border-radius:6px;padding:18px;overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{border:1px solid #dfe3e8;padding:10px;text-align:left;vertical-align:top}}th{{background:#edf2f7}}.bars{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.group{{background:white;border:1px solid #dfe3e8;border-radius:6px;padding:16px}}.group h3{{margin:0 0 12px;font-size:15px}}.barline{{display:grid;grid-template-columns:92px 1fr 45px;gap:8px;align-items:center;margin:8px 0;font-size:13px}}.track{{height:13px;background:#e8edf3}}.fill{{height:100%}}.dense{{background:#1769aa}}.bm25{{background:#16875d}}.hybrid{{background:#c47b16}}details summary{{cursor:pointer;color:#315f9b}}.evidence{{min-width:470px;margin:12px 0;padding:11px;background:#f8f9fb;border-left:3px solid #c47b16}}.evidence p{{color:#475467;line-height:1.55}}td.question{{min-width:240px}}@media(max-width:800px){{.bars{{grid-template-columns:1fr}}main{{padding:18px 14px}}}}
</style></head><body><header><h1>Hybrid Retrieval: Three-method Comparison</h1><p>Dense top 10 + BM25 top 5 + RRF_K=60 · Final Top 3</p></header><main>
<h2>三种方法总体对比</h2><section class="panel"><table><tr><th>Method</th>{''.join(f'<th>{label}</th>' for label,_key in metrics)}</tr>{comparison_rows}</table></section>
<h2>指标可视化</h2><section class="bars">{bars}</section>
<h2>Hybrid 按问题类型</h2><section class="panel"><table><tr><th>类型</th><th>问题数</th><th>Gold Hit@3</th><th>MRR@3</th><th>LLM Precision@3</th><th>Top-1 Direct</th></tr>{type_rows}</table></section>
<h2>Hybrid 逐题结果</h2><section class="panel"><table><tr><th>ID</th><th>问题</th><th>Gold rank</th><th>LLM labels</th><th>Top 3 融合来源</th></tr>{question_rows}</table></section>
</main></body></html>"""
    VISUALIZATION_FILE.write_text(page, encoding="utf-8")


def build_metric_group(label, key, comparison):
    colors = {"dense_only": "dense", "bm25_only": "bm25", "hybrid_rrf": "hybrid"}
    lines = []
    for method, values in comparison.items():
        value = values[key]
        lines.append(
            f'<div class="barline"><span>{method_label(method)}</span>'
            f'<div class="track"><div class="fill {colors[method]}" style="width:{value * 100:.1f}%"></div></div>'
            f'<b>{value:.3f}</b></div>'
        )
    return f'<div class="group"><h3>{html.escape(label)}</h3>{"".join(lines)}</div>'


def build_question_row(row):
    evidence_html = "".join(
        f"<div class='evidence'><b>#{item['rank']} · {html.escape(item['document'])} · chunk {item['chunk_index']}</b>"
        f"<div>RRF {item['rrf_score']:.6f} · Dense rank {item['dense_rank'] or '-'} · BM25 rank {item['bm25_rank'] or '-'} · label {item['llm_label']}</div>"
        f"<p>{html.escape(item['preview'])}</p></div>"
        for item in row["top_results"]
    )
    return (
        "<tr>"
        f"<td>{row['evaluation_id']}</td><td class='question'>{html.escape(row['question'])}</td>"
        f"<td>{row['gold_rank_top3'] or 'miss'}</td><td>{html.escape(str(row['llm_labels_top3']))}</td>"
        f"<td><details><summary>查看 Top 3</summary>{evidence_html}</details></td></tr>"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_jsonl(EVALUATION_FILE)
    chunks = load_corpus()

    embedder = E5Embedder(EMBEDDING_MODEL)
    embeddings, dense_index_seconds, dense_cache_used = load_or_create_corpus_embeddings(
        embedder,
        chunks,
    )
    tokenizer = JiebaTokenizer(CUSTOM_DICTIONARY)
    started = time.perf_counter()
    bm25 = BM25Index([chunk["text"] for chunk in chunks], tokenizer)
    bm25_seconds = time.perf_counter() - started

    rows, average_seconds = run_retrieval(
        embedder,
        embeddings,
        bm25,
        chunks,
        questions,
    )
    write_jsonl(RETRIEVAL_RESULTS, rows)
    judge_cache, reused, new, conflicts = run_judge(rows)
    rows = add_judge_results(rows, judge_cache)
    write_jsonl(RETRIEVAL_RESULTS, rows)

    summary = summarize(
        rows,
        average_seconds,
        dense_cache_used,
        dense_index_seconds,
        bm25_seconds,
        reused,
        new,
        conflicts,
    )
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_review(summary, rows)
    write_visualization(summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[Visualization] {VISUALIZATION_FILE}", flush=True)


if __name__ == "__main__":
    main()
