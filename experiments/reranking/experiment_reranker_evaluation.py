import gc
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
from experiments.retrieval.experiment_hybrid_rrf_evaluation import (
    BM25_TOP_K,
    DENSE_TOP_K,
    FINAL_TOP_K,
    RRF_K,
    reciprocal_rank_fusion,
    top_indices,
)


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "evaluation/results/reranker"
RESULTS_FILE = OUTPUT_DIR / "retrieval_results.jsonl"
JUDGE_RESULTS = OUTPUT_DIR / "judge_results.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"
REVIEW_FILE = OUTPUT_DIR / "review.md"
VISUALIZATION_FILE = OUTPUT_DIR / "visualization.html"

HYBRID_RESULTS = BASE_DIR / "evaluation/results/hybrid_rrf/retrieval_results.jsonl"
HYBRID_SUMMARY = BASE_DIR / "evaluation/results/hybrid_rrf/summary.json"
SHARED_JUDGE_FILES = [
    BASE_DIR / "evaluation/results/dense_only/judge_results.jsonl",
    BASE_DIR / "evaluation/results/bm25_only/judge_results.jsonl",
    BASE_DIR / "evaluation/results/hybrid_rrf/judge_results.jsonl",
]

RERANKER_MODELS = [
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3",
]
MAX_PAIR_TOKENS = 512
BATCH_SIZE = 4
JUDGE_SLEEP_SECONDS = 0.25


def method_name(model_name):
    return model_name.split("/")[-1]


class CrossEncoderReranker:
    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started

    def score(self, question, passages):
        scores = []
        started = time.perf_counter()
        with self.torch.no_grad():
            for start in range(0, len(passages), BATCH_SIZE):
                batch = passages[start : start + BATCH_SIZE]
                encoded = self.tokenizer(
                    [question] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_PAIR_TOKENS,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                if logits.shape[-1] == 1:
                    batch_scores = logits[:, 0]
                else:
                    batch_scores = logits[:, -1]
                scores.extend(float(value) for value in batch_scores.cpu().tolist())
        return scores, time.perf_counter() - started

    def close(self):
        self.model.to("cpu")
        del self.model
        del self.tokenizer
        gc.collect()
        if self.torch.backends.mps.is_available():
            self.torch.mps.empty_cache()


def build_candidate_pools(embedder, embeddings, bm25, questions):
    pools = {}
    for position, question in enumerate(questions, start=1):
        query_embedding = embedder.encode([QUERY_PREFIX + question["question"]])[0]
        dense_scores = np.dot(embeddings, query_embedding)
        bm25_scores, query_tokens = bm25.score(question["question"])
        dense_indices = top_indices(dense_scores, DENSE_TOP_K)
        bm25_indices = top_indices(bm25_scores, BM25_TOP_K)
        fused = reciprocal_rank_fusion(dense_indices, bm25_indices)
        for candidate in fused:
            index = candidate["chunk_position"]
            candidate["dense_cosine_similarity"] = float(dense_scores[index])
            candidate["bm25_score"] = float(bm25_scores[index])
        pools[question["evaluation_id"]] = {
            "question": question,
            "query_tokens": query_tokens,
            "candidates": fused,
        }
        print(
            f"[Candidates] {position}/{len(questions)} "
            f"{question['evaluation_id']} pool={len(fused)}",
            flush=True,
        )
    return pools


def rerank_model(model_name, pools, chunks):
    reranker = CrossEncoderReranker(model_name)
    rows = []
    timings = []
    name = method_name(model_name)
    try:
        for position, (evaluation_id, pool) in enumerate(pools.items(), start=1):
            question = pool["question"]
            candidates = pool["candidates"]
            passages = [chunks[item["chunk_position"]]["text"] for item in candidates]
            scores, elapsed = reranker.score(question["question"], passages)
            timings.append(elapsed)
            ranked = sorted(
                zip(candidates, scores),
                key=lambda item: (-item[1], item[0]["best_source_rank"]),
            )

            gold_rank = None
            results = []
            for rank, (candidate, score) in enumerate(ranked[:FINAL_TOP_K], start=1):
                chunk = chunks[candidate["chunk_position"]]
                if chunk["global_chunk_id"] == question["relevant_chunk_id"]:
                    gold_rank = rank
                results.append(
                    {
                        "rank": rank,
                        "reranker_score": round(float(score), 6),
                        "rrf_rank": candidates.index(candidate) + 1,
                        "rrf_score": round(candidate["rrf_score"], 8),
                        "dense_rank": candidate["dense_rank"],
                        "bm25_rank": candidate["bm25_rank"],
                        "dense_cosine_similarity": round(
                            candidate["dense_cosine_similarity"], 6
                        ),
                        "bm25_score": round(candidate["bm25_score"], 6),
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
                    "method": name,
                    "reranker_model": model_name,
                    "evaluation_id": evaluation_id,
                    "question": question["question"],
                    "standard_answer": question["standard_answer"],
                    "question_type": question["question_type"],
                    "difficulty": question["difficulty"],
                    "gold_chunk_id": question["relevant_chunk_id"],
                    "gold_document": question["source_document"],
                    "gold_chunk_index": question["source_chunk_index"],
                    "gold_rank_top3": gold_rank,
                    "gold_hit_at_3": gold_rank is not None,
                    "reciprocal_rank_at_3": round(1 / gold_rank, 6) if gold_rank else 0.0,
                    "candidate_count": len(candidates),
                    "rerank_seconds": round(elapsed, 6),
                    "top_results": results,
                }
            )
            print(
                f"[Rerank:{name}] {position}/{len(pools)} "
                f"{evaluation_id} gold_rank={gold_rank} {elapsed:.3f}s",
                flush=True,
            )
    finally:
        device = str(reranker.device)
        load_seconds = reranker.load_seconds
        reranker.close()

    timing_array = np.array(timings, dtype=float)
    return rows, {
        "model": model_name,
        "device": device,
        "model_load_seconds": round(load_seconds, 6),
        "average_rerank_seconds": round(float(timing_array.mean()), 6),
        "median_rerank_seconds": round(float(np.median(timing_array)), 6),
        "p95_rerank_seconds": round(float(np.percentile(timing_array, 95)), 6),
    }


def load_cache(path):
    if not path.exists():
        return {}
    return {
        (row["evaluation_id"], row["global_chunk_id"]): row
        for row in load_jsonl(path)
    }


def merged_judge_cache():
    merged = {}
    conflicts = []
    for path in [*SHARED_JUDGE_FILES, JUDGE_RESULTS]:
        for key, row in load_cache(path).items():
            if key in merged and merged[key]["label"] != row["label"]:
                conflicts.append(
                    {
                        "evaluation_id": key[0],
                        "global_chunk_id": key[1],
                        "kept_label": merged[key]["label"],
                        "conflicting_label": row["label"],
                        "file": str(path),
                    }
                )
                continue
            merged[key] = row
    return merged, conflicts


def judge_new_evidence(method_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache, conflicts = merged_judge_cache()
    own_cache = load_cache(JUDGE_RESULTS)
    api_key, chat_url = load_deepseek_config()
    unique_pairs = {}
    for rows in method_rows.values():
        for row in rows:
            for result in row["top_results"]:
                key = (row["evaluation_id"], result["global_chunk_id"])
                unique_pairs.setdefault(key, (row, result))

    reused = 0
    newly_judged = 0
    for position, (key, (row, result)) in enumerate(unique_pairs.items(), start=1):
        if key in cache:
            reused += 1
            print(f"[Judge] {position}/{len(unique_pairs)} reuse cache", flush=True)
            continue
        print(
            f"[Judge] {position}/{len(unique_pairs)} {row['evaluation_id']} "
            f"chunk={result['global_chunk_id']}",
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
                    "label": label,
                    "reason": reason,
                    "cache_source": "reranker_new",
                }
                with JUDGE_RESULTS.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(judged, ensure_ascii=False) + "\n")
                cache[key] = judged
                own_cache[key] = judged
                newly_judged += 1
                break
            except Exception as exc:
                last_error = exc
                print(f"  attempt {attempt} failed: {exc}", flush=True)
                time.sleep(1.5 * attempt)
        else:
            raise RuntimeError(f"Judge failed for {key}: {last_error}")
        time.sleep(JUDGE_SLEEP_SECONDS)

    for rows in method_rows.values():
        for row in rows:
            labels = []
            for result in row["top_results"]:
                judged = cache[(row["evaluation_id"], result["global_chunk_id"])]
                result["llm_label"] = judged["label"]
                result["llm_reason"] = judged["reason"]
                labels.append(judged["label"])
            row["llm_labels_top3"] = labels
            row["llm_precision_at_3_loose"] = round(
                sum(label in (1, 2) for label in labels) / FINAL_TOP_K, 6
            )
            row["llm_precision_at_3_strict"] = round(
                sum(label == 2 for label in labels) / FINAL_TOP_K, 6
            )
            row["llm_top1_direct"] = labels[0] == 2
            row["llm_top1_relevant"] = labels[0] in (1, 2)
    return reused, newly_judged, conflicts


def baseline_rows_for_output():
    rows = load_jsonl(HYBRID_RESULTS)
    for row in rows:
        row["method"] = "hybrid_rrf"
        row["candidate_count"] = row["fused_candidate_count"]
        row["rerank_seconds"] = 0.0
    return rows


def build_summary(method_rows, timings, judge_stats):
    hybrid_summary = json.loads(HYBRID_SUMMARY.read_text(encoding="utf-8"))
    comparison = {name: metric_block(rows) for name, rows in method_rows.items()}
    historical = {
        "dense_only": hybrid_summary["comparison"]["dense_only"],
        "bm25_only": hybrid_summary["comparison"]["bm25_only"],
        **comparison,
    }
    by_type = {}
    for name, rows in method_rows.items():
        groups = defaultdict(list)
        for row in rows:
            groups[row["question_type"]].append(row)
        by_type[name] = {
            kind: metric_block(group) for kind, group in sorted(groups.items())
        }
    return {
        "experiment": "hybrid_rrf_reranker_comparison",
        "configuration": {
            "chunking": "structure-aware + semantic sentence boundary 500/150",
            "dense_model": EMBEDDING_MODEL,
            "dense_top_k": DENSE_TOP_K,
            "bm25_top_k": BM25_TOP_K,
            "rrf_k": RRF_K,
            "rerank_candidate_pool": "deduplicated union of dense and BM25 candidates",
            "final_top_k": FINAL_TOP_K,
            "reranker_models": RERANKER_MODELS,
            "reranker_max_pair_tokens": MAX_PAIR_TOKENS,
        },
        "comparison": comparison,
        "historical_comparison": historical,
        "by_question_type": by_type,
        "timing": {
            "hybrid_average_retrieval_seconds": hybrid_summary[
                "average_retrieval_seconds"
            ],
            "rerankers": timings,
        },
        "llm_judge": judge_stats,
    }


def label(name):
    return {
        "dense_only": "Dense-only",
        "bm25_only": "BM25-only",
        "hybrid_rrf": "Hybrid RRF",
        "bge-reranker-base": "Hybrid + bge-reranker-base",
        "bge-reranker-v2-m3": "Hybrid + bge-reranker-v2-m3",
    }[name]


METRICS = [
    ("Gold Hit@3", "gold_chunk_hit_rate_at_3"),
    ("MRR@3", "mrr_at_3"),
    ("Gold Top-1", "gold_top1_accuracy"),
    ("LLM Precision@3", "llm_precision_at_3_loose"),
    ("LLM Top-1 Direct", "llm_top1_direct_accuracy"),
    ("LLM Top-1 Relevant", "llm_top1_relevant_accuracy"),
]


def comparison_table(comparison):
    rows = "".join(
        "<tr><td>" + html.escape(label(name)) + "</td>"
        + "".join(f"<td>{values[key]:.3f}</td>" for _title, key in METRICS)
        + "</tr>"
        for name, values in comparison.items()
    )
    headers = "".join(f"<th>{title}</th>" for title, _key in METRICS)
    return f"<table><tr><th>Method</th>{headers}</tr>{rows}</table>"


def write_outputs(summary, method_rows):
    all_rows = [row for rows in method_rows.values() for row in rows]
    write_jsonl(RESULTS_FILE, all_rows)
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Hybrid + Reranker Evaluation\n",
        "- Candidate retrieval: `Dense top10 + BM25 top5 + RRF_K=60`",
        "- Rerank pool: deduplicated Dense/BM25 candidate union",
        "- Final: `top3`",
        "",
        "## Main Comparison\n",
        "| Method | Gold Hit@3 | MRR@3 | Gold Top-1 | LLM Precision@3 | Top-1 Direct | Top-1 Relevant |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in summary["comparison"].items():
        lines.append(
            f"| {label(name)} | {pct(values['gold_chunk_hit_rate_at_3'])} | "
            f"{values['mrr_at_3']:.3f} | {pct(values['gold_top1_accuracy'])} | "
            f"{pct(values['llm_precision_at_3_loose'])} | "
            f"{pct(values['llm_top1_direct_accuracy'])} | "
            f"{pct(values['llm_top1_relevant_accuracy'])} |"
        )
    REVIEW_FILE.write_text("\n".join(lines), encoding="utf-8")

    timing_rows = "".join(
        f"<tr><td>{html.escape(label(item['model'].split('/')[-1]))}</td>"
        f"<td>{item['device']}</td><td>{item['model_load_seconds']:.3f}s</td>"
        f"<td>{item['average_rerank_seconds']:.3f}s</td>"
        f"<td>{item['p95_rerank_seconds']:.3f}s</td></tr>"
        for item in summary["timing"]["rerankers"]
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reranker Evaluation</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:#172033;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}header{{background:#314b3f;color:white;padding:34px max(28px,calc((100vw - 1220px)/2))}}header h1{{margin:0 0 8px;font-size:30px}}header p{{margin:0;color:#dce9e2}}main{{max-width:1220px;margin:auto;padding:26px 28px 60px}}h2{{margin:30px 0 12px;font-size:21px}}.panel{{background:white;border:1px solid #dfe3e8;border-radius:6px;padding:18px;overflow:auto}}.note{{line-height:1.65}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{border:1px solid #dfe3e8;padding:10px;text-align:left}}th{{background:#edf2ef}}details summary{{cursor:pointer;color:#2d644c}}.question{{min-width:230px}}.evidence{{min-width:480px;margin:10px 0;padding:10px;background:#f8faf9;border-left:3px solid #3f7d60}}.evidence p{{line-height:1.5;color:#4b5563}}@media(max-width:800px){{main{{padding:18px 14px}}}}</style></head>
<body><header><h1>Hybrid RRF + Reranker</h1><p>同一候选池、同一评测集、同一 LLM-as-judge 标准</p></header><main>
<h2>主要实验</h2><section class="panel">{comparison_table(summary['comparison'])}</section>
<h2>历史上下文</h2><section class="panel note"><p>这里加入 Dense-only 和 BM25-only，便于观察完整演进；reranker 的直接 baseline 仍是 Hybrid RRF。</p>{comparison_table(summary['historical_comparison'])}</section>
<h2>耗时</h2><section class="panel"><p>Hybrid 平均召回耗时：{summary['timing']['hybrid_average_retrieval_seconds']:.3f}s。模型加载时间单独统计。</p><table><tr><th>Model</th><th>Device</th><th>Load</th><th>Avg rerank/query</th><th>P95 rerank/query</th></tr>{timing_rows}</table></section>
<h2>逐题结果</h2><section class="panel"><table><tr><th>Method</th><th>ID</th><th>问题</th><th>Gold rank</th><th>Labels</th><th>Top 3</th></tr>{question_rows_html(method_rows)}</table></section>
</main></body></html>"""
    VISUALIZATION_FILE.write_text(page, encoding="utf-8")


def question_rows_html(method_rows):
    rows = []
    for name, values in method_rows.items():
        for row in values:
            evidences = "".join(
                f"<div class='evidence'><b>#{item['rank']} {html.escape(item['document'])} chunk {item['chunk_index']}</b>"
                f"<div>label {item['llm_label']} · reranker {item.get('reranker_score', '-')} · previous RRF rank {item.get('rrf_rank', item['rank'])}</div>"
                f"<p>{html.escape(item['preview'])}</p></div>"
                for item in row["top_results"]
            )
            rows.append(
                f"<tr><td>{html.escape(label(name))}</td><td>{row['evaluation_id']}</td>"
                f"<td class='question'>{html.escape(row['question'])}</td>"
                f"<td>{row['gold_rank_top3'] or 'miss'}</td><td>{row['llm_labels_top3']}</td>"
                f"<td><details><summary>查看</summary>{evidences}</details></td></tr>"
            )
    return "".join(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_jsonl(EVALUATION_FILE)
    chunks = load_corpus()
    embedder = E5Embedder(EMBEDDING_MODEL)
    embeddings, _seconds, _cache_used = load_or_create_corpus_embeddings(
        embedder, chunks
    )
    tokenizer = JiebaTokenizer(CUSTOM_DICTIONARY)
    bm25 = BM25Index([chunk["text"] for chunk in chunks], tokenizer)
    pools = build_candidate_pools(embedder, embeddings, bm25, questions)

    method_rows = {"hybrid_rrf": baseline_rows_for_output()}
    timings = []
    for model_name in RERANKER_MODELS:
        rows, timing = rerank_model(model_name, pools, chunks)
        method_rows[method_name(model_name)] = rows
        timings.append(timing)

    reused, newly_judged, conflicts = judge_new_evidence(method_rows)
    summary = build_summary(
        method_rows,
        timings,
        {
            "reused_pairs": reused,
            "newly_judged_pairs": newly_judged,
            "cache_label_conflicts": conflicts,
        },
    )
    write_outputs(summary, method_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[Visualization] {VISUALIZATION_FILE}", flush=True)


if __name__ == "__main__":
    main()
