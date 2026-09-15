import html
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import jieba
import numpy as np

from experiments.retrieval.experiment_dense_only_evaluation import (
    EVALUATION_FILE,
    judge_evidence,
    load_corpus,
    load_deepseek_config,
    load_jsonl,
    metric_block,
    pct,
    preview,
    write_jsonl,
)


BASE_DIR = Path(__file__).resolve().parents[2]
CUSTOM_DICTIONARY = BASE_DIR / "resources/jieba_dictionary/jieba_domain_dict.txt"
OUTPUT_DIR = BASE_DIR / "evaluation/results/bm25_only"
RETRIEVAL_RESULTS = OUTPUT_DIR / "retrieval_results.jsonl"
JUDGE_RESULTS = OUTPUT_DIR / "judge_results.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"
REVIEW_FILE = OUTPUT_DIR / "review.md"
VISUALIZATION_FILE = OUTPUT_DIR / "visualization.html"
DENSE_JUDGE_CACHE = BASE_DIR / "evaluation/results/dense_only/judge_results.jsonl"

TOP_K = 3
BM25_K1 = 1.5
BM25_B = 0.75
JUDGE_SLEEP_SECONDS = 0.25


class JiebaTokenizer:
    def __init__(self, dictionary_path):
        self.tokenizer = jieba.Tokenizer()
        self.tokenizer.initialize()
        self.tokenizer.load_userdict(str(dictionary_path))

    def tokenize(self, text):
        text = re.sub(r"\[Image:\s*[^\]]+\]", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"^Section:\s*", "", text, flags=re.MULTILINE)
        tokens = []
        pattern = re.compile(
            r"\d+(?:\.\d+)?%?|[A-Za-z][A-Za-z0-9]*|[\u4e00-\u9fff]+"
        )
        for match in pattern.finditer(text.lower()):
            part = match.group(0)
            if re.fullmatch(r"\d+(?:\.\d+)?%?|[a-z][a-z0-9]*", part):
                tokens.append(part)
            else:
                tokens.extend(
                    token.strip()
                    for token in self.tokenizer.cut(part, HMM=False)
                    if token.strip()
                )
        return tokens


class BM25Index:
    def __init__(self, documents, tokenizer, k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.tokenized_documents = [tokenizer.tokenize(text) for text in documents]
        self.document_lengths = np.array(
            [len(tokens) for tokens in self.tokenized_documents],
            dtype=np.float32,
        )
        self.average_document_length = float(self.document_lengths.mean())
        self.document_count = len(documents)
        self.postings = defaultdict(list)
        for document_index, tokens in enumerate(self.tokenized_documents):
            for term, frequency in Counter(tokens).items():
                self.postings[term].append((document_index, frequency))

    @property
    def vocabulary_size(self):
        return len(self.postings)

    def idf(self, term):
        document_frequency = len(self.postings.get(term, []))
        return math.log(
            1
            + (self.document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

    def score(self, query):
        query_tokens = self.tokenizer.tokenize(query)
        scores = np.zeros(self.document_count, dtype=np.float32)
        for term in dict.fromkeys(query_tokens):
            idf = self.idf(term)
            for document_index, frequency in self.postings.get(term, []):
                document_length = self.document_lengths[document_index]
                denominator = frequency + self.k1 * (
                    1
                    - self.b
                    + self.b * document_length / self.average_document_length
                )
                scores[document_index] += (
                    idf * frequency * (self.k1 + 1) / denominator
                )
        return scores, query_tokens


def run_retrieval(index, chunks, questions):
    rows = []
    retrieval_times = []
    for position, question in enumerate(questions, start=1):
        started = time.perf_counter()
        scores, query_tokens = index.score(question["question"])
        top_indices = np.argsort(-scores, kind="stable")[:TOP_K]
        elapsed = time.perf_counter() - started
        retrieval_times.append(elapsed)

        results = []
        gold_rank = None
        query_token_set = set(query_tokens)
        for rank, chunk_position in enumerate(top_indices, start=1):
            chunk_position = int(chunk_position)
            chunk = chunks[chunk_position]
            if chunk["global_chunk_id"] == question["relevant_chunk_id"]:
                gold_rank = rank
            results.append(
                {
                    "rank": rank,
                    "bm25_score": round(float(scores[chunk_position]), 6),
                    "global_chunk_id": chunk["global_chunk_id"],
                    "document": chunk["document"],
                    "chunk_index": chunk["chunk_index"],
                    "section": chunk.get("section", ""),
                    "token_count": chunk.get("token_count"),
                    "matched_query_tokens": sorted(
                        query_token_set
                        & set(index.tokenized_documents[chunk_position])
                    ),
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
                "retrieval_seconds": round(elapsed, 6),
                "top_results": results,
            }
        )
        print(
            f"[Retrieve] {position}/{len(questions)} {question['evaluation_id']} "
            f"gold_rank={gold_rank} tokens={query_tokens}",
            flush=True,
        )
    return rows, sum(retrieval_times) / len(retrieval_times)


def load_judge_cache(path):
    if not path.exists():
        return {}
    return {
        (row["evaluation_id"], row["global_chunk_id"]): row
        for row in load_jsonl(path)
    }


def run_judge(rows):
    own_cache = load_judge_cache(JUDGE_RESULTS)
    dense_cache = load_judge_cache(DENSE_JUDGE_CACHE)
    api_key, chat_url = load_deepseek_config()
    reused = 0
    newly_judged = 0
    total = len(rows) * TOP_K
    current = 0

    for row in rows:
        for result in row["top_results"]:
            current += 1
            key = (row["evaluation_id"], result["global_chunk_id"])
            if key in own_cache:
                print(f"[Judge] {current}/{total} skip own cache", flush=True)
                continue

            if key in dense_cache:
                source = dense_cache[key]
                judged = {
                    "evaluation_id": row["evaluation_id"],
                    "global_chunk_id": result["global_chunk_id"],
                    "rank": result["rank"],
                    "label": source["label"],
                    "reason": source["reason"],
                    "cache_source": "dense_only",
                }
                reused += 1
                print(f"[Judge] {current}/{total} reuse dense cache", flush=True)
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
                            "cache_source": "bm25_only_new",
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
    return own_cache, reused, newly_judged


def add_judge_results(rows, judge_cache):
    for row in rows:
        labels = []
        for result in row["top_results"]:
            judged = judge_cache[(row["evaluation_id"], result["global_chunk_id"])]
            result["llm_label"] = judged["label"]
            result["llm_reason"] = judged["reason"]
            labels.append(judged["label"])
        row["llm_labels_top3"] = labels
        row["llm_precision_at_3_loose"] = round(
            sum(label in (1, 2) for label in labels) / TOP_K,
            6,
        )
        row["llm_precision_at_3_strict"] = round(
            sum(label == 2 for label in labels) / TOP_K,
            6,
        )
        row["llm_top1_direct"] = labels[0] == 2
        row["llm_top1_relevant"] = labels[0] in (1, 2)
    return rows


def summarize(rows, index, index_seconds, average_seconds, reused, new):
    by_type = defaultdict(list)
    by_document = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)
        by_document[row["gold_document"]].append(row)
    return {
        "experiment": "bm25_only_top3",
        "tokenizer": "jieba precise mode, HMM=False",
        "custom_dictionary": str(CUSTOM_DICTIONARY),
        "custom_dictionary_terms": sum(
            bool(line.strip())
            for line in CUSTOM_DICTIONARY.read_text(encoding="utf-8").splitlines()
        ),
        "bm25_parameters": {"k1": BM25_K1, "b": BM25_B},
        "top_k": TOP_K,
        "corpus_chunks": index.document_count,
        "vocabulary_size": index.vocabulary_size,
        "average_document_tokens": round(index.average_document_length, 6),
        "index_build_seconds": round(index_seconds, 6),
        "average_retrieval_seconds": round(average_seconds, 6),
        "overall": metric_block(rows),
        "by_question_type": {
            key: metric_block(value) for key, value in sorted(by_type.items())
        },
        "by_source_document": {
            key: metric_block(value) for key, value in sorted(by_document.items())
        },
        "llm_judge": {
            "reused_from_dense": reused,
            "newly_judged": new,
            "loose_precision_rule": "labels 1 and 2 count as relevant",
            "strict_precision_rule": "only label 2 counts as directly answerable",
        },
        "gold_label_note": (
            "Each question has one annotated source chunk, so exact gold Recall@3 "
            "equals Gold Chunk HitRate@3."
        ),
    }


def write_review(summary, rows):
    overall = summary["overall"]
    lines = [
        "# BM25-only Retrieval Evaluation\n",
        "- Tokenizer: `jieba`, precise mode, `HMM=False`",
        f"- Custom dictionary terms: `{summary['custom_dictionary_terms']}`",
        f"- BM25: `k1={BM25_K1}`, `b={BM25_B}`",
        f"- Corpus chunks: `{summary['corpus_chunks']}`",
        f"- Final Top K: `{TOP_K}`",
        "",
        "## Overall Metrics\n",
        f"- Gold Chunk HitRate@3: `{pct(overall['gold_chunk_hit_rate_at_3'])}`",
        f"- MRR@3: `{overall['mrr_at_3']:.3f}`",
        f"- Gold Top-1 Accuracy: `{pct(overall['gold_top1_accuracy'])}`",
        f"- LLM Precision@3 (loose): `{pct(overall['llm_precision_at_3_loose'])}`",
        f"- LLM Precision@3 (strict): `{pct(overall['llm_precision_at_3_strict'])}`",
        f"- LLM Top-1 Direct Accuracy: `{pct(overall['llm_top1_direct_accuracy'])}`",
        f"- Average retrieval time: `{summary['average_retrieval_seconds']:.6f}s`",
        "",
        "## Exact Gold Misses\n",
    ]
    misses = [row for row in rows if not row["gold_hit_at_3"]]
    if not misses:
        lines.append("- None")
    for row in misses:
        lines.extend(
            [
                f"### {row['evaluation_id']} {row['question']}",
                f"- Query tokens: `{row['query_tokens']}`",
                f"- Gold: `{row['gold_chunk_id']}`",
                f"- LLM labels: `{row['llm_labels_top3']}`",
                "",
            ]
        )
    REVIEW_FILE.write_text("\n".join(lines), encoding="utf-8")


def write_visualization(summary, rows):
    overall = summary["overall"]
    cards = [
        ("Gold HitRate@3", pct(overall["gold_chunk_hit_rate_at_3"])),
        ("MRR@3", f"{overall['mrr_at_3']:.3f}"),
        ("LLM Precision@3", pct(overall["llm_precision_at_3_loose"])),
        ("Gold Top-1", pct(overall["gold_top1_accuracy"])),
        ("LLM Top-1 Direct", pct(overall["llm_top1_direct_accuracy"])),
        ("平均检索耗时", f"{summary['average_retrieval_seconds']:.4f}s"),
    ]
    cards_html = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    type_rows = "".join(
        "<tr>"
        f"<td>{html.escape(kind)}</td><td>{metrics['questions']}</td>"
        f"<td>{pct(metrics['gold_chunk_hit_rate_at_3'])}</td>"
        f"<td>{metrics['mrr_at_3']:.3f}</td>"
        f"<td>{pct(metrics['llm_precision_at_3_loose'])}</td>"
        f"<td>{pct(metrics['llm_top1_direct_accuracy'])}</td></tr>"
        for kind, metrics in summary["by_question_type"].items()
    )
    question_rows = "".join(build_question_html(row) for row in rows)
    document_rows = "".join(
        "<tr>"
        f"<td>{html.escape(document)}</td><td>{metrics['questions']}</td>"
        f"<td>{pct(metrics['gold_chunk_hit_rate_at_3'])}</td>"
        f"<td>{metrics['mrr_at_3']:.3f}</td>"
        f"<td>{pct(metrics['llm_precision_at_3_loose'])}</td></tr>"
        for document, metrics in summary["by_source_document"].items()
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BM25-only Evaluation</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:#172033;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}
header{{background:#173f35;color:white;padding:34px max(28px,calc((100vw - 1220px)/2))}}header h1{{margin:0 0 8px;font-size:30px}}header p{{margin:0;color:#d7eee5}}
main{{max-width:1220px;margin:auto;padding:26px 28px 60px}}h2{{margin:32px 0 14px;font-size:21px}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.metric,.panel{{background:white;border:1px solid #dfe3e8;border-radius:6px;padding:18px}}.metric span{{display:block;color:#667085}}.metric strong{{display:block;font-size:30px;margin-top:10px}}
.panel{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{border:1px solid #dfe3e8;padding:10px;vertical-align:top;text-align:left}}th{{background:#edf2f7}}td.question{{min-width:240px}}details summary{{cursor:pointer;color:#167d5a}}
.evidence{{min-width:470px;margin:12px 0;padding:11px;background:#f7faf9;border-left:3px solid #167d5a}}.evidence p{{color:#475467;line-height:1.55}}
@media(max-width:800px){{.metrics{{grid-template-columns:1fr}}main{{padding:18px 14px}}}}
</style></head><body><header><h1>BM25-only Retrieval Evaluation</h1><p>jieba precise mode · 58-term custom dictionary · BM25 k1={BM25_K1}, b={BM25_B} · Final Top 3</p></header>
<main><section class="metrics">{cards_html}</section>
<h2>实验配置</h2><section class="panel">语料 {summary['corpus_chunks']} chunks · 词表 {summary['vocabulary_size']} tokens · 平均 chunk {summary['average_document_tokens']:.1f} tokens · 建索引 {summary['index_build_seconds']:.3f}s</section>
<h2>按问题类型</h2><section class="panel"><table><tr><th>类型</th><th>问题数</th><th>Gold Hit@3</th><th>MRR@3</th><th>LLM Precision@3</th><th>Top-1 direct</th></tr>{type_rows}</table></section>
<h2>按来源报告</h2><section class="panel"><table><tr><th>报告</th><th>问题数</th><th>Gold Hit@3</th><th>MRR@3</th><th>LLM Precision@3</th></tr>{document_rows}</table></section>
<h2>逐题分词与 Evidence</h2><section class="panel"><table><tr><th>ID</th><th>问题</th><th>Jieba tokens</th><th>Gold rank</th><th>LLM labels</th><th>Top 3</th></tr>{question_rows}</table></section>
</main></body></html>"""
    VISUALIZATION_FILE.write_text(page, encoding="utf-8")


def build_question_html(row):
    evidence = "".join(
        f"<div class='evidence'><b>#{item['rank']} · {html.escape(item['document'])} · "
        f"chunk {item['chunk_index']} · BM25 {item['bm25_score']:.3f}</b>"
        f"<div>匹配词：{html.escape(' / '.join(item['matched_query_tokens']))}</div>"
        f"<p>{html.escape(item['preview'])}</p></div>"
        for item in row["top_results"]
    )
    return (
        "<tr>"
        f"<td>{row['evaluation_id']}</td>"
        f"<td class='question'>{html.escape(row['question'])}</td>"
        f"<td>{html.escape(' / '.join(row['query_tokens']))}</td>"
        f"<td>{row['gold_rank_top3'] or 'miss'}</td>"
        f"<td>{html.escape(str(row['llm_labels_top3']))}</td>"
        f"<td><details><summary>查看 Top 3</summary>{evidence}</details></td>"
        "</tr>"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_jsonl(EVALUATION_FILE)
    chunks = load_corpus()
    tokenizer = JiebaTokenizer(CUSTOM_DICTIONARY)

    started = time.perf_counter()
    index = BM25Index([chunk["text"] for chunk in chunks], tokenizer)
    index_seconds = time.perf_counter() - started
    rows, average_seconds = run_retrieval(index, chunks, questions)
    write_jsonl(RETRIEVAL_RESULTS, rows)

    judge_cache, reused, new = run_judge(rows)
    rows = add_judge_results(rows, judge_cache)
    write_jsonl(RETRIEVAL_RESULTS, rows)
    summary = summarize(rows, index, index_seconds, average_seconds, reused, new)
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
