import json
import os
import re
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = (
    BASE_DIR
    / "artifacts/chunking/2025年品牌营销趋势报告/hybrid_grid_search_eval"
)
CHUNKS_PATH = (
    BASE_DIR
    / "artifacts/chunking/2025年品牌营销趋势报告/"
    "cleaned_text_structure_semantic_boundary_500_min_150/chunks.jsonl"
)
ANNOTATION_MD = EVAL_DIR / "manual_precision_annotation.md"
OUTPUT_JSONL = EVAL_DIR / "llm_judge_precision_results.jsonl"
OUTPUT_MD = EVAL_DIR / "llm_judge_precision_review.md"
OUTPUT_SUMMARY = EVAL_DIR / "llm_judge_precision_summary.json"

MODEL_NAME = "deepseek-chat"
REQUEST_TIMEOUT = 60
SLEEP_SECONDS = 0.4


def load_deepseek_config():
    env_path = BASE_DIR / ".env"
    if load_dotenv:
        load_dotenv(env_path, override=True)
    elif env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("Missing DEEPSEEK_API_KEY or DEEPSEEK_BASE_URL in .env")
    return api_key, base_url.rstrip("/") + "/chat/completions"


def parse_manual_annotation(path: Path):
    text = path.read_text(encoding="utf-8")
    question_blocks = re.split(r"\n(?=## Q\d+ )", text)
    rows = []

    for block in question_blocks:
        question_match = re.match(r"## (Q\d+) (.+?)\n", block)
        if not question_match:
            continue

        question_id = question_match.group(1)
        question = question_match.group(2).strip()
        rank_blocks = re.split(r"\n(?=### Rank )", block)

        for rank_block in rank_blocks:
            rank_match = re.match(
                r"### Rank (\d+) \| chunk (\d+) \| section: (.*?)\n",
                rank_block,
            )
            if not rank_match:
                continue

            label_match = re.search(
                r"- Relevance label:\s*`?\s*([012])\s*`?",
                rank_block,
            )
            reason_match = re.search(r"- Reason:\s*(.*)", rank_block)
            preview_match = re.search(r">\s*(.*)", rank_block, flags=re.DOTALL)

            rows.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "rank": int(rank_match.group(1)),
                    "chunk_index": int(rank_match.group(2)),
                    "section": rank_match.group(3).strip(),
                    "human_label": int(label_match.group(1))
                    if label_match
                    else None,
                    "human_reason": reason_match.group(1).strip()
                    if reason_match
                    else "",
                    "evidence": preview_match.group(1).strip()
                    if preview_match
                    else "",
                }
            )

    return rows


def load_full_chunks(path: Path):
    chunks_by_index = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunks_by_index[int(chunk["chunk_index"])] = chunk
    return chunks_by_index


def attach_full_evidence(rows, chunks_by_index):
    enriched_rows = []
    for row in rows:
        chunk = chunks_by_index.get(row["chunk_index"])
        if chunk:
            row = {
                **row,
                "evidence": chunk["text"],
                "evidence_source": "full_chunk_text",
            }
        else:
            row = {**row, "evidence_source": "manual_preview_fallback"}
        enriched_rows.append(row)
    return enriched_rows


def extract_json(content: str):
    content = content.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced_match:
        content = fenced_match.group(1)
    else:
        object_match = re.search(r"\{.*\}", content, re.DOTALL)
        if object_match:
            content = object_match.group(0)
    return json.loads(content)


def judge_one(api_key: str, chat_url: str, row):
    system_prompt = (
        "You are an evaluator for a RAG retrieval experiment. "
        "Judge whether a retrieved evidence chunk is relevant to the question. "
        "Return only valid JSON."
    )
    user_prompt = f"""
请根据问题判断 evidence chunk 的相关性。

评分标准：
2 = 完全相关：chunk 可以直接回答问题，或包含标准答案所需的核心证据。
1 = 部分相关：chunk 与问题主题相关，可作为背景，但不能单独回答问题。
0 = 不相关：chunk 与问题无关，或属于明显噪声。

注意：
- 只评估 evidence 是否相关，不要根据措辞是否优美打分。
- 如果问题是数字事实题，只有包含关键数字或明确支持答案的证据才应给 2。
- 如果 evidence 只是同一大主题但不能回答问题，给 1。
- 如果 evidence 没有帮助回答问题，给 0。

Question:
{row["question"]}

Evidence section:
{row["section"]}

Evidence chunk:
{row["evidence"]}

请只返回 JSON：
{{
  "label": 0 或 1 或 2,
  "reason": "一句中文理由"
}}
""".strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "stream": False,
    }
    response = requests.post(
        chat_url,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    parsed = extract_json(content)
    label = int(parsed["label"])
    if label not in (0, 1, 2):
        raise ValueError(f"Invalid judge label: {label}")
    return {
        "llm_label": label,
        "llm_reason": str(parsed.get("reason", "")).strip(),
        "raw_response": content,
    }


def precision(labels, strict=False):
    if strict:
        relevant = sum(1 for label in labels if label == 2)
    else:
        relevant = sum(1 for label in labels if label in (1, 2))
    return relevant / len(labels)


def summarize(rows):
    by_question = {}
    for row in rows:
        by_question.setdefault(row["question_id"], []).append(row)

    question_summaries = []
    agreements = 0
    for question_id in sorted(by_question, key=lambda item: int(item[1:])):
        items = sorted(by_question[question_id], key=lambda item: item["rank"])
        human_labels = [item["human_label"] for item in items]
        llm_labels = [item["llm_label"] for item in items]
        agreements += sum(
            1
            for human_label, llm_label in zip(human_labels, llm_labels)
            if human_label == llm_label
        )
        question_summaries.append(
            {
                "question_id": question_id,
                "question": items[0]["question"],
                "human_labels": human_labels,
                "llm_labels": llm_labels,
                "human_precision_at_3_loose": round(
                    precision(human_labels, strict=False),
                    4,
                ),
                "human_precision_at_3_strict": round(
                    precision(human_labels, strict=True),
                    4,
                ),
                "llm_precision_at_3_loose": round(
                    precision(llm_labels, strict=False),
                    4,
                ),
                "llm_precision_at_3_strict": round(
                    precision(llm_labels, strict=True),
                    4,
                ),
            }
        )

    total = len(rows)
    return {
        "items": total,
        "exact_agreement": round(agreements / total, 4) if total else 0,
        "human_macro_precision_at_3_loose": round(
            sum(item["human_precision_at_3_loose"] for item in question_summaries)
            / len(question_summaries),
            4,
        ),
        "human_macro_precision_at_3_strict": round(
            sum(item["human_precision_at_3_strict"] for item in question_summaries)
            / len(question_summaries),
            4,
        ),
        "llm_macro_precision_at_3_loose": round(
            sum(item["llm_precision_at_3_loose"] for item in question_summaries)
            / len(question_summaries),
            4,
        ),
        "llm_macro_precision_at_3_strict": round(
            sum(item["llm_precision_at_3_strict"] for item in question_summaries)
            / len(question_summaries),
            4,
        ),
        "by_question": question_summaries,
    }


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review(path: Path, summary, rows):
    lines = [
        "# LLM-as-Judge Precision@3 Review\n",
        f"- Judge model: `{MODEL_NAME}`",
        "- Retrieval setting: `Dense top10 + BM25 top5 + RRF_K=60 + Final top3`",
        "- Label rule: `2 = directly relevant`, `1 = partially relevant`, `0 = irrelevant/noise`",
        "",
        "## Summary\n",
        f"- Exact agreement with human labels: `{summary['exact_agreement']:.3f}`",
        f"- Human loose Precision@3: `{summary['human_macro_precision_at_3_loose']:.3f}`",
        f"- Human strict Precision@3: `{summary['human_macro_precision_at_3_strict']:.3f}`",
        f"- LLM loose Precision@3: `{summary['llm_macro_precision_at_3_loose']:.3f}`",
        f"- LLM strict Precision@3: `{summary['llm_macro_precision_at_3_strict']:.3f}`",
        "",
        "## By Question\n",
        "| Question | Human labels | LLM labels | Human loose | LLM loose | Human strict | LLM strict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["by_question"]:
        lines.append(
            "| {qid} | {human} | {llm} | {h_loose:.3f} | {l_loose:.3f} | "
            "{h_strict:.3f} | {l_strict:.3f} |".format(
                qid=item["question_id"],
                human=item["human_labels"],
                llm=item["llm_labels"],
                h_loose=item["human_precision_at_3_loose"],
                l_loose=item["llm_precision_at_3_loose"],
                h_strict=item["human_precision_at_3_strict"],
                l_strict=item["llm_precision_at_3_strict"],
            )
        )

    lines.extend(["", "## Disagreements\n"])
    disagreements = [
        row for row in rows if row["human_label"] != row["llm_label"]
    ]
    if not disagreements:
        lines.append("- No disagreements.")
    for row in disagreements:
        lines.extend(
            [
                (
                    f"### {row['question_id']} Rank {row['rank']} "
                    f"chunk {row['chunk_index']}"
                ),
                f"- Human: `{row['human_label']}` - {row['human_reason']}",
                f"- LLM: `{row['llm_label']}` - {row['llm_reason']}",
                f"- Section: `{row['section']}`",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    api_key, chat_url = load_deepseek_config()
    chunks_by_index = load_full_chunks(CHUNKS_PATH)
    rows = attach_full_evidence(
        parse_manual_annotation(ANNOTATION_MD),
        chunks_by_index,
    )
    judged_rows = []

    for index, row in enumerate(rows, start=1):
        print(
            f"[Judge] {index}/{len(rows)} "
            f"{row['question_id']} rank {row['rank']} chunk {row['chunk_index']}"
        )
        judge_result = judge_one(api_key, chat_url, row)
        judged_rows.append({**row, **judge_result})
        time.sleep(SLEEP_SECONDS)

    summary = summarize(judged_rows)
    write_jsonl(OUTPUT_JSONL, judged_rows)
    OUTPUT_SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_review(OUTPUT_MD, summary, judged_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
