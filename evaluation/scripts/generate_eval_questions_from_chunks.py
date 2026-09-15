import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parents[2]
CHUNK_ROOT = BASE_DIR / "artifacts/chunking"
BASELINE_DIR_NAME = "cleaned_text_structure_semantic_boundary_500_min_150"
OUTPUT_DIR = BASE_DIR / "evaluation/candidates"
OUTPUT_JSONL = OUTPUT_DIR / "candidate_questions.jsonl"
OUTPUT_MD = OUTPUT_DIR / "candidate_questions_review.md"
OUTPUT_SAMPLE_JSON = OUTPUT_DIR / "sampled_chunks.json"

MODEL_NAME = "deepseek-chat"
REQUEST_TIMEOUT = 90
SLEEP_SECONDS = 0.4
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 20260829


def load_env():
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


def load_chunks():
    chunks = []
    for path in sorted(CHUNK_ROOT.glob(f"*/{BASELINE_DIR_NAME}/chunks.jsonl")):
        document_name = path.parent.parent.name
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                chunk["document"] = document_name
                chunk["chunk_file"] = str(path)
                chunks.append(chunk)
    return chunks


def chinese_char_count(text):
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def image_placeholder_count(text):
    return len(re.findall(r"\[Image:\s*images/.*?\]", text, flags=re.DOTALL))


def is_useful_for_question_generation(chunk):
    text = chunk.get("text", "")
    if chunk.get("token_count", 0) < 80:
        return False
    if chinese_char_count(text) < 60:
        return False
    if image_placeholder_count(text) >= 2 and chinese_char_count(text) < 120:
        return False
    return True


def sample_chunks(chunks, sample_size, seed):
    eligible = [chunk for chunk in chunks if is_useful_for_question_generation(chunk)]
    if len(eligible) < sample_size:
        raise RuntimeError(
            f"Only {len(eligible)} useful chunks found, cannot sample {sample_size}."
        )
    rng = random.Random(seed)
    sampled = rng.sample(eligible, sample_size)
    sampled.sort(key=lambda item: (item["document"], item["chunk_index"]))
    return sampled, eligible


def trim_chunk_text(text, max_chars=2600):
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def extract_json_array(content):
    content = content.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
    if fenced_match:
        content = fenced_match.group(1)
    else:
        array_match = re.search(r"\[.*\]", content, re.DOTALL)
        if array_match:
            content = array_match.group(0)
    return json.loads(content)


def call_deepseek(api_key, chat_url, chunk):
    system_prompt = (
        "You generate evaluation questions for a Chinese RAG retrieval dataset. "
        "Questions must be answerable from the given evidence chunk. "
        "Return only valid JSON."
    )
    user_prompt = f"""
请基于下面这个 knowledge chunk 生成 2-3 个中文 RAG 检索评测问题。

要求：
- 问题必须能从该 chunk 中找到直接证据。
- 优先生成真实用户会问的问题，不要照抄原文标题。
- 尽量覆盖不同类型：事实型、数字型、解释型、对比型。
- 如果 chunk 信息量不足，只生成 2 个问题。
- 标准答案必须严格基于 chunk，不要补充外部知识。
- keywords 写 2-5 个关键词，方便后续人工检查。

Document:
{chunk["document"]}

Section:
{chunk.get("section", "")}

Chunk index:
{chunk.get("chunk_index")}

Chunk text:
{trim_chunk_text(chunk.get("text", ""))}

请只返回 JSON array，格式如下：
[
  {{
    "question": "问题",
    "answer": "标准答案",
    "answer_section": "答案所在 section",
    "keywords": ["关键词1", "关键词2"],
    "question_type": "fact|number|explain|compare",
    "difficulty": "easy|medium|hard"
  }}
]
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
        "temperature": 0.2,
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
    questions = extract_json_array(content)
    if not isinstance(questions, list):
        raise ValueError("DeepSeek response is not a JSON array.")
    return questions[:3], content


def existing_chunk_keys(path):
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((row["document"], row["chunk_index"]))
    return keys


def append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review_markdown(jsonl_path, md_path):
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    parts = [
        "# RAG Evaluation Question Candidates\n",
        "说明：每个问题都由 DeepSeek 基于随机抽样 chunk 生成。人工筛选时，建议保留“问题清楚、答案可由证据直接支持、不是过度细节”的条目。\n",
    ]

    for row in rows:
        parts.append(
            f"\n\n## {row['candidate_id']} | Keep: [ ]\n"
            f"- Document: {row['document']}\n"
            f"- Chunk index: {row['chunk_index']}\n"
            f"- Section: {row['section']}\n"
            f"- Type: {row['question_type']}\n"
            f"- Difficulty: {row['difficulty']}\n"
            f"- Keywords: {', '.join(row['keywords'])}\n\n"
            f"**Question:** {row['question']}\n\n"
            f"**Standard answer:** {row['answer']}\n\n"
            f"**Evidence preview:**\n\n"
            f"> {row['evidence_preview'].replace(chr(10), chr(10) + '> ')}\n"
        )

    md_path.write_text("".join(parts).strip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Sample chunks and generate candidate RAG evaluation questions."
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        OUTPUT_JSONL.unlink(missing_ok=True)
        OUTPUT_MD.unlink(missing_ok=True)
        OUTPUT_SAMPLE_JSON.unlink(missing_ok=True)

    chunks = load_chunks()
    sampled, eligible = sample_chunks(chunks, args.sample_size, args.seed)
    OUTPUT_SAMPLE_JSON.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "sample_size": args.sample_size,
                "total_chunks": len(chunks),
                "eligible_chunks": len(eligible),
                "sampled_chunks": [
                    {
                        "document": chunk["document"],
                        "chunk_index": chunk["chunk_index"],
                        "section": chunk.get("section", ""),
                        "token_count": chunk.get("token_count"),
                        "chunk_file": chunk.get("chunk_file"),
                    }
                    for chunk in sampled
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    api_key, chat_url = load_env()
    done_keys = existing_chunk_keys(OUTPUT_JSONL)
    generated_question_count = 0

    for sample_index, chunk in enumerate(sampled, start=1):
        chunk_key = (chunk["document"], chunk["chunk_index"])
        if chunk_key in done_keys:
            print(
                f"[{sample_index}/{len(sampled)}] skip existing "
                f"{chunk['document']} chunk {chunk['chunk_index']}",
                flush=True,
            )
            continue

        print(
            f"[{sample_index}/{len(sampled)}] generating "
            f"{chunk['document']} chunk {chunk['chunk_index']}",
            flush=True,
        )

        last_error = None
        for attempt in range(1, 4):
            try:
                questions, raw_response = call_deepseek(api_key, chat_url, chunk)
                break
            except Exception as exc:
                last_error = exc
                print(f"  attempt {attempt} failed: {exc}", flush=True)
                time.sleep(1.5 * attempt)
        else:
            append_jsonl(
                OUTPUT_JSONL,
                [
                    {
                        "candidate_id": f"C{sample_index:03d}-ERR",
                        "document": chunk["document"],
                        "chunk_index": chunk["chunk_index"],
                        "section": chunk.get("section", ""),
                        "status": "generation_failed",
                        "error": str(last_error),
                        "evidence_preview": trim_chunk_text(chunk.get("text", ""), 600),
                    }
                ],
            )
            continue

        rows = []
        for question_index, item in enumerate(questions, start=1):
            rows.append(
                {
                    "candidate_id": f"C{sample_index:03d}-Q{question_index}",
                    "document": chunk["document"],
                    "chunk_index": chunk["chunk_index"],
                    "section": chunk.get("section", ""),
                    "token_count": chunk.get("token_count"),
                    "question": str(item.get("question", "")).strip(),
                    "answer": str(item.get("answer", "")).strip(),
                    "answer_section": str(
                        item.get("answer_section") or chunk.get("section", "")
                    ).strip(),
                    "keywords": item.get("keywords", []),
                    "question_type": str(item.get("question_type", "")).strip(),
                    "difficulty": str(item.get("difficulty", "")).strip(),
                    "evidence_preview": trim_chunk_text(chunk.get("text", ""), 900),
                    "raw_response": raw_response,
                    "status": "candidate",
                }
            )

        append_jsonl(OUTPUT_JSONL, rows)
        generated_question_count += len(rows)
        time.sleep(SLEEP_SECONDS)

    write_review_markdown(OUTPUT_JSONL, OUTPUT_MD)
    print(f"[Done] generated_new_questions={generated_question_count}")
    print(f"[Output] {OUTPUT_JSONL}")
    print(f"[Review] {OUTPUT_MD}")
    print(f"[Sample] {OUTPUT_SAMPLE_JSON}")


if __name__ == "__main__":
    main()
