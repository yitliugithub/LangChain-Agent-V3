import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODELS = [
    {
        "label": "baseline_miniLM",
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    {
        "label": "mpnet_base",
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    {
        "label": "e5_base",
        "name": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    {
        "label": "bge_m3",
        "name": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
    },
]

CHUNKS_PATH = Path(
    "artifacts/chunking/2025年品牌营销趋势报告/"
    "cleaned_text_structure_semantic_boundary_500_min_150/chunks.jsonl"
)
OUTPUT_PATH = Path(
    "artifacts/chunking/2025年品牌营销趋势报告/"
    "embedding_model_eval/embedding_model_benchmark.json"
)
QUESTIONS = [
    "广告主拓展新消费者的策略是什么？",
    "不同类别广告主有什么类？",
    "实际的营销实践中，“扩张”怎么体现？",
    "有多少游戏品牌广告主已经布局了小程序？",
    "食品饮料/美妆护肤/服装服饰三个行业No.1的触达率是多少？",
]
BATCH_SIZE = 8


def load_chunk_texts(path: Path):
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(json.loads(line)["text"])
    return chunks


def encode_texts(model, tokenizer, texts):
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            output = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (output.last_hidden_state * mask).sum(dim=1)
            pooled = pooled / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)


def benchmark_model(model_config, chunk_texts):
    print(f"[Benchmark] Loading {model_config['name']}...", flush=True)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    model = AutoModel.from_pretrained(model_config["name"])
    model.eval()
    load_seconds = time.perf_counter() - load_start

    passage_texts = [
        model_config["passage_prefix"] + text
        for text in chunk_texts
    ]
    chunk_start = time.perf_counter()
    chunk_embeddings = encode_texts(model, tokenizer, passage_texts)
    chunk_seconds = time.perf_counter() - chunk_start

    query_texts = [
        model_config["query_prefix"] + question
        for question in QUESTIONS
    ]
    query_start = time.perf_counter()
    query_embeddings = encode_texts(model, tokenizer, query_texts)
    query_seconds = time.perf_counter() - query_start

    return {
        "label": model_config["label"],
        "model": model_config["name"],
        "embedding_dim": int(chunk_embeddings.shape[1]),
        "chunk_count": len(chunk_texts),
        "query_count": len(QUESTIONS),
        "batch_size": BATCH_SIZE,
        "load_seconds": round(load_seconds, 3),
        "chunk_embedding_seconds": round(chunk_seconds, 3),
        "chunks_per_second": round(len(chunk_texts) / chunk_seconds, 3),
        "avg_ms_per_chunk": round(chunk_seconds / len(chunk_texts) * 1000, 2),
        "query_embedding_seconds": round(query_seconds, 3),
        "queries_per_second": round(len(QUESTIONS) / query_seconds, 3),
        "avg_ms_per_query": round(query_seconds / len(QUESTIONS) * 1000, 2),
        "query_embedding_shape": list(query_embeddings.shape),
    }


def main():
    chunk_texts = load_chunk_texts(CHUNKS_PATH)
    results = []

    for model_config in MODELS:
        results.append(benchmark_model(model_config, chunk_texts))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"output": str(OUTPUT_PATH), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
