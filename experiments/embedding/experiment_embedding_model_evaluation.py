import argparse
import json
import re
from pathlib import Path

import numpy as np

from experiments.retrieval.experiment_retrieval_evaluation import QUESTIONS, keyword_coverage, load_chunks


CHUNKS_PATH = Path(
    "artifacts/chunking/2025年品牌营销趋势报告/"
    "cleaned_text_structure_semantic_boundary_500_min_150/chunks.jsonl"
)
OUTPUT_DIR = Path("artifacts/chunking/2025年品牌营销趋势报告/embedding_model_eval")
TOP_K = 3


MODEL_CONFIGS = [
    {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "label": "baseline_miniLM",
        "query_prefix": "",
        "passage_prefix": "",
    },
    {
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "label": "mpnet_base",
        "query_prefix": "",
        "passage_prefix": "",
    },
    {
        "name": "intfloat/multilingual-e5-base",
        "label": "e5_base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    {
        "name": "BAAI/bge-m3",
        "label": "bge_m3",
        "query_prefix": "",
        "passage_prefix": "",
    },
]


def preview_text(text: str, limit: int = 420):
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


class SentenceTransformersEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class TransformersMeanPoolingEmbedder:
    """Fallback used when sentence-transformers is not installed."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts):
        embeddings = []
        batch_size = 8

        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                masked_embeddings = token_embeddings * attention_mask
                summed = masked_embeddings.sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1)
                batch_embeddings = summed / counts
                batch_embeddings = self.torch.nn.functional.normalize(
                    batch_embeddings,
                    p=2,
                    dim=1,
                )
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)


def get_embedder(model_name: str):
    try:
        return SentenceTransformersEmbedder(model_name), "sentence_transformers"
    except ModuleNotFoundError:
        return TransformersMeanPoolingEmbedder(model_name), "transformers_mean_pooling"


def apply_prefix(texts, prefix: str):
    if not prefix:
        return texts
    return [prefix + text for text in texts]


def evaluate_model(model_config, chunks):
    chunk_texts = [chunk["text"] for chunk in chunks]
    prefixed_chunk_texts = apply_prefix(chunk_texts, model_config["passage_prefix"])
    embedder, backend = get_embedder(model_config["name"])
    chunk_embeddings = embedder.encode(prefixed_chunk_texts)
    rows = []

    for question in QUESTIONS:
        query_text = model_config["query_prefix"] + question["question"]
        query_embedding = embedder.encode([query_text])[0]
        scores = np.dot(chunk_embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:TOP_K]

        top_results = []
        combined_top_text = ""
        for rank, chunk_index in enumerate(top_indices, start=1):
            chunk = chunks[int(chunk_index)]
            combined_top_text += "\n" + chunk["text"]
            top_results.append(
                {
                    "rank": rank,
                    "score": float(scores[int(chunk_index)]),
                    "chunk_index": chunk.get("chunk_index"),
                    "section": chunk.get("section", ""),
                    "token_count": chunk.get("token_count"),
                    "block_types": chunk.get("block_types", []),
                    "keyword_coverage": keyword_coverage(
                        chunk["text"],
                        question["expected_keywords"],
                    ),
                    "preview": preview_text(chunk["text"]),
                }
            )

        top3_coverage = keyword_coverage(combined_top_text, question["expected_keywords"])
        expected_section_hit = any(
            question["section"] in result["section"]
            or question["section"] in result["preview"]
            for result in top_results
        )

        rows.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "model": model_config["name"],
                "model_label": model_config["label"],
                "backend": backend,
                "top1_score": top_results[0]["score"],
                "top1_section": top_results[0]["section"],
                "top1_chunk_index": top_results[0]["chunk_index"],
                "top1_keyword_coverage": top_results[0]["keyword_coverage"],
                "top3_keyword_coverage": top3_coverage,
                "expected_section_hit_top3": expected_section_hit,
                "top_results": top_results,
            }
        )

    return rows


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_model_rows(rows):
    by_model = {}
    for row in rows:
        model_rows = by_model.setdefault(
            row["model_label"],
            {
                "model": row["model"],
                "backend": row["backend"],
                "top1_keyword_hits": 0,
                "top1_keyword_total": 0,
                "top3_keyword_hits": 0,
                "top3_keyword_total": 0,
                "section_hits": 0,
                "question_count": 0,
            },
        )
        top1 = row["top1_keyword_coverage"]
        top3 = row["top3_keyword_coverage"]
        model_rows["top1_keyword_hits"] += len(top1["found"])
        model_rows["top1_keyword_total"] += len(top1["found"]) + len(top1["missing"])
        model_rows["top3_keyword_hits"] += len(top3["found"])
        model_rows["top3_keyword_total"] += len(top3["found"]) + len(top3["missing"])
        model_rows["section_hits"] += 1 if row["expected_section_hit_top3"] else 0
        model_rows["question_count"] += 1

    for model_rows in by_model.values():
        model_rows["top1_keyword_rate"] = round(
            model_rows["top1_keyword_hits"] / model_rows["top1_keyword_total"],
            3,
        )
        model_rows["top3_keyword_rate"] = round(
            model_rows["top3_keyword_hits"] / model_rows["top3_keyword_total"],
            3,
        )
        model_rows["section_hit_rate"] = round(
            model_rows["section_hits"] / model_rows["question_count"],
            3,
        )

    return by_model


def write_review(path: Path, rows, summary):
    lines = [
        "# Embedding Model Evaluation\n",
        "- Chunking: `structure-aware + semantic sentence boundary 500/150`",
        f"- Top K: `{TOP_K}`",
        "- Score: normalized embedding dot product, equivalent to cosine similarity.",
        "- E5 uses official-style `query:` and `passage:` prefixes in this experiment.",
        "\n## Model Summary\n",
        "| Model | Backend | Top-1 keyword rate | Top-3 keyword rate | Section hit rate |",
        "| --- | --- | --- | --- | --- |",
    ]

    for label, item in summary.items():
        lines.append(
            "| {label} | {backend} | {top1} | {top3} | {section} |".format(
                label=label,
                backend=item["backend"],
                top1=item["top1_keyword_rate"],
                top3=item["top3_keyword_rate"],
                section=item["section_hit_rate"],
            )
        )

    lines.extend(
        [
            "\n## Question Summary\n",
            "| Question | Model | Top-1 section | Top-1 keywords | Top-3 keywords | Section hit top-3 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in rows:
        top1 = row["top1_keyword_coverage"]
        top3 = row["top3_keyword_coverage"]
        lines.append(
            "| {qid} | {model} | {section} | {top1_found}/{top1_total} | "
            "{top3_found}/{top3_total} | {section_hit} |".format(
                qid=row["question_id"],
                model=row["model_label"],
                section=row["top1_section"].replace("|", "/") or "-",
                top1_found=len(top1["found"]),
                top1_total=len(top1["found"]) + len(top1["missing"]),
                top3_found=len(top3["found"]),
                top3_total=len(top3["found"]) + len(top3["missing"]),
                section_hit="yes" if row["expected_section_hit_top3"] else "no",
            )
        )

    lines.append("\n## Details\n")
    for row in rows:
        lines.append(f"### {row['question_id']} - {row['model_label']}\n")
        for result in row["top_results"]:
            coverage = result["keyword_coverage"]
            lines.append(
                "- Rank {rank}: score={score:.4f}, chunk={chunk}, "
                "section=`{section}`, keywords={found}/{total}".format(
                    rank=result["rank"],
                    score=result["score"],
                    chunk=result["chunk_index"],
                    section=result["section"],
                    found=len(coverage["found"]),
                    total=len(coverage["found"]) + len(coverage["missing"]),
                )
            )
            lines.append(f"  Preview: {result['preview']}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def selected_model_configs(model_label: str):
    if not model_label:
        return MODEL_CONFIGS

    selected = [
        model_config
        for model_config in MODEL_CONFIGS
        if model_config["label"] == model_label
    ]
    if not selected:
        labels = ", ".join(model_config["label"] for model_config in MODEL_CONFIGS)
        raise ValueError(f"Unknown model label: {model_label}. Available labels: {labels}")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Embedding model retrieval evaluation.")
    parser.add_argument(
        "--model-label",
        default="",
        help="Run one model only. Available: baseline_miniLM, mpnet_base, e5_base, bge_m3.",
    )
    args = parser.parse_args()

    chunks = load_chunks(CHUNKS_PATH)
    all_rows = []
    model_configs = selected_model_configs(args.model_label)

    for model_config in model_configs:
        print(f"[Model] Evaluating {model_config['name']}...", flush=True)
        try:
            all_rows.extend(evaluate_model(model_config, chunks))
        except Exception as exc:
            all_rows.append(
                {
                    "question_id": "ERROR",
                    "question": "",
                    "model": model_config["name"],
                    "model_label": model_config["label"],
                    "backend": "failed",
                    "error": str(exc),
                    "top1_score": 0,
                    "top1_section": "",
                    "top1_chunk_index": None,
                    "top1_keyword_coverage": {"found": [], "missing": [], "coverage": 0},
                    "top3_keyword_coverage": {"found": [], "missing": [], "coverage": 0},
                    "expected_section_hit_top3": False,
                    "top_results": [],
                }
            )
            print(f"[Warning] {model_config['name']} failed: {exc}")

    success_rows = [row for row in all_rows if row["question_id"] != "ERROR"]
    summary = summarize_model_rows(success_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.model_label}" if args.model_label else ""
    write_json(OUTPUT_DIR / f"model_configs{suffix}.json", model_configs)
    write_json(OUTPUT_DIR / f"summary{suffix}.json", summary)
    write_jsonl(OUTPUT_DIR / f"embedding_model_results{suffix}.jsonl", all_rows)
    write_review(OUTPUT_DIR / f"review{suffix}.md", success_rows, summary)

    print(
        json.dumps(
            {
                "chunking": "structure_semantic_boundary_500_150",
                "top_k": TOP_K,
                "models": len(model_configs),
                "successful_rows": len(success_rows),
                "outputs": {
                    "summary": str(OUTPUT_DIR / f"summary{suffix}.json"),
                    "results": str(OUTPUT_DIR / f"embedding_model_results{suffix}.jsonl"),
                    "review": str(OUTPUT_DIR / f"review{suffix}.md"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
