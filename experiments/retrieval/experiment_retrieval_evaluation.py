import json
from pathlib import Path

import numpy as np


EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)
OUTPUT_DIR = Path("artifacts/chunking/2025年品牌营销趋势报告/retrieval_eval")
TOP_K = 3


CHUNKING_METHODS = {
    "fixed_400_80": Path(
        "artifacts/chunking/2025年品牌营销趋势报告/"
        "cleaned_text_fixed_400_overlap_80/chunks.jsonl"
    ),
    "fixed_400_80_cleaned_keep_tables": Path(
        "artifacts/chunking/2025年品牌营销趋势报告/"
        "cleaned_text_fixed_400_overlap_80_cleaned_keep_tables/chunks.jsonl"
    ),
    "structure_400_120": Path(
        "artifacts/chunking/2025年品牌营销趋势报告/"
        "cleaned_text_structure_aware_400_min_120/chunks.jsonl"
    ),
    "structure_500_150": Path(
        "artifacts/chunking/2025年品牌营销趋势报告/"
        "cleaned_text_structure_aware_500_min_150/chunks.jsonl"
    ),
    "baseline_structure_semantic_500_150": Path(
        "artifacts/chunking/2025年品牌营销趋势报告/"
        "cleaned_text_structure_semantic_boundary_500_min_150/chunks.jsonl"
    ),
    "structure_600_180": Path(
        "artifacts/chunking/2025年品牌营销趋势报告/"
        "cleaned_text_structure_aware_600_min_180/chunks.jsonl"
    ),
}


QUESTIONS = [
    {
        "id": "Q1",
        "type": "小主题连续性",
        "question": "广告主拓展新消费者的策略是什么？",
        "section": "”向内求“和”向外看“，广告主的差异化增量博弈",
        "expected_keywords": ["拓展人群", "拓展市场", "丰富产品品类", "开拓新市场"],
        "expected_answer": (
            "拓展人群：通过丰富产品品类来扩大受众群体；拓展市场："
            "通过开拓新市场来实现增量增长。"
        ),
    },
    {
        "id": "Q2",
        "type": "分类总结型",
        "question": "不同类别广告主有什么类？",
        "section": "不同类别广告主的核心指标倾向",
        "expected_keywords": ["探索型", "过程型", "结果型"],
        "expected_answer": (
            "不同类别广告主包括探索型、过程型、结果型。探索型以扩张为核心；"
            "过程型强调平台深度合作和降本增效；结果型强调 ROI 维持和销售转化。"
        ),
    },
    {
        "id": "Q3",
        "type": "具体论述型",
        "question": "实际的营销实践中，“扩张”怎么体现？",
        "section": "销售转化评估是探索型品牌广告主验证营销效果的关键所在",
        "expected_keywords": ["产品层面", "市场布局层面", "爆款营销", "下沉市场"],
        "expected_answer": (
            "扩张体现在产品层面和市场布局层面：优化产品组合、持续投入"
            "爆款营销，并向下沉市场进军。"
        ),
    },
    {
        "id": "Q4",
        "type": "数字事实型",
        "question": "有多少游戏品牌广告主已经布局了小程序？",
        "section": "存量竞争下的强效果线上生意：需可验证渠道破营销难题",
        "expected_keywords": ["游戏", "25%", "小程序", "15%"],
        "expected_answer": "25%的游戏品牌广告主已经布局了小程序，高于平均的15%。",
    },
    {
        "id": "Q5",
        "type": "表格事实型",
        "question": "食品饮料/美妆护肤/服装服饰三个行业No.1的触达率是多少？",
        "section": "评估户外广告投放/ 营销效果关于ROI 最核心的指标",
        "expected_keywords": ["食品饮料", "52%", "美妆护肤", "37%", "服装服饰", "50%"],
        "expected_answer": (
            "食品饮料触达率52%，美妆护肤触达率37%，服装服饰触达率50%。"
        ),
    },
    {
        "id": "Q6",
        "type": "专有名词/平台型问题",
        "question": "广告主更关注小红书、抖音、快手这类内容平台，还是淘天、京东这类传统电商平台？",
        "section": "加大新兴内容电商平台布局，探索多元内容营销",
        "expected_keywords": [
            "小红书",
            "抖音",
            "快手",
            "淘天",
            "京东",
            "新兴内容电商平台",
            "短剧营销",
        ],
        "expected_answer": (
            "从投放增量视角看，广告主更愿意尝试新兴内容种草平台的电商功能，"
            "并对小红书、抖音、快手等平台的本地生活、社交种草、短视频功能表现出兴趣；"
            "同时部分广告主也会尝试播客和短剧营销。"
        ),
    },
    {
        "id": "Q7",
        "type": "模糊语义问题",
        "question": "广告主在下沉市场面临什么营销难题？",
        "section": "潜力驱动品牌下沉，但营销难题仍待解决",
        "expected_keywords": [
            "下沉",
            "消费者需求",
            "不同地区",
            "因地制宜",
            "难以复制",
        ],
        "expected_answer": (
            "广告主在下沉市场面临消费者需求差异明显、不同地区需要因地制宜制定产品营销规划、"
            "营销卖点需要重新挖掘、成功模式难以简单复制等问题。"
        ),
    },
    {
        "id": "Q8",
        "type": "数字+语义混合问题",
        "question": "2025年广告主选择谨慎扩张的比例是多少，和2024年初相比有什么变化？",
        "section": "更为明确的竞争博弈方向",
        "expected_keywords": [
            "13%",
            "4%",
            "谨慎扩张",
            "方向不明",
            "占比锐减",
        ],
        "expected_answer": (
            "2024年初有13%的广告主秉持谨慎扩张理念，2025年这一比例下降至4%。"
            "这说明方向不明的广告主占比明显减少，广告主的竞争和增长方向更清晰。"
        ),
    },
]


def load_chunks(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def get_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except ModuleNotFoundError:
        return TransformersMeanPoolingEmbedder(EMBEDDING_MODEL_NAME)


class TransformersMeanPoolingEmbedder:
    """Small fallback when sentence-transformers is not installed."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        del show_progress_bar
        embeddings = []
        batch_size = 16

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

                if normalize_embeddings:
                    batch_embeddings = self.torch.nn.functional.normalize(
                        batch_embeddings,
                        p=2,
                        dim=1,
                    )

                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)


def embed_texts(model, texts):
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def keyword_coverage(text: str, keywords):
    found = [keyword for keyword in keywords if keyword in text]
    return {
        "found": found,
        "missing": [keyword for keyword in keywords if keyword not in found],
        "coverage": round(len(found) / len(keywords), 3) if keywords else 0,
    }


def preview_text(text: str, limit: int = 500):
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def evaluate_method(model, method_name, chunks, questions):
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(model, chunk_texts)
    results = []

    for question in questions:
        query_embedding = embed_texts(model, [question["question"]])[0]
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

        top3_coverage = keyword_coverage(
            combined_top_text,
            question["expected_keywords"],
        )
        expected_section_hit = any(
            question["section"] in result["section"]
            or question["section"] in result["preview"]
            for result in top_results
        )

        results.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "method": method_name,
                "top1_score": top_results[0]["score"],
                "top1_section": top_results[0]["section"],
                "top1_chunk_index": top_results[0]["chunk_index"],
                "top1_keyword_coverage": top_results[0]["keyword_coverage"],
                "top3_keyword_coverage": top3_coverage,
                "expected_section_hit_top3": expected_section_hit,
                "top_results": top_results,
            }
        )

    return results


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_review(path: Path, questions, rows):
    lines = [
        "# Retrieval Evaluation\n",
        f"- Embedding model: `{EMBEDDING_MODEL_NAME}`",
        f"- Top K: `{TOP_K}`",
        "\n## Summary Table\n",
        "| Question | Method | Top-1 section | Top-1 keywords | Top-3 keywords | Section hit top-3 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        top1 = row["top1_keyword_coverage"]
        top3 = row["top3_keyword_coverage"]
        lines.append(
            "| {qid} | {method} | {section} | {top1_found}/{top1_total} | "
            "{top3_found}/{top3_total} | {section_hit} |".format(
                qid=row["question_id"],
                method=row["method"],
                section=row["top1_section"].replace("|", "/") or "-",
                top1_found=len(top1["found"]),
                top1_total=len(top1["found"]) + len(top1["missing"]),
                top3_found=len(top3["found"]),
                top3_total=len(top3["found"]) + len(top3["missing"]),
                section_hit="yes" if row["expected_section_hit_top3"] else "no",
            )
        )

    lines.append("\n## Question Details\n")
    for question in questions:
        lines.append(f"### {question['id']} {question['question']}\n")
        lines.append(f"- Type: `{question['type']}`")
        lines.append(f"- Expected section: `{question['section']}`")
        lines.append(f"- Expected keywords: `{', '.join(question['expected_keywords'])}`")
        lines.append(f"- Expected answer: {question['expected_answer']}\n")

        for row in [item for item in rows if item["question_id"] == question["id"]]:
            lines.append(f"#### {row['method']}\n")
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


def main():
    model = get_model()
    all_rows = []

    for method_name, path in CHUNKING_METHODS.items():
        if not path.exists():
            print(f"[Skip] missing chunks file: {path}")
            continue
        chunks = load_chunks(path)
        method_rows = evaluate_method(model, method_name, chunks, QUESTIONS)
        all_rows.extend(method_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "questions.json", QUESTIONS)
    write_jsonl(OUTPUT_DIR / "retrieval_results.jsonl", all_rows)
    write_review(OUTPUT_DIR / "review.md", QUESTIONS, all_rows)

    print(
        json.dumps(
            {
                "questions": len(QUESTIONS),
                "methods": len(CHUNKING_METHODS),
                "results": len(all_rows),
                "outputs": {
                    "questions": str(OUTPUT_DIR / "questions.json"),
                    "results": str(OUTPUT_DIR / "retrieval_results.jsonl"),
                    "review": str(OUTPUT_DIR / "review.md"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
