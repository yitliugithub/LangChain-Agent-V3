import json
from pathlib import Path

from experiments.retrieval.experiment_dense_only_evaluation import EVALUATION_FILE, load_jsonl, write_jsonl


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "evaluation/datasets/query_handling"
DRAFT_FILE = DATA_DIR / "clarify_candidates_round2_draft.jsonl"
REVIEW_FILE = DATA_DIR / "clarify_candidates_round2_review.md"

SPECS = [
    {
        "source_id": "Q003",
        "query": "这种新的投资形式，其核心是什么？",
        "missing": ["‘这种新的投资形式’所指的具体形式"],
        "clarification": "您所说的‘这种新的投资形式’具体指什么？",
        "explanation": "可能指种草、品牌广告、效果广告等不同形式，核心概念会不同。",
    },
    {
        "source_id": "Q005",
        "query": "这一年小红书达人商单的CPE呈现什么趋势？",
        "missing": ["‘这一年’所指的年份"],
        "clarification": "您想了解哪一年的小红书达人商单CPE趋势？",
        "explanation": "不同年份的CPE趋势可能不同，‘这一年’缺少对话中的时间指代。",
    },
    {
        "source_id": "Q006",
        "query": "这两个大促期间的CPE表现有什么不同？",
        "missing": ["需要比较的两个大促", "平台或商单范围"],
        "clarification": "您想比较哪两个大促，以及哪个平台的CPE表现？",
        "explanation": "618、双十一、年货节等组合及不同平台会产生不同答案。",
    },
    {
        "source_id": "Q007",
        "query": "这个品类在哪个月份的达人商单投放费用达到峰值？",
        "missing": ["‘这个品类’所指的产品品类", "年份"],
        "clarification": "您指的是哪个产品品类、哪一年的达人商单投放？",
        "explanation": "3C、美妆、母婴等品类以及不同年份的费用峰值月份可能不同。",
    },
    {
        "source_id": "Q015",
        "query": "这两家公司合并后获得了哪些投资，投资金额和估值是多少？",
        "missing": ["‘这两家公司’所指的公司名称"],
        "clarification": "您所说的两家公司分别是哪两家？",
        "explanation": "不同公司合并事件对应的投资方、金额和估值完全不同。",
    },
    {
        "source_id": "Q022",
        "query": "截至当时，门店数量排名前五的品牌有哪些？",
        "missing": ["‘当时’所指的时间", "所属行业"],
        "clarification": "您想了解哪个时间点、哪个行业的门店数量排名？",
        "explanation": "时间和行业不同，门店数量前五品牌会发生变化。",
    },
    {
        "source_id": "Q027",
        "query": "这两个年龄群体合计占宠物市场多少份额？",
        "missing": ["‘这两个年龄群体’所指的群体", "年份"],
        "clarification": "您指的是哪两个年龄群体，以及哪一年的宠物市场份额？",
        "explanation": "90后与00后、80后与90后等组合会得到不同占比，年份也会影响结果。",
    },
    {
        "source_id": "Q033",
        "query": "一线城市中，采用这种工具的宠物医院占比是多少？",
        "missing": ["‘这种工具’所指的工具"],
        "clarification": "您所说的‘这种工具’具体指哪一种工具？",
        "explanation": "AI辅助诊断、线上问诊或其他医疗工具的采用比例并不相同。",
    },
    {
        "source_id": "Q043",
        "query": "为什么这些行业的品牌广告主会精细调整整体投放策略？",
        "missing": ["‘这些行业’所指的行业类型"],
        "clarification": "您所说的‘这些行业’具体指哪些行业？",
        "explanation": "同质化竞争行业、低数字化行业或其他行业调整策略的原因可能不同。",
    },
    {
        "source_id": "Q052",
        "query": "这类广告主对数字广告和户外广告预算的态度如何？",
        "missing": ["‘这类广告主’所指的广告主类型"],
        "clarification": "您所说的‘这类广告主’具体是哪一类广告主？",
        "explanation": "探索型、过程型和结果型广告主的预算态度不同。",
    },
]


def write_review(rows):
    lines = [
        "# Clarify Candidates Round 2 Review\n",
        "这一批专门测试没有前文时的指代缺失。",
        "",
        "- 真正无法确定指代对象：填写 `yes`。",
        "- 即使没有前文仍可直接回答：填写 `no`。",
        "- 可直接修改问题、缺失信息和参考反问。",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['query_id']}",
                "",
                "- **review_decision:** ``",
                "- **review_notes:** ``",
                f"- **user_query:** {row['user_query']}",
                f"- **canonical_query:** {row['canonical_query']}",
                f"- **source_evaluation_id:** `{row['source_evaluation_id']}`",
                f"- **source_document:** {row['source_document']}",
                f"- **standard_answer:** {row['standard_answer']}",
                f"- **missing_required_information:** {json.dumps(row['missing_required_information'], ensure_ascii=False)}",
                f"- **expected_clarifying_question:** {json.dumps(row['expected_clarifying_question'], ensure_ascii=False)}",
                f"- **ambiguity_explanation:** {row['ambiguity_explanation']}",
                "",
                "---",
                "",
            ]
        )
    REVIEW_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    evaluation = {
        row["evaluation_id"]: row for row in load_jsonl(EVALUATION_FILE)
    }
    rows = []
    for index, spec in enumerate(SPECS, start=1):
        source = evaluation[spec["source_id"]]
        rows.append(
            {
                "query_id": f"QHD{index:02d}",
                "source_evaluation_id": source["evaluation_id"],
                "user_query": spec["query"],
                "canonical_query": source["question"],
                "standard_answer": source["standard_answer"],
                "source_document": source["source_document"],
                "source_section": source["source_section"],
                "expected_action": "clarify",
                "missing_required_information": spec["missing"],
                "expected_clarifying_question": spec["clarification"],
                "ambiguity_explanation": spec["explanation"],
                "relevant_chunk_id": None,
                "review_decision": "",
                "review_notes": "",
            }
        )
    write_jsonl(DRAFT_FILE, rows)
    write_review(rows)
    print(json.dumps({"candidates": len(rows), "review": str(REVIEW_FILE)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
