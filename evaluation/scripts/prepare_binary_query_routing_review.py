import json
from pathlib import Path

from experiments.retrieval.experiment_dense_only_evaluation import load_jsonl


BASE_DIR = Path(__file__).resolve().parents[2]
EVALUATION_FILE = BASE_DIR / "evaluation/datasets/query_handling/final.jsonl"
PREDICTIONS_FILE = BASE_DIR / "evaluation/results/query_route/predictions.jsonl"
OUTPUT_FILE = BASE_DIR / "evaluation/datasets/query_handling/binary_routing_review.md"


def json_value(value):
    return json.dumps(value, ensure_ascii=False)


def main():
    evaluation_rows = load_jsonl(EVALUATION_FILE)
    predictions = {
        row["query_id"]: row for row in load_jsonl(PREDICTIONS_FILE)
    }
    lines = [
        "# Binary Query Routing Review\n",
        "本轮只判断：问题应该直接进入检索，还是必须先反问。",
        "",
        "## 修改规则\n",
        "- `retrieve`：当前信息足以形成一个合理答案，即使问题比较宽泛或口语化。",
        "- `clarify`：缺少的信息会导致多个明显不同的答案，不能安全检索。",
        "- 如果改成 `retrieve`，请将 `missing_required_information` 改为 `[]`，将 `expected_clarifying_question` 改为 `null`。",
        "- 如果保留或改成 `clarify`，请检查缺失信息和参考反问是否真正必要。",
        "- `review_decision` 最终填写 `yes`；认为问题本身不适合评测时填写 `no`。",
        "- 模型预测只用于帮助发现争议，不是标准答案。",
        "",
    ]
    for row in evaluation_rows:
        prediction = predictions[row["query_id"]]
        expected_action = (
            "clarify" if row["expected_route"] == "ambiguous" else "retrieve"
        )
        missing = (
            row["clarification_slots"] if expected_action == "clarify" else []
        )
        clarifying_question = (
            row["expected_clarifying_question"]
            if expected_action == "clarify"
            else None
        )
        lines.extend(
            [
                f"## {row['query_id']}",
                "",
                "- **review_decision:** ``",
                "- **review_notes:** ``",
                f"- **user_query:** {row['user_query']}",
                f"- **previous_route:** `{row['expected_route']}`",
                f"- **expected_action:** `{expected_action}`",
                f"- **missing_required_information:** {json_value(missing)}",
                f"- **expected_clarifying_question:** {json_value(clarifying_question)}",
                f"- **model_predicted_route:** `{prediction['predicted_route']}`",
                f"- **model_reason:** {prediction['reason']}",
                "",
                "---",
                "",
            ]
        )
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "questions": len(evaluation_rows),
                "output": str(OUTPUT_FILE),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
