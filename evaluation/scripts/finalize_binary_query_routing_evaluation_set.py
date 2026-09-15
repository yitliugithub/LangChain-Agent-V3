import json
import re
from collections import Counter
from pathlib import Path

from experiments.retrieval.experiment_dense_only_evaluation import EVALUATION_FILE, load_jsonl, write_jsonl


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "evaluation/datasets/query_handling"
SOURCE_FILE = INPUT_DIR / "final.jsonl"
REVIEW_FILE = INPUT_DIR / "binary_routing_review.md"
FINAL_FILE = INPUT_DIR / "binary_routing_final.jsonl"
SUMMARY_FILE = INPUT_DIR / "binary_routing_final_summary.json"

EDITABLE_FIELDS = {
    "review_decision",
    "review_notes",
    "user_query",
    "expected_action",
    "missing_required_information",
    "expected_clarifying_question",
}


def clean_scalar(value):
    value = value.strip()
    # Tolerate a missing closing backtick in a manually edited scalar.
    return value.strip("`").strip()


def parse_review():
    text = REVIEW_FILE.read_text(encoding="utf-8")
    sections = re.split(r"(?=^## QH\d{2}-\d\s*$)", text, flags=re.MULTILINE)
    parsed = {}
    for section in sections:
        heading = re.match(r"^## (QH\d{2}-\d)\s*(?:\n|$)", section)
        if not heading:
            continue
        query_id = heading.group(1)
        values = {}
        for match in re.finditer(
            r"^- \*\*([a-z_]+):\*\*\s*(.*)$", section, flags=re.MULTILINE
        ):
            field, raw = match.groups()
            if field not in EDITABLE_FIELDS:
                continue
            raw = raw.strip()
            if field == "missing_required_information":
                values[field] = json.loads(raw)
            elif field == "expected_clarifying_question":
                values[field] = json.loads(raw)
            else:
                values[field] = clean_scalar(raw)
        parsed[query_id] = values
    return parsed


def main():
    source_rows = {row["query_id"]: row for row in load_jsonl(SOURCE_FILE)}
    original_questions = {
        row["evaluation_id"]: row for row in load_jsonl(EVALUATION_FILE)
    }
    reviewed = parse_review()
    if set(source_rows) != set(reviewed):
        missing = sorted(set(source_rows) - set(reviewed))
        extra = sorted(set(reviewed) - set(source_rows))
        raise ValueError(
            f"Binary review IDs differ from source. Missing={missing}, extra={extra}"
        )

    accepted = []
    rejected = []
    normalizations = []
    undecided = []
    for query_id, source in source_rows.items():
        values = reviewed[query_id]
        decision = values.get("review_decision", "").lower()
        if decision == "no":
            rejected.append(query_id)
            continue
        if decision != "yes":
            undecided.append(query_id)
            continue

        action = values.get("expected_action", "").lower()
        if action not in {"retrieve", "clarify"}:
            raise ValueError(f"{query_id}: invalid expected_action {action!r}")
        missing = values.get("missing_required_information", [])
        clarification = values.get("expected_clarifying_question")
        if action == "retrieve":
            if missing or clarification is not None:
                normalizations.append(
                    {
                        "query_id": query_id,
                        "change": "Cleared stale clarification fields for retrieve",
                    }
                )
            missing = []
            clarification = None
            relevant_chunk_id = original_questions[source["source_evaluation_id"]][
                "relevant_chunk_id"
            ]
        else:
            if not missing or not clarification:
                raise ValueError(
                    f"{query_id}: clarify requires missing information and a question"
                )
            relevant_chunk_id = None

        accepted.append(
            {
                **source,
                "user_query": values.get("user_query", source["user_query"]),
                "expected_action": action,
                "missing_required_information": missing,
                "expected_clarifying_question": clarification,
                "relevant_chunk_id": relevant_chunk_id,
                "review_decision": decision,
                "review_notes": values.get("review_notes", ""),
            }
        )

    if undecided:
        raise ValueError(f"Unreviewed query IDs: {undecided}")
    if len({row["query_id"] for row in accepted}) != len(accepted):
        raise ValueError("Duplicate query IDs")
    if len({row["user_query"] for row in accepted}) != len(accepted):
        raise ValueError("Duplicate user queries")

    distribution = Counter(row["expected_action"] for row in accepted)
    majority_baseline = max(distribution.values()) / len(accepted)
    summary = {
        "accepted_questions": len(accepted),
        "rejected_questions": len(rejected),
        "action_distribution": dict(distribution),
        "majority_class_baseline_accuracy": round(majority_baseline, 6),
        "normalizations": normalizations,
        "rejected_query_ids": rejected,
        "ready_for_formal_evaluation": min(distribution.values()) >= 8,
        "readiness_note": (
            "Add more clarify examples before formal evaluation"
            if min(distribution.values()) < 8
            else "Class support is sufficient for a prototype evaluation"
        ),
    }
    write_jsonl(FINAL_FILE, accepted)
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[Final] {FINAL_FILE}", flush=True)


if __name__ == "__main__":
    main()
