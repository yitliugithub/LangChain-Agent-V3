import json
import re
from collections import Counter
from pathlib import Path

from experiments.retrieval.experiment_dense_only_evaluation import load_jsonl, write_jsonl


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "evaluation/datasets/query_handling"
BASE_FILE = DATA_DIR / "binary_routing_expanded.jsonl"
DRAFT_FILE = DATA_DIR / "clarify_candidates_round2_draft.jsonl"
REVIEW_FILE = DATA_DIR / "clarify_candidates_round2_review.md"
ACCEPTED_FILE = DATA_DIR / "clarify_candidates_round2_accepted.jsonl"
FINAL_FILE = DATA_DIR / "binary_routing_complete.jsonl"
SUMMARY_FILE = DATA_DIR / "binary_routing_complete_summary.json"


def parse_review():
    text = REVIEW_FILE.read_text(encoding="utf-8")
    sections = re.split(r"(?=^## QHD\d{2}\s*$)", text, flags=re.MULTILINE)
    parsed = {}
    for section in sections:
        heading = re.match(r"^## (QHD\d{2})\s*(?:\n|$)", section)
        if not heading:
            continue
        query_id = heading.group(1)
        values = {}
        for match in re.finditer(
            r"^- \*\*([a-z_]+):\*\*\s*(.*)$", section, flags=re.MULTILINE
        ):
            field, raw = match.groups()
            raw = raw.strip()
            if field in {"review_decision", "review_notes", "user_query"}:
                values[field] = raw.strip("`").strip()
            elif field == "missing_required_information":
                values[field] = json.loads(raw)
            elif field == "expected_clarifying_question":
                values[field] = json.loads(raw)
        parsed[query_id] = values
    return parsed


def main():
    base_rows = load_jsonl(BASE_FILE)
    drafts = {row["query_id"]: row for row in load_jsonl(DRAFT_FILE)}
    reviewed = parse_review()
    if set(drafts) != set(reviewed):
        raise ValueError("Round-2 review IDs do not match candidate draft IDs")

    accepted = []
    rejected = []
    undecided = []
    for query_id, draft in drafts.items():
        values = reviewed[query_id]
        decision = values.get("review_decision", "").lower()
        if decision == "no":
            rejected.append(query_id)
            continue
        if decision != "yes":
            undecided.append(query_id)
            continue
        missing = values.get("missing_required_information", [])
        clarification = values.get("expected_clarifying_question")
        if not missing or not clarification:
            raise ValueError(f"{query_id}: accepted clarify item is incomplete")
        accepted.append(
            {
                **draft,
                "user_query": values.get("user_query", draft["user_query"]),
                "missing_required_information": missing,
                "expected_clarifying_question": clarification,
                "review_decision": "yes",
                "review_notes": values.get("review_notes", ""),
            }
        )
    if undecided:
        raise ValueError(f"Unreviewed candidate IDs: {undecided}")

    merged = base_rows + accepted
    ids = [row["query_id"] for row in merged]
    queries = [row["user_query"] for row in merged]
    if len(ids) != len(set(ids)) or len(queries) != len(set(queries)):
        raise ValueError("Duplicate ID or query after round-2 merge")

    distribution = Counter(row["expected_action"] for row in merged)
    summary = {
        "base_questions": len(base_rows),
        "accepted_round2": len(accepted),
        "rejected_round2": len(rejected),
        "total_questions": len(merged),
        "action_distribution": dict(distribution),
        "majority_class_baseline_accuracy": round(
            max(distribution.values()) / len(merged), 6
        ),
        "accepted_query_ids": [row["query_id"] for row in accepted],
        "rejected_query_ids": rejected,
    }
    write_jsonl(ACCEPTED_FILE, accepted)
    write_jsonl(FINAL_FILE, merged)
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[Final] {FINAL_FILE}", flush=True)


if __name__ == "__main__":
    main()
