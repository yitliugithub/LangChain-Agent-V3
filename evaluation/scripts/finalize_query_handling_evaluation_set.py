import json
import re
from collections import Counter
from pathlib import Path

from experiments.retrieval.experiment_dense_only_evaluation import load_jsonl, write_jsonl


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "evaluation/datasets/query_handling"
DRAFT_FILE = INPUT_DIR / "draft.jsonl"
REVIEW_FILE = INPUT_DIR / "review.md"
FINAL_FILE = INPUT_DIR / "final.jsonl"
SUMMARY_FILE = INPUT_DIR / "final_summary.json"

EDITABLE_FIELDS = {
    "review_decision",
    "review_notes",
    "user_query",
    "expected_route",
    "canonical_query",
    "must_preserve",
    "must_not_add",
    "expected_clarifying_question",
    "clarification_slots",
}
JSON_LIST_FIELDS = {"must_preserve", "must_not_add", "clarification_slots"}


def strip_code_ticks(value):
    value = value.strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def parse_review():
    text = REVIEW_FILE.read_text(encoding="utf-8")
    sections = re.split(r"(?=^## QH\d{2}-\d\s+·)", text, flags=re.MULTILINE)
    parsed = {}
    for section in sections:
        heading = re.match(r"^## (QH\d{2}-\d)\s+·", section)
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
            value = strip_code_ticks(raw)
            if field in JSON_LIST_FIELDS:
                value = json.loads(value)
                if not isinstance(value, list):
                    raise ValueError(f"{query_id} {field} must be a JSON list")
            values[field] = value
        parsed[query_id] = values
    return parsed


def validate(rows, rejected):
    errors = []
    valid_routes = {"clear", "expandable", "ambiguous"}
    ids = [row["query_id"] for row in rows]
    queries = [row["user_query"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate query_id")
    if len(queries) != len(set(queries)):
        errors.append("Duplicate user_query")

    for row in rows:
        query_id = row["query_id"]
        if row["expected_route"] not in valid_routes:
            errors.append(f"{query_id}: invalid route {row['expected_route']}")
        for field in ("user_query", "canonical_query", "must_preserve"):
            if not row[field]:
                errors.append(f"{query_id}: empty {field}")
        if row["expected_route"] == "ambiguous":
            if row["relevant_chunk_id"] is not None:
                errors.append(f"{query_id}: ambiguous query must not have gold chunk")
            if not row["expected_clarifying_question"]:
                errors.append(f"{query_id}: missing clarifying question")
            if not row["clarification_slots"]:
                errors.append(f"{query_id}: missing clarification slots")
        elif row["relevant_chunk_id"] is None:
            errors.append(f"{query_id}: retrievable query is missing gold chunk")

    if errors:
        raise ValueError("\n".join(errors))
    return {
        "accepted_questions": len(rows),
        "rejected_questions": len(rejected),
        "route_distribution": dict(Counter(row["expected_route"] for row in rows)),
        "source_question_count": len({row["source_evaluation_id"] for row in rows}),
        "source_document_count": len({row["source_document"] for row in rows}),
        "accepted_query_ids": ids,
        "rejected_query_ids": [row["query_id"] for row in rejected],
    }


def main():
    drafts = {row["query_id"]: row for row in load_jsonl(DRAFT_FILE)}
    reviewed = parse_review()
    if set(drafts) != set(reviewed):
        missing = sorted(set(drafts) - set(reviewed))
        extra = sorted(set(reviewed) - set(drafts))
        raise ValueError(f"Review IDs differ from draft. Missing={missing}, extra={extra}")

    accepted = []
    rejected = []
    undecided = []
    for query_id, draft in drafts.items():
        row = {**draft, **reviewed[query_id]}
        decision = row["review_decision"].strip().lower()
        row["review_decision"] = decision
        if decision == "yes":
            accepted.append(row)
        elif decision == "no":
            rejected.append(row)
        else:
            undecided.append(query_id)
    if undecided:
        raise ValueError(f"Unreviewed query IDs: {undecided}")

    summary = validate(accepted, rejected)
    write_jsonl(FINAL_FILE, accepted)
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[Final] {FINAL_FILE}", flush=True)


if __name__ == "__main__":
    main()
