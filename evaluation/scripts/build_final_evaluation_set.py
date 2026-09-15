import json
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
CANDIDATE_DIR = BASE_DIR / "evaluation/candidates"
REVIEW_FILE = CANDIDATE_DIR / "candidate_questions_review.md"
CANDIDATE_FILE = CANDIDATE_DIR / "candidate_questions.jsonl"
CHUNK_ROOT = BASE_DIR / "artifacts/chunking"
CHUNK_DIR_NAME = "cleaned_text_structure_semantic_boundary_500_min_150"

OUTPUT_DIR = BASE_DIR / "evaluation/datasets"
OUTPUT_JSONL = OUTPUT_DIR / "final_evaluation_set.jsonl"
OUTPUT_JSON = OUTPUT_DIR / "final_evaluation_set.json"
OUTPUT_MD = OUTPUT_DIR / "final_evaluation_set.md"
OUTPUT_SUMMARY = OUTPUT_DIR / "summary.json"

KEEP_PATTERN = re.compile(
    r"^##\s+(?P<candidate_id>\S+)\s+\|\s+Keep:\s*\[\s*yes\s*\]\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def load_selected_candidate_ids():
    text = REVIEW_FILE.read_text(encoding="utf-8")
    return [match.group("candidate_id") for match in KEEP_PATTERN.finditer(text)]


def load_candidates():
    candidates = {}
    with CANDIDATE_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            candidates[row["candidate_id"]] = row
    return candidates


def load_chunks():
    chunks = {}
    pattern = f"*/{CHUNK_DIR_NAME}/chunks.jsonl"
    for path in sorted(CHUNK_ROOT.glob(pattern)):
        document = path.parent.parent.name
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                chunks[(document, chunk["chunk_index"])] = chunk
    return chunks


def build_rows(selected_ids, candidates, chunks):
    rows = []
    for index, candidate_id in enumerate(selected_ids, start=1):
        if candidate_id not in candidates:
            raise KeyError(f"Selected candidate not found: {candidate_id}")

        candidate = candidates[candidate_id]
        chunk_key = (candidate["document"], candidate["chunk_index"])
        if chunk_key not in chunks:
            raise KeyError(f"Source chunk not found: {chunk_key}")

        chunk = chunks[chunk_key]
        rows.append(
            {
                "evaluation_id": f"Q{index:03d}",
                "candidate_id": candidate_id,
                "question": candidate["question"],
                "standard_answer": candidate["answer"],
                "keywords": candidate.get("keywords", []),
                "answer_section": candidate.get("answer_section", ""),
                "question_type": candidate.get("question_type", ""),
                "difficulty": candidate.get("difficulty", ""),
                "source_document": candidate["document"],
                "source_chunk_index": candidate["chunk_index"],
                "relevant_chunk_id": (
                    f"{candidate['document']}::chunk_{candidate['chunk_index']}"
                ),
                "source_section": chunk.get("section", ""),
                "source_token_count": chunk.get("token_count"),
                "evidence_text": chunk.get("text", ""),
            }
        )
    return rows


def write_jsonl(rows):
    with OUTPUT_JSONL.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(rows):
    parts = [
        "# Final RAG Evaluation Set\n",
        f"Selected questions: {len(rows)}\n",
    ]
    for row in rows:
        keywords = ", ".join(row["keywords"])
        evidence = row["evidence_text"].replace("\n", "\n> ")
        parts.append(
            f"\n## {row['evaluation_id']}\n\n"
            f"- Candidate ID: {row['candidate_id']}\n"
            f"- Document: {row['source_document']}\n"
            f"- Chunk index: {row['source_chunk_index']}\n"
            f"- Section: {row['source_section']}\n"
            f"- Type: {row['question_type']}\n"
            f"- Difficulty: {row['difficulty']}\n"
            f"- Keywords: {keywords}\n\n"
            f"**Question:** {row['question']}\n\n"
            f"**Standard answer:** {row['standard_answer']}\n\n"
            f"**Evidence:**\n\n> {evidence}\n"
        )
    OUTPUT_MD.write_text("".join(parts), encoding="utf-8")


def write_summary(rows):
    summary = {
        "selected_questions": len(rows),
        "unique_source_chunks": len(
            {(row["source_document"], row["source_chunk_index"]) for row in rows}
        ),
        "documents": dict(Counter(row["source_document"] for row in rows)),
        "question_types": dict(Counter(row["question_type"] for row in rows)),
        "difficulties": dict(Counter(row["difficulty"] for row in rows)),
        "selection_rule": "Keep marker contains yes, ignoring spaces and case",
    }
    OUTPUT_SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    selected_ids = load_selected_candidate_ids()
    if not selected_ids:
        raise RuntimeError("No selected questions found in the review file.")
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("Duplicate selected candidate IDs found in the review file.")

    candidates = load_candidates()
    chunks = load_chunks()
    rows = build_rows(selected_ids, candidates, chunks)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows)
    OUTPUT_JSON.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(rows)
    summary = write_summary(rows)

    print(f"[Done] selected_questions={summary['selected_questions']}")
    print(f"[Done] unique_source_chunks={summary['unique_source_chunks']}")
    print(f"[Output] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
