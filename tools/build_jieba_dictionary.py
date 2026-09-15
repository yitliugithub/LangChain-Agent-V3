import json
import re
from pathlib import Path

import jieba


BASE_DIR = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = BASE_DIR / "resources" / "jieba_dictionary_candidates"
REVIEW_FILE = CANDIDATE_DIR / "candidate_terms_review.md"
CANDIDATE_FILE = CANDIDATE_DIR / "candidate_terms.json"

OUTPUT_DIR = BASE_DIR / "resources" / "jieba_dictionary"
DICTIONARY_FILE = OUTPUT_DIR / "jieba_domain_dict.txt"
SELECTED_JSON = OUTPUT_DIR / "selected_terms.json"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"
MANUAL_EXTRA_TERMS = ["内容"]

SELECTED_ROW_PATTERN = re.compile(
    r"^\|\s*\[\s*yes\s*\]\s*\|\s*(?P<term>[^|]+?)\s*\|",
    flags=re.IGNORECASE | re.MULTILINE,
)


def load_selected_terms():
    text = REVIEW_FILE.read_text(encoding="utf-8")
    return [match.group("term").strip() for match in SELECTED_ROW_PATTERN.finditer(text)]


def load_candidates():
    rows = json.loads(CANDIDATE_FILE.read_text(encoding="utf-8"))
    return {row["term"]: row for row in rows}


def find_nested_pairs(terms):
    pairs = []
    for shorter in terms:
        for longer in terms:
            if shorter == longer:
                continue
            if len(shorter) < len(longer) and shorter in longer:
                pairs.append([shorter, longer])
    return sorted(pairs)


def validate_selected_terms(terms, candidates):
    problems = []
    if len(terms) != len(set(terms)):
        problems.append("duplicate selected terms")

    for term in terms:
        if term not in candidates:
            problems.append(f"selected term missing from candidate JSON: {term}")
        if not term.strip() or re.search(r"[\r\n|]", term):
            problems.append(f"invalid dictionary term: {term!r}")
    return problems


def validate_jieba_loading(terms):
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    before = {term: list(tokenizer.cut(term, HMM=False)) for term in terms}
    tokenizer.load_userdict(str(DICTIONARY_FILE))
    after = {term: list(tokenizer.cut(term, HMM=False)) for term in terms}

    failures = {
        term: tokens
        for term, tokens in after.items()
        if tokens != [term]
    }
    changed_count = sum(before[term] != after[term] for term in terms)
    return before, after, failures, changed_count


def main():
    selected_terms = load_selected_terms()
    selected_terms.extend(
        term for term in MANUAL_EXTRA_TERMS if term not in selected_terms
    )
    candidates = load_candidates()
    validation_candidates = {
        **candidates,
        **{
            term: {
                "term": term,
                "score": None,
                "frequency": None,
                "minimum_pmi": None,
                "boundary_entropy": None,
                "document_count": None,
                "sources": ["manual_extra_term"],
                "example_documents": [],
            }
            for term in MANUAL_EXTRA_TERMS
        },
    }
    problems = validate_selected_terms(selected_terms, validation_candidates)
    if problems:
        raise RuntimeError("; ".join(problems))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DICTIONARY_FILE.write_text("\n".join(selected_terms) + "\n", encoding="utf-8")

    before, after, failures, changed_count = validate_jieba_loading(selected_terms)
    if failures:
        raise RuntimeError(f"Jieba failed to preserve selected terms: {failures}")

    selected_rows = []
    for term in selected_terms:
        selected_rows.append(
            {
                **validation_candidates[term],
                "default_jieba_tokens": before[term],
                "custom_jieba_tokens": after[term],
            }
        )
    SELECTED_JSON.write_text(
        json.dumps(selected_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    nested_pairs = find_nested_pairs(selected_terms)
    summary = {
        "selected_term_count": len(selected_terms),
        "manual_extra_terms": MANUAL_EXTRA_TERMS,
        "duplicate_count": len(selected_terms) - len(set(selected_terms)),
        "jieba_changed_segmentation_count": changed_count,
        "jieba_validation_failures": failures,
        "nested_term_pairs": nested_pairs,
        "nested_terms_policy": (
            "Retained because both terms were manually selected and are valid "
            "domain concepts; retrieval tokenization must be checked in context."
        ),
        "dictionary_format": "one UTF-8 term per line; compatible with jieba.load_userdict",
        "evaluation_data_used": False,
        "dictionary_file": str(DICTIONARY_FILE),
        "selected_terms_file": str(SELECTED_JSON),
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
