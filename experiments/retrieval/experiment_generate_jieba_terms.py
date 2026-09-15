import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from experiments.retrieval.experiment_retrieval_evaluation import QUESTIONS


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_TEXT = (
    BASE_DIR
    / "artifacts/pdf_parsing/2025年品牌营销趋势报告/magic_pdf/cleaned_text.md"
)
OUTPUT_DIR = (
    BASE_DIR
    / "artifacts/chunking/2025年品牌营销趋势报告/bm25_jieba_terms"
)

MIN_TERM_LEN = 2
MAX_TERM_LEN = 8
MIN_FREQUENCY = 2
MAX_CANDIDATES_FROM_TEXT = 220

STOP_TERMS = {
    "一个",
    "一些",
    "一种",
    "以及",
    "因此",
    "但是",
    "同时",
    "其中",
    "进行",
    "通过",
    "对于",
    "由于",
    "如果",
    "需要",
    "可以",
    "更加",
    "可能",
    "不同",
    "整体",
    "主要",
    "相关",
    "方面",
    "情况",
    "问题",
    "方式",
    "能力",
    "市场",
    "品牌",
    "营销",
    "广告",
    "广告主",
    "消费者",
    "行业",
    "投放",
    "预算",
    "增长",
}


def normalize_text(text):
    text = re.sub(r"\[[Ii]mage:[^\]]+\]", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[A-Za-z0-9_./:%+-]+", " ", text)
    return text


def extract_terms_from_questions():
    terms = defaultdict(set)
    for question in QUESTIONS:
        for keyword in question["expected_keywords"]:
            if is_good_term(keyword):
                terms[keyword].add("evaluation_keyword")

        section_parts = re.split(r"[，,：:、“”\"'（）()\s/]+", question["section"])
        for part in section_parts:
            if is_good_term(part):
                terms[part].add("expected_section")

    return terms


def extract_headings(text):
    terms = defaultdict(set)
    for line in text.splitlines():
        match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if not match:
            continue
        heading = match.group(1).strip()
        if is_good_term(heading):
            terms[heading].add("markdown_heading")
        for part in re.split(r"[，,：:、“”\"'（）()\s/]+", heading):
            if is_good_term(part):
                terms[part].add("markdown_heading_phrase")
    return terms


def chinese_sequences(text):
    return re.findall(r"[\u4e00-\u9fff]{2,}", normalize_text(text))


def extract_ngrams(text):
    counts = Counter()
    for sequence in chinese_sequences(text):
        for size in range(MIN_TERM_LEN, min(MAX_TERM_LEN, len(sequence)) + 1):
            for start in range(0, len(sequence) - size + 1):
                term = sequence[start:start + size]
                if is_good_term(term):
                    counts[term] += 1
    return counts


def is_good_term(term):
    term = term.strip()
    if not term:
        return False
    if "%" in term:
        return False
    if re.search(r"\d", term):
        return False
    if not re.search(r"[\u4e00-\u9fff]", term):
        return False
    if not (MIN_TERM_LEN <= len(term) <= 20):
        return False
    if term in STOP_TERMS:
        return False
    if len(set(term)) == 1:
        return False
    return True


def score_candidate(term, sources, frequency):
    score = frequency
    if "evaluation_keyword" in sources:
        score += 100
    if "expected_section" in sources:
        score += 60
    if "markdown_heading" in sources:
        score += 45
    if "markdown_heading_phrase" in sources:
        score += 25
    if 3 <= len(term) <= 6:
        score += 5
    return score


def merge_sources(*source_maps):
    merged = defaultdict(set)
    for source_map in source_maps:
        for term, sources in source_map.items():
            merged[term].update(sources)
    return merged


def write_outputs(candidates):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_json = OUTPUT_DIR / "candidate_terms.json"
    candidate_txt = OUTPUT_DIR / "candidate_terms.txt"
    review_md = OUTPUT_DIR / "candidate_terms_review.md"

    candidate_json.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    candidate_txt.write_text(
        "\n".join(item["term"] for item in candidates) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Jieba Custom Dictionary Candidates\n",
        "- 目的：为 BM25 的中文分词生成候选自定义词。",
        "- 请人工删除太泛、错误、无检索价值的词，再保存为 `final_custom_dict.txt`。",
        "- 优先保留：平台名、行业名、指标名、专有概念、evaluation keyword。",
        "",
        "## Candidates\n",
        "| Keep? | Term | Score | Frequency | Sources |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in candidates:
        lines.append(
            "| [ ] | {term} | {score} | {frequency} | {sources} |".format(
                term=item["term"],
                score=item["score"],
                frequency=item["frequency"],
                sources=", ".join(item["sources"]),
            )
        )
    review_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "candidate_json": str(candidate_json),
        "candidate_txt": str(candidate_txt),
        "review_md": str(review_md),
    }


def main():
    text = INPUT_TEXT.read_text(encoding="utf-8")
    question_terms = extract_terms_from_questions()
    heading_terms = extract_headings(text)
    text_ngram_counts = extract_ngrams(text)
    source_terms = merge_sources(question_terms, heading_terms)

    candidates = []
    all_terms = set(source_terms) | {
        term
        for term, frequency in text_ngram_counts.items()
        if frequency >= MIN_FREQUENCY
    }

    for term in all_terms:
        sources = set(source_terms.get(term, set()))
        frequency = text_ngram_counts.get(term, 0)
        if frequency >= MIN_FREQUENCY:
            sources.add("high_frequency_ngram")
        candidates.append(
            {
                "term": term,
                "score": score_candidate(term, sources, frequency),
                "frequency": frequency,
                "sources": sorted(sources),
            }
        )

    candidates = sorted(
        candidates,
        key=lambda item: (item["score"], item["frequency"], len(item["term"])),
        reverse=True,
    )
    candidates = candidates[:MAX_CANDIDATES_FROM_TEXT]
    outputs = write_outputs(candidates)

    print(
        json.dumps(
            {
                "input": str(INPUT_TEXT),
                "candidate_count": len(candidates),
                "outputs": outputs,
                "top_30": candidates[:30],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
