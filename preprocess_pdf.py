import os
import re

import pymupdf4llm


INPUT_DIR = "rag_data"
OUTPUT_DIR = "normalized_data"


def clean_markdown(text: str) -> str:
    """
    Light Markdown cleaning only.

    This step keeps paragraph blank lines because build_rag.py relies on them
    for paragraph-aware chunking.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.splitlines():
        lines.append(line.strip())

    text = "\n".join(lines)
    text = re.sub(r"(?m)^\s*\d{1,2}\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(
        r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])",
        "",
        text,
    )

    return text.strip()


def pdf_to_markdown(pdf_path: str) -> str:
    print(f"\n正在解析 PDF：{pdf_path}")
    markdown_text = pymupdf4llm.to_markdown(pdf_path)
    print("PDF -> Markdown 完成。")
    return markdown_text


def save_markdown(markdown_text: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(markdown_text)


def process_pdf(pdf_path: str):
    filename = os.path.basename(pdf_path)
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR, base_name + ".md")

    markdown_text = pdf_to_markdown(pdf_path)
    markdown_text = clean_markdown(markdown_text)
    save_markdown(markdown_text, output_path)

    print(f"已保存：{output_path}")
    print("\n========== Markdown Preview ==========")
    print(markdown_text[:3000])
    print("\n======================================")


def preprocess_all_pdfs():
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"找不到目录：{INPUT_DIR}")

    pdf_files = [
        filename
        for filename in sorted(os.listdir(INPUT_DIR))
        if filename.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("rag_data 中没有找到 PDF。")
        return

    print(f"找到 {len(pdf_files)} 个 PDF。")

    for filename in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, filename)
        try:
            process_pdf(pdf_path)
        except Exception as exc:
            print(f"\n处理 {filename} 失败：{exc}")


if __name__ == "__main__":
    preprocess_all_pdfs()
