# Author: Yiting Liu
# Time 22/8/2026 AM12:53
import os
import re

import pymupdf4llm


# ============================================================
# 1. 配置
# ============================================================

INPUT_DIR = "rag_data"

OUTPUT_DIR = "normalized_data"


# ============================================================
# 2. 创建输出目录
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. 基础文本清理
# ============================================================

def clean_markdown(text: str) -> str:
    """
    对 PDF 转换出来的 Markdown 做轻量清洗。

    注意：
    这里不会进行特别激进的删除，
    防止误删真正有价值的研究数据。
    """

    if not text:
        return ""


    # --------------------------------------------------------
    # 统一换行
    # --------------------------------------------------------

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )


    # --------------------------------------------------------
    # 删除行首行尾空格
    # --------------------------------------------------------

    lines = []

    for line in text.splitlines():

        line = line.strip()

        lines.append(
            line
        )


    text = "\n".join(
        lines
    )


    # --------------------------------------------------------
    # 删除单独页码
    #
    # 比如：
    #
    # 03
    # 04
    # 15
    #
    # 但不会删除正文中的 2025、43% 等数据。
    # --------------------------------------------------------

    text = re.sub(
        r"(?m)^\s*\d{1,2}\s*$",
        "",
        text
    )


    # --------------------------------------------------------
    # 删除大量连续空行
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    # --------------------------------------------------------
    # 修复中文字符之间明显不合理的多余空格
    #
    # 例如：
    #
    # 品 牌 营 销
    #
    # →
    #
    # 品牌营销
    #
    # --------------------------------------------------------

    text = re.sub(
        r"(?<=[\u4e00-\u9fff]) "
        r"(?=[\u4e00-\u9fff])",
        "",
        text
    )


    return text.strip()


# ============================================================
# 4. PDF → Markdown
# ============================================================

def pdf_to_markdown(
    pdf_path: str
) -> str:
    """
    使用 pymupdf4llm 将 PDF 转换为 Markdown。

    相比：
        page.get_text("text")

    它会尽量保留：
        - 标题
        - 段落
        - 页面结构
        - 表格等信息
    """

    print(
        f"\n正在解析 PDF：{pdf_path}"
    )


    markdown_text = (
        pymupdf4llm.to_markdown(
            pdf_path
        )
    )


    print(
        "PDF → Markdown 完成。"
    )


    return markdown_text


# ============================================================
# 5. 保存 Markdown
# ============================================================

def save_markdown(
    markdown_text: str,
    output_path: str
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            markdown_text
        )


# ============================================================
# 6. 处理单个 PDF
# ============================================================

def process_pdf(
    pdf_path: str
):

    filename = os.path.basename(
        pdf_path
    )


    # --------------------------------------------------------
    # 文件名：
    #
    # report.pdf
    #
    # →
    #
    # report.md
    # --------------------------------------------------------

    base_name = os.path.splitext(
        filename
    )[0]


    output_filename = (
        base_name
        +
        ".md"
    )


    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )


    # ========================================================
    # STEP 1
    # PDF → Markdown
    # ========================================================

    markdown_text = pdf_to_markdown(
        pdf_path
    )


    # ========================================================
    # STEP 2
    # Markdown Cleaning
    # ========================================================

    markdown_text = clean_markdown(
        markdown_text
    )


    # ========================================================
    # STEP 3
    # Save
    # ========================================================

    save_markdown(
        markdown_text,
        output_path
    )


    print(
        f"已保存：{output_path}"
    )


    # ========================================================
    # Preview
    # ========================================================

    print(
        "\n========== Markdown Preview =========="
    )


    print(
        markdown_text[:3000]
    )


    print(
        "\n======================================"
    )


# ============================================================
# 7. 批量处理 rag_data 中所有 PDF
# ============================================================

def preprocess_all_pdfs():

    if not os.path.exists(
        INPUT_DIR
    ):

        raise FileNotFoundError(
            f"找不到目录：{INPUT_DIR}"
        )


    pdf_files = [

        filename

        for filename
        in os.listdir(
            INPUT_DIR
        )

        if filename.lower().endswith(
            ".pdf"
        )

    ]


    if not pdf_files:

        print(
            "rag_data 中没有找到 PDF。"
        )

        return


    print(
        f"找到 {len(pdf_files)} 个 PDF。"
    )


    for filename in pdf_files:

        pdf_path = os.path.join(
            INPUT_DIR,
            filename
        )


        try:

            process_pdf(
                pdf_path
            )


        except Exception as e:

            print(
                f"\n处理 {filename} 失败："
                f"{e}"
            )


# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":

    preprocess_all_pdfs()