import os
import re
import hashlib

import chromadb

from sentence_transformers import SentenceTransformer


# ============================================================
# 1. 配置
# ============================================================

INPUT_DIR = "normalized_data"

CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "marketing_knowledge"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# Chunk 参数
# ============================================================

# 一个 Chunk 最理想不要超过这个长度
MAX_CHUNK_CHARS = 1000

# 太小的 Chunk 尽量和前一个合并
MIN_CHUNK_CHARS = 200


# ============================================================
# 2. Embedding Model
# ============================================================

print("正在加载 Embedding Model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Embedding Model 加载完成。")


# ============================================================
# 3. ChromaDB
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)


# Demo 阶段每次重新构建知识库
try:

    client.delete_collection(
        COLLECTION_NAME
    )

    print("旧知识库已删除。")

except Exception:

    pass


collection = client.create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# 4. 读取 Markdown
# ============================================================

def load_markdown(file_path: str) -> str:

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ============================================================
# 5. 按 Markdown Heading 分 Section
# ============================================================

def split_by_headings(text: str):
    """
    输入：

    # 标题1

    内容...

    ## 标题2

    内容...

    输出：

    [
        {
            "heading": "标题1",
            "text": "..."
        },
        ...
    ]
    """

    lines = text.splitlines()

    sections = []

    current_heading = "Document"

    current_content = []


    for line in lines:

        line = line.strip()


        # Markdown Heading
        if re.match(
            r"^#{1,6}\s+",
            line
        ):

            # 先保存之前的 Section
            if current_content:

                sections.append(
                    {
                        "heading":
                            current_heading,

                        "text":
                            "\n".join(
                                current_content
                            ).strip()
                    }
                )


            # 新 Heading
            current_heading = re.sub(
                r"^#{1,6}\s+",
                "",
                line
            ).strip()


            current_content = []


        else:

            if line:

                current_content.append(
                    line
                )


    # 最后一个 Section
    if current_content:

        sections.append(
            {
                "heading":
                    current_heading,

                "text":
                    "\n".join(
                        current_content
                    ).strip()
            }
        )


    return sections


# ============================================================
# 6. 按段落切
# ============================================================

def split_paragraphs(text: str):

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )


    return [

        paragraph.strip()

        for paragraph in paragraphs

        if paragraph.strip()

    ]


# ============================================================
# 7. 按句子切
# ============================================================

def split_sentences(text: str):

    sentences = re.split(
        r"(?<=[。！？.!?])\s*",
        text
    )


    return [

        sentence.strip()

        for sentence in sentences

        if sentence.strip()

    ]


# ============================================================
# 8. 最后兜底：按字符切
# ============================================================

def hard_split(text: str):

    chunks = []

    start = 0


    while start < len(text):

        end = (
            start
            +
            MAX_CHUNK_CHARS
        )


        chunk = text[
            start:end
        ].strip()


        if chunk:

            chunks.append(
                chunk
            )


        start = end


    return chunks


# ============================================================
# 9. Recursive Chunking
# ============================================================

def recursive_chunk(
    text: str
):

    # ========================================================
    # 情况 1：
    # 本身已经足够短
    # ========================================================

    if len(text) <= MAX_CHUNK_CHARS:

        return [
            text
        ]


    # ========================================================
    # 情况 2：
    # 先尝试 Paragraph
    # ========================================================

    paragraphs = split_paragraphs(
        text
    )


    if len(paragraphs) > 1:

        chunks = []

        current_chunk = ""


        for paragraph in paragraphs:

            candidate = (

                current_chunk
                +
                "\n\n"
                +
                paragraph

            ).strip()


            if (
                len(candidate)
                <=
                MAX_CHUNK_CHARS
            ):

                current_chunk = candidate


            else:

                if current_chunk:

                    chunks.append(
                        current_chunk
                    )


                # paragraph 自己仍然很大
                if (
                    len(paragraph)
                    >
                    MAX_CHUNK_CHARS
                ):

                    chunks.extend(
                        recursive_chunk(
                            paragraph
                        )
                    )

                    current_chunk = ""


                else:

                    current_chunk = paragraph


        if current_chunk:

            chunks.append(
                current_chunk
            )


        return chunks


    # ========================================================
    # 情况 3：
    # Paragraph 本身还是太长
    # → Sentence
    # ========================================================

    sentences = split_sentences(
        text
    )


    if len(sentences) > 1:

        chunks = []

        current_chunk = ""


        for sentence in sentences:

            candidate = (

                current_chunk
                +
                sentence

            )


            if (
                len(candidate)
                <=
                MAX_CHUNK_CHARS
            ):

                current_chunk = candidate


            else:

                if current_chunk:

                    chunks.append(
                        current_chunk
                    )


                if (
                    len(sentence)
                    >
                    MAX_CHUNK_CHARS
                ):

                    chunks.extend(
                        hard_split(
                            sentence
                        )
                    )

                    current_chunk = ""


                else:

                    current_chunk = sentence


        if current_chunk:

            chunks.append(
                current_chunk
            )


        return chunks


    # ========================================================
    # 情况 4：
    # 最后硬切
    # ========================================================

    return hard_split(
        text
    )


# ============================================================
# 10. 合并过短 Chunk
# ============================================================

def merge_small_chunks(
    chunks
):

    if not chunks:

        return []


    merged = []


    for chunk in chunks:


        if (
            len(chunk)
            <
            MIN_CHUNK_CHARS

            and

            merged
        ):

            candidate = (

                merged[-1]
                +
                "\n\n"
                +
                chunk

            )


            # 合并后不要太夸张
            if (
                len(candidate)
                <=
                MAX_CHUNK_CHARS
                *
                1.2
            ):

                merged[-1] = candidate

            else:

                merged.append(
                    chunk
                )


        else:

            merged.append(
                chunk
            )


    return merged


# ============================================================
# 11. 将 Markdown 转成最终 Chunk
# ============================================================

def create_chunks_from_markdown(
    markdown_text
):

    sections = split_by_headings(
        markdown_text
    )


    final_chunks = []


    for section in sections:


        heading = section[
            "heading"
        ]


        section_text = section[
            "text"
        ]


        chunks = recursive_chunk(
            section_text
        )


        chunks = merge_small_chunks(
            chunks
        )


        for chunk in chunks:


            # =================================================
            # 把 Heading 加入 Chunk
            #
            # 非常重要：
            # Embedding 时保留 Section Context。
            # =================================================

            full_text = (

                f"Section: {heading}\n\n"

                f"{chunk}"

            )


            final_chunks.append(
                {
                    "section":
                        heading,

                    "text":
                        full_text
                }
            )


    return final_chunks


# ============================================================
# 12. Embedding
# ============================================================

def create_embeddings(
    texts
):

    return embedding_model.encode(

        texts,

        normalize_embeddings=True,

        show_progress_bar=False

    )


# ============================================================
# 13. Chunk ID
# ============================================================

def create_chunk_id(
    source,
    section,
    index
):

    raw = (
        f"{source}|"
        f"{section}|"
        f"{index}"
    )


    return hashlib.md5(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# 14. Index Markdown
# ============================================================

def index_markdown(
    file_path
):


    filename = os.path.basename(
        file_path
    )


    print(
        f"\n正在处理：{filename}"
    )


    markdown_text = load_markdown(
        file_path
    )


    chunks = create_chunks_from_markdown(
        markdown_text
    )


    print(
        f"生成 {len(chunks)} 个 Chunks。"
    )


    # ========================================================
    # Embedding
    # ========================================================

    texts = [

        chunk[
            "text"
        ]

        for chunk
        in chunks

    ]


    embeddings = create_embeddings(
        texts
    )


    # ========================================================
    # 写入 Chroma
    # ========================================================

    for index, (
        chunk,
        embedding
    ) in enumerate(

        zip(
            chunks,
            embeddings
        )

    ):


        chunk_id = create_chunk_id(

            filename,

            chunk[
                "section"
            ],

            index

        )


        collection.add(

            ids=[
                chunk_id
            ],

            documents=[
                chunk[
                    "text"
                ]
            ],

            embeddings=[
                embedding.tolist()
            ],

            metadatas=[
                {
                    "source":
                        filename,

                    "section":
                        chunk[
                            "section"
                        ],

                    "chunk_index":
                        index,

                    "chunk_method":
                        "heading_recursive"
                }
            ]

        )


    # ========================================================
    # Preview
    # ========================================================

    print(
        "\n========== Chunk Preview =========="
    )


    for index, chunk in enumerate(
        chunks[:10]
    ):


        print(
            f"\n--- Chunk {index + 1} ---"
        )


        print(
            f"Section: {chunk['section']}"
        )


        print(
            chunk["text"][:500]
        )


    print(
        "\n==================================="
    )


# ============================================================
# 15. 构建整个 RAG 知识库
# ============================================================

def build_knowledge_base():


    if not os.path.exists(
        INPUT_DIR
    ):

        raise FileNotFoundError(
            f"找不到目录：{INPUT_DIR}"
        )


    markdown_files = [

        filename

        for filename
        in os.listdir(
            INPUT_DIR
        )

        if filename.lower().endswith(
            ".md"
        )

    ]


    if not markdown_files:

        print(
            "normalized_data 中没有 Markdown 文件。"
        )

        return


    print(
        f"找到 {len(markdown_files)} 个 Markdown 文件。"
    )


    for filename in markdown_files:


        file_path = os.path.join(

            INPUT_DIR,

            filename

        )


        try:

            index_markdown(
                file_path
            )


        except Exception as e:

            print(
                f"处理 {filename} 失败：{e}"
            )


    print(
        "\nRAG Knowledge Base 构建完成。"
    )


# ============================================================
# 16. Main
# ============================================================

if __name__ == "__main__":

    build_knowledge_base()