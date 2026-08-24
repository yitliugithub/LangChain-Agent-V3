import hashlib
import os
import re

INPUT_DIR = "normalized_data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "marketing_knowledge"
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Prototype heuristic parameters. This demo uses character length rather than
# token length so the chunking behavior stays simple and easy to explain.
MAX_CHUNK_CHARS = 1000
MIN_CHUNK_CHARS = 200

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        print("正在加载 Embedding Model...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding Model 加载完成。")
    return _embedding_model


def get_collection(rebuild: bool = False):
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("旧知识库已删除。")
        except Exception:
            pass

    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def load_markdown(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_by_headings(text: str):
    """
    Heading-aware section split.

    Blank lines are intentionally preserved. Paragraph chunking later depends on
    the original Markdown blank-line boundary: re.split(r"\\n\\s*\\n", text).
    """
    sections = []
    current_heading = "Document"
    current_content = []

    for raw_line in text.splitlines():
        heading_line = raw_line.strip()

        if re.match(r"^#{1,6}\s+", heading_line):
            section_text = "\n".join(current_content).strip()
            if section_text:
                sections.append(
                    {
                        "heading": current_heading,
                        "text": section_text,
                    }
                )

            current_heading = re.sub(r"^#{1,6}\s+", "", heading_line).strip()
            current_content = []
        else:
            current_content.append(raw_line.rstrip())

    section_text = "\n".join(current_content).strip()
    if section_text:
        sections.append(
            {
                "heading": current_heading,
                "text": section_text,
            }
        )

    return sections


def split_paragraphs(text: str):
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]


def split_sentences(text: str):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？.!?])\s*", text)
        if sentence.strip()
    ]


def hard_split(text: str):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + MAX_CHUNK_CHARS].strip()
        if chunk:
            chunks.append(chunk)
        start += MAX_CHUNK_CHARS

    return chunks


def recursive_chunk(text: str):
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    paragraphs = split_paragraphs(text)
    if len(paragraphs) > 1:
        return pack_units(paragraphs, separator="\n\n")

    sentences = split_sentences(text)
    if len(sentences) > 1:
        return pack_units(sentences, separator="")

    return hard_split(text)


def pack_units(units, separator: str):
    chunks = []
    current_chunk = ""

    for unit in units:
        candidate = (
            f"{current_chunk}{separator}{unit}".strip()
            if current_chunk
            else unit
        )

        if len(candidate) <= MAX_CHUNK_CHARS:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)

        if len(unit) > MAX_CHUNK_CHARS:
            chunks.extend(recursive_chunk(unit))
            current_chunk = ""
        else:
            current_chunk = unit

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def merge_small_chunks(chunks):
    if not chunks:
        return []

    merged = []
    for chunk in chunks:
        if len(chunk) < MIN_CHUNK_CHARS and merged:
            candidate = merged[-1] + "\n\n" + chunk
            if len(candidate) <= MAX_CHUNK_CHARS:
                merged[-1] = candidate
            else:
                merged.append(chunk)
        else:
            merged.append(chunk)

    return merged


def create_chunks_from_markdown(markdown_text):
    sections = split_by_headings(markdown_text)
    final_chunks = []

    for section in sections:
        heading = section["heading"]
        section_text = section["text"]
        chunks = merge_small_chunks(recursive_chunk(section_text))

        for chunk in chunks:
            final_chunks.append(
                {
                    "section": heading,
                    "text": f"Section: {heading}\n\n{chunk}",
                }
            )

    return final_chunks


def create_embeddings(texts):
    model = get_embedding_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def create_chunk_id(source, section, index):
    raw = f"{source}|{section}|{index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def index_markdown(file_path, collection):
    filename = os.path.basename(file_path)
    print(f"\n正在处理：{filename}")

    markdown_text = load_markdown(file_path)
    chunks = create_chunks_from_markdown(markdown_text)
    print(f"生成 {len(chunks)} 个 Chunks。")

    if not chunks:
        return

    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    ids = []
    metadatas = []
    for index, chunk in enumerate(chunks):
        ids.append(create_chunk_id(filename, chunk["section"], index))
        metadatas.append(
            {
                "source": filename,
                "section": chunk["section"],
                "chunk_index": index,
                "chunk_method": "heading_recursive",
            }
        )

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=[embedding.tolist() for embedding in embeddings],
        metadatas=metadatas,
    )

    print("\n========== Chunk Preview ==========")
    for index, chunk in enumerate(chunks[:10], start=1):
        print(f"\n--- Chunk {index} ---")
        print(f"Section: {chunk['section']}")
        print(chunk["text"][:500])
    print("\n===================================")


def build_knowledge_base():
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"找不到目录：{INPUT_DIR}")

    markdown_files = [
        filename
        for filename in sorted(os.listdir(INPUT_DIR))
        if filename.lower().endswith(".md")
    ]

    if not markdown_files:
        print("normalized_data 中没有 Markdown 文件。")
        return

    collection = get_collection(rebuild=True)
    print(f"找到 {len(markdown_files)} 个 Markdown 文件。")

    for filename in markdown_files:
        file_path = os.path.join(INPUT_DIR, filename)
        try:
            index_markdown(file_path, collection)
        except Exception as exc:
            print(f"处理 {filename} 失败：{exc}")

    print("\nRAG Knowledge Base 构建完成。")


if __name__ == "__main__":
    build_knowledge_base()
