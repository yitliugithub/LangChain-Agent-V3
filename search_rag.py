CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "marketing_knowledge"
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

_embedding_model = None
_collection = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        print("正在加载 Embedding Model...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding Model 加载完成。")
    return _embedding_model


def get_collection():
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def create_query_embedding(question: str):
    model = get_embedding_model()
    return model.encode(
        question,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def search_rag(question: str, top_k: int = 3):
    collection = get_collection()
    query_embedding = create_query_embedding(question)

    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def format_results(results):
    formatted = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        formatted.append(
            {
                "source": metadata.get("source"),
                "section": metadata.get("section"),
                "chunk_index": metadata.get("chunk_index"),
                "chunk_method": metadata.get("chunk_method"),
                "distance": float(distance),
                "content": document,
            }
        )

    return formatted


def print_results(question, results):
    print("\n====================================")
    print(f"问题：{question}")
    print("====================================")

    for index, item in enumerate(format_results(results), start=1):
        print(f"\n---------- Result {index} ----------")
        print(f"Source: {item['source']}")
        print(f"Section: {item['section']}")
        print(f"Distance: {item['distance']:.4f}")
        print("\nContent:")
        print(item["content"])


def main():
    print("\nRAG Retriever 已启动。")
    print("输入 quit 退出。\n")

    while True:
        question = input("请输入问题：").strip()

        if question.lower() == "quit":
            print("程序结束。")
            break
        if not question:
            continue

        try:
            results = search_rag(question, top_k=3)
            print_results(question, results)
        except Exception as exc:
            print(f"检索失败：{exc}")


if __name__ == "__main__":
    main()
