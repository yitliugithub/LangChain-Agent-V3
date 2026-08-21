# Author: Yiting Liu
# Time 22/8/2026 AM12:54
import chromadb

from sentence_transformers import SentenceTransformer


# ============================================================
# 1. 配置
# ============================================================

CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "marketing_knowledge"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# 2. 加载 Embedding Model
# ============================================================

print("正在加载 Embedding Model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Embedding Model 加载完成。")


# ============================================================
# 3. 连接 ChromaDB
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# 4. Query Embedding
# ============================================================

def create_query_embedding(
    question: str
):

    embedding = embedding_model.encode(

        question,

        normalize_embeddings=True,

        show_progress_bar=False

    )

    return embedding


# ============================================================
# 5. RAG Search
# ============================================================

def search_rag(
    question: str,
    top_k: int = 3
):

    # --------------------------------------------------------
    # Step 1
    # Question → Embedding
    # --------------------------------------------------------

    query_embedding = (
        create_query_embedding(
            question
        )
    )


    # --------------------------------------------------------
    # Step 2
    # Chroma Similarity Search
    # --------------------------------------------------------

    results = collection.query(

        query_embeddings=[
            query_embedding.tolist()
        ],

        n_results=top_k,

        include=[
            "documents",
            "metadatas",
            "distances"
        ]

    )


    return results


# ============================================================
# 6. 打印结果
# ============================================================

def print_results(
    question,
    results
):

    print(
        "\n===================================="
    )

    print(
        f"问题：{question}"
    )

    print(
        "===================================="
    )


    documents = (
        results["documents"][0]
    )

    metadatas = (
        results["metadatas"][0]
    )

    distances = (
        results["distances"][0]
    )


    for index, (
        document,
        metadata,
        distance
    ) in enumerate(

        zip(
            documents,
            metadatas,
            distances
        ),

        start=1

    ):


        print(
            f"\n---------- Result {index} ----------"
        )


        print(
            f"Source: "
            f"{metadata.get('source')}"
        )


        print(
            f"Section: "
            f"{metadata.get('section')}"
        )


        print(
            f"Distance: "
            f"{distance:.4f}"
        )


        print(
            "\nContent:"
        )


        print(
            document
        )


# ============================================================
# 7. CLI
# ============================================================

def main():

    print(
        "\nRAG Retriever 已启动。"
    )

    print(
        "输入 quit 退出。\n"
    )


    while True:

        question = input(
            "请输入问题："
        ).strip()


        if question.lower() == "quit":

            print(
                "程序结束。"
            )

            break


        if not question:

            continue


        try:

            results = search_rag(
                question,
                top_k=3
            )


            print_results(
                question,
                results
            )


        except Exception as e:

            print(
                f"检索失败：{e}"
            )


# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":

    main()