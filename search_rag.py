import math
import re
from collections import Counter, defaultdict

import numpy as np

from rag_config import (
    BM25_B,
    BM25_K1,
    BM25_TOP_K,
    CHROMA_DIR,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    FINAL_TOP_K,
    JIEBA_DICTIONARY,
    RERANKER_MODEL,
    RRF_K,
)
from rag_models import CrossEncoderReranker, E5Embedder


class JiebaTokenizer:
    def __init__(self, dictionary_path):
        import jieba

        if not dictionary_path.exists():
            raise FileNotFoundError(f"Jieba自定义词典不存在：{dictionary_path}")
        self.tokenizer = jieba.Tokenizer()
        self.tokenizer.initialize()
        self.tokenizer.load_userdict(str(dictionary_path))

    def tokenize(self, text):
        text = re.sub(r"\[Image(?: omitted)?:\s*[^\]]+\]", " ", text, flags=re.I)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"^Section:\s*", "", text, flags=re.MULTILINE)
        tokens = []
        pattern = re.compile(
            r"\d+(?:\.\d+)?%?|[A-Za-z][A-Za-z0-9]*|[\u4e00-\u9fff]+"
        )
        for match in pattern.finditer(text.lower()):
            part = match.group(0)
            if re.fullmatch(r"\d+(?:\.\d+)?%?|[a-z][a-z0-9]*", part):
                tokens.append(part)
            else:
                tokens.extend(
                    token.strip()
                    for token in self.tokenizer.cut(part, HMM=False)
                    if token.strip()
                )
        return tokens


class BM25Index:
    def __init__(self, documents, tokenizer, k1=BM25_K1, b=BM25_B):
        if not documents:
            raise ValueError("Cannot build BM25 index from an empty corpus")
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.tokenized_documents = [tokenizer.tokenize(text) for text in documents]
        self.document_lengths = np.array(
            [len(tokens) for tokens in self.tokenized_documents],
            dtype=np.float32,
        )
        self.average_document_length = max(
            float(self.document_lengths.mean()),
            1.0,
        )
        self.document_count = len(documents)
        self.postings = defaultdict(list)
        for document_index, tokens in enumerate(self.tokenized_documents):
            for term, frequency in Counter(tokens).items():
                self.postings[term].append((document_index, frequency))

    def idf(self, term):
        document_frequency = len(self.postings.get(term, []))
        return math.log(
            1
            + (self.document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

    def score(self, query):
        query_tokens = self.tokenizer.tokenize(query)
        scores = np.zeros(self.document_count, dtype=np.float32)
        for term in dict.fromkeys(query_tokens):
            idf = self.idf(term)
            for document_index, frequency in self.postings.get(term, []):
                document_length = self.document_lengths[document_index]
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * document_length / self.average_document_length
                )
                scores[document_index] += (
                    idf * frequency * (self.k1 + 1) / denominator
                )
        return scores, query_tokens


def reciprocal_rank_fusion(dense_ids, bm25_ids):
    dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, 1)}
    bm25_ranks = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, 1)}
    fused = []
    for chunk_id in set(dense_ranks) | set(bm25_ranks):
        score = 0.0
        if chunk_id in dense_ranks:
            score += 1 / (RRF_K + dense_ranks[chunk_id])
        if chunk_id in bm25_ranks:
            score += 1 / (RRF_K + bm25_ranks[chunk_id])
        fused.append(
            {
                "id": chunk_id,
                "rrf_score": score,
                "dense_rank": dense_ranks.get(chunk_id),
                "bm25_rank": bm25_ranks.get(chunk_id),
                "best_source_rank": min(
                    dense_ranks.get(chunk_id, 10**9),
                    bm25_ranks.get(chunk_id, 10**9),
                ),
            }
        )
    return sorted(
        fused,
        key=lambda item: (
            -item["rrf_score"],
            item["best_source_rank"],
            item["id"],
        ),
    )


class HybridRAGRetriever:
    def __init__(self):
        import chromadb

        print("[RAG] 正在加载Chroma知识库...")
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(name=COLLECTION_NAME)
        stored = self.collection.get(include=["documents", "metadatas"])
        ids = stored.get("ids", [])
        documents = stored.get("documents", [])
        metadatas = stored.get("metadatas", [])
        if not ids:
            raise RuntimeError("RAG知识库为空，请先运行 python build_rag.py")

        self.corpus = [
            {"id": chunk_id, "document": document, "metadata": metadata or {}}
            for chunk_id, document, metadata in zip(ids, documents, metadatas)
        ]
        self.by_id = {item["id"]: item for item in self.corpus}
        self.id_by_position = [item["id"] for item in self.corpus]
        self.position_by_id = {
            chunk_id: position
            for position, chunk_id in enumerate(self.id_by_position)
        }
        self.tokenizer = JiebaTokenizer(JIEBA_DICTIONARY)
        self.bm25 = BM25Index(
            [item["document"] for item in self.corpus],
            self.tokenizer,
        )

        print(f"[RAG] 正在加载Embedding模型：{EMBEDDING_MODEL}")
        self.embedder = E5Embedder(EMBEDDING_MODEL)
        print(f"[RAG] 正在加载Reranker：{RERANKER_MODEL}")
        self.reranker = CrossEncoderReranker(RERANKER_MODEL)
        print(f"[RAG] Retriever就绪，共 {len(self.corpus)} 个Chunks。")

    def search(self, question, top_k=FINAL_TOP_K):
        if not question or not question.strip():
            raise ValueError("RAG query不能为空")
        if top_k < 1:
            raise ValueError("top_k必须大于0")

        query_embedding = self.embedder.encode(
            [EMBEDDING_QUERY_PREFIX + question]
        )[0]
        dense = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(DENSE_TOP_K, len(self.corpus)),
            include=["distances"],
        )
        dense_ids = dense.get("ids", [[]])[0]
        dense_distances = dense.get("distances", [[]])[0]
        dense_distance_by_id = {
            chunk_id: float(distance)
            for chunk_id, distance in zip(dense_ids, dense_distances)
        }

        bm25_scores, query_tokens = self.bm25.score(question)
        bm25_positions = np.argsort(-bm25_scores, kind="stable")[:BM25_TOP_K]
        bm25_ids = [self.id_by_position[int(position)] for position in bm25_positions]
        bm25_score_by_id = {
            self.id_by_position[int(position)]: float(bm25_scores[int(position)])
            for position in bm25_positions
        }

        candidates = reciprocal_rank_fusion(dense_ids, bm25_ids)
        passages = [self.by_id[item["id"]]["document"] for item in candidates]
        reranker_scores = self.reranker.score(question, passages)
        ranked = sorted(
            zip(candidates, reranker_scores),
            key=lambda item: (
                -item[1],
                -item[0]["rrf_score"],
                item[0]["best_source_rank"],
            ),
        )

        results = []
        query_token_set = set(query_tokens)
        for rank, (candidate, reranker_score) in enumerate(ranked[:top_k], 1):
            stored_item = self.by_id[candidate["id"]]
            metadata = stored_item["metadata"]
            position = self.position_by_id[candidate["id"]]
            matched_tokens = sorted(
                query_token_set
                & set(self.bm25.tokenized_documents[position])
            )
            dense_distance = dense_distance_by_id.get(candidate["id"])
            results.append(
                {
                    "rank": rank,
                    "source": metadata.get("source", "Unknown"),
                    "source_file": metadata.get("source_file", ""),
                    "section": metadata.get("section", "Unknown"),
                    "chunk_index": metadata.get("chunk_index"),
                    "token_count": metadata.get("token_count"),
                    "chunk_method": metadata.get("chunk_method"),
                    "content": stored_item["document"],
                    "reranker_score": float(reranker_score),
                    "rrf_score": float(candidate["rrf_score"]),
                    "dense_rank": candidate["dense_rank"],
                    "dense_cosine_similarity": (
                        1.0 - dense_distance if dense_distance is not None else None
                    ),
                    "bm25_rank": candidate["bm25_rank"],
                    "bm25_score": bm25_score_by_id.get(candidate["id"]),
                    "matched_query_tokens": matched_tokens,
                }
            )
        return results


_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRAGRetriever()
    return _retriever


def search_rag(question, top_k=FINAL_TOP_K):
    return get_retriever().search(question, top_k=top_k)


def print_results(question, results):
    print("\n====================================")
    print(f"问题：{question}")
    print("====================================")
    for item in results:
        print(f"\n---------- Result {item['rank']} ----------")
        print(f"Source: {item['source']}")
        print(f"Section: {item['section']}")
        print(f"Reranker: {item['reranker_score']:.4f}")
        print(
            f"Dense rank: {item['dense_rank']} | "
            f"BM25 rank: {item['bm25_rank']} | RRF: {item['rrf_score']:.6f}"
        )
        print("\n" + item["content"])


def main():
    print("RAG Hybrid Retriever 已启动。输入 quit 退出。")
    while True:
        question = input("请输入问题：").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        try:
            print_results(question, search_rag(question))
        except Exception as exc:
            print(f"检索失败：{exc}")


if __name__ == "__main__":
    main()
