from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

RAG_DATA_DIR = BASE_DIR / "data/raw_pdfs"
NORMALIZED_DATA_DIR = BASE_DIR / "data/normalized"
PARSER_OUTPUT_DIR = BASE_DIR / "artifacts/pdf_parsing"
CHROMA_DIR = BASE_DIR / "storage/chroma_db"
JIEBA_DICTIONARY = BASE_DIR / "resources/jieba_dictionary/jieba_domain_dict.txt"

COLLECTION_NAME = "marketing_knowledge"

# Frozen from the completed chunking and retrieval experiments.
MAX_CHUNK_TOKENS = 500
MIN_CHUNK_TOKENS = 150
CHUNK_TOKENIZER_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_PASSAGE_PREFIX = "passage: "
EMBEDDING_MAX_LENGTH = 512
EMBEDDING_BATCH_SIZE = 16

BM25_K1 = 1.5
BM25_B = 0.75
DENSE_TOP_K = 10
BM25_TOP_K = 5
RRF_K = 60

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_LENGTH = 512
RERANKER_BATCH_SIZE = 4
FINAL_TOP_K = 3

CHUNK_METHOD = "structure_aware_semantic_sentence_boundary_500_150"
