import gc

import numpy as np

from rag_config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_LENGTH,
    RERANKER_BATCH_SIZE,
    RERANKER_MAX_LENGTH,
)


def select_device(torch):
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class E5Embedder:
    """E5 mean-pooling embedder matching the evaluated implementation."""

    def __init__(self, model_name):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = select_device(torch)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size=EMBEDDING_BATCH_SIZE):
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=EMBEDDING_MAX_LENGTH,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                }
                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                summed = (token_embeddings * attention_mask).sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1)
                pooled = summed / counts
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(embeddings)

    def close(self):
        self.model.to("cpu")
        del self.model
        del self.tokenizer
        gc.collect()
        if self.torch.backends.mps.is_available():
            self.torch.mps.empty_cache()


class CrossEncoderReranker:
    """Scores each original-query/chunk pair and returns comparable logits."""

    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = select_device(torch)
        self.model.to(self.device)
        self.model.eval()

    def score(self, query, passages):
        scores = []
        with self.torch.no_grad():
            for start in range(0, len(passages), RERANKER_BATCH_SIZE):
                batch = passages[start:start + RERANKER_BATCH_SIZE]
                encoded = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=RERANKER_MAX_LENGTH,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                }
                logits = self.model(**encoded).logits
                batch_scores = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
                scores.extend(float(value) for value in batch_scores.cpu().tolist())
        return scores

    def close(self):
        self.model.to("cpu")
        del self.model
        del self.tokenizer
        gc.collect()
        if self.torch.backends.mps.is_available():
            self.torch.mps.empty_cache()
