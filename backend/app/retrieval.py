from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.ingestion import DocumentChunk, build_chunks
from app.rbac import collections_for_role


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_.-]+", text.lower())


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    rerank_score: float | None = None


class HybridRetriever:
    def __init__(self, data_dir: Path):
        self.chunks = build_chunks(data_dir)
        self._tokenized = [tokenize(chunk.text) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        self._embedder = None
        self._reranker = None
        self._embeddings: np.ndarray | None = None
        self._load_optional_models()

    def _load_optional_models(self) -> None:
        settings = get_settings()
        self._embeddings = np.array([self._hashed_embedding(chunk.text) for chunk in self.chunks])
        if not settings.enable_local_ml_models:
            return
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer

            self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._embeddings = np.array(self._embedder.encode([chunk.text for chunk in self.chunks], normalize_embeddings=True))
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            self._embedder = None
            self._reranker = None

    @property
    def indexed_count(self) -> int:
        return len(self.chunks)

    def retrieve(self, question: str, role: str, broad_k: int = 10, final_k: int = 3) -> list[RetrievedChunk]:
        allowed = set(collections_for_role(role))
        allowed_indices = [i for i, chunk in enumerate(self.chunks) if chunk.collection in allowed and role in chunk.access_roles]
        if not allowed_indices:
            return []

        query_tokens = tokenize(question)
        bm25_scores = self._bm25.get_scores(query_tokens) if self._bm25 else np.zeros(len(self.chunks))
        dense_scores = np.zeros(len(self.chunks))
        if self._embedder and self._embeddings is not None:
            query_embedding = np.array(self._embedder.encode([question], normalize_embeddings=True))[0]
            dense_scores = self._embeddings @ query_embedding
        elif self._embeddings is not None:
            query_embedding = self._hashed_embedding(question)
            dense_scores = self._embeddings @ query_embedding

        fused: list[RetrievedChunk] = []
        max_bm25 = max(float(max(bm25_scores)), 1.0)
        for index in allowed_indices:
            normalized_bm25 = float(bm25_scores[index]) / max_bm25
            normalized_dense = (float(dense_scores[index]) + 1.0) / 2.0
            lexical_exact = self._lexical_overlap(query_tokens, self._tokenized[index])
            score = 0.48 * normalized_bm25 + 0.42 * normalized_dense + 0.10 * lexical_exact
            fused.append(RetrievedChunk(self.chunks[index], score))

        candidates = sorted(fused, key=lambda item: item.score, reverse=True)[:broad_k]
        return self._rerank(question, candidates)[:final_k]

    def _lexical_overlap(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens:
            return 0.0
        doc = set(doc_tokens)
        return sum(1 for token in set(query_tokens) if token in doc) / math.sqrt(len(set(query_tokens)) + 1)

    def _rerank(self, question: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if self._reranker:
            pairs = [(question, item.chunk.text) for item in candidates]
            scores = self._reranker.predict(pairs)
            for item, score in zip(candidates, scores):
                item.rerank_score = float(score)
            return sorted(candidates, key=lambda item: item.rerank_score or item.score, reverse=True)
        for item in candidates:
            item.rerank_score = item.score
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def _hashed_embedding(self, text: str, dimensions: int = 256) -> np.ndarray:
        vector = np.zeros(dimensions, dtype=float)
        for token in tokenize(text):
            bucket = hash(token) % dimensions
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(get_settings().data_dir)
    return _retriever
