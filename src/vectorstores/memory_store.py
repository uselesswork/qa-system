import numpy as np
from typing import List
from .base import BaseVectorStore
from .search_result import SearchResult
from src.embeddings.embedded_chunk import EmbeddedChunk


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class InMemoryVectorStore(BaseVectorStore):

    def __init__(self):
        self._chunks: List[EmbeddedChunk] = []

    def add(self, chunks: List[EmbeddedChunk]) -> None:
        self._chunks.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:

        results: List[SearchResult] = []

        for chunk in self._chunks:
            score = cosine_similarity(query_vector, chunk.vector)
            results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        self._chunks.clear()
