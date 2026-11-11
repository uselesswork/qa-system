import numpy as np
from typing import List
from src.rerankers.base import BaseReranker
from src.vectorstores.search_result import SearchResult
from src.utils.similarity import cosine_similarity


class EmbeddingReranker(BaseReranker):
    """
    使用 query embedding 与 chunk embedding 的余弦相似度进行 rerank
    """

    def rerank(
        self,
        query_vector: np.ndarray,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:

        if not results:
            return []

        reranked = []

        for r in results:
            chunk_vector = r.chunk.vector
            score = cosine_similarity(query_vector, chunk_vector)

            reranked.append(
                SearchResult(
                    chunk=r.chunk,
                    score=score
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
