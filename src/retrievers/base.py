from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np
from src.vectorstores.search_result import SearchResult


@dataclass
class RetrievalBatch:
    results: List[SearchResult]
    query_vector: np.ndarray | None = None


class BaseRetriever(ABC):

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        根据 query 返回 Top-K 相关 chunk
        """
        pass

    async def retrieve_with_vector(self, query: str, top_k: int = 5) -> RetrievalBatch:
        """兼容只实现 retrieve 的检索器；可选复用检索时的问题向量。"""
        return RetrievalBatch(await self.retrieve(query=query, top_k=top_k))
