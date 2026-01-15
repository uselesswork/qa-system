from abc import ABC, abstractmethod
from typing import List
from src.vectorstores.search_result import SearchResult


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
