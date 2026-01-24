from abc import ABC, abstractmethod
from typing import List
from src.vectorstores.search_result import SearchResult
from .schemas import AggregatedAnswer


class BaseAnswerAggregator(ABC):

    @abstractmethod
    async def aggregate(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        对检索到的 chunks 进行聚合 / 高亮 / 合并
        返回仍然是 SearchResult 列表
        """
        pass
