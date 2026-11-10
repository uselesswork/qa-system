from abc import ABC, abstractmethod
from typing import List
from src.vectorstores.search_result import SearchResult
from .schemas import AggregatedAnswer, EvidenceChunk


class BaseAnswerAggregator(ABC):

    @abstractmethod
    async def aggregate(
        self,
        query: str,
        results: List[SearchResult]
    ) -> AggregatedAnswer:
        """生成回答并挑选证据。"""
        pass


class BaseEvidenceAggregator(ABC):
    @abstractmethod
    async def aggregate(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[EvidenceChunk]:
        """只挑选证据，不生成回答。"""
        pass
