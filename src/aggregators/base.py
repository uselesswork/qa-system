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
    ) -> AggregatedAnswer:
        pass
