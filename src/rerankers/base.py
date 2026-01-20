from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.vectorstores.search_result import SearchResult


class BaseReranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query_vector: np.ndarray,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        对初始检索结果进行二次排序
        """
        pass
