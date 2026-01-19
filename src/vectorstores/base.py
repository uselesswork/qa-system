from abc import ABC, abstractmethod
from typing import List
import numpy as np
from .search_result import SearchResult
from src.embeddings.embedded_chunk import EmbeddedChunk


class BaseVectorStore(ABC):

    @abstractmethod
    def add(self, chunks: List[EmbeddedChunk]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
