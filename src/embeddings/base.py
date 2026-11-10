from abc import ABC, abstractmethod
from typing import List
from ..processors.chunk import TextChunk
from .embedded_chunk import EmbeddedChunk


class BaseEmbedding(ABC):

    @abstractmethod
    async def embed(
            self,
            chunks: List[TextChunk]
    ) -> List[EmbeddedChunk]:
        pass

    @abstractmethod
    async def embed_texts(
            self,
            texts: List[str]
    ) -> List[EmbeddedChunk]:
        """
        用于 query embedding
        """
        pass
