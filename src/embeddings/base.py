from abc import ABC, abstractmethod
from typing import List
from ..processors.chunk import TextChunk
from .embedded_chunk import EmbeddedChunk


class BaseEmbeddingModel(ABC):

    # async 接口冻结，未来可无缝替换为本地模型 / 其他云模型
    @abstractmethod
    async def embed(
        self, chunks: List[TextChunk]
    ) -> List[EmbeddedChunk]:
        """
        将 TextChunk 批量转换为 EmbeddedChunk（异步）
        """
        pass