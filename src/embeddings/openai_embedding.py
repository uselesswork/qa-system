import os
import numpy as np
from typing import List
from openai import AsyncOpenAI

from ..processors.chunk import TextChunk
from .base import BaseEmbeddingModel
from .embedded_chunk import EmbeddedChunk


class OpenAIEmbedding(BaseEmbeddingModel):
    def __init__(
            self,
            model_name: str = "text-embedding-3-small",
            client: AsyncOpenAI | None = None,
    ):
        self.model_name = model_name
        self.client = client or AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    async def embed(
            self, chunks: List[TextChunk]
    ) -> List[EmbeddedChunk]:

        if not chunks:
            return []

        texts = [c.content for c in chunks]

        # === 核心：一次性批量 embedding ===
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        vectors = [
            np.array(item.embedding, dtype=np.float32)
            for item in response.data
        ]

        results: List[EmbeddedChunk] = []

        for chunk, vector in zip(chunks, vectors):
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "embedding_model": self.model_name,
                    "embedding_dim": vector.shape[0],
                }
            )

            results.append(
                EmbeddedChunk(
                    vector=vector,
                    metadata=metadata,
                )
            )

        return results

    # 新增实现
    async def embed_texts(self, texts: List[str]):
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        results = []
        for text, item in zip(texts, response.data):
            results.append(
                EmbeddedChunk(
                    vector=np.array(item.embedding),
                    metadata={
                        "embedding_model": self.model_name,
                        "embedding_dim": len(item.embedding),
                    }
                )
            )
        return results
