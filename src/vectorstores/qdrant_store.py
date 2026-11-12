"""按知识库过滤的 Qdrant 向量存储；本地模式可持久化到磁盘。"""

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from src.embeddings.embedded_chunk import EmbeddedChunk
from src.vectorstores.base import BaseVectorStore
from src.vectorstores.search_result import SearchResult


class VectorDimensionError(ValueError):
    """现有集合的向量维度与当前 Embedding 模型不一致。"""


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, client: QdrantClient, collection: str = "qa_chunks",
                 knowledge_base_id: str | None = None,
                 ready_document_ids: list[str] | None = None):
        self.client = client
        self.collection = collection
        self.knowledge_base_id = knowledge_base_id
        self.ready_document_ids = ready_document_ids

    @classmethod
    def local(cls, path: Path, collection: str = "qa_chunks"):
        path.mkdir(parents=True, exist_ok=True)
        return cls(QdrantClient(path=str(path)), collection)

    def ensure_collection(self, dimension: int) -> None:
        if self.client.collection_exists(self.collection):
            existing = self.client.get_collection(self.collection).config.params.vectors
            if existing.size != dimension:
                raise VectorDimensionError(
                    f"向量维度不匹配：集合为 {existing.size}，新向量为 {dimension}。"
                )
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return
        self.ensure_collection(len(chunks[0].vector))
        points = []
        for chunk in chunks:
            metadata = chunk.metadata
            points.append(models.PointStruct(
                id=metadata["chunk_id"],
                vector=np.asarray(chunk.vector).tolist(),
                payload={**metadata, "content": chunk.content},
            ))
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if not self.knowledge_base_id or not self.ready_document_ids:
            return []
        if not self.client.collection_exists(self.collection):
            return []
        points = self.client.query_points(
            collection_name=self.collection,
            query=np.asarray(query_vector).tolist(),
            query_filter=models.Filter(must=[
                models.FieldCondition(key="knowledge_base_id", match=models.MatchValue(value=self.knowledge_base_id)),
                models.FieldCondition(key="document_id", match=models.MatchAny(any=self.ready_document_ids)),
            ]),
            with_payload=True,
            with_vectors=True,
            limit=top_k,
        ).points
        return [SearchResult(
            chunk=EmbeddedChunk(
                content=point.payload["content"],
                vector=np.asarray(point.vector, dtype=np.float32),
                metadata={**point.payload, "chunk_id": str(point.id)},
            ),
            score=point.score,
        ) for point in points]

    def delete_points(self, ids: list[str]) -> None:
        if ids and self.client.collection_exists(self.collection):
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.PointIdsList(points=ids),
                wait=True,
            )

    def clear(self) -> None:
        raise RuntimeError("不能清空共享集合；请按文档 ID 删除向量。")

    def close(self) -> None:
        self.client.close()
