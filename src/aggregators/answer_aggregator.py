from typing import List
from src.aggregators.base import BaseAnswerAggregator
from src.aggregators.schemas import EvidenceChunk
from src.vectorstores.search_result import SearchResult


class AnswerAggregator(BaseAnswerAggregator):

    def __init__(self, max_chunks: int = 3):
        self.max_chunks = max_chunks

    async def aggregate(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[EvidenceChunk]:
        """
        将 rerank 后的 SearchResult
        转换为最终对外展示的 EvidenceChunk
        """

        evidence_chunks: List[EvidenceChunk] = []

        for idx, r in enumerate(results[: self.max_chunks]):
            chunk = r.chunk

            evidence_chunks.append(
                EvidenceChunk(
                    rank=idx + 1,
                    score=r.score,
                    content=chunk.content,
                    source=chunk.metadata.get("source"),
                    chunk_id=chunk.metadata.get("chunk_id"),
                )
            )

        return evidence_chunks
