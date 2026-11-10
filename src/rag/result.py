from dataclasses import dataclass, field
from typing import Any

from src.aggregators.schemas import EvidenceChunk
from src.vectorstores.search_result import SearchResult


@dataclass
class RAGResult:
    answer: str
    sources: list[SearchResult]
    metadata: dict[str, Any]
    evidence_chunks: list[EvidenceChunk] = field(default_factory=list)


@dataclass
class RAGStreamEvent:
    answer_chunk: str | None = None
    result: RAGResult | None = None
