from dataclasses import dataclass
from typing import List, Any


@dataclass
class EvidenceChunk:
    rank: int
    score: float
    content: str
    source: Any
    chunk_id: int


@dataclass
class AggregatedAnswer:
    answer: str
    evidence_chunks: List[EvidenceChunk]
