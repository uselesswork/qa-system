from dataclasses import dataclass
from typing import List, Dict, Any
from src.processors.chunk import TextChunk


@dataclass
class RAGResult:
    answer: str
    sources: List[TextChunk]
    metadata: Dict[str, Any]
    evidence_chunks: Dict[str, Any] = None # 新增
