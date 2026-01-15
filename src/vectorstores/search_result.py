from dataclasses import dataclass
from typing import Any

@dataclass
class SearchResult:
    chunk: Any      # EmbeddedChunk
    score: float
