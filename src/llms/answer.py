from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class LLMAnswer:
    content: str
    metadata: Dict[str, Any]
