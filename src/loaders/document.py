from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
