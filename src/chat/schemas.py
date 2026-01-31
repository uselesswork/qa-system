from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ChatMessage:
    role: str          # "user" | "assistant"
    content: str

@dataclass
class Evidence:
    rank: int
    score: float
    source: str
    chunk_id: str
    content: str

@dataclass
class ChatTurn:
    user_message: ChatMessage
    assistant_message: ChatMessage
    rewritten_query: str
    evidence: List[Evidence]
    metadata: Optional[dict] = None
