from dataclasses import dataclass, field
from typing import Dict, Any
import time
import uuid


@dataclass
class ChatSession:
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new() -> "ChatSession":
        return ChatSession(session_id=str(uuid.uuid4()))

    def touch(self):
        self.updated_at = time.time()
