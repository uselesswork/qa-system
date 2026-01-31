from collections import defaultdict
from typing import List
from .schemas import ChatMessage

class ChatMemory:

    def __init__(self, max_turns: int = 10):
        self.store = defaultdict(list)
        self.max_turns = max_turns

    def append(self, session_id: str, message: ChatMessage):
        self.store[session_id].append(message)

        # 限制长度（防 prompt 爆炸）
        if len(self.store[session_id]) > self.max_turns * 2:
            self.store[session_id] = self.store[session_id][-self.max_turns * 2:]

    def get(self, session_id: str) -> List[ChatMessage]:
        return self.store.get(session_id, [])
