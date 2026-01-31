from abc import ABC, abstractmethod
from typing import List
from .schemas import ChatMessage

class BaseQueryRewriter(ABC):

    @abstractmethod
    async def rewrite(
        self,
        query: str,
        history: List[ChatMessage]
    ) -> str:
        pass
