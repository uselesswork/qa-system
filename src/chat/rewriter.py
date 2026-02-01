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
        """
            输入：当前用户问题 + 对话历史
            输出：阶段感知后的“检索用 query”
        """
        pass
