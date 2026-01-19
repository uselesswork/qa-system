from abc import ABC, abstractmethod
from typing import List

from src.vectorstores.search_result import SearchResult

class BasePromptBuilder(ABC):

    @abstractmethod
    def build(
            self,
            query: str,
            chunks: List[SearchResult] # 修改这里
    ) -> str:
        """
        根据 query 和上下文内容构建 prompt
        """

    pass
