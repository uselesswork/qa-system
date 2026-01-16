from abc import ABC, abstractmethod
from typing import List


class BasePromptBuilder(ABC):

    @abstractmethod
    def build(
            self,
            query: str,
            contexts: List[str]
    ) -> str:
        """
        根据 query 和上下文内容构建 prompt
        """

    pass
