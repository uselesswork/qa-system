from abc import ABC, abstractmethod
from .answer import LLMAnswer


class BaseLLM(ABC):

    @abstractmethod
    async def generate(self, prompt: str) -> LLMAnswer:
        """
        输入 Prompt，返回模型生成的回答
        """
        pass
