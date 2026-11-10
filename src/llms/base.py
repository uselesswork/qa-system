from abc import ABC, abstractmethod
from typing import AsyncIterator
from .answer import LLMAnswer


class BaseLLM(ABC):

    @abstractmethod
    async def generate(self, prompt: str) -> LLMAnswer:
        """
        输入 Prompt，返回模型生成的回答
        """
        pass

    @abstractmethod
    def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """逐段生成文本。"""
        pass
