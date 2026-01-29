import time
from openai import AsyncOpenAI
from .base import BaseLLM
from .answer import LLMAnswer


class OpenAIChatLLM(BaseLLM):

    def __init__(
            self,
            model: str = "gpt-4o-mini",
            temperature: float = 0.2,
    ):
        self.client = AsyncOpenAI()
        self.model = model
        self.temperature = temperature

    async def generate(self, prompt: str) -> LLMAnswer:
        start_time = time.time()

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )

        elapsed = time.time() - start_time

        message = response.choices[0].message.content

        return LLMAnswer(
            content=message,
            metadata={
                "model": self.model,
                "temperature": self.temperature,
                "elapsed_time": round(elapsed, 3),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        )

    # --------------------- 新增 ---------------------
    async def stream_generate(self, prompt: str):
        """
        async generator: 边生成边 yield chunk
        """
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=True  # 启用流式响应
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
