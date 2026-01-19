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
