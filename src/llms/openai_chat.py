import time
from openai import AsyncOpenAI
from .base import BaseLLM
from .answer import LLMAnswer


class OpenAIChatLLM(BaseLLM):

    def __init__(
            self,
            model: str = "gpt-6-luna",
            temperature: float | None = None,
            client: AsyncOpenAI | None = None,
    ):
        self.client = client or AsyncOpenAI()
        self.model = model
        self.temperature = temperature

    def _request_options(self, prompt: str) -> dict:
        options = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        # 有些模型仅接受服务端默认温度；None 表示不发送该参数。
        if self.temperature is not None:
            options["temperature"] = self.temperature
        return options

    async def generate(self, prompt: str) -> LLMAnswer:
        start_time = time.time()

        response = await self.client.chat.completions.create(
            **self._request_options(prompt)
        )

        elapsed = time.time() - start_time

        message = response.choices[0].message.content

        metadata = {
                "model": self.model,
                "elapsed_time": round(elapsed, 3),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
        }
        if self.temperature is not None:
            metadata["temperature"] = self.temperature

        return LLMAnswer(content=message, metadata=metadata)

    # --------------------- 新增 ---------------------
    async def stream_generate(self, prompt: str):
        """
        async generator: 边生成边 yield chunk
        """
        stream = await self.client.chat.completions.create(
            **self._request_options(prompt),
            stream=True  # 启用流式响应
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
