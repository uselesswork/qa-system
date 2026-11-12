import asyncio
from types import SimpleNamespace

from src.llms.openai_chat import OpenAIChatLLM


class FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            async def chunks():
                yield SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="分块回答")
                )])
            return chunks()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="完整回答"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        )


def test_default_temperature_is_not_sent_for_models_with_fixed_temperature():
    async def scenario():
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        llm = OpenAIChatLLM(client=client)

        answer = await llm.generate("问题")
        parts = [part async for part in llm.stream_generate("问题")]

        assert answer.content == "完整回答"
        assert "temperature" not in answer.metadata
        assert parts == ["分块回答"]
        assert len(completions.calls) == 2
        assert all("temperature" not in call for call in completions.calls)
        assert completions.calls[0]["model"] == "gpt-6-luna"
        assert completions.calls[1]["stream"] is True

    asyncio.run(scenario())


def test_explicit_temperature_is_sent_when_requested():
    async def scenario():
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        llm = OpenAIChatLLM(model="gpt-4o-mini", temperature=0.2, client=client)

        answer = await llm.generate("问题")
        _ = [part async for part in llm.stream_generate("问题")]

        assert answer.metadata["temperature"] == 0.2
        assert all(call["temperature"] == 0.2 for call in completions.calls)

    asyncio.run(scenario())
