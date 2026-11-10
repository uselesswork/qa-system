import asyncio
import json
from types import SimpleNamespace

from app.main import ChatRequest, app, ask_stream, chat
from src.aggregators.answer_aware_aggregator import AnswerAwareAggregator
from src.aggregators.schemas import AggregatedAnswer, EvidenceChunk
from src.aggregators.simple_aggregator import AnswerAggregator
from src.chat.memory import ChatMemory
from src.chat.service import ChatService
from src.rag.rag_core import RAGCore
from src.rag.result import RAGResult
from src.vectorstores.search_result import SearchResult


class FakeRetriever:
    def __init__(self):
        self.calls = []

    async def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        return [SearchResult(SimpleNamespace(content="source text", metadata={}), 0.9)]


class FakeLLM:
    def __init__(self):
        self.generate_calls = []
        self.stream_calls = []

    async def generate(self, prompt):
        self.generate_calls.append(prompt)
        return SimpleNamespace(content="plain answer", metadata={"model": "fake"})

    async def stream_generate(self, prompt):
        self.stream_calls.append(prompt)
        yield "plain "
        yield "answer"


class FakeAggregator:
    def __init__(self):
        self.calls = 0

    async def aggregate(self, query, results):
        self.calls += 1
        return AggregatedAnswer(
            answer="aggregated answer",
            evidence_chunks=[EvidenceChunk(1, 0.8, "source text", "doc.md", 1)],
        )


def make_core(aggregator=None):
    retriever = FakeRetriever()
    llm = FakeLLM()
    core = RAGCore(
        embedder=None,
        vectorstore=None,
        retriever=retriever,
        prompt_builder=SimpleNamespace(build=lambda query, chunks: f"Q: {query}"),
        llm=llm,
        aggregator=aggregator,
        top_k=3,
    )
    return core, retriever, llm


def test_aggregated_answer_does_not_generate_twice():
    aggregator = FakeAggregator()
    core, retriever, llm = make_core(aggregator)

    result = asyncio.run(core.answer("question"))

    assert result.answer == "aggregated answer"
    assert result.evidence_chunks[0].source == "doc.md"
    assert retriever.calls == [("question", 6)]
    assert aggregator.calls == 1
    assert llm.generate_calls == []
    assert result.metadata["llm"] == {}


def test_plain_answer_generates_once():
    core, retriever, llm = make_core()

    result = asyncio.run(core.answer("question"))

    assert result.answer == "plain answer"
    assert result.evidence_chunks == []
    assert result.metadata["llm"] == {"model": "fake"}
    assert retriever.calls == [("question", 6)]
    assert llm.generate_calls == ["Q: question"]


def test_stream_returns_answer_and_evidence_from_one_run():
    async def collect():
        aggregator = FakeAggregator()
        core, retriever, llm = make_core(aggregator)
        app.state.rag_core = core
        response = await ask_stream("question")
        events = []
        async for data in response.body_iterator:
            events.append(json.loads(data.removeprefix("data: ")))
        return events, retriever, aggregator, llm

    events, retriever, aggregator, llm = asyncio.run(collect())

    assert "".join(event["answer_chunk"] for event in events[:-1]) == "aggregated answer"
    assert events[-1]["answer"] == "aggregated answer"
    assert events[-1]["evidence"][0]["source"] == "doc.md"
    assert retriever.calls == [("question", 6)]
    assert aggregator.calls == 1
    assert llm.generate_calls == []


def test_plain_stream_returns_same_final_answer():
    async def collect():
        core, retriever, llm = make_core()
        events = [event async for event in core.stream_answer_with_result("question")]
        return events, retriever, llm

    events, retriever, llm = asyncio.run(collect())

    assert "".join(event.answer_chunk for event in events[:-1]) == "plain answer"
    assert isinstance(events[-1].result, RAGResult)
    assert events[-1].result.answer == "plain answer"
    assert retriever.calls == [("question", 6)]
    assert llm.generate_calls == []
    assert llm.stream_calls == ["Q: question"]


def test_failed_chat_does_not_record_half_turn():
    class FailingRAG:
        async def answer(self, query):
            raise RuntimeError("model unavailable")

    class IdentityRewriter:
        async def rewrite(self, query, history):
            return query

    memory = ChatMemory()
    service = ChatService(FailingRAG(), IdentityRewriter(), memory)

    async def run():
        try:
            await service.chat("session", "question")
        except RuntimeError:
            pass

    asyncio.run(run())
    assert memory.get("session") == []


def test_chat_api_records_complete_turn_and_evidence():
    class IdentityRewriter:
        async def rewrite(self, query, history):
            return query

    core, retriever, _ = make_core(FakeAggregator())
    memory = ChatMemory()
    app.state.chat_service = ChatService(core, IdentityRewriter(), memory)

    payload = asyncio.run(chat(ChatRequest(session_id="session", message="question")))

    assert payload["answer"] == "aggregated answer"
    assert payload["evidence"][0]["source"] == "doc.md"
    assert [message.role for message in memory.get("session")] == ["user", "assistant"]
    assert retriever.calls == [("question", 6)]


def test_aggregator_contracts_separate_answers_from_evidence_only():
    result = SearchResult(
        SimpleNamespace(
            content="source text",
            vector=[1.0, 0.0],
            metadata={"source": "doc.md", "chunk_id": 1},
        ),
        0.9,
    )

    class FakeEmbedder:
        async def embed_texts(self, texts):
            return [SimpleNamespace(vector=[1.0, 0.0]) for _ in texts]

    async def run():
        evidence = await AnswerAggregator(max_chunks=1).aggregate("question", [result])
        llm = FakeLLM()
        answer_aggregator = AnswerAwareAggregator(
            llm=llm,
            prompt_builder=SimpleNamespace(build=lambda query, chunks: "prompt"),
            embedder=FakeEmbedder(),
            max_chunks=1,
        )
        answer = await answer_aggregator.aggregate("question", [result])
        return evidence, answer

    evidence, answer = asyncio.run(run())

    assert len(evidence) == 1
    assert answer.answer == "plain answer"
    assert answer.evidence_chunks[0].source == "doc.md"
