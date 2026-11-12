import asyncio
import json
from types import SimpleNamespace

import httpx

from app.main import app, ask_stream
from app.api.auth import require_csrf, require_user
from app.api.chat import get_repository
from app.api.knowledge import get_service
from src.auth.service import AuthContext
from fastapi import Request
import importlib
from src.aggregators.answer_aware_aggregator import AnswerAwareAggregator
from src.aggregators.answer_multitude_aggregator import CrossDocumentAnswerAggregator
from src.aggregators.schemas import AggregatedAnswer, EvidenceChunk
from src.aggregators.simple_aggregator import AnswerAggregator
from src.chat.memory import ChatMemory
from src.chat.service import ChatService
from src.rag.rag_core import RAGCore
from src.rag.result import RAGResult
from src.retrievers.base import RetrievalBatch
from src.vectorstores.search_result import SearchResult


class FakeRetriever:
    def __init__(self):
        self.calls = []

    async def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        return [SearchResult(SimpleNamespace(content="source text", metadata={}), 0.9)]

    async def retrieve_with_vector(self, query, top_k):
        return RetrievalBatch(await self.retrieve(query, top_k))


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


def test_stream_returns_answer_and_evidence_from_one_run(monkeypatch):
    async def collect():
        aggregator = FakeAggregator()
        core, retriever, llm = make_core(aggregator)
        app.state.rag_core = core
        monkeypatch.setattr(importlib.import_module("app.main"), "user_core", lambda *_: core)
        response = await ask_stream(
            "question", Request({"type": "http", "app": app}),
            AuthContext("u", "user", "csrf"), object()
        )
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


def test_chat_service_records_complete_turn_and_evidence():
    class IdentityRewriter:
        async def rewrite(self, query, history):
            return query

    core, retriever, _ = make_core(FakeAggregator())
    memory = ChatMemory()
    app.state.chat_service = ChatService(core, IdentityRewriter(), memory)

    payload = asyncio.run(app.state.chat_service.chat("session", "question"))

    assert payload.assistant_message.content == "aggregated answer"
    assert payload.evidence[0].source == "doc.md"
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


def test_answer_aware_rerank_is_shared_by_plain_and_stream_paths():
    class ScoringLLM(FakeLLM):
        async def generate(self, prompt):
            self.generate_calls.append(prompt)
            return SimpleNamespace(content="0.8", metadata={})

    async def run(stream):
        core, _, _ = make_core(FakeAggregator())
        llm = ScoringLLM()
        core.llm = llm
        core.enable_answer_aware = True
        if stream:
            events = [event async for event in core.stream_answer_with_result("question")]
            result = events[-1].result
        else:
            result = await core.answer("question")
        return result, llm

    plain_result, plain_llm = asyncio.run(run(False))
    stream_result, stream_llm = asyncio.run(run(True))

    assert plain_result.answer == stream_result.answer == "aggregated answer"
    assert len(plain_llm.generate_calls) == len(stream_llm.generate_calls) == 2


def test_answer_aggregators_handle_no_results_without_model_calls():
    async def run():
        llm = FakeLLM()
        old = AnswerAwareAggregator(llm, None, None)
        cross_document = CrossDocumentAnswerAggregator(llm, None, None)
        return (
            await old.aggregate("question", []),
            await cross_document.aggregate("question", []),
            llm,
        )

    old, cross_document, llm = asyncio.run(run())
    assert old.evidence_chunks == cross_document.evidence_chunks == []
    assert "未检索到" in old.answer
    assert "未检索到" in cross_document.answer
    assert llm.generate_calls == []


def test_http_routes_keep_their_response_contract(monkeypatch):
    class IdentityRewriter:
        async def rewrite(self, query, history):
            return query

    aggregator = FakeAggregator()
    core, retriever, _ = make_core(aggregator)
    monkeypatch.setattr(app.state, "rag_core", core, raising=False)
    monkeypatch.setattr(importlib.import_module("app.main"), "user_core", lambda *_: core)
    monkeypatch.setattr(app.state, "query_rewriter", IdentityRewriter(), raising=False)
    class FakeRepository:
        def history(self, user_id, session_id):
            return []
        def append_turn(self, *args):
            pass
    app.dependency_overrides[require_user] = lambda: AuthContext("u", "user", "csrf")
    app.dependency_overrides[require_csrf] = lambda: AuthContext("u", "user", "csrf")
    app.dependency_overrides[get_repository] = lambda: FakeRepository()
    app.dependency_overrides[get_service] = lambda: object()

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            ask_response = await client.post("/ask", json={"query": "question"})
            chat_response = await client.post(
                "/chat", json={"session_id": "session", "message": "question"}
            )
            stream_response = await client.get("/ask/stream", params={"query": "question"})
        return ask_response, chat_response, stream_response

    try:
        ask_response, chat_response, stream_response = asyncio.run(run())
    finally:
        app.dependency_overrides.clear()

    assert ask_response.status_code == chat_response.status_code == stream_response.status_code == 200
    assert ask_response.json()["evidence"][0]["source"] == "doc.md"
    assert chat_response.json()["rewritten_query"] == "question"
    stream_events = [
        json.loads(line.removeprefix("data: "))
        for line in stream_response.text.splitlines() if line.startswith("data: ")
    ]
    assert stream_events[-1]["status"] == "Done"
    assert stream_events[-1]["answer"] == "aggregated answer"
    assert len(retriever.calls) == aggregator.calls == 3
