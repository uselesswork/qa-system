"""阶段 12：显式联网、引用和会话边界的离线回归。"""

import asyncio
from types import SimpleNamespace

import httpx
from sqlalchemy import create_engine

from app.main import app
from src.auth.service import AuthService
from src.chat.web_search import WebSearchService, WebSearchUnavailable
from src.db import Base, Document
from src.db.session import create_session_factory
from src.knowledge.service import KnowledgeService


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.output


def test_web_search_uses_only_current_question_and_extracts_citations():
    annotation = SimpleNamespace(
        type="url_citation", url="https://example.com/report", title="示例报告",
        start_index=3, end_index=6,
    )
    response = SimpleNamespace(
        output_text="结论 [1]",
        output=[SimpleNamespace(type="web_search_call", content=[]),
                SimpleNamespace(type="message", content=[SimpleNamespace(
                    type="output_text", annotations=[annotation, annotation])])],
    )
    responses = FakeResponses(response)
    service = WebSearchService(SimpleNamespace(responses=responses), model="search-test")
    result = asyncio.run(service.search("当前问题"))

    assert result["answer"] == "结论 [1]"
    assert result["evidence"] == [{"source_type": "web", "title": "示例报告",
                                   "url": "https://example.com/report", "rank": 1}]
    assert result["metadata"]["citations"][0]["start_index"] == 3
    assert responses.requests[0]["input"] == "当前问题"
    assert responses.requests[0]["tool_choice"] == "required"
    assert responses.requests[0]["store"] is False


def test_web_search_rejects_answer_without_web_source():
    response = SimpleNamespace(output_text="没有来源的回答", output=[])
    service = WebSearchService(SimpleNamespace(responses=FakeResponses(response)),
                               model="search-test")
    try:
        asyncio.run(service.search("问题"))
    except WebSearchUnavailable:
        pass
    else:
        raise AssertionError("没有网页来源的回答不应返回")


def test_web_mode_requires_explicit_choice_and_persists_sources(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite+pysqlite:///{(tmp_path / 'web.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    knowledge = KnowledgeService(factory, None, None, tmp_path / "uploads")
    monkeypatch.setattr(app.state, "auth_service", AuthService(factory), raising=False)
    monkeypatch.setattr(app.state, "knowledge_service", knowledge, raising=False)
    monkeypatch.setattr(app.state, "query_rewriter", SimpleNamespace(), raising=False)

    class FakeWeb:
        def __init__(self):
            self.queries = []
            self.fail = False

        async def search(self, question):
            self.queries.append(question)
            if self.fail:
                raise WebSearchUnavailable("没有网页来源")
            return {"answer": "网页答案 [1]",
                    "evidence": [{"source_type": "web", "rank": 1,
                                  "title": "来源", "url": "https://example.com/a"}],
                    "metadata": {"mode": "web", "model": "fake",
                                 "citations": [{"url": "https://example.com/a",
                                                "title": "来源", "start_index": 5,
                                                "end_index": 8}]}}

    web = FakeWeb()
    monkeypatch.setattr(app.state, "web_search_service", web, raising=False)

    async def sign_in(client, name):
        body = {"username": name, "password": "long-test-password-123"}
        assert (await client.post("/auth/register", json=body)).status_code == 201
        login = await client.post("/auth/login", json=body)
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as alice, \
                   httpx.AsyncClient(transport=transport, base_url="http://test") as bob:
            await sign_in(alice, "alice")
            await sign_in(bob, "bob")
            kb = (await alice.post("/knowledge-bases", json={"name": "空库"})).json()
            session_id = (await alice.post(
                "/chat/sessions", json={"knowledge_base_id": kb["id"]}
            )).json()["id"]
            path = f"/chat/sessions/{session_id}/messages"

            assert (await alice.post(path, json={"message": "默认问题"})).status_code == 409
            assert web.queries == []
            assert (await bob.post(path, json={"message": "越权", "mode": "web"})).status_code == 404
            assert (await alice.post(path, json={"message": "无 CSRF", "mode": "web"},
                                     headers={"X-CSRF-Token": "wrong"})).status_code == 403
            assert web.queries == []

            answer = await alice.post(path, json={"message": "  当前问题  ", "mode": "web"})
            assert answer.status_code == 200, answer.text
            assert answer.json()["metadata"]["mode"] == "web"
            assert answer.json()["evidence"][0]["url"] == "https://example.com/a"
            assert web.queries == ["当前问题"]
            saved = (await alice.get(f"/chat/sessions/{session_id}")).json()
            assert len(saved["messages"]) == 2
            assert saved["messages"][1]["metadata"]["mode"] == "web"
            assert saved["messages"][1]["evidence"][0]["source_type"] == "web"

            web.fail = True
            assert (await alice.post(path, json={"message": "失败问题", "mode": "web"})).status_code == 502
            assert len((await alice.get(f"/chat/sessions/{session_id}")).json()["messages"]) == 2

            with factory.begin() as db:
                db.add(Document(knowledge_base_id=kb["id"], original_name="ready.md",
                                file_type="md", sha256="a" * 64, size_bytes=1,
                                status="ready"))
            assert (await alice.post(path, json={"message": "已有资料", "mode": "web"})).status_code == 409
            assert web.queries == ["当前问题", "失败问题"]

    try:
        asyncio.run(scenario())
    finally:
        engine.dispose()
