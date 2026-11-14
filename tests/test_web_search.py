"""阶段 12：显式联网、引用和会话边界的离线回归。"""

import asyncio
from types import SimpleNamespace

import httpx
from sqlalchemy import create_engine

from app.main import app
from src.auth.service import AuthService
from src.chat.repository import ChatRepository
from src.chat.web_search import WebSearchService, WebSearchUnavailable
from src.chat.web_context import answer_from_web_context
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
    assert responses.requests[0]["include"] == ["web_search_call.action.sources"]


def test_web_search_accepts_search_call_sources_without_inline_annotations():
    response = SimpleNamespace(
        output_text="网页答案",
        output=[SimpleNamespace(type="web_search_call", action=SimpleNamespace(
            sources=[SimpleNamespace(url="https://example.com/report", title="报告")]
        )), SimpleNamespace(type="message", content=[SimpleNamespace(
            type="output_text", annotations=[]
        )])],
    )
    service = WebSearchService(SimpleNamespace(responses=FakeResponses(response)), model="search-test")
    result = asyncio.run(service.search("问题"))
    assert result["evidence"][0]["url"] == "https://example.com/report"
    assert result["metadata"]["citations"][0]["url"] == "https://example.com/report"
    assert "来源：" in result["answer"]


def test_saved_web_answer_can_support_followup_without_new_search():
    class Repository:
        def __init__(self):
            self.saved = []

        def get(self, user_id, session_id):
            assert (user_id, session_id) == ("owner", "session")
            return {"messages": [
                {"role": "user", "content": "明朝何时灭亡？"},
                {"role": "assistant", "content": "明朝于1644年灭亡。",
                 "metadata": {"mode": "web"},
                 "evidence": [{"source_type": "web", "url": "https://example.com/ming",
                               "title": "史料", "rank": 1}]},
            ]}

        def append_turn(self, *args):
            self.saved.append(args)

    class LLM:
        model = "fake"
        async def generate(self, prompt):
            if "只返回 JSON" in prompt:
                return SimpleNamespace(content=(
                    '{"supported": true, "index": 1, "quote": "1644年灭亡"}'
                ))
            return SimpleNamespace(content="1644年。")

    repo = Repository()
    result = asyncio.run(answer_from_web_context(
        repo, LLM(), "owner", "session", "那是哪一年？"
    ))
    assert result["answer"] == "1644年。"
    assert result["evidence"][0]["url"] == "https://example.com/ming"
    assert result["metadata"]["mode"] == "session_web_context"
    assert len(repo.saved) == 1


def test_unrelated_saved_web_answer_is_not_used_as_evidence():
    class Repository:
        def get(self, user_id, session_id):
            return {"messages": [
                {"role": "user", "content": "明朝何时灭亡？"},
                {"role": "assistant", "content": "明朝于1644年灭亡。",
                 "metadata": {"mode": "web"},
                 "evidence": [{"source_type": "web", "url": "https://example.com/ming"}]},
            ]}

        def append_turn(self, *args):
            raise AssertionError("无关问题不应保存网页上下文回答")

    class LLM:
        async def generate(self, prompt):
            return SimpleNamespace(content='{"supported": false, "index": 0, "quote": ""}')

    assert asyncio.run(answer_from_web_context(
        Repository(), LLM(), "owner", "session", "唐朝何时建立？"
    )) is None


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

    class ContextLLM:
        model = "fake"
        async def generate(self, prompt):
            if "只返回 JSON" in prompt:
                return SimpleNamespace(content='{"supported": true, "index": 1, "quote": "网页答案"}')
            return SimpleNamespace(content="根据已保存的网页答案。")

    monkeypatch.setattr(app.state, "rag_core", SimpleNamespace(llm=ContextLLM()), raising=False)

    async def sign_in(client, name):
        body = {"username": name, "password": "long-test-password-123"}
        assert (await client.post("/auth/register", json=body)).status_code == 201
        login = await client.post("/auth/login", json=body)
        client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        return login.json()["id"]

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as alice, \
                   httpx.AsyncClient(transport=transport, base_url="http://test") as bob:
            alice_id = await sign_in(alice, "alice")
            await sign_in(bob, "bob")
            assert (await alice.post("/chat/sessions", json={})).status_code == 400
            kb = (await alice.post("/knowledge-bases", json={"name": "空库"})).json()
            assert (await alice.post("/chat/sessions", json={
                "knowledge_base_id": kb["id"], "session_type": "web"
            })).status_code == 400
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
            web.fail = False
            assert (await alice.post(path, json={"message": "已有资料也联网", "mode": "web"})).status_code == 200

            web_only = await alice.post("/chat/sessions", json={"session_type": "web"})
            assert web_only.status_code == 201
            assert web_only.json()["knowledge_base_id"] is None
            assert web_only.json()["session_type"] == "web"
            web_path = f"/chat/sessions/{web_only.json()['id']}/messages"
            assert (await alice.post(web_path, json={"message": "不开联网"})).status_code == 409
            assert (await alice.post(web_path, json={"message": "无知识库联网", "mode": "web"})).status_code == 200
            assert (await alice.get(f"/chat/sessions/{web_only.json()['id']}")).json()["messages"][1]["metadata"]["mode"] == "web"
            cached = await alice.post(web_path, json={"message": "那答案是什么？"})
            assert cached.status_code == 200, cached.text
            assert cached.json()["metadata"]["mode"] == "session_web_context"
            assert cached.json()["evidence"][0]["url"] == "https://example.com/a"
            assert len(web.queries) == 4
            retried = await alice.post(web_path, json={"message": "重新回答", "mode": "web"})
            assert retried.status_code == 200
            assert "前一个问题：那答案是什么？" in web.queries[-1]

            legacy_id = ChatRepository(factory).create(alice_id)["id"]
            legacy_path = f"/chat/sessions/{legacy_id}/messages"
            assert (await alice.post(legacy_path, json={"message": "旧会话联网", "mode": "web"})).status_code == 200
            assert web.queries[:4] == ["当前问题", "失败问题", "已有资料也联网",
                                      "无知识库联网"]
            assert web.queries[-1] == "旧会话联网"

    try:
        asyncio.run(scenario())
    finally:
        engine.dispose()
