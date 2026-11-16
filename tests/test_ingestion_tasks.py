"""Stage 13 ingestion state and idempotency using local temporary stores."""

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pytest
from sqlalchemy import create_engine, func, select

from src.db import Base, Chunk, Document, IngestionTask
from src.db.session import create_session_factory
from src.embeddings.embedded_chunk import EmbeddedChunk
from src.knowledge.service import KnowledgeError, KnowledgeService
from src.vectorstores.qdrant_store import QdrantVectorStore
from app.main import app
from app.api import knowledge as knowledge_api
from src.auth.service import AuthService
from src.tasks import ingest


class Embedding:
    fail = False

    async def embed(self, chunks):
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [EmbeddedChunk(c.content, np.array([1.0, 0.0], dtype=np.float32),
                              dict(c.metadata)) for c in chunks]


def test_queued_ingestion_is_idempotent_and_retryable(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{(tmp_path / 'tasks.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    vectors = QdrantVectorStore.local(tmp_path / "vectors")
    embedding = Embedding()
    service = KnowledgeService(factory, vectors, embedding, tmp_path / "uploads")
    kb_id = service.create_kb("测试库", None, "owner")["id"]
    queued = service.queue_upload(kb_id, "one.md", b"# Topic\nEnough source text", "owner")
    assert queued["status"] == "pending"
    with pytest.raises(KnowledgeError):
        service.queue_upload(kb_id, "copy.md", b"# Topic\nEnough source text", "owner")
    assert service.list_documents(kb_id, "owner")[0]["status"] == "pending"
    with pytest.raises(KnowledgeError):
        service.get_task(kb_id, queued["id"], "other")

    embedding.fail = True
    asyncio.run(service.run_task(queued["id"]))
    assert service.get_task(kb_id, queued["id"], "owner")["status"] == "failed"
    assert "embedding unavailable" in service.get_task(kb_id, queued["id"], "owner")["error_message"]
    assert service.list_documents(kb_id, "owner")[0]["status"] == "failed"
    embedding.fail = False
    service.retry_task(kb_id, queued["id"], "owner")
    asyncio.run(service.run_task(queued["id"]))
    asyncio.run(service.run_task(queued["id"]))  # Celery may redeliver the same task
    assert service.get_task(kb_id, queued["id"], "owner")["attempts"] == 2
    assert service.list_documents(kb_id, "owner")[0]["status"] == "ready"
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(IngestionTask)) == 1
        assert session.scalar(select(func.count()).select_from(Document)) == 1
        assert session.scalar(select(func.count()).select_from(Chunk)) == 1
    vectors.close()
    engine.dispose()


def test_http_upload_returns_task_before_indexing(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite+pysqlite:///{(tmp_path / 'api.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    vectors = QdrantVectorStore.local(tmp_path / "api_vectors")
    service = KnowledgeService(factory, vectors, Embedding(), tmp_path / "api_uploads")
    monkeypatch.setattr(app.state, "auth_service", AuthService(factory), raising=False)
    monkeypatch.setattr(app.state, "knowledge_service", service, raising=False)
    monkeypatch.setenv("REDIS_URL", "redis://test/0")
    monkeypatch.setenv("QDRANT_URL", "http://test:6333")
    dispatched = []
    monkeypatch.setattr(knowledge_api, "dispatch_ingestion", dispatched.append)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            credentials = {"username": "owner", "password": "long-password-123"}
            assert (await client.post("/auth/register", json=credentials)).status_code == 201
            login = await client.post("/auth/login", json=credentials)
            client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
            kb_id = (await client.post("/knowledge-bases", json={"name": "异步库"})).json()["id"]
            upload = await client.post(f"/knowledge-bases/{kb_id}/documents", files={
                "file": ("one.md", b"# Topic\nEnough source text", "text/markdown")
            })
            assert upload.status_code == 202, upload.text
            task_id = upload.json()["id"]
            assert dispatched == [task_id]
            status = await client.get(f"/knowledge-bases/{kb_id}/tasks/{task_id}")
            assert status.json()["status"] == "pending"
            documents = (await client.get(f"/knowledge-bases/{kb_id}/documents")).json()
            assert documents[0]["task_id"] == task_id
            assert documents[0]["status"] == "pending"

    asyncio.run(scenario())
    vectors.close()
    engine.dispose()


def test_worker_requeues_stale_indexing_task(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite+pysqlite:///{(tmp_path / 'recovery.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    vectors = QdrantVectorStore.local(tmp_path / "recovery_vectors")
    service = KnowledgeService(factory, vectors, Embedding(), tmp_path / "recovery_uploads")
    kb_id = service.create_kb("恢复库", None, "owner")["id"]
    task_id = service.queue_upload(kb_id, "one.md", b"# Topic\nSource", "owner")["id"]
    with factory.begin() as session:
        task = session.get(IngestionTask, task_id)
        task.status = "indexing"
        task.updated_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
    monkeypatch.setattr(ingest, "create_database_engine", lambda: engine)
    dispatched = []
    monkeypatch.setattr(ingest, "dispatch_ingestion", dispatched.append)
    assert ingest.requeue_pending() == 1
    assert dispatched == [task_id]
    assert service.get_task(kb_id, task_id, "owner")["status"] == "pending"
    vectors.close()
