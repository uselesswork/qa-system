"""Redis/Celery worker entry point for durable document indexing."""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from celery import Celery
from dotenv import load_dotenv
from sqlalchemy import and_, or_, select

from src.db import IngestionTask
from src.db.session import create_database_engine, create_session_factory
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.knowledge.service import KnowledgeService
from src.vectorstores.qdrant_store import QdrantVectorStore


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
celery_app = Celery("qa_ingestion", broker=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))
celery_app.conf.update(task_acks_late=True, task_reject_on_worker_lost=True,
                       worker_prefetch_multiplier=1, task_ignore_result=True)
celery_app.conf.beat_schedule = {
    "recover-ingestion-tasks": {"task": "qa.requeue_pending", "schedule": 60.0}
}


def dispatch_ingestion(task_id: str) -> None:
    celery_app.send_task("qa.ingest_document", args=[task_id], task_id=task_id)


@celery_app.task(name="qa.ingest_document")
def ingest_document(task_id: str) -> None:
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise RuntimeError("异步索引需要 QDRANT_URL；本地 Qdrant 不能由 Web 与 worker 同时打开。")
    engine = create_database_engine()
    vectors = QdrantVectorStore.remote(qdrant_url)
    try:
        service = KnowledgeService(create_session_factory(engine), vectors,
                                   OpenAIEmbedding(), ROOT / "data" / "uploads")
        asyncio.run(service.run_task(task_id))
    finally:
        vectors.close()
        engine.dispose()


@celery_app.task(name="qa.requeue_pending")
def requeue_pending() -> int:
    engine = create_database_engine()
    try:
        sessions = create_session_factory(engine)
        stale_before = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=30)
        with sessions.begin() as session:
            stale = session.scalars(select(IngestionTask).where(
                IngestionTask.status == "indexing", IngestionTask.updated_at < stale_before
            ).with_for_update()).all()
            for task in stale:
                task.status = "pending"
                task.error_message = "worker 中断，任务已重新排队。"
            ids = list(session.scalars(select(IngestionTask.id).where(
                IngestionTask.status == "pending"
            ))) + [task.id for task in stale]
        ids = list(dict.fromkeys(ids))
        for task_id in ids:
            dispatch_ingestion(task_id)
        return len(ids)
    finally:
        engine.dispose()
