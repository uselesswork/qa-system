"""知识库管理与按库问答接口。"""

from pathlib import Path
import os
from threading import Lock

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.encoders import jsonable_encoder
from openai import APIError
from pydantic import BaseModel, Field

from src.db.session import create_database_engine, create_session_factory
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.knowledge.service import KnowledgeError, KnowledgeService, MAX_FILE_BYTES
from src.rag.rag_core import RAGCore
from src.retrievers.simple_retriever import SimpleRetriever
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.tasks.ingest import dispatch_ingestion
from app.api.auth import require_csrf, require_user
from src.auth.service import AuthContext
from src.security.rate_limit import enforce_limit


router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])
ROOT = Path(__file__).resolve().parents[2]
_service_lock = Lock()


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: str | None = None


class KnowledgeAsk(BaseModel):
    query: str = Field(max_length=2000)


def get_service(request: Request) -> KnowledgeService:
    service = getattr(request.app.state, "knowledge_service", None)
    if service is not None:
        return service
    with _service_lock:
        service = getattr(request.app.state, "knowledge_service", None)
        if service is not None:
            return service
        engine = vectors = None
        try:
            engine = create_database_engine()
            qdrant_url = os.getenv("QDRANT_URL")
            vectors = (QdrantVectorStore.remote(qdrant_url) if qdrant_url
                       else QdrantVectorStore.local(ROOT / "data" / "qdrant"))
            service = KnowledgeService(
                create_session_factory(engine), vectors, OpenAIEmbedding(),
                ROOT / "data" / "uploads",
            )
            request.app.state.knowledge_service = service
            request.app.state.knowledge_engine = engine
            return service
        except Exception as exc:
            if vectors is not None:
                vectors.close()
            if engine is not None:
                engine.dispose()
            raise HTTPException(503, "知识库存储未配置或不可用。") from exc


def translate(exc: KnowledgeError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


def scoped_core(core: RAGCore, service: KnowledgeService,
                ready_ids: list[str], kb_id: str | None = None) -> RAGCore:
    store = QdrantVectorStore(
        service.vectors.client, service.vectors.collection, kb_id, ready_ids
    )
    return RAGCore(
        embedder=core.embedder, vectorstore=store,
        retriever=SimpleRetriever(core.embedder, store),
        prompt_builder=core.prompt_builder, llm=core.llm,
        reranker=core.reranker, aggregator=core.aggregator,
        top_k=core.top_k, enable_answer_aware=core.enable_answer_aware,
    )


def user_core(request: Request, service: KnowledgeService, user_id: str) -> RAGCore:
    core = getattr(request.app.state, "rag_core", None)
    if core is None:
        raise HTTPException(503, "问答模型尚未就绪。")
    return scoped_core(core, service, service.owner_ready_document_ids(user_id))


@router.post("", status_code=201)
def create_knowledge_base(body: KnowledgeBaseCreate,
                          user: AuthContext = Depends(require_csrf),
                          service: KnowledgeService = Depends(get_service)):
    try:
        return service.create_kb(body.name, body.description, user.user_id)
    except KnowledgeError as exc:
        raise translate(exc) from exc


@router.get("")
def list_knowledge_bases(user: AuthContext = Depends(require_user),
                         service: KnowledgeService = Depends(get_service)):
    return service.list_kbs(user.user_id)


@router.get("/{kb_id}")
def get_knowledge_base(kb_id: str, user: AuthContext = Depends(require_user),
                       service: KnowledgeService = Depends(get_service)):
    try:
        return service.get_kb(kb_id, user.user_id)
    except KnowledgeError as exc:
        raise translate(exc) from exc


@router.get("/{kb_id}/documents")
def list_documents(kb_id: str, user: AuthContext = Depends(require_user),
                   service: KnowledgeService = Depends(get_service)):
    try:
        return service.list_documents(kb_id, user.user_id)
    except KnowledgeError as exc:
        raise translate(exc) from exc


def async_index_enabled() -> bool:
    return bool(os.getenv("QDRANT_URL") and os.getenv("REDIS_URL"))


@router.post("/{kb_id}/documents", status_code=201)
async def upload_document(kb_id: str, response: Response, request: Request, file: UploadFile = File(...),
                          user: AuthContext = Depends(require_csrf),
                          service: KnowledgeService = Depends(get_service)):
    try:
        content = await file.read(MAX_FILE_BYTES + 1)
        enforce_limit(request, user, "upload", 20)
        if async_index_enabled():
            task = service.queue_upload(kb_id, file.filename or "", content, user.user_id)
            try:
                dispatch_ingestion(task["id"])
            except Exception as exc:
                raise HTTPException(503, f"索引任务已保存但无法入队：{type(exc).__name__}。可稍后重试。") from exc
            response.status_code = 202
            return task
        return await service.upload(kb_id, file.filename or "", content, user.user_id)
    except KnowledgeError as exc:
        raise translate(exc) from exc
    finally:
        await file.close()


@router.delete("/{kb_id}/documents/{document_id}", status_code=204)
def delete_document(kb_id: str, document_id: str,
                    user: AuthContext = Depends(require_csrf),
                    service: KnowledgeService = Depends(get_service)):
    try:
        service.delete_document(kb_id, document_id, user.user_id)
    except KnowledgeError as exc:
        raise translate(exc) from exc


@router.post("/{kb_id}/documents/{document_id}/reindex")
async def reindex_document(kb_id: str, document_id: str, response: Response,
                           user: AuthContext = Depends(require_csrf),
                           service: KnowledgeService = Depends(get_service)):
    try:
        if async_index_enabled():
            task = service.queue_reindex(kb_id, document_id, user.user_id)
            try:
                dispatch_ingestion(task["id"])
            except Exception as exc:
                raise HTTPException(503, f"索引任务已保存但无法入队：{type(exc).__name__}。可稍后重试。") from exc
            response.status_code = 202
            return task
        return await service.reindex_document(kb_id, document_id, user.user_id)
    except KnowledgeError as exc:
        raise translate(exc) from exc


@router.get("/{kb_id}/tasks/{task_id}")
def get_ingestion_task(kb_id: str, task_id: str,
                       user: AuthContext = Depends(require_user),
                       service: KnowledgeService = Depends(get_service)):
    try:
        return service.get_task(kb_id, task_id, user.user_id)
    except KnowledgeError as exc:
        raise translate(exc) from exc


@router.post("/{kb_id}/tasks/{task_id}/retry", status_code=202)
def retry_ingestion_task(kb_id: str, task_id: str,
                         user: AuthContext = Depends(require_csrf),
                         service: KnowledgeService = Depends(get_service)):
    try:
        task = service.retry_task(kb_id, task_id, user.user_id)
        dispatch_ingestion(task_id)
        return task
    except KnowledgeError as exc:
        raise translate(exc) from exc
    except Exception as exc:
        raise HTTPException(503, f"任务已保存但无法入队：{type(exc).__name__}。") from exc


@router.post("/{kb_id}/ask")
async def ask_knowledge_base(kb_id: str, body: KnowledgeAsk, request: Request,
                             user: AuthContext = Depends(require_csrf),
                             service: KnowledgeService = Depends(get_service)):
    try:
        ready_ids = service.ready_document_ids(kb_id, user.user_id)
    except KnowledgeError as exc:
        raise translate(exc) from exc
    if not ready_ids:
        raise HTTPException(409, "该知识库还没有已索引的文档。")
    core = getattr(request.app.state, "rag_core", None)
    if core is None:
        raise HTTPException(503, "问答模型尚未就绪。")
    scoped_answer_core = scoped_core(core, service, ready_ids, kb_id)
    enforce_limit(request, user, "knowledge", 120)
    try:
        result = await scoped_answer_core.answer(body.query)
    except APIError as exc:
        raise HTTPException(
            502, "模型服务调用失败，请检查模型名称、请求参数和服务状态。"
        ) from exc
    return {
        "answer": result.answer,
        "evidence": [{"rank": e.rank, "score": round(e.score, 4),
                      "source": str(e.source) if e.source else None,
                      "chunk_id": e.chunk_id, "content": e.content}
                     for e in result.evidence_chunks],
        "metadata": jsonable_encoder(result.metadata or {}),
    }
