from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, StreamingResponse
import json
from openai import APIError

from app.rag_app import build_rag_core
from app.api.knowledge import router as knowledge_router
from app.api.knowledge import get_service, user_core
from app.api.auth import router as auth_router, require_csrf, require_user
from app.api.chat import router as chat_router, get_repository
from src.auth.service import AuthContext
from src.chat.persistent import PersistentChatService
from src.chat.repository import ChatNotFound, ChatRepository
from src.knowledge.service import KnowledgeService
from src.chat.service import LLMQueryRewriter


class AskRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_core = await build_rag_core()

    app.state.rag_core = rag_core
    app.state.query_rewriter = LLMQueryRewriter(rag_core.llm, max_turns=5)
    print("[Lifespan] RAG system ready")

    yield

    service = getattr(app.state, "knowledge_service", None)
    if service is not None:
        service.vectors.close()
        app.state.knowledge_engine.dispose()
        app.state.knowledge_service = None
        app.state.knowledge_engine = None
    auth_engine = getattr(app.state, "auth_engine", None)
    if auth_engine is not None:
        auth_engine.dispose()
        app.state.auth_service = None
        app.state.auth_engine = None
    print("[Lifespan] shutdown")


app = FastAPI(
    title="Minimal RAG Backend",
    lifespan=lifespan
)
app.include_router(knowledge_router)
app.include_router(auth_router)
app.include_router(chat_router)


def serialize_evidence(items):
    return [
        {
            "rank": item.rank,
            "score": round(item.score, 4),
            "source": str(item.source) if item.source is not None else None,
            "chunk_id": item.chunk_id,
            "content": item.content,
        }
        for item in items
    ]


def serialize_metadata(metadata):
    return jsonable_encoder(metadata or {})


@app.post("/ask")
async def ask(req: AskRequest, request: Request,
              user: AuthContext = Depends(require_csrf),
              knowledge: KnowledgeService = Depends(get_service)):
    rag_core = user_core(request, knowledge, user.user_id)
    try:
        result = await rag_core.answer(req.query)
    except APIError as exc:
        raise HTTPException(502, "模型服务调用失败，请检查模型配置与服务状态。") from exc

    return {
        "answer": result.answer,
        "evidence": serialize_evidence(result.evidence_chunks),
        "metadata": serialize_metadata(result.metadata),
    }


@app.post("/chat")
async def chat(req: ChatRequest, request: Request,
               user: AuthContext = Depends(require_csrf),
               repository: ChatRepository = Depends(get_repository),
               knowledge: KnowledgeService = Depends(get_service)):
    service = PersistentChatService(repository, user_core(request, knowledge, user.user_id),
                                    app.state.query_rewriter)
    try:
        return await service.chat(user.user_id, req.session_id, req.message)
    except ChatNotFound as exc:
        raise HTTPException(404, "会话不存在。") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except APIError as exc:
        raise HTTPException(502, "模型服务调用失败，请检查模型配置与服务状态。") from exc


@app.get("/ask/stream")
async def ask_stream(query: str, request: Request,
                     user: AuthContext = Depends(require_csrf),
                     knowledge: KnowledgeService = Depends(get_service)):
    """
    使用 SSE 流式返回 answer + 最终 evidence/metadata
    """
    rag_core = user_core(request, knowledge, user.user_id)

    async def event_generator():
        async for event in rag_core.stream_answer_with_result(query):
            if event.answer_chunk is not None:
                yield f"data: {json.dumps({'answer_chunk': event.answer_chunk}, ensure_ascii=False)}\n\n"
                continue
            result = event.result

            final_data = {
                'status': 'Done',
                'answer': result.answer,
                'evidence': serialize_evidence(result.evidence_chunks),
                'metadata': serialize_metadata(result.metadata),
            }

            yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        template = Path(__file__).resolve().parent / "templates" / "index.html"
        with template.open("r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Welcome to RAG System</h1><p>Template file not found.</p>"
    except Exception as e:
        return f"<h1>Error loading page</h1><p>{str(e)}</p>"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8090)
