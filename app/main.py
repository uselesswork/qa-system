from pathlib import Path

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, StreamingResponse
import json

from app.rag_app import build_rag_core
from app.api.knowledge import router as knowledge_router
from src.chat.memory import ChatMemory
from src.chat.service import ChatService, LLMQueryRewriter


class AskRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_core = await build_rag_core()

    app.state.rag_core = rag_core
    app.state.chat_service = ChatService(
        rag_core=rag_core,
        rewriter=LLMQueryRewriter(rag_core.llm, max_turns=5),
        memory=ChatMemory(max_turns=10),
    )
    print("[Lifespan] RAG system ready")

    yield

    service = getattr(app.state, "knowledge_service", None)
    if service is not None:
        service.vectors.close()
        app.state.knowledge_engine.dispose()
        app.state.knowledge_service = None
        app.state.knowledge_engine = None
    print("[Lifespan] shutdown")


app = FastAPI(
    title="Minimal RAG Backend",
    lifespan=lifespan
)
app.include_router(knowledge_router)


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
async def ask(req: AskRequest):
    rag_core = app.state.rag_core
    result = await rag_core.answer(req.query)

    return {
        "answer": result.answer,
        "evidence": serialize_evidence(result.evidence_chunks),
        "metadata": serialize_metadata(result.metadata),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    turn = await app.state.chat_service.chat(req.session_id, req.message)
    return {
        "session_id": req.session_id,
        "user_message": turn.user_message.content,
        "answer": turn.assistant_message.content,
        "rewritten_query": turn.rewritten_query,
        "evidence": serialize_evidence(turn.evidence),
        "metadata": serialize_metadata(turn.metadata),
    }


@app.get("/ask/stream")
async def ask_stream(query: str):
    """
    使用 SSE 流式返回 answer + 最终 evidence/metadata
    """
    rag_core = app.state.rag_core

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
