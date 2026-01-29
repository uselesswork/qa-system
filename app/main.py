from pathlib import Path

from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, StreamingResponse
import json

from app.rag_app import build_rag_core


class AskRequest(BaseModel):
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_core = build_rag_core()

    store = rag_core.vectorstore
    embedder = rag_core.embedder

    embedded_chunks = await embedder.embed(store._pending_chunks)
    store.add(embedded_chunks)

    app.state.rag_core = rag_core
    print("[Lifespan] RAG system ready")

    yield

    print("[Lifespan] shutdown")


app = FastAPI(
    title="Minimal RAG Backend",
    lifespan=lifespan
)


@app.post("/ask")
async def ask(req: AskRequest):
    rag_core = app.state.rag_core
    result = await rag_core.answer(req.query)

    return {
        "answer": result.answer,
        "evidence": [
            {
                "rank": e.rank,
                "score": round(e.score, 4),
                "source": e.source,
                "chunk_id": e.chunk_id,
                "content": e.content,
            }
            for e in result.evidence_chunks
        ],
        "metadata": result.metadata
    }


@app.get("/ask/stream")
async def ask_stream(query: str):
    """
    使用 SSE 流式返回 answer + 最终 evidence/metadata
    """
    rag_core = app.state.rag_core

    async def event_generator():
        # 逐步生成 answer
        async for chunk in rag_core.stream_answer(query):
            yield f"data: {json.dumps({'answer_chunk': chunk})}\n\n"

        # 最终获取 Evidence / metadata
        result = await rag_core.answer(query)

        # 构建证据列表，避免在f-string中使用多行表达式
        evidence_list = []
        for e in result.evidence_chunks:
            evidence_item = {
                'rank': e.rank,
                'score': round(e.score, 4),
                'source': str(e.source),
                'chunk_id': e.chunk_id,
                'content': e.content
            }
            evidence_list.append(evidence_item)

        # 处理 metadata 中可能包含的不可序列化对象
        metadata_serializable = {}
        for key, value in result.metadata.items():
            if isinstance(value, (Path, type(None))) or hasattr(value, '__fspath__'):
                metadata_serializable[key] = str(value)
            else:
                metadata_serializable[key] = value

        final_data = {
            'status': 'Done',
            'answer': result.answer,
            'evidence': evidence_list,
            'metadata': metadata_serializable
        }

        yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Welcome to RAG System</h1><p>Template file not found.</p>"
    except Exception as e:
        return f"<h1>Error loading page</h1><p>{str(e)}</p>"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8090)
