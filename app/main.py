from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from pathlib import Path

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
