"""持久化会话列表、历史与续问。"""

from fastapi import APIRouter, Depends, HTTPException, Request
from openai import APIError
from pydantic import BaseModel
from typing import Literal

from app.api.auth import get_auth_service, require_csrf, require_user
from app.api.knowledge import get_service, scoped_core, user_core
from src.auth.service import AuthContext, AuthService
from src.knowledge.service import KnowledgeError, KnowledgeService
from src.chat.persistent import PersistentChatService
from src.chat.repository import ChatNotFound, ChatRepository
from src.chat.web_search import WebSearchService, WebSearchUnavailable
from src.chat.web_context import answer_from_web_context


router = APIRouter(prefix="/chat/sessions", tags=["chat-sessions"])


class MessageIn(BaseModel):
    message: str
    mode: Literal["knowledge", "web"] = "knowledge"


class SessionIn(BaseModel):
    knowledge_base_id: str | None = None
    session_type: Literal["knowledge", "web"] = "knowledge"


def get_repository(service: AuthService = Depends(get_auth_service)) -> ChatRepository:
    return ChatRepository(service.sessions)


def get_web_search(request: Request) -> WebSearchService:
    service = getattr(request.app.state, "web_search_service", None)
    if service is None:
        service = WebSearchService()
        request.app.state.web_search_service = service
    return service


@router.post("", status_code=201)
def create_session(body: SessionIn, user: AuthContext = Depends(require_csrf),
                   repository: ChatRepository = Depends(get_repository)):
    if body.session_type == "web" and body.knowledge_base_id is not None:
        raise HTTPException(400, "联网专用会话不绑定知识库。")
    if body.session_type == "knowledge" and body.knowledge_base_id is None:
        raise HTTPException(400, "知识库会话需要选择知识库。")
    try:
        return repository.create(user.user_id, body.knowledge_base_id, body.session_type)
    except ChatNotFound as exc:
        raise HTTPException(404, "知识库不存在。") from exc


def session_core(request: Request, knowledge: KnowledgeService,
                 repository: ChatRepository, user_id: str, session_id: str):
    kb_id, session_type = repository.scope_info(user_id, session_id)
    if session_type == "web":
        raise HTTPException(409, "此会话仅用于联网搜索，请开启联网搜索按钮。")
    if kb_id is None:
        return user_core(request, knowledge, user_id)  # 迁移前创建的全库会话
    try:
        ready_ids = knowledge.ready_document_ids(kb_id, user_id)
    except KnowledgeError as exc:
        raise ChatNotFound from exc
    if not ready_ids:
        raise HTTPException(409, "当前知识库还没有已索引的文档，请先上传文件。")
    core = getattr(request.app.state, "rag_core", None)
    if core is None:
        raise HTTPException(503, "问答服务尚未就绪。")
    return scoped_core(core, knowledge, ready_ids, kb_id)


@router.get("")
def list_sessions(user: AuthContext = Depends(require_user),
                  repository: ChatRepository = Depends(get_repository)):
    return repository.list(user.user_id)


@router.get("/{session_id}")
def get_session(session_id: str, user: AuthContext = Depends(require_user),
                repository: ChatRepository = Depends(get_repository)):
    try:
        return repository.get(user.user_id, session_id)
    except ChatNotFound as exc:
        raise HTTPException(404, "会话不存在。") from exc


@router.post("/{session_id}/messages")
async def send_message(session_id: str, body: MessageIn, request: Request,
                       user: AuthContext = Depends(require_csrf),
                       repository: ChatRepository = Depends(get_repository)):
    try:
        if not body.message.strip():
            raise HTTPException(400, "消息不能为空。")
        if body.mode == "web":
            question = body.message.strip()
            if not question or len(question) > 2000:
                raise HTTPException(400, "联网搜索问题不能为空或超过 2000 字符。")
            repository.scope_info(user.user_id, session_id)
            search_input = question
            if question.startswith(("重新回答", "再回答", "那", "它", "这个", "刚才", "上面", "继续")):
                earlier = repository.history(user.user_id, session_id)
                previous = next((item.content for item in reversed(earlier)
                                 if item.role == "user"), "")
                if previous:
                    search_input = (f"前一个问题：{previous}\n当前追问：{question}\n"
                                    "请回答当前追问并引用可核对的网页来源。")
            result = await get_web_search(request).search(search_input)
            repository.append_turn(
                user.user_id, session_id, question, result["answer"], question,
                result["evidence"], result["metadata"],
            )
            return {"session_id": session_id, "user_message": question,
                    "answer": result["answer"], "rewritten_query": question,
                    "evidence": result["evidence"], "metadata": result["metadata"]}
        rewriter = getattr(request.app.state, "query_rewriter", None)
        if rewriter is None:
            raise HTTPException(503, "问答服务尚未就绪。")
        repository.scope_info(user.user_id, session_id)
        rag_core = getattr(request.app.state, "rag_core", None)
        if rag_core is not None:
            contextual = await answer_from_web_context(
                repository, rag_core.llm, user.user_id, session_id, body.message.strip()
            )
            if contextual is not None:
                return contextual
        service = PersistentChatService(
            repository, session_core(request, get_service(request), repository,
                                     user.user_id, session_id),
            rewriter,
        )
        return await service.chat(user.user_id, session_id, body.message)
    except ChatNotFound as exc:
        raise HTTPException(404, "会话不存在。") from exc
    except WebSearchUnavailable as exc:
        raise HTTPException(502, str(exc)) from exc
    except KnowledgeError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except APIError as exc:
        raise HTTPException(502, "模型服务调用失败，请检查模型配置与服务状态。") from exc
