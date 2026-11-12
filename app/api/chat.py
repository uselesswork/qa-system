"""持久化会话列表、历史与续问。"""

from fastapi import APIRouter, Depends, HTTPException, Request
from openai import APIError
from pydantic import BaseModel

from app.api.auth import get_auth_service, require_csrf, require_user
from app.api.knowledge import get_service, user_core
from src.auth.service import AuthContext, AuthService
from src.knowledge.service import KnowledgeService
from src.chat.persistent import PersistentChatService
from src.chat.repository import ChatNotFound, ChatRepository


router = APIRouter(prefix="/chat/sessions", tags=["chat-sessions"])


class MessageIn(BaseModel):
    message: str


def get_repository(service: AuthService = Depends(get_auth_service)) -> ChatRepository:
    return ChatRepository(service.sessions)


@router.post("", status_code=201)
def create_session(user: AuthContext = Depends(require_csrf),
                   repository: ChatRepository = Depends(get_repository)):
    return repository.create(user.user_id)


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
                       repository: ChatRepository = Depends(get_repository),
                       knowledge: KnowledgeService = Depends(get_service)):
    rewriter = getattr(request.app.state, "query_rewriter", None)
    if rewriter is None:
        raise HTTPException(503, "问答服务尚未就绪。")
    service = PersistentChatService(repository, user_core(request, knowledge, user.user_id),
                                    rewriter)
    try:
        return await service.chat(user.user_id, session_id, body.message)
    except ChatNotFound as exc:
        raise HTTPException(404, "会话不存在。") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except APIError as exc:
        raise HTTPException(502, "模型服务调用失败，请检查模型配置与服务状态。") from exc
