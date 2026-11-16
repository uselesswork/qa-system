"""不记录请求正文或凭据的结构化访问日志。"""

import json
import logging
import time
from uuid import uuid4

from starlette.requests import Request


logger = logging.getLogger("qa.http")


async def request_log_middleware(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(json.dumps({"event": "http_error", "request_id": request_id,
                                     "method": request.method, "path": request.url.path,
                                     "latency_ms": round((time.perf_counter()-started)*1000, 3)}))
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info(json.dumps({"event": "http_request", "request_id": request_id,
                            "method": request.method, "path": request.url.path,
                            "status": response.status_code,
                            "latency_ms": round((time.perf_counter()-started)*1000, 3)}))
    return response
