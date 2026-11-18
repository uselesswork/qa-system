"""按账号限制付费接口的请求次数；多个 Web 容器共享 Redis 计数。"""

from fastapi import HTTPException, Request
from redis import Redis, RedisError

from src.auth.service import AuthContext


_INCREMENT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return {count, redis.call('TTL', KEYS[1])}
"""


class RedisRateLimiter:
    def __init__(self, url: str):
        self.client = Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)

    def check(self, user_id: str, scope: str, limit: int, window_seconds: int = 3600) -> int:
        count, ttl = self.client.eval(
            _INCREMENT, 1, f"qa:rate:{scope}:{user_id}", window_seconds
        )
        if count > limit:
            raise HTTPException(
                429, f"请求过于频繁，请稍后重试（{scope} 每小时最多 {limit} 次）。",
                headers={"Retry-After": str(max(1, ttl))},
            )
        return count

    def close(self) -> None:
        self.client.close()


def enforce_limit(request: Request, user: AuthContext, scope: str, limit: int) -> None:
    enforce_subject_limit(request, user.user_id, scope, limit)


def enforce_subject_limit(request: Request, subject: str, scope: str, limit: int) -> None:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    try:
        limiter.check(subject, scope, limit)
    except RedisError as exc:
        raise HTTPException(503, "请求限流服务暂不可用，请稍后重试。") from exc
