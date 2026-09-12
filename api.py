import asyncio
import time
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from config import settings
from core.agent_loop import AgentLoop
from core.rag import get_rag_engine
from memory.redis import Redis
from tools.registry import TOOLS_SCHEMA

logger = structlog.get_logger(__name__)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    answer: str
    trace_id: str


class ReindexResponse(BaseModel):
    indexed_chunks: int
    events: list[str] = []


class InMemoryRateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        async with self._lock:
            timestamps = [value for value in self._requests.get(key, []) if value > cutoff]
            if len(timestamps) >= self.limit:
                self._requests[key] = timestamps
                return False
            timestamps.append(now)
            self._requests[key] = timestamps
            return True


def _parse_rate_limit(value: str) -> tuple[int, int]:
    try:
        amount, period = value.split("/", 1)
        limit = int(amount)
        window_seconds = {"second": 1, "minute": 60, "hour": 3600}[period.rstrip("s")]
        if limit < 1:
            raise ValueError
        return limit, window_seconds
    except (KeyError, ValueError):
        raise RuntimeError("API_RATE_LIMIT 必须使用类似 100/minute 的格式") from None


def require_runtime_config() -> None:
    errors = settings.validate()
    if errors:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "服务配置不完整", "errors": errors},
        )


rate_limit, rate_window = _parse_rate_limit(settings.API_RATE_LIMIT)
limiter = InMemoryRateLimiter(rate_limit, rate_window)
agent = AgentLoop()


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if not settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API 尚未配置 API_KEY", 
        )
    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    client_host = request.client.host if request.client else "unknown"
    if not await limiter.allow(client_host):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("api_started", host=settings.API_HOST, port=settings.API_PORT)
    yield


app = FastAPI(
    title="Office Agent API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    redis = Redis()
    try:
        await redis.connect()
        await redis.redis.ping()
        redis_status = "ok"
    except Exception as exc:
        logger.warning("health_redis_error", error=str(exc))
        redis_status = "unavailable"
    finally:
        await redis.close()
    return {
        "status": "ok" if redis_status == "ok" else "degraded",
        "redis": redis_status,
        "configuration": "ok" if not settings.validate() else "incomplete",
    }


@app.post("/api/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat(payload: ChatRequest) -> ChatResponse:
    require_runtime_config()
    redis = Redis()
    try:
        await redis.connect()
        history = await redis.get_history(payload.session_id)
        answer, trace = await agent.run(
            session_id=payload.session_id,
            user_id=payload.user_id,
            query=payload.query,
            tools=TOOLS_SCHEMA,
            redis=redis,
            history=history,
        )
        return ChatResponse(answer=answer, trace_id=trace.trace_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("api_chat_error", error=str(exc))
        raise HTTPException(status_code=502, detail="助手处理失败，请稍后重试") from exc
    finally:
        await redis.close()


@app.get("/api/v1/traces/{trace_id}", dependencies=[Depends(require_api_key)])
async def get_trace(trace_id: str) -> dict:
    redis = Redis()
    try:
        await redis.connect()
        trace = await redis.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace 不存在或已过期")
        return trace
    finally:
        await redis.close()


@app.post(
    "/api/v1/knowledge/reindex",
    response_model=ReindexResponse,
    dependencies=[Depends(require_api_key)],
)
async def reindex() -> ReindexResponse:
    require_runtime_config()
    try:
        result = await get_rag_engine().index_documents()
        return ReindexResponse(indexed_chunks=result["indexed_chunks"], events=result["events"])
    except Exception as exc:
        logger.exception("api_reindex_error", error=str(exc))
        raise HTTPException(status_code=502, detail="知识库重建失败，请稍后重试") from exc
