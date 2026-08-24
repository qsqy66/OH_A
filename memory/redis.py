import json
import structlog
from redis import asyncio as aioredis
from config import settings
from core.tarce import TraceLog
logger = structlog.get_logger(__name__)
class Redis:
    def __init__(self, redis_url: str | None = settings.REDIS_URL):
        self.redis_url = redis_url
        self.redis: aioredis.Redis | None = None
        
    async def connect(self):
        if self.redis is None:
            self.redis = aioredis.from_url(self.redis_url)
        
    @staticmethod
    def _key(name: str) -> str:
        return f"{settings.REDIS_PREFIX}{name}"

    @staticmethod
    def _trace_key(trace_id: str,) -> str:
        return Redis._key(f"trace:{trace_id}")
    
    @staticmethod
    def _history_key(trace_id: str) -> str:
        return Redis._key(f"history:{trace_id}")

    @staticmethod
    def _session_summary_key(session_id: str) -> str:
        return Redis._key(f"memory:st:{session_id}:summary")

    @staticmethod
    def _profile_key(user_id: str) -> str:
        return Redis._key(f"memory:lt:{user_id}:profile")

    @staticmethod
    def _episodic_key(user_id: str) -> str:
        return Redis._key(f"memory:lt:{user_id}:episodic")
    
    async def append_history(self, session_id: str, message: dict):
        key = self._history_key(session_id)
        value = json.dumps(message, ensure_ascii=False)
        await self.redis.rpush(key, value)
        await self.redis.expire(key, 3600)
        await self.redis.ltrim(key, -20, -1)
        
    async def get_history(self, session_id: str, limit: int | None = None) -> list:
        key = self._history_key(session_id)
        if limit is not None:
            history = await self.redis.lrange(key, -limit, -1)
        else:
            history = await self.redis.lrange(key, 0, -1)
            
        messages = []
        for msg in history:
            try:
                text = msg.decode("utf-8")
                messages.append(json.loads(text))
            except json.JSONDecodeError as e:
                logger.error(f"{session_id}_history_decode_error", error=str(e))
        return messages

    # ==================== 短期记忆：会话摘要 ====================
    async def get_session_summary(self, session_id: str) -> str | None:
        key = self._session_summary_key(session_id)
        value = await self.redis.get(key)
        if value is None:
            return None
        return value.decode("utf-8")

    async def save_session_summary(self, session_id: str, text: str):
        key = self._session_summary_key(session_id)
        await self.redis.set(key, text)
        await self.redis.expire(key, settings.SESSION_TTL)

    # ==================== 长期记忆：用户画像（语义记忆） ====================
    async def get_profile(self, user_id: str) -> dict:
        key = self._profile_key(user_id)
        value = await self.redis.get(key)
        if value is None:
            return {}
        try:
            return json.loads(value.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"{user_id}_profile_decode_error", error=str(e))
            return {}

    async def save_profile(self, user_id: str, profile: dict):
        key = self._profile_key(user_id)
        await self.redis.set(key, json.dumps(profile, ensure_ascii=False))
        await self.redis.expire(key, settings.LT_MEMORY_TTL_DAYS * 86400)

    # ==================== 长期记忆：情景记忆（带向量） ====================
    async def get_episodic(self, user_id: str) -> list[dict]:
        key = self._episodic_key(user_id)
        value = await self.redis.get(key)
        if value is None:
            return []
        try:
            return json.loads(value.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"{user_id}_episodic_decode_error", error=str(e))
            return []

    async def save_episodic(self, user_id: str, records: list[dict]):
        key = self._episodic_key(user_id)
        await self.redis.set(key, json.dumps(records, ensure_ascii=False))
        await self.redis.expire(key, settings.LT_MEMORY_TTL_DAYS * 86400)


    async def save_trace(self, trace: TraceLog):
        key = self._trace_key(trace.trace_id)

        # 手动遍历每一个TraceStep，不用step.model_dump()，直接读取属性
        step_list = []
        for step in trace.steps:
            step_dict = {
                "step_type": getattr(step, "step_type", None),
                "description": getattr(step, "description", None),
                "input_messages_count": getattr(step, "input_messages_count", None),
                "output_content": getattr(step, "output_content", None),
                "model_name": getattr(step, "model_name", None),
                "tokens_used": getattr(step, "tokens_used", None),
                "tool_name": getattr(step, "tool_name", None),
                "tool_input": getattr(step, "tool_input", None),
                "tool_output": getattr(step, "tool_output", None),
                "timestamp": getattr(step, "timestamp", None)
            }
            # 过滤掉全空的step
            step_dict = {k: v for k, v in step_dict.items() if v is not None}
            step_list.append(step_dict)

        # 顶层TraceLog手动组装，完全脱离pydantic model_dump递归
        dump_dict = {
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "start_time": trace.start_time,
            "end_time": trace.end_time,
            "final_answer": trace.final_answer,
            "total_tokens": trace.total_tokens,
            "error_summary": trace.error_summary,
            "steps": step_list
        }

        # 直接json.dumps存入Redis
        json_str = json.dumps(dump_dict, ensure_ascii=False)
        await self.redis.set(key, json_str)
        await self.redis.expire(key, 3600)
    async def get_trace(self, trace_id: str) -> dict | None:
        key = self._trace_key(trace_id)
        value = await self.redis.get(key)
        if value is None:
            return None
        try:
            text = value.decode("utf-8")
            data_dict = json.loads(text)
            # 用Pydantic反向校验还原完整嵌套steps结构
            trace_obj = TraceLog.model_validate(data_dict)
            # 重新dump完整可打印字典
            full_dump = trace_obj.model_dump(mode="json", exclude_none=True)
            return full_dump
        except json.JSONDecodeError as e:
            logger.error(f"{trace_id}_trace_decode_error", error=str(e))
            return None
        except Exception as e:
            logger.error(f"{trace_id}_trace_validate_error", error=str(e))
            return None
    async def close(self):
        if self.redis is not None:
            await self.redis.close()
