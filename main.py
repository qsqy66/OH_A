import asyncio
import json

from structlog import get_logger

from config import settings
from core.agent_loop import AgentLoop
from core.rag import get_rag_engine
from memory.redis import Redis
from tools.registry import TOOLS_SCHEMA

logger = get_logger(__name__)


async def main():
    settings.require_valid()
    select = input("请选择模式：1.多轮对话 2.TRACE查询 3.知识库重建\n").strip()

    if select == "2":
        trace_id = input("请输入TRACE ID：").strip()
        if not trace_id:
            print("TRACE ID不能为空")
            return
        redis_trace = Redis()
        await redis_trace.connect()
        trace = await redis_trace.get_trace(trace_id)
        if trace:
            print("找到对应的TRACE信息：")
            print(json.dumps(trace, ensure_ascii=False, indent=2))
        else:
            print("未找到对应的TRACE信息")
        await redis_trace.close()
        return

    if select == "3":
        print("开始重建知识库索引...")
        count = await get_rag_engine().index_documents(force=True)
        print(f"知识库重建完成，共 {count} 条内容已写入索引。")
        return

    agent = AgentLoop()
    session_id = "text_1"
    while True:
        query = input("请输入问题：")
        if query in {"exit", "quit", "不聊了"}:
            break
        redis = Redis()
        await redis.connect()
        history = await redis.get_history(session_id)
        logger.debug("history_loaded", count=f"读取到{len(history)}条历史记录")

        answer, trace = await agent.run(
            session_id=session_id,
            user_id="USER_001",
            query=query,
            tools=TOOLS_SCHEMA,
            redis=redis,
            history=history,
        )
        print(answer)
        print(f"本次对话的trace_id为：{trace.trace_id}，请妥善保存以便后续查询")
        await redis.close()


if __name__ == "__main__":
    asyncio.run(main())