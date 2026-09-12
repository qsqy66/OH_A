from core.agent_loop import AgentLoop
from core.rag import get_rag_engine
from tools.registry import TOOLS_SCHEMA
import asyncio
import json
from memory.redis import Redis
from structlog import get_logger
from config import settings
logger = get_logger(__name__)

async def reindex_knowledge():
    try:
        result = await get_rag_engine().index_documents()
        for event in result["events"]:
            print(event)
        if result["indexed_chunks"] == 0:
            print("知识库为空：请把文档放到 docs 目录下，然后重试。")
        else:
            print(f"知识库更新完成，共入库 {result['indexed_chunks']} 个片段。")
    except Exception as exc:
        logger.exception("cli_reindex_error", error=str(exc))
        print(f"知识库更新失败：{exc}")


async def main():
    settings.require_valid()
    select = input("请选择模式：1.多轮对话 2.TRACE查询 3.更新RAG知识库\n")
    if select == "2":
        trace_id = input("请输入TRACE ID：")
        if not trace_id:
            print("TRACE ID不能为空")
            return
        redis_trace = Redis()
        await redis_trace.connect()
        trace = await redis_trace.get_trace(trace_id)
        if trace:
            print("找到对应的TRACE信息：")
            
            full_text = json.dumps(trace, ensure_ascii=False, indent=2)
            print(full_text)
        else:
            print("未找到对应的TRACE信息")
        await redis_trace.close()
    elif select == "3":
        await reindex_knowledge()
    else:
        agent = AgentLoop()
        session_id = "text_1"
        while(True):
            query = input("请输入问题：")
            if query == "exit" or query == "quit" or query == "不聊了":
                break
            redis = Redis()
            await redis.connect()
            history = await redis.get_history(session_id)
            logger.debug(f"history_loaded", count=f"读取到{len(history)}条历史记录")
            
            answer, trace = await agent.run(
                        session_id = session_id,
                        user_id = "USER_001",
                        query = query,
                        tools = TOOLS_SCHEMA,
                        redis = redis,
                        history = history
                    )
            print(answer)
            print(f"本次对话的trace_id为：{trace.trace_id}，请妥善保存以便后续查询")
            await redis.close()

if __name__=="__main__":
    asyncio.run(main())