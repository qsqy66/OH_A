import asyncio
import structlog
import json
import re
from pathlib import Path
import sys
from tools.registry import tools_excutor, build_tool_message
from core.tarce import TraceLog, TraceStep
from core.llm_client import LLMClient, get_llm_client
from memory.manager import get_memory_manager
logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """
你是一个办公助手智能体，具有调用计算器，读取excel文档进行数据查询，联网搜索，以及检索本地知识库(search_docs)的能力。
当用户问的问题涉及本地文档、内部资料、公司制度时，优先调用 search_docs 工具检索知识库，而不是联网搜索。
所有回答用中文进行回复，回复必须简洁。
当遇到不知道的问题就回复不知道，禁止胡编乱造。
回答问题进行逐步思考最后给出答案。
回复格式先输出[THINK]标签之后衔接思考过程。[ANSWER]标签后面衔接回答结果。
回复示例：[THINK]\n用户问xx，我需要...\n[ANSWER]\n根据数据...
回复禁止携带xml标签

你在调用工具时，**禁止输出任何思考过程、标签、[THINK]、[ANSWER]这类中间推理文本**。
只输出标准的function‑call工具调用，不要输出自然语言思考内容。
所有推理只在function‑call内部完成，不要把思考写在输出文本中。
如果需要向用户提问，直接在tool调用结束后的assistant content返回，不要混入工具参数
如果使用了 search_docs，最终回答必须包含“来源：文件名”形式的来源信息；不能只给结论而省略来源。
"""


def _append_sources(answer: str, sources: list[str]) -> str:
    unique_sources = list(dict.fromkeys(source for source in sources if source))
    if not unique_sources or "来源：" in answer:
        return answer
    return answer.rstrip() + "\n\n来源：" + "、".join(unique_sources)


def _extract_sources(tool_output: str) -> list[str]:
    return re.findall(r"来源：([^（\n]+)", tool_output or "")


def _build_memory_block(memory_ctx: dict) -> str:
    """把记忆上下文拼成可注入 system prompt 的文本段；无可注入内容返回空串。"""
    parts = []
    summary = memory_ctx.get("summary")
    if summary:
        parts.append("[短期记忆·本会话摘要]\n" + summary)
    profile = memory_ctx.get("profile") or {}
    if profile:
        parts.append("[长期记忆·用户画像]\n" + "\n".join(f"- {k}：{v}" for k, v in profile.items()))
    hits = memory_ctx.get("episodic_hits") or []
    if hits:
        parts.append("[长期记忆·相关历史]\n" + "\n".join(f"- {h['text']}" for h in hits))
    return "\n\n".join(parts)


class AgentLoop():
    def __init__(self):
        self.llm = get_llm_client()
        
    async def run(
        self,
        session_id,
        user_id: str | None = None,
        tools: list = None,
        redis = None,
        history: list = None,
        query: str = "",
    ):

        user_id = user_id or session_id
        messages = history.copy() if history else []
        messages.insert(0, {"role":"system","content":SYSTEM_PROMPT})
        messages.append({"role":"user","content":query})
        await redis.append_history(session_id, {"role":"user","content":query})
        trace = TraceLog(session_id=session_id)
        retrieved_sources: list[str] = []

        # —— 记忆注入：短期(会话摘要) + 长期(用户画像 / 相关情景记忆) ——
        # 记忆为空或加载失败时 memory_ctx 为空 dict，不注入任何内容，不影响主流程
        memory_ctx = await get_memory_manager().load(user_id, session_id, query, redis)
        memory_block = _build_memory_block(memory_ctx)
        if memory_block:
            messages[0]["content"] = (
                SYSTEM_PROMPT
                + "\n\n以下是为本次回答准备的记忆信息：\n"
                + memory_block
            )
            logger.info(
                "memory_injected",
                profile = len(memory_ctx.get("profile", {})),
                episodic = len(memory_ctx.get("episodic_hits", [])),
            )

        for loop in range(15):
            try:
                resp = await self.llm.chat(
                    messages = messages,
                    tools = tools if tools else None
                )
                # 创建本轮LLM推理Step
                step_llm: TraceStep = trace.add_step(
                    step_type = "llm_infer",
                    input_messages_count = len(messages),
                    description = f"第{loop+1}轮模型推理",
                )
                
            except Exception as e:
                err_msg = f"模型调用异常，请稍后重试，错误信息：{str(e)}"
                logger.error(
                    f"loop_{loop+1}_llm_error",
                    error = err_msg
                )
                trace.add_step(
                    step_type = "error",
                    description = f"第{loop+1}轮模型调用异常，错误信息：{str(e)}"
                ).finish()
                continue
            
            choice = resp.choices[0]
            chat_content = choice.message.content or ""
            tool_calls = choice.message.tool_calls

            # 给当前LLM步骤填充完整数据并闭环
            step_llm.output_content = chat_content
            step_llm.output_tool_calls = tool_calls
            step_llm.model_name = "glm-4.5-air"
            step_llm.tokens_used = resp.usage.total_tokens
            step_llm.finish()
            
            if tool_calls:
                # 追加assistant工具调用消息
                messages.append({
                    "role":"assistant",
                    "content":chat_content,
                    "tool_calls":[
                        {"id":tc.id,"type":"function","function":{"name":tc.function.name,"arguments":tc.function.arguments}}
                        for tc in tool_calls
                    ]
                })
                
                # 遍历执行所有工具调用并埋点
                for tc in tool_calls:
                    tc_name = tc.function.name
                    tc_args = tc.function.arguments
                    result = await tools_excutor(tc_name, tc_args)
                    if tc_name == "search_docs":
                        retrieved_sources.extend(_extract_sources(result))
                    messages.append(build_tool_message(tc.id, result))
                    
                    logger.debug(
                        f"loop_{loop+1}_has_toolcall",
                        tcName = tc.function.name,
                        arguments = tc_args,
                        result = result
                    )
                    # 工具调用Step直接add并finish
                    trace.add_step(
                        step_type = "tool_call",
                        description = f"第{loop+1}轮工具调用，函数名：{tc.function.name}",
                        tool_name = tc.function.name,
                        tool_input = tc_args,
                        tool_output = result
                    ).finish()
            else:
                # 无工具调用，直接收尾本轮对话
                logger.debug(
                    f"loop_{loop+1}_has_no_toolcall",
                    content = choice.message.content
                )
                # 设置最终答案（内部自动赋值end_time）
                chat_content = _append_sources(chat_content, retrieved_sources)
                trace.set_final_answer(chat_content)
                await redis.append_history(session_id, {"role":"assistant","content":chat_content})
                # 记忆写入：回答完成后提取画像/情景记忆，并在会话达到阈值时压缩摘要
                messages.append({"role":"assistant","content":chat_content})
                mem_res = await get_memory_manager().update(
                    user_id, session_id, query, chat_content, messages, redis
                )
                trace.add_step(
                    step_type = "memory",
                    description = (
                        f"记忆加载：画像{len(memory_ctx.get('profile', {}))}条、"
                        f"情景{len(memory_ctx.get('episodic_hits', []))}条；"
                        f"记忆更新：画像+{mem_res['profile_added']}、"
                        f"情景+{mem_res['episodic_added']}、"
                        f"摘要{'已更新' if mem_res['summary_updated'] else '未变'}"
                    ),
                ).finish()
                await redis.save_trace(trace)
                return chat_content, trace
        
        # 循环15轮耗尽上限兜底
        over_limit_msg = "处理步骤达到上限，请简化问题重试"
        trace.set_error(over_limit_msg)
        trace.add_step(
            step_type = "memory",
            description = (
                f"记忆加载：画像{len(memory_ctx.get('profile', {}))}条、"
                f"情景{len(memory_ctx.get('episodic_hits', []))}条（本轮超限未写入记忆）"
            ),
        ).finish()
        await redis.save_trace(trace)
        return over_limit_msg, trace