import time
import uuid
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Optional, List
from enum import StrEnum
from contextvars import ContextVar


# 当前请求的 TraceLog，由 AgentLoop 设置，供 RAG 检索等深层组件写入 Trace 步骤
current_trace: ContextVar = ContextVar("current_trace", default=None)


class StepType(StrEnum):
    LLM_INFER = "llm_infer"
    TOOL_CALL = "tool_call"
    TOOL_RETRY = "tool_retry"
    ERROR = "error"
    RAG_RETRIEVE = "rag_retrieve"
    SUMMARY = "summary"
    MEMORY = "memory"


class TraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_step_id: Optional[str] = None
    step_type: StepType
    timestamp: float = Field(default_factory=time.time)
    duration_ms: Optional[int] = None
    description: Optional[str] = None

    # LLM
    input_messages_count: Optional[int] = None
    output_content: Optional[str] = None
    output_tool_calls: Optional[List[dict]] = None
    model_name: Optional[str] = None
    tokens_used: Optional[int] = None

    # Tool
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    tool_output: Optional[str] = None
    tool_error: Optional[str] = None
    tool_retry_count: Optional[int] = None

    # RAG
    rewritten_query: Optional[str] = None
    retrieved_chunks: Optional[List[dict]] = None

    def finish(self):
        self.duration_ms = int((time.time() - self.timestamp) * 1000)


class TraceLog(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    steps: List[TraceStep] = Field(default_factory=list)
    final_answer: Optional[str] = None
    total_tokens: Optional[int] = None
    error_summary: Optional[str] = None

    def add_step(self, **kwargs) -> TraceStep:
        step = TraceStep(**kwargs)
        self.steps.append(step)
        return step

    def set_final_answer(self, answer: str):
        self.final_answer = answer
        self.end_time = time.time()

    def set_error(self, err_msg: str):
        self.error_summary = err_msg
        self.end_time = time.time()