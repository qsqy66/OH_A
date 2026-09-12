import asyncio
from types import SimpleNamespace

from core import agent_loop


class FakeRedis:
    def __init__(self):
        self.history = []
        self.saved_trace = None

    async def append_history(self, _session_id, message):
        self.history.append(message)

    async def save_trace(self, trace):
        self.saved_trace = trace


class FakeMemory:
    async def load(self, *_args):
        return {}

    async def update(self, *_args):
        return {"profile_added": 0, "episodic_added": 0, "summary_updated": False}


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def chat(self, **_kwargs):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def response(content, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=SimpleNamespace(total_tokens=3),
    )


def tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def run_agent(agent, redis, query="你好"):
    return asyncio.run(
        agent.run(
            session_id="session-1",
            user_id="user-1",
            query=query,
            tools=[],
            redis=redis,
            history=[],
        )
    )


def test_agent_returns_answer_and_persists_trace(monkeypatch):
    monkeypatch.setattr(agent_loop, "get_memory_manager", lambda: FakeMemory())
    agent = agent_loop.AgentLoop.__new__(agent_loop.AgentLoop)
    agent.llm = FakeLLM([response("你好，我可以帮你处理办公问题。")])
    redis = FakeRedis()

    answer, trace = run_agent(agent, redis)

    assert answer.startswith("你好")
    assert trace.final_answer == answer
    assert redis.saved_trace is trace
    assert redis.history[-1]["role"] == "assistant"


def test_agent_executes_tool_and_appends_source(monkeypatch):
    monkeypatch.setattr(agent_loop, "get_memory_manager", lambda: FakeMemory())

    async def fake_tool(_name, _arguments):
        return "已检索制度。\n[片段1] 来源：公司制度.md（SHA256：abc123，版本：2026-10-21 15:47:00，页码：N/A，第 1-2 行）\n相关度：0.9\n正文内容"

    monkeypatch.setattr(agent_loop, "tools_excutor", fake_tool)
    calls = [tool_call("search_docs", '{"query":"年假"}')]
    agent = agent_loop.AgentLoop.__new__(agent_loop.AgentLoop)
    agent.llm = FakeLLM([response("", calls), response("年假信息如下")])
    redis = FakeRedis()

    answer, trace = run_agent(agent, redis, query="年假怎么规定？")

    assert "来源：公司制度.md（SHA256：abc123，版本：2026-10-21 15:47:00，页码：N/A，第 1-2 行）" in answer
    assert any(step.tool_name == "search_docs" for step in trace.steps)
    assert agent.llm.calls == 2


def test_extract_sources_captures_full_provenance():
    tool_output = (
        "已从本地知识库检索到 2 条相关内容：\n"
        "[片段1] 来源：公司制度.md（SHA256：abc123，版本：2026-10-21 15:47:00，页码：N/A，第 1-2 行）\n相关度：0.9\n正文A\n"
        "[片段2] 来源：差旅标准.csv（SHA256：def456，版本：2026-10-21 15:47:00，页码：N/A，第 3-3 行）\n相关度：0.8\n正文B"
    )
    assert agent_loop._extract_sources(tool_output) == [
        "公司制度.md（SHA256：abc123，版本：2026-10-21 15:47:00，页码：N/A，第 1-2 行）",
        "差旅标准.csv（SHA256：def456，版本：2026-10-21 15:47:00，页码：N/A，第 3-3 行）",
    ]


def test_append_sources_includes_version_page_line():
    answer = "年假最多顺延至次年三月。"
    sources = ["公司制度.md（SHA256：abc123，版本：2026-10-21 15:47:00，页码：N/A，第 1-2 行）"]
    result = agent_loop._append_sources(answer, sources)
    assert "来源：公司制度.md（SHA256：abc123，版本：2026-10-21 15:47:00，页码：N/A，第 1-2 行）" in result


def test_agent_retries_llm_failure_then_returns(monkeypatch):
    monkeypatch.setattr(agent_loop, "get_memory_manager", lambda: FakeMemory())
    agent = agent_loop.AgentLoop.__new__(agent_loop.AgentLoop)
    agent.llm = FakeLLM([RuntimeError("temporary"), response("恢复后的答案")])
    redis = FakeRedis()

    answer, trace = run_agent(agent, redis)

    assert answer == "恢复后的答案"
    assert agent.llm.calls == 2
    assert any(step.step_type == "error" for step in trace.steps)
