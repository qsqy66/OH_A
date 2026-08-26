import asyncio

from tools import registry


def run_tool(name: str, arguments: str) -> str:
    return asyncio.run(registry.tools_excutor(name, arguments))


def test_tool_registry_validates_and_forbids_extra_arguments():
    assert run_tool("calculator", '{"expression":"2+2"}') == "4"
    assert run_tool("calculator", '{"expression":"2+2","extra":1}') == (
        "工具参数无效，请检查参数格式。"
    )


def test_tool_registry_returns_safe_message_for_invalid_json():
    assert run_tool("calculator", "not-json") == "工具参数无效，请检查参数格式。"


def test_tool_registry_applies_timeout(monkeypatch):
    async def slow_tool(**_: object) -> str:
        await asyncio.sleep(0.02)
        return "done"

    monkeypatch.setitem(registry.tools_registry, "calculator", slow_tool)
    monkeypatch.setattr(registry.settings, "TOOL_TIMEOUT", 0.001)

    assert run_tool("calculator", '{"expression":"2+2"}') == "工具执行失败，请稍后重试。"