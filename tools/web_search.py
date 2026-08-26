import asyncio
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import settings


def _request_search(query: str) -> dict:
    params = urlencode({"q": query, "count": settings.SEARCH_MAX_RESULTS})
    request = Request(
        f"{settings.SEARCH_ENDPOINT}?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": settings.SEARCH_API_KEY,
            "User-Agent": "office-agent/1.0",
        },
    )
    with urlopen(request, timeout=settings.SEARCH_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


async def web_search(query: str) -> str:
    query = (query or "").strip()
    if not query or len(query) > 500:
        return "搜索失败：查询内容不能为空且不能超过 500 个字符。"
    if not settings.SEARCH_API_KEY:
        return "搜索不可用：尚未配置 SEARCH_API_KEY。"
    try:
        data = await asyncio.to_thread(_request_search, query)
        results = data.get("web", {}).get("results", [])[: settings.SEARCH_MAX_RESULTS]
        if not results:
            return f"搜索'{query}'：未找到结果。"
        lines = [f"搜索'{query}'结果："]
        for index, item in enumerate(results, 1):
            title = str(item.get("title", "无标题"))
            description = str(item.get("description", "无摘要"))
            url = str(item.get("url", ""))
            lines.append(f"{index}. {title}\n{description}\n来源：{url}")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"搜索失败：{type(exc).__name__}。"