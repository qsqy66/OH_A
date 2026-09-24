import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

# 允许直接 `python evals/run_rag_eval.py` 运行时定位到项目根目录下的 config/core。
# 用 insert(0) 而非 append：venv 里装了同名 PyPI 包 `config`，必须让本项目优先。
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings
from core.llm_client import get_llm_client
from core.rag import get_rag_engine, _rerank, _decompose, _rewrite_query, _dedup_by_id


ROOT = Path(__file__).parent


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"来源：{item['source']}，行号：{item['line_start']}-{item['line_end']}\n{item['text']}"
        for item in results
    )


def _parse_json_list(text: str) -> list[str]:
    """从模型输出中稳健地解析 JSON 数组；失败返回空列表。"""
    if not text:
        return []
    text = text.strip()
    # 去掉 markdown 代码围栏
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()]


def _verdict_supported(verdict: str) -> bool:
    """把核验模型的自由文本输出归约为布尔：仅「支持」判 True，其余（不支持/无法判断）判 False。"""
    v = (verdict or "").strip().rstrip("。.；;，,")
    if v.startswith("不支持") or v.startswith("无法"):
        return False
    return v.startswith("支持")


def _norm(text: str) -> str:
    """去空白归一化，避免关键词因「凌晨2点」vs「凌晨 2 点」这类空格差异而误判。"""
    return re.sub(r"\s+", "", text or "")


def _relevant_sources(question: dict) -> set[str]:
    return set(question.get("relevant_sources") or question.get("expected_sources") or [])


def _source_recall(sources: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return len(set(sources) & relevant) / len(relevant)


def _chunk_recall(results: list[dict], question: dict) -> float | None:
    relevant_chunks = question.get("relevant_chunks")
    if not relevant_chunks:
        return None
    texts = [item["text"] for item in results]
    matched = 0
    for chunk in relevant_chunks:
        needle = chunk.strip()
        if any(needle in t or t in needle for t in texts):
            matched += 1
    return matched / len(relevant_chunks)


_EXTRACT_CLAIMS_PROMPT = (
    "你是信息抽取助手。请把下面「回答」拆解为若干原子论断（atomic claims），"
    "每条论断只包含一个可独立判断真伪的事实主张，且必须自包含：保留主语与限定条件（如城市、职级、金额单位），"
    "不要丢失上下文导致过度泛化（例如应写「杭州的住宿标准为400元/晚」，而不是「住宿标准为400元/晚」）。"
    "严格输出 JSON 数组，每个元素是一个字符串，不要输出任何其他内容。"
    "若回答为空或没有可拆解的事实，输出 []。"
)

_VERIFY_CLAIM_PROMPT = (
    "你是事实核验助手。请判断「论断」是否被「资料」支持。"
    "资料明确包含该论断，或该论断可由资料中的规则直接推理得出（例如时间先后、数值大小、条件满足），都算支持；"
    "资料未提及、与资料矛盾、或需要资料之外的额外假设，都不算支持。"
    "只输出一个词：支持、不支持 或 无法判断，不要输出任何解释或其他内容。"
)


async def evaluate_faithfulness(llm, answer: str, context: str) -> dict:
    """RAGAS 式忠诚度：先拆论断，再逐条核验是否被检索资料支持，faithfulness = 被支持数 / 总数。"""
    claims_text = await llm.lite_chat(_EXTRACT_CLAIMS_PROMPT, f"回答：\n{answer}")
    claims = _parse_json_list(claims_text)
    if not claims:
        return {"faithfulness": None, "claims": []}

    verdicts = []
    supported = 0
    for claim in claims:
        # 不设 max_tokens：GLM 模型会先消耗推理 token，设 16 会导致 content 为空
        verdict = (await llm.lite_chat(
            _VERIFY_CLAIM_PROMPT, f"资料：\n{context}\n\n论断：\n{claim}"
        )).strip()
        if not verdict:
            # 偶发空回复时再问一次，明确要求输出
            verdict = (await llm.lite_chat(
                _VERIFY_CLAIM_PROMPT + "必须输出一个词，不要留空。",
                f"资料：\n{context}\n\n论断：\n{claim}",
            )).strip()
        ok = _verdict_supported(verdict)
        supported += int(ok)
        verdicts.append({
            "claim": claim,
            "verdict": verdict.splitlines()[0].strip() if verdict else "",
            "supported": ok,
        })

    return {
        "faithfulness": round(supported / len(claims), 4),
        "claims": verdicts,
    }


async def evaluate(path: Path, with_faithfulness: bool = True) -> dict:
    questions = load_questions(path)
    engine = get_rag_engine()
    if not engine.is_indexed():
        await engine.index_documents()

    llm = get_llm_client()
    recall_hit_count = 0
    recall_sum = 0.0
    recall_mrr_sum = 0.0
    rerank_hit_count = 0
    rerank_loss_sum = 0.0
    rerank_loss_count = 0
    keyword_hits = 0
    keyword_total = 0
    faithfulness_sum = 0.0
    faithfulness_count = 0
    details = []

    for question in questions:
        query = question["query"]
        relevant = _relevant_sources(question)

        # 1) 召回阶段：拆解 → 每路向量召回（不 rerank），衡量召回能力
        sub_queries = await _decompose(query)
        recall_children = []       # 所有路的子块
        recall_entries = []        # (rewritten, parents) 每路
        for sq in sub_queries:
            rewritten = await _rewrite_query(sq)
            children = await engine.retrieve(rewritten, settings.RAG_RECALL_TOP_K)
            recall_children.extend(children)
            recall_entries.append((rewritten, engine.resolve_parents(children)))

        recall_sources = [item["source"] for item in recall_children]
        recall_hit = bool(set(recall_sources) & relevant)
        recall_hit_count += int(recall_hit)

        chunk_recall = _chunk_recall(recall_children, question)
        recall = chunk_recall if chunk_recall is not None else _source_recall(recall_sources, relevant)
        recall_sum += recall

        recall_first_rank = next((i + 1 for i, s in enumerate(recall_sources) if s in relevant), None)
        recall_mrr_sum += 1.0 / recall_first_rank if recall_first_rank else 0.0

        # 2) 最终阶段：每路 rerank 取 Top-K，合并去重（与生产 search_docs 一致）
        merged = []
        for rewritten, parents in recall_entries:
            merged.extend(await _rerank(rewritten, parents, settings.RAG_TOP_K))
        final_results = _dedup_by_id(merged)[: settings.RAG_MAX_PARENTS]
        final_sources = [item["source"] for item in final_results]

        rerank_hit = bool(set(final_sources) & relevant)
        rerank_hit_count += int(rerank_hit)

        # 重排丢失率：多路召回（合并去重前）的相关来源里，最终结果丢掉了多少（越低越好，0 = 无丢失）
        recall_parents = [p for _, parents in recall_entries for p in parents]
        recall_relevant = set(p["source"] for p in recall_parents) & relevant
        recall_relevant_count = len(recall_relevant)
        final_relevant_count = len(set(final_sources) & relevant)
        loss = (recall_relevant_count - final_relevant_count) / recall_relevant_count if recall_relevant_count else None
        if loss is not None:
            rerank_loss_sum += loss
            rerank_loss_count += 1

        # 3) 用重排后的最终片段生成回答并评估质量
        context = build_context(final_results)
        answer = await llm.lite_chat(
            "只根据提供的资料回答问题。不要编造；只输出答案正文。",
            f"问题：{query}\n资料：\n{context}",
        )
        answer_text = answer or ""

        expected_keywords = question.get("expected_keywords") or []
        answer_hit = all(_norm(k) in _norm(answer_text) for k in expected_keywords) if expected_keywords else None
        if expected_keywords:
            keyword_total += 1
            keyword_hits += int(answer_hit)

        faithfulness_result = None
        if with_faithfulness:
            faithfulness_result = await evaluate_faithfulness(llm, answer_text, context)
            if faithfulness_result["faithfulness"] is not None:
                faithfulness_sum += faithfulness_result["faithfulness"]
                faithfulness_count += 1

        details.append({
            "id": question["id"],
            "recall_hit": recall_hit,
            "recall": round(recall, 4),
            "recall_first_rank": recall_first_rank,
            "rerank_hit": rerank_hit,
            "rerank_loss": round(loss, 4) if loss is not None else None,
            "answer_keyword_hit": answer_hit,
            "faithfulness": faithfulness_result["faithfulness"] if faithfulness_result else None,
            "claims": faithfulness_result["claims"] if faithfulness_result else [],
            "recall_sources": recall_sources,
            "final_sources": final_sources,
        })

    total = len(questions)
    return {
        "dataset": str(path),
        "total": total,
        # 召回阶段（重排前，衡量检索召回能力）
        "recall_hit_rate": round(recall_hit_count / total, 4) if total else 0.0,
        "recall_rate": round(recall_sum / total, 4) if total else 0.0,
        "recall_mrr": round(recall_mrr_sum / total, 4) if total else 0.0,
        # 重排后（最终注入 LLM，衡量重排收益）
        "rerank_hit_rate": round(rerank_hit_count / total, 4) if total else 0.0,
        "rerank_loss_rate": round(rerank_loss_sum / rerank_loss_count, 4) if rerank_loss_count else None,
        # 生成质量
        "answer_keyword_accuracy": round(keyword_hits / keyword_total, 4) if keyword_total else None,
        "faithfulness": round(faithfulness_sum / faithfulness_count, 4) if faithfulness_count else None,
        "details": details,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="评测 RAG：召回(hit/recall/MRR) + 重排(hit/loss) + 回答质量")
    parser.add_argument("--dataset", type=Path, default=ROOT / "rag_questions.jsonl")
    parser.add_argument("--no-faithfulness", action="store_true", help="跳过 LLM-as-judge 忠诚度评测，省 API 调用")
    args = parser.parse_args()
    print(json.dumps(
        await evaluate(args.dataset, with_faithfulness=not args.no_faithfulness),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    asyncio.run(main())
