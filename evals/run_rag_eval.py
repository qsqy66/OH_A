import argparse
import asyncio
import json
from pathlib import Path

from config import settings
from core.llm_client import get_llm_client
from core.rag import get_rag_engine


ROOT = Path(__file__).parent


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"来源：{item['source']}，行号：{item['line_start']}-{item['line_end']}\n{item['text']}"
        for item in results
    )


async def evaluate(path: Path) -> dict:
    questions = load_questions(path)
    engine = get_rag_engine()
    if not engine.chunks:
        await engine.index_documents()

    llm = get_llm_client()
    retrieval_hits = 0
    answer_hits = 0
    details = []
    for question in questions:
        results = await engine.retrieve(question["query"], settings.RAG_TOP_K)
        sources = {item["source"] for item in results}
        retrieval_hit = bool(sources.intersection(question["expected_sources"]))
        retrieval_hits += int(retrieval_hit)

        answer = await llm.lite_chat(
            "只根据提供的资料回答问题。不要编造；只输出答案正文。",
            f"问题：{question['query']}\n资料：\n{build_context(results)}",
        )
        answer_text = answer or ""
        answer_hit = all(keyword in answer_text for keyword in question["expected_keywords"])
        answer_hits += int(answer_hit)
        details.append({
            "id": question["id"],
            "retrieval_hit": retrieval_hit,
            "answer_hit": answer_hit,
            "sources": sorted(sources),
        })

    total = len(questions)
    return {
        "dataset": str(path),
        "total": total,
        "retrieval_hit_rate": round(retrieval_hits / total, 4) if total else 0.0,
        "answer_accuracy": round(answer_hits / total, 4) if total else 0.0,
        "details": details,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="评测本地知识库检索命中率和回答准确率")
    parser.add_argument("--dataset", type=Path, default=ROOT / "rag_questions.jsonl")
    args = parser.parse_args()
    print(json.dumps(await evaluate(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
