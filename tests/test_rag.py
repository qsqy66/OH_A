import asyncio

import numpy as np

from core import rag
from core.rag import chunk_text, cosine_similarity
from core.tarce import TraceLog


def test_chunk_text_preserves_overlap():
    chunks = chunk_text("abcdefghij", chunk_size=6, overlap=2)

    assert chunks == ["abcdef", "efghij"]


def test_chunk_text_handles_empty_text():
    assert chunk_text("   ") == []


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity([0, 0], [1, 2]) == 0.0
    assert np.isclose(cosine_similarity([1, 0], [1, 0]), 1.0)


class FakeEmbedLLM:
    def __init__(self):
        self.embed_calls = 0
        self.embedded_texts = []

    async def embed_texts(self, texts):
        self.embed_calls += 1
        self.embedded_texts.extend(texts)
        return [[0.1] * 8 for _ in texts]


def _chunk(source, sha, version="2026-10-20 10:00:00"):
    return {
        "id": f"{source}:{sha}:0",
        "source": source,
        "sha256": sha,
        "version": version,
        "page": "N/A",
        "line_start": 1,
        "line_end": 1,
        "text": "正文",
        "embedding": [0.0] * 8,
    }


def test_index_documents_incremental(monkeypatch):
    # 当前 docs：a 未变、b 内容更新、c 改名自 old_c、d 新增
    docs = [
        {"source": "a.md", "text": "A内容", "sha256": "sha_a"},
        {"source": "b.md", "text": "B新内容", "sha256": "sha_b2"},
        {"source": "c_新名.md", "text": "C内容", "sha256": "sha_c"},
        {"source": "d.md", "text": "D内容", "sha256": "sha_d"},
    ]
    engine = rag.RAGEngine.__new__(rag.RAGEngine)
    engine.chunks = [
        _chunk("a.md", "sha_a"),
        _chunk("b.md", "sha_b1"),
        _chunk("old_c.md", "sha_c"),
    ]
    monkeypatch.setattr(engine, "_save", lambda: None)
    fake = FakeEmbedLLM()
    monkeypatch.setattr(rag, "load_documents", lambda _dir: docs)
    monkeypatch.setattr(rag, "get_llm_client", lambda: fake)

    result = asyncio.run(engine.index_documents())

    # 只有 b（更新）和 d（新增）需要重新向量化；a（未变）、c（改名）不需要
    assert fake.embed_calls == 2
    assert {c["source"] for c in engine.chunks} == {"a.md", "b.md", "c_新名.md", "d.md"}

    a = next(c for c in engine.chunks if c["source"] == "a.md")
    assert a["version"] == "2026-10-20 10:00:00"  # 未变，版本不变

    c = next(c for c in engine.chunks if c["source"] == "c_新名.md")
    assert c["sha256"] == "sha_c"  # 改名，内容指纹不变

    b = next(c for c in engine.chunks if c["source"] == "b.md")
    assert b["sha256"] == "sha_b2"
    assert b["version"] != "2026-10-20 10:00:00"  # 更新，版本刷新

    events = result["events"]
    assert any("文件更新：b.md" in e for e in events)
    assert any("文件名变更：old_c.md → c_新名.md" in e for e in events)
    assert any("新增文件：d.md" in e for e in events)


def test_rewrite_query_disabled_returns_original(monkeypatch):
    monkeypatch.setattr(rag.settings, "RAG_QUERY_REWRITE", False)
    assert asyncio.run(rag._rewrite_query("去杭州出差住宿标准多少")) == "去杭州出差住宿标准多少"


def test_rewrite_query_fails_open(monkeypatch):
    class BoomLLM:
        async def lite_chat(self, *_args, **_kwargs):
            raise RuntimeError("no api")

    monkeypatch.setattr(rag.settings, "RAG_QUERY_REWRITE", True)
    monkeypatch.setattr(rag, "get_llm_client", lambda: BoomLLM())
    assert asyncio.run(rag._rewrite_query("原始问题")) == "原始问题"


def test_rewrite_query_returns_rewritten(monkeypatch):
    class FakeLLM:
        async def lite_chat(self, *_args, **_kwargs):
            return "杭州 出差 住宿 报销 标准"

    monkeypatch.setattr(rag.settings, "RAG_QUERY_REWRITE", True)
    monkeypatch.setattr(rag, "get_llm_client", lambda: FakeLLM())
    assert asyncio.run(rag._rewrite_query("我去杭州住一晚能报多少钱")) == "杭州 出差 住宿 报销 标准"


def test_search_docs_records_rag_retrieve_trace(monkeypatch):
    class FakeLLM:
        async def lite_chat(self, *_args, **_kwargs):
            return "改写后的检索式"

    class FakeEngine:
        chunks = [{"source": "x"}]

        async def retrieve(self, _query, _top_k=None):
            return [{
                "source": "公司考勤与请假制度.md",
                "sha256": "abc123",
                "version": "2026-08-27 19:36:13",
                "page": "N/A",
                "line_start": 1,
                "line_end": 2,
                "text": "正文",
                "similarity": 0.9,
            }]

    monkeypatch.setattr(rag, "get_rag_engine", lambda: FakeEngine())
    monkeypatch.setattr(rag, "get_llm_client", lambda: FakeLLM())
    monkeypatch.setattr(rag.settings, "RAG_QUERY_REWRITE", True)

    trace = TraceLog(session_id="s")
    token = rag.current_trace.set(trace)
    try:
        asyncio.run(rag.search_docs("我去杭州住一晚能报多少"))
    finally:
        rag.current_trace.reset(token)

    rag_steps = [s for s in trace.steps if s.step_type == "rag_retrieve"]
    assert len(rag_steps) == 1
    step = rag_steps[0]
    assert step.rewritten_query == "改写后的检索式"
    assert step.retrieved_chunks[0]["source"] == "公司考勤与请假制度.md"
    assert step.retrieved_chunks[0]["similarity"] == 0.9


def test_rerank_disabled_falls_back(monkeypatch):
    chunks = [
        {"text": "a", "similarity": 0.9},
        {"text": "b", "similarity": 0.8},
        {"text": "c", "similarity": 0.7},
    ]
    monkeypatch.setattr(rag.settings, "RERANK_ENABLED", False)
    assert asyncio.run(rag._rerank("q", chunks, 2)) == chunks[:2]


def test_rerank_reorders_by_model(monkeypatch):
    chunks = [
        {"text": "a", "similarity": 0.9},
        {"text": "b", "similarity": 0.8},
        {"text": "c", "similarity": 0.7},
    ]
    monkeypatch.setattr(rag.settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(rag.settings, "RERANK_API_KEY", "fake-key")
    monkeypatch.setattr(
        rag,
        "_request_rerank",
        lambda _q, _d: {"output": {"results": [
            {"index": 2, "relevance_score": 0.99},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.3},
        ]}},
    )
    result = asyncio.run(rag._rerank("q", chunks, 2))
    assert [c["text"] for c in result] == ["c", "a"]


def test_rerank_fails_open(monkeypatch):
    chunks = [
        {"text": "a", "similarity": 0.9},
        {"text": "b", "similarity": 0.8},
    ]

    def boom(_q, _d):
        raise RuntimeError("network")

    monkeypatch.setattr(rag.settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(rag.settings, "RERANK_API_KEY", "fake-key")
    monkeypatch.setattr(rag, "_request_rerank", boom)
    assert asyncio.run(rag._rerank("q", chunks, 1)) == chunks[:1]