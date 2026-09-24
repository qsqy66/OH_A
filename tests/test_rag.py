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


def test_index_documents_incremental(monkeypatch):
    # 当前 docs：a 未变、b 内容更新、c 改名自 old_c、d 新增
    docs = [
        {"source": "a.md", "text": "A内容", "sha256": "sha_a"},
        {"source": "b.md", "text": "B新内容", "sha256": "sha_b2"},
        {"source": "c_新名.md", "text": "C内容", "sha256": "sha_c"},
        {"source": "d.md", "text": "D内容", "sha256": "sha_d"},
    ]
    engine = rag.RAGEngine.__new__(rag.RAGEngine)
    monkeypatch.setattr(
        engine, "_existing_sha_by_source",
        lambda: {"a.md": "sha_a", "b.md": "sha_b1", "old_c.md": "sha_c"},
    )

    inserted = []
    deleted = []
    renamed = []

    def fake_insert(children, parents):
        inserted.extend(c["source"] for c in children)

    def fake_delete(source):
        deleted.append(source)

    def fake_rename(old, new):
        renamed.append((old, new))

    monkeypatch.setattr(engine, "_insert", fake_insert)
    monkeypatch.setattr(engine, "_delete_source", fake_delete)
    monkeypatch.setattr(engine, "_rename_source", fake_rename)
    monkeypatch.setattr(engine, "_count_chunks", lambda: 99)

    fake = FakeEmbedLLM()
    monkeypatch.setattr(rag, "load_documents", lambda _dir: docs)
    monkeypatch.setattr(rag, "get_llm_client", lambda: fake)

    result = asyncio.run(engine.index_documents())

    # b 更新、d 新增 → 重新向量化插入；a 未变、c 改名 → 不重新向量化
    assert set(inserted) == {"b.md", "d.md"}
    assert renamed == [("old_c.md", "c_新名.md")]
    assert deleted == ["b.md"]  # b 更新时先删旧
    assert result["indexed_chunks"] == 99

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
        def is_indexed(self):
            return True

        async def retrieve(self, _query, _top_k=None):
            return [{
                "id": "c1",
                "parent_id": "p1",
                "source": "公司考勤与请假制度.md",
                "sha256": "abc123",
                "version": "2026-08-27 19:36:13",
                "page": "N/A",
                "line_start": 1,
                "line_end": 2,
                "text": "子块正文",
                "similarity": 0.9,
            }]

        def resolve_parents(self, _child_chunks):
            return [{
                "id": "p1",
                "source": "公司考勤与请假制度.md",
                "sha256": "abc123",
                "version": "2026-08-27 19:36:13",
                "page": "N/A",
                "line_start": 1,
                "line_end": 2,
                "text": "父块正文",
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


def test_split_parents_merges_paragraphs_no_overlap():
    text = "第一段。\n\n第二段。\n\n第三段。"
    assert rag._split_parents(text, 100) == ["第一段。\n\n第二段。\n\n第三段。"]


def test_split_parents_splits_when_over_limit():
    # 每段 2 token，max=4 只放得下两段
    assert rag._split_parents("甲。\n\n乙。\n\n丙。", 4) == ["甲。\n\n乙。", "丙。"]


def test_split_children_sentence_based_with_overlap():
    # 每句 2 token，max=4、overlap 25% → 相邻子块共享尾部一个句
    assert rag._split_children("甲。乙。丙。丁。戊。", 4, 0.25) == ["甲。乙。", "乙。丙。", "丙。丁。", "丁。戊。"]


def test_resolve_parents_dedup(monkeypatch):
    class FakeClient:
        def has_collection(self, _name):
            return True

        def load_collection(self, _name):
            return None

        def query(self, collection_name=None, filter=None, output_fields=None, limit=None):
            return [
                {"id": "sha:p:0", "source": "a.md", "text": "A", "version": "v", "page": "N/A", "line_start": 1, "line_end": 1, "sha256": "sha"},
                {"id": "sha:p:1", "source": "b.md", "text": "B", "version": "v", "page": "N/A", "line_start": 1, "line_end": 1, "sha256": "sha"},
            ]

    engine = rag.RAGEngine.__new__(rag.RAGEngine)
    monkeypatch.setattr(engine, "_get_client", lambda: FakeClient())

    children = [
        {"parent_id": "sha:p:0", "similarity": 0.9},
        {"parent_id": "sha:p:0", "similarity": 0.7},  # 重复父块
        {"parent_id": "sha:p:1", "similarity": 0.8},
    ]
    resolved = engine.resolve_parents(children)
    assert [p["source"] for p in resolved] == ["a.md", "b.md"]  # 去重
    assert resolved[0]["similarity"] == 0.9  # 取最大子块相似度


def test_decompose_disabled_returns_single(monkeypatch):
    monkeypatch.setattr(rag.settings, "RAG_QUERY_DECOMPOSE", False)
    assert asyncio.run(rag._decompose("问题")) == ["问题"]


def test_decompose_single_intent(monkeypatch):
    class FakeLLM:
        async def lite_chat(self, *_args, **_kwargs):
            return '{"single": true, "sub_queries": []}'

    monkeypatch.setattr(rag.settings, "RAG_QUERY_DECOMPOSE", True)
    monkeypatch.setattr(rag, "get_llm_client", lambda: FakeLLM())
    assert asyncio.run(rag._decompose("加班费怎么算")) == ["加班费怎么算"]


def test_decompose_multi_intent(monkeypatch):
    class FakeLLM:
        async def lite_chat(self, *_args, **_kwargs):
            return '{"single": false, "sub_queries": ["入职流程", "员工福利"]}'

    monkeypatch.setattr(rag.settings, "RAG_QUERY_DECOMPOSE", True)
    monkeypatch.setattr(rag, "get_llm_client", lambda: FakeLLM())
    assert asyncio.run(rag._decompose("新员工入职流程和福利分别是什么")) == ["入职流程", "员工福利"]


def test_decompose_fails_open(monkeypatch):
    class BoomLLM:
        async def lite_chat(self, *_args, **_kwargs):
            raise RuntimeError("no api")

    monkeypatch.setattr(rag.settings, "RAG_QUERY_DECOMPOSE", True)
    monkeypatch.setattr(rag, "get_llm_client", lambda: BoomLLM())
    assert asyncio.run(rag._decompose("原始问题")) == ["原始问题"]


def test_dedup_by_id_keeps_max_similarity():
    results = [
        {"id": "p1", "similarity": 0.5, "source": "a.md"},
        {"id": "p2", "similarity": 0.7, "source": "b.md"},
        {"id": "p1", "similarity": 0.9, "source": "a.md"},  # 重复，相似度更高
    ]
    deduped = rag._dedup_by_id(results)
    assert [r["id"] for r in deduped] == ["p1", "p2"]
    assert deduped[0]["similarity"] == 0.9  # 保留最大相似度