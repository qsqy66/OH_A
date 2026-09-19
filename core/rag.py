import asyncio
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import structlog

from config import settings
from core.llm_client import get_llm_client
from core.tarce import current_trace

logger = structlog.get_logger(__name__)

_TEXT_EXTS = {".txt", ".md", ".markdown"}
_TABLE_EXTS = {".csv", ".xlsx", ".xls"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_table_file(path: Path) -> str:
    import pandas as pd

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return f"文件：{path.name}\n" + df.to_string(index=False)

    sheets = pd.read_excel(path, sheet_name=None)
    parts = []
    for sheet_name, df in sheets.items():
        parts.append(f"工作表：{sheet_name}\n{df.to_string(index=False)}")
    return f"文件：{path.name}\n" + "\n\n".join(parts)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now_version() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_documents(docs_dir) -> list[dict]:
    docs_dir = Path(docs_dir)
    docs = []
    if not docs_dir.exists():
        return docs

    for path in sorted(docs_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            suffix = path.suffix.lower()
            if suffix in _TEXT_EXTS:
                text = _read_text_file(path)
            elif suffix in _TABLE_EXTS:
                text = _read_table_file(path)
            else:
                logger.warning("rag_unsupported_file", file=path.name)
                continue
            if text.strip():
                docs.append({
                    "source": path.name,
                    "text": text,
                    "sha256": _file_sha256(path),
                })
        except Exception as exc:
            logger.error("rag_read_file_error", file=path.name, error=str(exc))
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size 必须大于 overlap，且 overlap 不能为负数")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def _chunk_metadata(text: str, pieces: list[str]) -> list[dict]:
    metadata = []
    search_start = 0
    for piece in pieces:
        start = text.find(piece, search_start)
        if start < 0:
            start = search_start
        end = start + len(piece)
        metadata.append({
            "page": "N/A",
            "line_start": text.count("\n", 0, start) + 1,
            "line_end": text.count("\n", 0, end) + 1,
        })
        search_start = max(start + 1, end - 1)
    return metadata


def _count_tokens(text: str) -> int:
    """粗略估算 token 数：本项目文档以中文为主，按 1 字符 ≈ 1 token 计。"""
    return len(text)


_SENTENCE_END = "。！？；!?;"


def _split_sentences(text: str) -> list[str]:
    """按句末标点拆句，标点保留在句尾。"""
    sentences = []
    start = 0
    for index, ch in enumerate(text):
        if ch in _SENTENCE_END:
            sentences.append(text[start:index + 1])
            start = index + 1
    if start < len(text):
        sentences.append(text[start:])
    return [s for s in sentences if s.strip()]


def _tail_sentences(sentences: list[str], keep_chars: int) -> list[str]:
    """从句子列表尾部保留约 keep_chars 个字符（按整句截取），用于子块 overlap。"""
    tail = []
    remaining = keep_chars
    for sent in reversed(sentences):
        if remaining <= 0:
            break
        tail.insert(0, sent)
        remaining -= _count_tokens(sent)
    return tail


def _split_children(parent_text: str, max_tokens: int, overlap_ratio: float) -> list[str]:
    """在父块内按句切子块，最大 max_tokens，overlap = 子块长度 * overlap_ratio，不越父块边界。"""
    sentences = _split_sentences(parent_text)
    if not sentences:
        return []
    children = []
    buf = []
    buf_len = 0
    for sent in sentences:
        slen = _count_tokens(sent)
        if buf and buf_len + slen > max_tokens:
            children.append("".join(buf))
            keep = int(buf_len * overlap_ratio)
            buf = _tail_sentences(buf, keep)
            buf_len = _count_tokens("".join(buf))
        buf.append(sent)
        buf_len += slen
    if buf:
        children.append("".join(buf))
    return children


def _split_parents(text: str, max_tokens: int) -> list[str]:
    """按空行段落合并成父块，尽量不超过 max_tokens，父块之间无重叠。"""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    parents = []
    buf = []
    buf_len = 0
    for para in paragraphs:
        plen = _count_tokens(para)
        if buf and buf_len + plen > max_tokens:
            parents.append("\n\n".join(buf))
            buf, buf_len = [], 0
        if plen > max_tokens:
            # 单个段落超长，按句硬切成多个父块（无重叠）
            parents.extend(_split_children(para, max_tokens, 0.0))
            continue
        buf.append(para)
        buf_len += plen
    if buf:
        parents.append("\n\n".join(buf))
    return parents


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class RAGEngine:
    def __init__(self):
        self.chunks: list[dict] = []   # 子块（有 embedding，含 parent_id）
        self.parents: list[dict] = []  # 父块（无 embedding，存溯源元数据与全文）
        self._persist_dir = Path(settings.VECTOR_PERSIST_DIR)
        self._persist_file = self._persist_dir / f"{settings.VECTOR_COLLECTION}.json"
        self._load()

    def _load(self) -> None:
        if not self._persist_file.exists():
            return
        try:
            data = json.loads(self._persist_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.chunks = data.get("chunks", [])
                self.parents = data.get("parents", [])
            else:
                # 旧格式：只有 chunks 列表，无父块
                self.chunks = data
                self.parents = []
            logger.info("rag_index_loaded", chunks=len(self.chunks), parents=len(self.parents), backend="json")
        except Exception as exc:
            logger.error("rag_index_load_error", error=str(exc))
            self.chunks = []
            self.parents = []

    def _save(self) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._persist_file.write_text(
            json.dumps({"chunks": self.chunks, "parents": self.parents}, ensure_ascii=False),
            encoding="utf-8",
        )

    async def _embed_doc(self, llm, doc: dict, version: str) -> tuple[list[dict], list[dict]]:
        """把文档切成父子块：父块不向量化，子块向量化并关联父块 ID。"""
        parent_texts = _split_parents(doc["text"], settings.PARENT_CHUNK_TOKENS)
        parent_meta = _chunk_metadata(doc["text"], parent_texts)

        children: list[dict] = []
        parents: list[dict] = []
        child_index = 0
        for p_idx, (p_text, p_meta) in enumerate(zip(parent_texts, parent_meta)):
            parent_id = f"{doc['sha256']}:parent:{p_idx}"
            parents.append({
                "id": parent_id,
                "source": doc["source"],
                "sha256": doc["sha256"],
                "version": version,
                "page": p_meta["page"],
                "line_start": p_meta["line_start"],
                "line_end": p_meta["line_end"],
                "text": p_text,
            })

            child_texts = _split_children(p_text, settings.CHILD_CHUNK_TOKENS, settings.CHILD_CHUNK_OVERLAP)
            child_meta = _chunk_metadata(p_text, child_texts)
            embeddings = await llm.embed_texts(child_texts)
            for c_idx, (c_text, c_meta, vector) in enumerate(zip(child_texts, child_meta, embeddings)):
                children.append({
                    "id": f"{doc['sha256']}:child:{child_index}",
                    "parent_id": parent_id,
                    "source": doc["source"],
                    "sha256": doc["sha256"],
                    "version": version,
                    "page": c_meta["page"],
                    "line_start": p_meta["line_start"] + c_meta["line_start"] - 1,
                    "line_end": p_meta["line_start"] + c_meta["line_end"] - 1,
                    "text": c_text,
                    "embedding": [float(value) for value in vector],
                })
                child_index += 1
        return children, parents

    async def index_documents(self, force: bool = False) -> dict:
        """按「文件名 + SHA256」增量更新向量库（父子块）。

        返回 {"indexed_chunks": 子块总数, "events": [人类可读的变更提示]}。

        - 文件名与 SHA256 都不变 → 内容未变，不重新向量化，版本不变；
        - 文件名不变、SHA256 变 → 内容更新，重新向量化，版本设为当前时间；
        - 文件名变、SHA256 不变 → 只更新片段元数据里的文件名；
        - 两者都变或全新 → 按新文件处理；
        - 旧索引里存在但当前 docs 中不存在的 → 视为删除。
        """
        docs = load_documents(settings.DOCS_DIR)
        if not docs:
            self.chunks = []
            self.parents = []
            self._save()
            return {"indexed_chunks": 0, "events": ["知识库为空：docs 目录下没有可入库的文档。"]}

        if force:
            return await self._full_reindex(docs)

        existing_children_by_source: dict[str, list[dict]] = {}
        for chunk in self.chunks:
            if "sha256" not in chunk:
                continue
            existing_children_by_source.setdefault(chunk["source"], []).append(chunk)
        existing_parents_by_source: dict[str, list[dict]] = {}
        for parent in self.parents:
            if "sha256" not in parent:
                continue
            existing_parents_by_source.setdefault(parent["source"], []).append(parent)
        existing_sha_by_source = {
            source: chunks[0]["sha256"]
            for source, chunks in existing_children_by_source.items()
            if chunks
        }

        llm = get_llm_client()
        new_children: list[dict] = []
        new_parents: list[dict] = []
        events: list[str] = []
        claimed: set[str] = set()

        for doc in docs:
            source = doc["source"]
            sha = doc["sha256"]

            # 1) 文件名与 SHA256 都不变 → 内容未变
            if source in existing_children_by_source and existing_sha_by_source.get(source) == sha:
                new_children.extend(existing_children_by_source[source])
                new_parents.extend(existing_parents_by_source.get(source, []))
                claimed.add(source)
                continue

            # 2) 文件名不变但 SHA256 变 → 内容更新
            if source in existing_children_by_source:
                version = _now_version()
                children, parents = await self._embed_doc(llm, doc, version)
                new_children.extend(children)
                new_parents.extend(parents)
                claimed.add(source)
                events.append(f"文件更新：{source}（版本 {version}）")
                continue

            # 3) 文件名变但 SHA256 不变 → 只改元数据里的文件名
            rename_candidates = [
                old for old, old_sha in existing_sha_by_source.items()
                if old_sha == sha and old not in claimed
            ]
            if rename_candidates:
                old = rename_candidates[0]
                for chunk in existing_children_by_source[old]:
                    renamed = dict(chunk)
                    renamed["source"] = source
                    new_children.append(renamed)
                for parent in existing_parents_by_source.get(old, []):
                    renamed = dict(parent)
                    renamed["source"] = source
                    new_parents.append(renamed)
                claimed.add(old)
                events.append(f"文件名变更：{old} → {source}")
                continue

            # 4) 全新文件
            version = _now_version()
            children, parents = await self._embed_doc(llm, doc, version)
            new_children.extend(children)
            new_parents.extend(parents)
            events.append(f"新增文件：{source}（版本 {version}）")

        # 5) 删除检测：旧索引里未被认领的来源
        for old in existing_children_by_source:
            if old not in claimed:
                events.append(f"文件已删除：{old}")

        self.chunks = new_children
        self.parents = new_parents
        self._save()
        logger.info("rag_indexed", chunks=len(self.chunks), parents=len(self.parents), backend="json", events=len(events))
        return {"indexed_chunks": len(self.chunks), "events": events}

    async def _full_reindex(self, docs: list[dict]) -> dict:
        llm = get_llm_client()
        version = _now_version()
        new_children = []
        new_parents = []
        for doc in docs:
            children, parents = await self._embed_doc(llm, doc, version)
            new_children.extend(children)
            new_parents.extend(parents)
        self.chunks = new_children
        self.parents = new_parents
        self._save()
        logger.info("rag_reindexed", chunks=len(self.chunks), parents=len(self.parents), backend="json")
        return {
            "indexed_chunks": len(self.chunks),
            "events": [f"全量重建：{len(docs)} 个文件，版本 {version}"],
        }

    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """对子块做向量召回，返回 top_k 个子块（含 parent_id 与相似度）。"""
        top_k = top_k or settings.RAG_TOP_K
        if not self.chunks:
            return []

        query_vector = (await get_llm_client().embed_texts([query]))[0]
        results = []
        for chunk in self.chunks:
            results.append({
                **{key: value for key, value in chunk.items() if key != "embedding"},
                "similarity": round(cosine_similarity(query_vector, chunk["embedding"]), 4),
            })
        results.sort(key=lambda item: item["similarity"], reverse=True)
        return results[:top_k]

    def resolve_parents(self, child_chunks: list[dict]) -> list[dict]:
        """把召回的父块映射回父块并去重，返回父块（附带其子块最大相似度）。"""
        parent_by_id = {parent["id"]: parent for parent in self.parents}
        resolved = []
        seen = set()
        for child in child_chunks:
            pid = child.get("parent_id")
            if pid is None or pid in seen:
                continue
            parent = parent_by_id.get(pid)
            if parent is None:
                continue
            seen.add(pid)
            resolved.append({**parent, "similarity": child.get("similarity", 0.0)})
        return resolved

    async def close(self) -> None:
        return None


_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


_QUERY_REWRITE_PROMPT = (
    "你是检索查询改写助手。把用户的口语化问题改写成适合在文档知识库中做语义检索的检索式："
    "补全关键词、保留专有名词与数字、去除寒暄和无关措辞。"
    "只输出改写后的检索式，不要任何解释。"
)


async def _rewrite_query(query: str) -> str:
    """用轻量模型把口语化问题改写成检索式；关闭或失败时回退原始问题。"""
    if not settings.RAG_QUERY_REWRITE:
        return query
    try:
        rewritten = (await get_llm_client().lite_chat(
            _QUERY_REWRITE_PROMPT, f"问题：{query}"
        )).strip()
        return rewritten or query
    except Exception as exc:
        logger.warning("rag_query_rewrite_failed", error=str(exc))
        return query


def _request_rerank(query: str, documents: list[str]) -> dict:
    payload = json.dumps({
        "model": settings.RERANK_MODEL,
        "input": {"query": query, "documents": documents},
        "parameters": {"top_n": len(documents), "return_documents": True},
    }).encode("utf-8")
    request = Request(
        settings.RERANK_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {settings.RERANK_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "office-agent/1.0",
        },
        method="POST",
    )
    with urlopen(request, timeout=settings.RERANK_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


async def _rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """用 Cross-encoder 重排召回片段；未配置或失败时回退余弦相似度顺序取 top_k。"""
    if not chunks or top_k <= 0:
        return []
    if not settings.RERANK_ENABLED or not settings.RERANK_API_KEY:
        return chunks[:top_k]
    try:
        documents = [chunk["text"] for chunk in chunks]
        data = await asyncio.to_thread(_request_rerank, query, documents)
        results = data.get("output", {}).get("results", [])
        if not results:
            return chunks[:top_k]

        reranked = []
        seen = set()
        for item in results:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
                reranked.append(chunks[idx])
                seen.add(idx)
        # 兜底：补齐 API 未返回的片段（按原余弦顺序）
        for idx, chunk in enumerate(chunks):
            if idx not in seen:
                reranked.append(chunk)
        return reranked[:top_k]
    except Exception as exc:
        logger.warning("rag_rerank_failed", error=str(exc))
        return chunks[:top_k]


async def search_docs(query: str, top_k: int | None = None) -> str:
    engine = get_rag_engine()
    if not engine.chunks:
        result = await engine.index_documents()
        if result["indexed_chunks"] == 0:
            return "知识库为空：请把文档放到 docs 目录下，然后重试。"

    search_query = await _rewrite_query(query)
    final_k = top_k or settings.RAG_TOP_K
    child_results = await engine.retrieve(search_query, settings.RAG_RECALL_TOP_K)
    parents = engine.resolve_parents(child_results)
    results = await _rerank(search_query, parents, final_k)

    trace = current_trace.get()
    if trace is not None:
        trace.add_step(
            step_type="rag_retrieve",
            description="知识库检索",
            rewritten_query=search_query if search_query != query else None,
            retrieved_chunks=[
                {
                    "source": item["source"],
                    "version": item["version"],
                    "line_start": item["line_start"],
                    "line_end": item["line_end"],
                    "similarity": item.get("similarity"),
                }
                for item in results
            ],
        ).finish()

    if not results:
        return "未在知识库中检索到相关内容。"

    lines = [f"已从本地知识库检索到 {len(results)} 条相关内容："]
    for index, result in enumerate(results, 1):
        location = f"第 {result['line_start']}-{result['line_end']} 行"
        lines.append(
            f"\n[片段{index}] 来源：{result['source']}（版本：{result['version']}，"
            f"页码：{result['page']}，{location}）"
            f"\n相关度：{result.get('similarity', '')}\n{result['text']}"
        )
    return "\n".join(lines)
