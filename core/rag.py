import hashlib
import json
from datetime import datetime
from pathlib import Path

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
        self.chunks: list[dict] = []
        self._persist_dir = Path(settings.VECTOR_PERSIST_DIR)
        self._persist_file = self._persist_dir / f"{settings.VECTOR_COLLECTION}.json"
        self._load()

    def _load(self) -> None:
        if not self._persist_file.exists():
            return
        try:
            self.chunks = json.loads(self._persist_file.read_text(encoding="utf-8"))
            logger.info("rag_index_loaded", count=len(self.chunks), backend="json")
        except Exception as exc:
            logger.error("rag_index_load_error", error=str(exc))
            self.chunks = []

    def _save(self) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._persist_file.write_text(
            json.dumps(self.chunks, ensure_ascii=False),
            encoding="utf-8",
        )

    async def _embed_doc(self, llm, doc: dict, version: str) -> list[dict]:
        pieces = chunk_text(doc["text"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        positions = _chunk_metadata(doc["text"], pieces)
        embeddings = await llm.embed_texts(pieces)
        chunks = []
        for index, (piece, vector, position) in enumerate(zip(pieces, embeddings, positions)):
            chunks.append({
                "id": f"{doc['source']}:{doc['sha256']}:{index}",
                "source": doc["source"],
                "sha256": doc["sha256"],
                "version": version,
                "page": position["page"],
                "line_start": position["line_start"],
                "line_end": position["line_end"],
                "text": piece,
                "embedding": [float(value) for value in vector],
            })
        return chunks

    async def index_documents(self, force: bool = False) -> dict:
        """按「文件名 + SHA256」增量更新向量库。

        返回 {"indexed_chunks": 总片段数, "events": [人类可读的变更提示]}。

        - 文件名与 SHA256 都不变 → 内容未变，不重新向量化，版本不变；
        - 文件名不变、SHA256 变 → 内容更新，重新向量化，版本设为当前时间；
        - 文件名变、SHA256 不变 → 只更新片段元数据里的文件名；
        - 两者都变或全新 → 按新文件处理；
        - 旧索引里存在但当前 docs 中不存在的 → 视为删除。
        """
        docs = load_documents(settings.DOCS_DIR)
        if not docs:
            self.chunks = []
            self._save()
            return {"indexed_chunks": 0, "events": ["知识库为空：docs 目录下没有可入库的文档。"]}

        if force:
            return await self._full_reindex(docs)

        # 只保留带 sha256 的片段（旧索引无 sha256，会被当作新文件重新入库）
        existing_by_source: dict[str, list[dict]] = {}
        for chunk in self.chunks:
            if "sha256" not in chunk:
                continue
            existing_by_source.setdefault(chunk["source"], []).append(chunk)
        existing_sha_by_source = {
            source: chunks[0]["sha256"]
            for source, chunks in existing_by_source.items()
            if chunks
        }

        llm = get_llm_client()
        new_chunks: list[dict] = []
        events: list[str] = []
        claimed: set[str] = set()

        for doc in docs:
            source = doc["source"]
            sha = doc["sha256"]

            # 1) 文件名与 SHA256 都不变 → 内容未变
            if source in existing_by_source and existing_sha_by_source.get(source) == sha:
                new_chunks.extend(existing_by_source[source])
                claimed.add(source)
                continue

            # 2) 文件名不变但 SHA256 变 → 内容更新
            if source in existing_by_source:
                version = _now_version()
                new_chunks.extend(await self._embed_doc(llm, doc, version))
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
                for chunk in existing_by_source[old]:
                    renamed = dict(chunk)
                    renamed["source"] = source
                    renamed["id"] = f"{source}:{sha}:{renamed['id'].rsplit(':', 1)[-1]}"
                    new_chunks.append(renamed)
                claimed.add(old)
                events.append(f"文件名变更：{old} → {source}")
                continue

            # 4) 全新文件
            version = _now_version()
            new_chunks.extend(await self._embed_doc(llm, doc, version))
            events.append(f"新增文件：{source}（版本 {version}）")

        # 5) 删除检测：旧索引里未被认领的来源
        for old in existing_by_source:
            if old not in claimed:
                events.append(f"文件已删除：{old}")

        self.chunks = new_chunks
        self._save()
        logger.info("rag_indexed", chunks=len(self.chunks), backend="json", events=len(events))
        return {"indexed_chunks": len(self.chunks), "events": events}

    async def _full_reindex(self, docs: list[dict]) -> dict:
        llm = get_llm_client()
        version = _now_version()
        new_chunks = []
        for doc in docs:
            new_chunks.extend(await self._embed_doc(llm, doc, version))
        self.chunks = new_chunks
        self._save()
        logger.info("rag_reindexed", chunks=len(self.chunks), backend="json")
        return {
            "indexed_chunks": len(self.chunks),
            "events": [f"全量重建：{len(docs)} 个文件，版本 {version}"],
        }

    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
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


async def search_docs(query: str, top_k: int | None = None) -> str:
    engine = get_rag_engine()
    if not engine.chunks:
        result = await engine.index_documents()
        if result["indexed_chunks"] == 0:
            return "知识库为空：请把文档放到 docs 目录下，然后重试。"

    search_query = await _rewrite_query(query)
    results = await engine.retrieve(search_query, top_k)

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
                    "similarity": item["similarity"],
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
            f"\n相关度：{result['similarity']}\n{result['text']}"
        )
    return "\n".join(lines)
