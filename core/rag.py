import hashlib
import json
from pathlib import Path

import numpy as np
import structlog

from config import settings
from core.llm_client import get_llm_client

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


def _file_version(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


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
                    "version": _file_version(path),
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

    async def index_documents(self, force: bool = False) -> int:
        if self.chunks and not force:
            return 0

        docs = load_documents(settings.DOCS_DIR)
        if not docs:
            self.chunks = []
            return 0

        llm = get_llm_client()
        new_chunks = []
        index = 0
        for doc in docs:
            pieces = chunk_text(doc["text"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            positions = _chunk_metadata(doc["text"], pieces)
            embeddings = await llm.embed_texts(pieces)
            for piece, vector, position in zip(pieces, embeddings, positions):
                new_chunks.append({
                    "id": f"{doc['source']}:{doc['version']}:{index}",
                    "source": doc["source"],
                    "version": doc["version"],
                    "page": position["page"],
                    "line_start": position["line_start"],
                    "line_end": position["line_end"],
                    "text": piece,
                    "embedding": [float(value) for value in vector],
                })
                index += 1

        self.chunks = new_chunks
        self._save()
        logger.info("rag_indexed", chunks=len(self.chunks), backend="json")
        return len(self.chunks)

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


async def search_docs(query: str, top_k: int | None = None) -> str:
    engine = get_rag_engine()
    if not engine.chunks:
        count = await engine.index_documents()
        if count == 0:
            return "知识库为空：请把文档放到 docs 目录下，然后重试。"

    results = await engine.retrieve(query, top_k)
    if not results:
        return "未在知识库中检索到相关内容。"

    lines = [f"已从本地知识库检索到 {len(results)} 条相关内容："]
    for index, result in enumerate(results, 1):
        location = f"第 {result['line_start']}-{result['line_end']} 行"
        lines.append(
            f"\n[片段{index}] 来源：{result['source']}（版本：{result['version']}，"
            f"页码：{result['page']}，{location}，相关度：{result['similarity']}）\n"
            f"{result['text']}"
        )
    return "\n".join(lines)
