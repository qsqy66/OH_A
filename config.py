import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

class Settings:

    # =====================================================================
    #  一、GLM 智谱大模型配置（必填，从 .env 读取）
    # =====================================================================

    GLM_API_KEY: str = os.getenv("GLM_API_KEY", "")

    # API 网关地址（OpenAI 兼容协议）
    GLM_BASE_URL: str = os.getenv("GLM_BASE_URL", "")

    # 对话补全端点（相对于 BASE_URL）
    GLM_CHAT_ENDPOINT: str = os.getenv("GLM_CHAT_ENDPOINT", "/chat/completions")

    # =====================================================================
    #  二、模型选型
    # =====================================================================
    # 主模型 — 负责对话、工具调用、复杂推理
    GLM_MODEL_NAME: str = os.getenv("GLM_MODEL_NAME", "")
    GLM_CHAT_MODEL: str = os.getenv("GLM_CHAT_MODEL", "")

    # Embedding 模型 — 负责文本转向量（RAG 检索用）
    GLM_EMBED_MODEL: str = os.getenv("GLM_EMBED_MODEL", "")

    # 轻量模型 — 负责查询改写、检索评估、摘要等"后台任务"
    GLM_LITE_MODEL: str = os.getenv("GLM_LITE_MODEL", "")

    # =====================================================================
    #  三、LLM 调用参数
    # =====================================================================
    # 单次 API 调用超时（秒），网络慢或模型推理时间长就调大
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "120.0"))

    # 失败重试次数（指数退避：等1秒→2秒→4秒）
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # 最大输出 token 数
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    # 温度参数（0=确定 1=创造），工具调用场景建议偏低
    LLM_TEMPERATURE: float = float(os.getenv("GLM_TEMPERATURE", "0.1"))

    # Thinking（深度思考）开关 — glm-4.5 系列支持思维链模式
    GLM_THINKING_SWITCH: str = os.getenv("GLM_THINKING_SWITCH", "enabled")

    # =====================================================================
    #  四、Redis 配置
    # =====================================================================
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Key 前缀 — 多个项目共用一个 Redis 时用于区分
    REDIS_PREFIX: str = os.getenv("REDIS_PREFIX", "office_agent:")

    # 会话过期时间（秒），默认 24 小时，超时 Redis 自动清理
    SESSION_TTL: int = int(os.getenv("SESSION_TTL", "86400"))

    # =====================================================================
    #  五、对话记忆配置
    # =====================================================================
    # 最多保留几轮对话（超过后旧消息被裁剪或压缩为摘要）
    MAX_HISTORY_ROUNDS: int = int(os.getenv("MAX_HISTORY_ROUNDS", "10"))

    # 上下文最大 token 估算值（超了触发截断/摘要）
    MAX_HISTORY_TOKENS: int = int(os.getenv("MAX_HISTORY_TOKENS", "32000"))

    # 触发摘要的轮数阈值（超过这个轮数就把旧对话压缩为摘要）
    SUMMARY_TRIGGER_ROUNDS: int = int(os.getenv("SUMMARY_TRIGGER_ROUNDS", "6"))

    # =====================================================================
    #  五·一、长期记忆（LT）配置
    # =====================================================================
    # 记忆总开关 — 关闭后行为与无记忆时完全一致（读空、写空、零延迟）
    MEMORY_ENABLED: bool = os.getenv("MEMORY_ENABLED", "true").lower() in ("1", "true", "yes")

    # 是否每轮回答后用轻量模型自动提取新记忆（画像更新 + 情景记忆）
    MEMORY_EXTRACT: bool = os.getenv("MEMORY_EXTRACT", "true").lower() in ("1", "true", "yes")

    # 情景记忆按相关性最多召回几条注入上下文
    MEMORY_TOP_K: int = int(os.getenv("MEMORY_TOP_K", "3"))

    # 长期记忆有效期（天），超期 Redis 自动清理
    LT_MEMORY_TTL_DAYS: int = int(os.getenv("LT_MEMORY_TTL_DAYS", "90"))

    # 长期记忆容量上限（画像事实条数 / 情景记录条数），防止无限膨胀
    LT_PROFILE_MAX: int = int(os.getenv("LT_PROFILE_MAX", "30"))
    LT_EPISODIC_MAX: int = int(os.getenv("LT_EPISODIC_MAX", "200"))

    # =====================================================================
    #  六、RAG（检索增强生成）配置
    # =====================================================================
    # 文档存放目录 — PDF/Word/Excel 放这里，程序自动扫描
    DOCS_DIR: str = os.getenv("DOCS_DIR", "./docs")

    # 本地 JSON 向量索引配置（无需额外数据库）
    VECTOR_PERSIST_DIR: str = os.getenv("VECTOR_PERSIST_DIR", "./vector_data")
    VECTOR_COLLECTION: str = os.getenv("VECTOR_COLLECTION", "office_docs")

    # 文本分块（旧版简单分块，已被父子切块取代，保留兼容）
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # 父子切块：父块（段落/页码级，无 overlap、不向量化）/ 子块（句级、向量化、关联父块）
    PARENT_CHUNK_TOKENS: int = int(os.getenv("PARENT_CHUNK_TOKENS", "2500"))
    CHILD_CHUNK_TOKENS: int = int(os.getenv("CHILD_CHUNK_TOKENS", "150"))
    CHILD_CHUNK_OVERLAP: float = float(os.getenv("CHILD_CHUNK_OVERLAP", "0.25"))

    # 向量召回候选数（Cross-encoder 重排前的候选片段数）
    RAG_RECALL_TOP_K: int = int(os.getenv("RAG_RECALL_TOP_K", "10"))

    # 最终注入 LLM 上下文的片段数（重排后取前 N 条）
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))

    # 检索相关性阈值（0~1），低于此值自动切换联网搜索
    RAG_RELEVANCE_THRESHOLD: float = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.5"))

    # 查询改写开关：用轻量模型把口语化问题改写成检索式，失败自动回退原问题
    RAG_QUERY_REWRITE: bool = os.getenv("RAG_QUERY_REWRITE", "true").lower() in ("1", "true", "yes")

    # Cross-encoder 重排（Qwen Reranker）：未配置 Key 或调用失败时回退余弦顺序
    RERANK_ENABLED: bool = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
    RERANK_API_KEY: str = os.getenv("RERANK_API_KEY", "")
    RERANK_ENDPOINT: str = os.getenv(
        "RERANK_ENDPOINT",
        "https://cn-beijing.maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank",
    )
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "qwen3.7-text-rerank")
    RERANK_TIMEOUT: float = float(os.getenv("RERANK_TIMEOUT", "15.0"))

    # =====================================================================
    #  七、联网搜索配置
    # =====================================================================
    SEARCH_TIMEOUT: float = float(os.getenv("SEARCH_TIMEOUT", "15.0"))
    SEARCH_MAX_RESULTS: int = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
    SEARCH_API_KEY: str = os.getenv("SEARCH_API_KEY", "")
    SEARCH_ENDPOINT: str = os.getenv(
        "SEARCH_ENDPOINT", "https://api.search.brave.com/res/v1/web/search"
    )

    # =====================================================================
    #  八、工具调用配置
    # =====================================================================
    # 单个工具失败后最多重试几次
    TOOL_MAX_RETRIES: int = int(os.getenv("TOOL_MAX_RETRIES", "3"))
    TOOL_TIMEOUT: float = float(os.getenv("TOOL_TIMEOUT", "30.0"))
    TOOL_OUTPUT_MAX_CHARS: int = int(os.getenv("TOOL_OUTPUT_MAX_CHARS", "12000"))

    # =====================================================================
    #  九、FastAPI 服务配置
    # =====================================================================
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RATE_LIMIT: str = os.getenv("API_RATE_LIMIT", "100/minute")
    API_KEY: str = os.getenv("API_KEY", "")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

    # =====================================================================
    #  十、日志配置
    # =====================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> list[str]:

        errors = []
        required = {
            "GLM_API_KEY": self.GLM_API_KEY,
            "GLM_BASE_URL": self.GLM_BASE_URL,
            "GLM_MODEL_NAME": self.GLM_MODEL_NAME,
            "GLM_EMBED_MODEL": self.GLM_EMBED_MODEL,
            "GLM_LITE_MODEL": self.GLM_LITE_MODEL,
        }
        errors.extend(
            f"缺少必填配置：{name}"
            for name, value in required.items()
            if not value.strip()
        )
        if self.LLM_MAX_RETRIES < 1:
            errors.append("LLM_MAX_RETRIES 必须大于等于 1")
        if self.LLM_TIMEOUT <= 0:
            errors.append("LLM_TIMEOUT 必须大于 0")
        if self.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE 必须大于 0")
        if not 0 <= self.CHUNK_OVERLAP < self.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP 必须大于等于 0 且小于 CHUNK_SIZE")
        if self.PARENT_CHUNK_TOKENS <= 0:
            errors.append("PARENT_CHUNK_TOKENS 必须大于 0")
        if self.CHILD_CHUNK_TOKENS <= 0:
            errors.append("CHILD_CHUNK_TOKENS 必须大于 0")
        if not 0 <= self.CHILD_CHUNK_OVERLAP < 1:
            errors.append("CHILD_CHUNK_OVERLAP 必须大于等于 0 且小于 1")
        if self.TOOL_TIMEOUT <= 0:
            errors.append("TOOL_TIMEOUT 必须大于 0")
        if self.TOOL_OUTPUT_MAX_CHARS <= 0:
            errors.append("TOOL_OUTPUT_MAX_CHARS 必须大于 0")
        if self.RAG_TOP_K <= 0:
            errors.append("RAG_TOP_K 必须大于 0")
        if self.RAG_RECALL_TOP_K <= 0:
            errors.append("RAG_RECALL_TOP_K 必须大于 0")
        return errors

    def require_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise RuntimeError("配置校验失败：\n- " + "\n- ".join(errors))


# ———— 全局单例 ————
# 模块级变量在第一次 import 时初始化一次，之后所有地方共享同一实例。
# 这就是最简单的"单例模式"——不需要任何框架。
settings = Settings()
