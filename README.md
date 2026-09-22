# Office Agent

基于 OpenAI 兼容接口的企业办公智能助手，当前支持：

- 多轮对话 + ReAct 式 Function Calling 工具调用（计算器 / Excel / 联网搜索 / 知识库检索）
- 本地知识库 RAG：父子切块 + 查询改写 + Milvus 向量检索 + Cross-encoder 重排 + 增量索引 + 内容溯源
- 短期会话摘要 + 长期用户画像 / 情景记忆
- 结构化 Trace 调用链（step 级，按 trace_id 查询）
- structlog 结构化日志
- FastAPI HTTP 服务（X-API-Key 鉴权 + 滑动窗口限流）

## 快速开始

### 1. 准备环境

要求：Python 3.11+、一个 OpenAI 兼容的 LLM API；Redis 用于会话历史、长期记忆与 Trace；向量库使用 Milvus（默认 Milvus Lite 嵌入式，无需独立部署）。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，填写 API Key 和三个模型名称。`.env` 不要提交到 Git。

### 2. 启动 Redis（可选）

```powershell
docker compose up -d redis
docker compose ps
```

### 3. 启动助手

```powershell
python main.py
```

首次使用本地知识库时，程序会读取 `docs/` 下的 `.txt`、`.md`、`.csv` 和 `.xlsx` 文件并建立父子切块，向量存储到 **Milvus**（默认 Milvus Lite 嵌入式，数据落盘在 `milvus_data/`；也可通过 `MILVUS_URI` 指向 Milvus 服务版）。

文档采用**父子切块**：父块按段落合并（约 2000-3000 token、无重叠、不向量化），子块在父块内按句切分（约 100-200 token、25% 重叠、做向量化并关联父块 ID）。检索时先召回子块，再映射回父块去重，注入 LLM 的是父块全文。

每个入库片段都会记录来源文件、SHA256 内容指纹、版本（更新时间戳）、页码和行号；其中 SHA256 仅用于增量更新比对、不对外展示。当前纯文本和表格文件的页码记录为 `N/A`，行号用于定位原文位置。使用知识库回答时，最终回答会自动附加溯源信息（来源文件 + 版本 + 页码 + 行号）。

更新知识库采用增量方式，按「文件名 + SHA256」比对：文件名与 SHA256 都不变则不重新向量化、版本不变；文件名不变但 SHA256 变则重新向量化并把版本更新为当前时间；文件名变但 SHA256 不变则只更新片段元数据里的文件名。文件名变更与文件更新都会输出提示。

检索前会先用轻量模型对口语化问题做**查询改写**（补全关键词、保留专有名词与数字），提升语义检索命中率；改写失败会自动回退原问题。可用 `RAG_QUERY_REWRITE=false` 关闭。

检索采用**三段式**：先向量召回 Top-10 子块，再映射回父块去重，最后用 Cross-encoder 重排（Qwen Reranker）精排父块取 Top-3 注入 LLM 上下文；未配置 `RERANK_API_KEY` 或重排失败时自动回退余弦相似度顺序。子块召回数由 `RAG_RECALL_TOP_K`、最终父块数由 `RAG_TOP_K` 控制。

输入 `1` 进入多轮对话，输入 `2` 根据 Trace ID 查询调用链，输入 `3` 增量更新本地知识库索引（新增、修改或重命名 `docs/` 下的文档后使用）；输入 `exit`、`quit` 或 `不聊了`退出。

### 4. 启动 HTTP API

先在 `.env` 中设置一个 API Key，然后运行：

```powershell
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

接口文档地址为 `http://localhost:8000/docs`。业务接口需要携带请求头：

```text
X-API-Key: 你的API_KEY
```

发送对话请求：

```powershell
$headers = @{ "X-API-Key" = "你的API_KEY" }
$body = @{ session_id = "session-001"; user_id = "user-001"; query = "年假最多可以顺延多久？" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/chat -Headers $headers -ContentType "application/json" -Body $body
```

服务提供 `/health` 健康检查、`/api/v1/traces/{trace_id}` Trace 查询和 `/api/v1/knowledge/reindex` 知识库重建接口。

### 5. 运行测试

```powershell
python -m pytest -q
```

测试不依赖真实 LLM、Embedding、联网搜索、Redis 或 Milvus，会自动隔离外部服务并验证核心安全边界；当前 35 个用例，覆盖率阈值 40%。

项目已配置 GitHub Actions。推送代码或创建 Pull Request 时，会在 Python 3.11 和 3.12 环境自动执行测试、覆盖率统计和源码编译检查。

### 6. 运行 RAG 评测

项目提供 20 条多文档复杂问题作为评测集，位置为 `evals/rag_questions.jsonl`，每条问题的答案需综合两个及以上文档，且提问采用口语化、多样化问法（在哪/多少钱/怎么办/啥时候等）以检验检索对改写表述的鲁棒性。配置真实 LLM 和 Embedding 后运行：

```powershell
python evals/run_rag_eval.py
```

脚本输出两段式指标：

**召回阶段**（重排前，衡量检索召回能力，取 `RAG_RECALL_TOP_K` 候选）：

- `recall_hit_rate`：相关来源是否进入召回 Top-N（命中率）；
- `recall_rate`：标注相关片段被召回的比例（召回率，来源级或片段级）；
- `recall_mrr`：首个相关来源排名的倒数均值。

**重排后**（衡量 Cross-encoder 重排收益，取 `RAG_TOP_K` 最终片段）：

- `rerank_hit_rate`：相关来源是否进入重排后的最终 Top-N；
- `rerank_precision`：最终 Top-N 里相关来源片段的占比。

**生成质量**：

- `answer_keyword_accuracy`：回答是否命中预设关键词（快速回归项）；
- `faithfulness`：回答忠诚度，RAGAS 式 LLM-as-judge（拆论断 → 逐条核验是否被检索资料支持）。

忠诚度评测会额外调用轻量模型，想省 API 调用可加 `--no-faithfulness` 跳过；每个问题的论断级核验明细在输出的 `details[].claims` 中，便于人工抽查。重排需配置 `RERANK_API_KEY`，未配置时自动回退余弦顺序（此时 rerank 指标等价于召回阶段截断）。

## 可观测性

**Trace 调用链**：每次对话生成 step 级结构化 Trace，记录 LLM 推理、工具调用、知识库检索、记忆读写与错误各阶段的 Token 消耗与耗时；知识库检索步骤会额外记录查询改写结果与命中片段（来源/版本/行号/相似度）。Trace 持久化到 Redis，可用 `trace_id` 查询。

**日志**：全链路使用 structlog 输出 key=value 结构化日志，便于日志采集与检索。

## 配置校验

程序启动时会检查必填模型配置、超时、重试次数、RAG 分块与重排参数。配置不完整时会直接输出缺失项，而不是运行到第一次请求才失败。

## 当前边界

联网搜索使用 Brave Search API，需要在 `.env` 中配置 `SEARCH_API_KEY`；未配置时会明确返回不可用，不会伪造搜索结果。Cross-encoder 重排使用 Qwen Reranker，需配置 `RERANK_API_KEY`，未配置时回退余弦顺序。工具参数会经过严格校验，计算器不执行任意 Python 代码，Excel 只能读取 `data/` 目录内的表格文件。

向量库默认使用 Milvus Lite（嵌入式，适合单机原型），生产多实例可切换 `MILVUS_URI` 到 Milvus 服务版并把索引类型改为 HNSW。当前仍未覆盖生产级分布式限流和多实例共享限流状态，这些属于后续增强项。
