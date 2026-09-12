# Office Agent

基于 OpenAI 兼容接口的企业办公智能助手原型，当前支持：

- 多轮对话与 Redis 会话历史
- Function Calling 工具调用
- 本地 Markdown、TXT、CSV、Excel 文档检索
- Excel 工作表查询
- 对话 Trace 查询
- 用户画像与情景记忆

## 快速开始

### 1. 准备环境

要求：Python 3.11+、一个 OpenAI 兼容的 LLM API；Redis 仅用于会话历史和长期记忆。

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

首次使用本地知识库时，程序会读取 `docs/` 下的 `.txt`、`.md`、`.csv` 和 `.xlsx` 文件并建立本地 JSON 向量索引，索引文件保存在 `vector_data/`，不需要额外数据库。

每个入库片段都会记录来源文件、SHA256 内容指纹、版本（更新时间戳）、页码和行号；当前纯文本和表格文件的页码记录为 `N/A`，行号用于定位原文位置。使用知识库回答时，最终回答会自动附加溯源信息（来源文件 + SHA256 + 版本 + 页码 + 行号）。

更新知识库采用增量方式，按「文件名 + SHA256」比对：文件名与 SHA256 都不变则不重新向量化、版本不变；文件名不变但 SHA256 变则重新向量化并把版本更新为当前时间；文件名变但 SHA256 不变则只更新片段元数据里的文件名。文件名变更与文件更新都会输出提示。

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

测试不依赖真实 LLM、联网搜索或 Redis，会自动隔离外部服务并验证核心安全边界。

项目已配置 GitHub Actions。推送代码或创建 Pull Request 时，会在 Python 3.11 和 3.12 环境自动执行测试、覆盖率统计和源码编译检查。

### 6. 运行 RAG 评测

项目提供 20 条多文档复杂问题作为评测集，位置为 `evals/rag_questions.jsonl`，每条问题的答案需综合两个及以上文档，且提问采用口语化、多样化问法（在哪/多少钱/怎么办/啥时候等）以检验检索对改写表述的鲁棒性。配置真实 LLM 和 Embedding 后运行：

```powershell
python evals/run_rag_eval.py
```

脚本输出五类指标：

- `retrieval_hit_rate`：期望来源是否进入 top-k 检索结果（命中率）；
- `retrieval_recall`：标注相关片段被检索到的比例（召回率，来源级或片段级）；
- `retrieval_mrr`：首个相关结果的排名倒数均值；
- `answer_keyword_accuracy`：回答是否命中预设关键词（快速回归项）；
- `faithfulness`：回答忠诚度，RAGAS 式 LLM-as-judge（拆论断 → 逐条核验是否被检索资料支持）。

忠诚度评测会额外调用轻量模型，想省 API 调用可加 `--no-faithfulness` 跳过；每个问题的论断级核验明细在输出的 `details[].claims` 中，便于人工抽查。

## 配置校验

程序启动时会检查必填模型配置、超时、重试次数和 RAG 分块参数。配置不完整时会直接输出缺失项，而不是运行到第一次请求才失败。

## 当前边界

联网搜索使用 Brave Search API，需要在 `.env` 中配置 `SEARCH_API_KEY`；未配置时会明确返回不可用，不会伪造搜索结果。工具参数会经过严格校验，计算器不执行任意 Python 代码，Excel 只能读取 `data/` 目录内的表格文件。

当前仍未覆盖生产级分布式限流和多实例共享限流状态，这些属于后续增强项。
