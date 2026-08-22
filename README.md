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

要求：Python 3.10+、Docker Desktop，以及一个 OpenAI 兼容的 LLM API。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，填写 API Key 和三个模型名称。`.env` 不要提交到 Git。

### 2. 启动 Redis

```powershell
docker compose up -d redis
docker compose ps
```

### 3. 启动助手

```powershell
python main.py
```

首次使用本地知识库时，程序会读取 `docs/` 下的 `.txt`、`.md`、`.csv` 和 `.xlsx` 文件并建立本地向量索引。索引文件保存在 `chroma_data/`，该目录不应提交到 Git。

输入 `1` 进入多轮对话，输入 `2` 根据 Trace ID 查询调用链；输入 `exit`、`quit` 或 `不聊了`退出。

## 配置校验

程序启动时会检查必填模型配置、超时、重试次数和 RAG 分块参数。配置不完整时会直接输出缺失项，而不是运行到第一次请求才失败。

## 当前边界

当前入口是 CLI，`web_search` 仍是演示用 Mock，项目暂未提供 HTTP API、用户鉴权和自动化测试。这些是后续服务化阶段的工作项。
