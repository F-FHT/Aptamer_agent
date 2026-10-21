# 🧬 Aptamer-Agent: 智能适配体筛选全栈系统

基于 **LangGraph + MCP 协议 + Streamlit** 构建的 AI 智能体系统，用于自动化适配体的测序数据清洗、同源聚类、Motif 挖掘、二级结构预测与综合打分。

## 🌟 核心特性

- **智能人机协同**：支持自然语言指令，Agent 自动规划分析路径。
- **MCP 微服务架构**：底层生信算法封装为 Model Context Protocol 服务。
- **全链路数据可视**：Streamlit 前端实时展示思考过程与多 Tab 数据看板。
- **全链路可观测性**：内置轻量级结构化日志系统，基于 `contextvars` 自动注入 `trace_id` 与 `node_name`，同时支持彩色控制台输出与 JSON 本地落盘，精准追踪 Agent 与 MCP 交互全过程。
- **防死循环熔断**：自研数据分流与状态隔离机制。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 `LLM_API_KEY`。

```bash
cp .env.example .env
```

项目默认使用 **DeepSeek**（`LLM_BASE_URL=https://api.deepseek.com`、`LLM_MODEL_NAME=deepseek-chat`），但支持所有兼容 OpenAI 接口标准的大模型。修改 `.env` 中的 `LLM_BASE_URL` 和 `LLM_MODEL_NAME` 即可自由切换，例如：

- OpenAI：`LLM_BASE_URL=https://api.openai.com/v1`、`LLM_MODEL_NAME=gpt-4o`
- 通义千问：`LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- Kimi：`LLM_BASE_URL=https://api.moonshot.cn/v1`
- 本地 Ollama：`LLM_BASE_URL=http://localhost:11434/v1`、`LLM_MODEL_NAME=qwen2.5`

> 旧变量 `DEEPSEEK_API_KEY` 仍可作为 `LLM_API_KEY` 的回退值使用。

### 3. 启动 MCP 工具服务 (终端 1)

```bash
python mcp_server/server.py
```

### 4. 启动前端 UI (终端 2)

```bash
streamlit run frontend/app_ui.py
```

## 📸 界面演示

![对话与数据看板](docs/demo1.png)

## 📜 协议

MIT License
