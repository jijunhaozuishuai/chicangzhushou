# 🤖 AI 持仓管家（AI Portfolio Assistant）

一个基于多 Agent 架构的 AI 投资助手，支持管理真实持仓、行情分析、K线图生成与每日复盘报告。

## 🛠 技术栈
- **大模型框架**：LangChain, LangGraph (Supervisor 多 Agent 路由)
- **RAG 检索优化**：多路改写 + 混合检索 (BM25+向量) + 父文档检索 + Rerank 精排 + Self-RAG 自我反思
- **数据源**：腾讯行情（实时价、前复权K线）, AKShare (基本面)
- **后端**：FastAPI + SSE 流式输出 + SQLite 长期记忆 + 本地缓存防限流
- **前端**：原生 HTML / JS (支持打字机效果、对话中断)

## ✨ 核心功能
- 对话式管理真实持仓（新增 / 删除 / 修改成本）
- 股票实时行情与基本面查询
- 生成 K 线图（日线 / 周线 / 月线）
- 一键生成每日持仓复盘报告

## 🚀 如何运行
1. 安装依赖：`pip install -r requirements.txt`
2. 配置 `.env` 文件，填入百炼 API Key。
3. 启动后端：`uvicorn server:app --reload`
4. 浏览器打开 `new_file.html` 即可体验。
