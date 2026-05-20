# macOS 从零一键部署说明

本版默认加入 Ollama Embedding 模型拉取，默认模型为 embeddinggemma。

## 默认模型

```text
QA_MODEL=gemma4:e4b
NOTICE_MODEL=gemma4:e4b
PAPER_MODEL=gemma4:27b
EMBEDDING_MODEL=embeddinggemma
OLLAMA_EMBEDDING_MODEL=embeddinggemma
```

## 制作发布包

```bash
chmod +x install_mac_full.sh run_mac.sh doctor_mac.sh make_release_zip_mac_full.sh
bash make_release_zip_mac_full.sh
```

## 部署

```bash
bash install_mac_full.sh
bash run_mac.sh
```

## 手动拉取模型

```bash
ollama pull gemma4:e4b
ollama pull gemma4:27b
ollama pull embeddinggemma
```

如果目标电脑不能拉取 gemma4 或 embeddinggemma，可部署时输入其他模型。

# Gemma 4 Hybrid RAG Dual Model

一个在本地运行的智能问答与写作系统，基于 **FastAPI + Ollama + Chroma + Hybrid Retrieval** 构建，支持：

- 知识库问答
- 通知生成
- 论文写作
- Excel / CSV 数据分析
- 知识库文件上传、删除、替换更新
- 分类管理
- Markdown 渲染
- 数学公式渲染（MathJax）
- 打字机式结果输出

## 模型分工

本项目采用双模型路由：

- **问答**：Gemma 4 E4B
- **通知**：Gemma 4 E4B
- **论文**：Gemma 4 27B
- **Excel 分析**：Gemma 4 E4B

这样做的目的很直接：

- 日常问答与通知更快
- 长篇论文质量更高
- 本地 Mac 部署更平衡

---

## 技术栈

- **后端**：FastAPI
- **本地模型服务**：Ollama
- **向量库**：Chroma
- **检索策略**：Dense + Sparse + RRF
- **前端**：原生 HTML + Tailwind CSS
- **公式渲染**：MathJax
- **Markdown 渲染**：marked.js

> 本版本已经 **彻底移除 FlashRank 依赖**。

---

## 主要功能

### 1. 统一工作台
访问：

```bash
http://127.0.0.1:8000/ask
```

包含三个标签页：

- 问答分析
- 通知生成
- 论文写作

### 2. 知识库管理
访问：

```bash
http://127.0.0.1:8000/kb
```

支持：

- 拖拽上传
- 文件列表
- 删除文件
- 替换更新
- 下载原文件
- 查看文件详情
- 类别增删管理

### 3. 输出体验
支持：

- Markdown 显示
- LaTeX 数学公式渲染
- 打字机式输出
- 来源文件名简洁展示

---

## 目录结构

```bash
gemma4-rag-hybrid-dual-model/
├── app/
│   ├── main.py
│   ├── rag.py
│   ├── retrieval.py
│   ├── ingest.py
│   ├── loaders.py
│   ├── excel_analysis.py
│   ├── config.py
│   ├── schemas.py
│   ├── prompts.py
│   ├── categories.py
│   ├── history.py
│   ├── exporters.py
│   └── store.py
├── frontend/
│   ├── workspace.html
│   └── kb.html
├── data/
│   ├── docs/
│   ├── chroma_db/
│   └── state/
├── requirements.txt
├── .env
├── start.sh
├── stop.sh
└── restart.sh
```

---

## 环境要求

建议环境：

- macOS
- Python 3.11 / 3.12 / 3.13
- Homebrew
- Ollama

推荐机器：

- Apple Silicon Mac
- 48GB 内存及以上体验更好

---

## 安装与运行

### 1. 安装 Ollama

如果没有安装：

```bash
brew install ollama
```

启动：

```bash
brew services start ollama
```

### 2. 拉取模型

```bash
ollama pull gemma4:e4b
ollama pull gemma4:27b
ollama pull embeddinggemma
```

### 3. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. 安装依赖

```bash
pip install -U pip
pip install -r requirements.txt
```

### 5. 启动项目

```bash
./start.sh
```

或者手动启动：

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 启动脚本

### 启动
```bash
./start.sh
```

### 停止 Ollama
```bash
./stop.sh
```

### 重启
```bash
./restart.sh
```

---

## 检索方案

本项目使用：

- Dense Retrieval
- Sparse Retrieval（BM25）
- RRF 融合

即：

**Dense + Sparse + RRF**

这样做比单一路径更稳，适合本地知识库问答。

---

## 支持的文件类型

知识库文件：

- `.pdf`
- `.docx`
- `.md`
- `.txt`

表格分析文件：

- `.xlsx`
- `.xls`
- `.csv`

---

## 常见问题

### 1. 为什么问答会比较慢？
因为完整链路包括：

- 检索
- embedding
- 上下文组织
- 大模型生成

如果想更快，可以：

- 用 E4B 做日常问答
- 控制知识库规模
- 调小召回数量
- 减少同时运行的重任务

### 2. 为什么论文生成比通知慢？
因为论文路由到了 **Gemma 4 26B**，质量更高，但速度更慢。

### 3. 为什么来源只显示文件名？
这是有意做的简化，目的是让前端更干净，不显示冗余 metadata。

---

## 后续可继续扩展

- 真正的流式输出
- 多用户支持
- 登录鉴权
- 管理后台
- 更强的知识库标签过滤
- 导出 Word 样式优化
- 历史记录回填

---

## License



```text
QIXIN License
```



```text
仅供学习与研究使用
```
