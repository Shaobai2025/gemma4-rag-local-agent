#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "===================================================="
echo "  Gemma4 RAG 项目 - macOS 从零一键部署"
echo "  包含 Homebrew / Ollama / LLM模型 / Embedding模型"
echo "===================================================="
echo "项目目录：$PROJECT_DIR"
echo ""

if [ "$(uname -s)" != "Darwin" ]; then
  echo "❌ 当前脚本主要支持 macOS。"
  exit 1
fi

echo "[0/12] 检查 Xcode Command Line Tools..."
if ! xcode-select -p >/dev/null 2>&1; then
  echo "⚠️ 未检测到 Xcode Command Line Tools，开始安装..."
  xcode-select --install || true
  echo "请完成弹窗安装后，重新运行：bash install_mac_full.sh"
  exit 1
else
  echo "✅ Xcode Command Line Tools 已安装"
fi

echo ""
echo "[1/12] 检查 Homebrew..."
if ! command -v brew >/dev/null 2>&1; then
  echo "⚠️ 未检测到 Homebrew，开始安装..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  if [ -x "/opt/homebrew/bin/brew" ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
    if ! grep -q "/opt/homebrew/bin/brew shellenv" "$HOME/.zprofile" 2>/dev/null; then
      echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
    fi
  fi

  if [ -x "/usr/local/bin/brew" ]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
else
  echo "✅ Homebrew 已安装"
fi

brew --version

echo ""
echo "[2/12] 更新 Homebrew..."
brew update || true

echo ""
echo "[3/12] 检查/安装 Python 3.11..."
if ! command -v python3.11 >/dev/null 2>&1; then
  brew install python@3.11
else
  echo "✅ python3.11 已安装"
fi

PY_CMD="$(brew --prefix python@3.11)/bin/python3.11"
if [ ! -x "$PY_CMD" ]; then
  PY_CMD="python3"
fi
"$PY_CMD" --version

echo ""
echo "[4/12] 检查/安装 Git..."
if ! command -v git >/dev/null 2>&1; then
  brew install git
else
  echo "✅ Git 已安装"
fi

echo ""
echo "[5/12] 检查/安装 Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
  brew install ollama
else
  echo "✅ Ollama 已安装"
fi
ollama --version || true

echo ""
echo "[6/12] 检查/启动 Ollama 服务..."
mkdir -p logs
if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "✅ Ollama 服务已运行"
else
  echo "⚠️ Ollama 服务未运行，尝试后台启动..."
  nohup ollama serve > logs/ollama.log 2>&1 &
  sleep 5
  if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama 服务已启动"
  else
    echo "⚠️ Ollama 自动启动可能失败。请另开终端执行：ollama serve"
  fi
fi

echo ""
echo "[7/12] 配置模型名称..."
DEFAULT_QA_MODEL="gemma4:e4b"
DEFAULT_PAPER_MODEL="gemma4:27b"
DEFAULT_EMBEDDING_MODEL="embeddinggemma"

echo "说明："
echo "- 默认使用你的模型配置：gemma4:e4b / gemma4:27b。"
echo "- 默认 Embedding 模型：embeddinggemma。"
echo "- 如果目标电脑无法拉取 gemma4 或 embeddinggemma，可临时输入其他模型。"
echo ""

read -r -p "请输入问答/通知模型 [默认 gemma4:e4b]：" QA_MODEL_INPUT
QA_MODEL_VALUE="${QA_MODEL_INPUT:-$DEFAULT_QA_MODEL}"

read -r -p "请输入论文/长报告模型 [默认 gemma4:27b]：" PAPER_MODEL_INPUT
PAPER_MODEL_VALUE="${PAPER_MODEL_INPUT:-$DEFAULT_PAPER_MODEL}"

read -r -p "请输入 Ollama Embedding 模型 [默认 embeddinggemma]：" EMBEDDING_MODEL_INPUT
EMBEDDING_MODEL_VALUE="${EMBEDDING_MODEL_INPUT:-$DEFAULT_EMBEDDING_MODEL}"

read -r -p "是否自动拉取 LLM 模型？可能较慢，输入 y 拉取 [y/N]：" PULL_LLM_MODELS

if [ "$PULL_LLM_MODELS" = "y" ] || [ "$PULL_LLM_MODELS" = "Y" ]; then
  echo "开始拉取模型：$QA_MODEL_VALUE"
  ollama pull "$QA_MODEL_VALUE"

  if [ "$PAPER_MODEL_VALUE" != "$QA_MODEL_VALUE" ]; then
    echo "开始拉取模型：$PAPER_MODEL_VALUE"
    ollama pull "$PAPER_MODEL_VALUE"
  fi
else
  echo "跳过 LLM 模型拉取。后续可手动执行："
  echo "  ollama pull $QA_MODEL_VALUE"
  echo "  ollama pull $PAPER_MODEL_VALUE"
fi

read -r -p "是否自动拉取 Embedding 模型 $EMBEDDING_MODEL_VALUE？建议输入 y [Y/n]：" PULL_EMBED_MODEL
PULL_EMBED_MODEL="${PULL_EMBED_MODEL:-y}"

if [ "$PULL_EMBED_MODEL" = "y" ] || [ "$PULL_EMBED_MODEL" = "Y" ]; then
  echo "开始拉取 Embedding 模型：$EMBEDDING_MODEL_VALUE"
  ollama pull "$EMBEDDING_MODEL_VALUE"
else
  echo "跳过 Embedding 模型拉取。后续可手动执行："
  echo "  ollama pull $EMBEDDING_MODEL_VALUE"
fi

echo ""
echo "[8/12] 检查项目文件..."
if [ ! -f "app/main.py" ]; then
  echo "❌ 未找到 app/main.py"
  echo "请确认脚本位于项目根目录，并在项目根目录运行。"
  exit 1
fi

if [ ! -d "frontend" ]; then
  mkdir -p frontend
fi
echo "✅ 项目结构正常"

echo ""
echo "[9/12] 创建 Python 虚拟环境..."
if [ ! -d ".venv" ]; then
  "$PY_CMD" -m venv .venv
  echo "✅ 已创建 .venv"
else
  echo "✅ .venv 已存在，跳过创建"
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo ""
echo "[10/12] 安装 Python 依赖..."
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  pip install -U fastapi "uvicorn[standard]" python-multipart pandas openpyxl numpy matplotlib python-docx reportlab langchain langchain-community langchain-ollama langgraph chromadb jieba
fi

echo ""
echo "[11/12] 创建目录和 .env..."
mkdir -p data/excel_sessions data/uploads data/chroma data/vectorstore data/files docs uploads logs

if [ ! -f ".env" ]; then
  cat > .env <<EOF
OLLAMA_BASE_URL=http://127.0.0.1:11434

QA_MODEL=$QA_MODEL_VALUE
NOTICE_MODEL=$QA_MODEL_VALUE
PAPER_MODEL=$PAPER_MODEL_VALUE
VISION_MODEL=llava

# Ollama Embedding 模型，用于知识库向量化/RAG
EMBEDDING_MODEL=$EMBEDDING_MODEL_VALUE
OLLAMA_EMBEDDING_MODEL=$EMBEDDING_MODEL_VALUE

HOST=127.0.0.1
PORT=8000

# 局域网访问可改为：HOST=0.0.0.0

EXCEL_SESSION_MAX_AGE_HOURS=24
EXCEL_CLEANUP_INTERVAL_MINUTES=60
EOF
  echo "✅ 已生成 .env"
else
  echo "✅ .env 已存在，跳过生成"
  echo "如需修改默认模型，请手动编辑 .env"
fi

echo ""
echo "[12/12] 语法检查..."
python -m py_compile app/main.py
if [ -d "app/excel_agent" ]; then
  python -m py_compile app/excel_agent/*.py
fi

echo ""
echo "===================================================="
echo "✅ 从零部署完成"
echo "===================================================="
echo ""
echo "当前模型配置："
echo "  QA_MODEL=$QA_MODEL_VALUE"
echo "  NOTICE_MODEL=$QA_MODEL_VALUE"
echo "  PAPER_MODEL=$PAPER_MODEL_VALUE"
echo "  EMBEDDING_MODEL=$EMBEDDING_MODEL_VALUE"
echo ""
echo "启动服务："
echo "  bash run_mac.sh"
echo ""
echo "访问地址："
echo "  http://127.0.0.1:8000/ask"
echo ""
