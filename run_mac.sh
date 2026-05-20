#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
  echo "❌ 未发现 .venv，请先运行：bash install_mac_full.sh"
  exit 1
fi

source .venv/bin/activate

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

mkdir -p logs

if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "⚠️ Ollama 服务似乎未运行，尝试后台启动..."
  nohup ollama serve > logs/ollama.log 2>&1 &
  sleep 5
fi

echo "===================================================="
echo "  启动 Gemma4 RAG 服务"
echo "===================================================="
echo "模型配置："
echo "  QA_MODEL=${QA_MODEL:-gemma4:e4b}"
echo "  NOTICE_MODEL=${NOTICE_MODEL:-gemma4:e4b}"
echo "  PAPER_MODEL=${PAPER_MODEL:-gemma4:27b}"
echo "  EMBEDDING_MODEL=${EMBEDDING_MODEL:-embeddinggemma}"
echo ""
echo "访问地址："
echo "  http://${HOST}:${PORT}/ask"
echo ""
echo "按 Ctrl + C 停止后端服务"
echo "===================================================="

uvicorn app.main:app --host "$HOST" --port "$PORT"
