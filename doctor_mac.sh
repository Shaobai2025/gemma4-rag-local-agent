#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "===================================================="
echo "  部署环境诊断"
echo "===================================================="

echo ""
echo "[Homebrew]"
if command -v brew >/dev/null 2>&1; then
  echo "✅ brew"
  brew --version | head -n 1
else
  echo "❌ brew 未安装"
fi

echo ""
echo "[Python]"
python3 --version || true
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "虚拟环境 Python：$(python --version)"
else
  echo "❌ .venv 不存在"
fi

echo ""
echo "[Ollama]"
if command -v ollama >/dev/null 2>&1; then
  echo "✅ ollama"
  ollama --version || true
  if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama 服务运行中"
    ollama list || true
  else
    echo "❌ Ollama 服务未运行"
  fi
else
  echo "❌ Ollama 未安装"
fi

echo ""
echo "[项目文件]"
test -f app/main.py && echo "✅ app/main.py" || echo "❌ app/main.py 缺失"
test -f frontend/workspace.html && echo "✅ frontend/workspace.html" || echo "⚠️ frontend/workspace.html 缺失"
test -f .env && echo "✅ .env" || echo "⚠️ .env 缺失"

echo ""
echo "[.env 模型配置]"
if [ -f ".env" ]; then
  grep -E "^(QA_MODEL|NOTICE_MODEL|PAPER_MODEL|EMBEDDING_MODEL|OLLAMA_EMBEDDING_MODEL|OLLAMA_BASE_URL|HOST|PORT)=" .env || true
else
  echo "⚠️ 未找到 .env"
fi

echo ""
echo "[Python依赖]"
if [ -d ".venv" ]; then
  source .venv/bin/activate
  python - <<'PY'
mods = ["fastapi", "uvicorn", "pandas", "openpyxl", "matplotlib", "docx", "reportlab", "langchain_ollama"]
for m in mods:
    try:
        __import__(m)
        print(f"✅ {m}")
    except Exception as e:
        print(f"❌ {m}: {e}")
PY
fi

echo ""
echo "[端口]"
PORT_VALUE="8000"
if [ -f ".env" ]; then
  PORT_VALUE="$(grep '^PORT=' .env | cut -d '=' -f2 || echo 8000)"
fi

if lsof -i :"$PORT_VALUE" >/dev/null 2>&1; then
  echo "⚠️ 端口 $PORT_VALUE 已被占用"
  lsof -i :"$PORT_VALUE" || true
else
  echo "✅ 端口 $PORT_VALUE 未被占用"
fi

echo ""
echo "===================================================="
echo "诊断完成"
echo "===================================================="
