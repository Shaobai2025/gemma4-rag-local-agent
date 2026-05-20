#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "未检测到 .venv，正在创建虚拟环境..."
  python3 -m venv .venv || exit 1
fi
source .venv/bin/activate || exit 1
if ! command -v uvicorn >/dev/null 2>&1; then
  python -m pip install -U pip
  python -m pip install -r requirements.txt || exit 1
fi
brew services start ollama || true
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
