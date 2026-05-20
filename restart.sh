#!/usr/bin/env bash
cd "$(dirname "$0")"
brew services stop ollama || true
brew services start ollama || true
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
