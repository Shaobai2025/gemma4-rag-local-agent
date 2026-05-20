import json
import os
import uuid
from datetime import datetime
from app.config import HISTORY_FILE

def _ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def ensure_history_file():
    if not os.path.exists(HISTORY_FILE):
        _ensure_parent(HISTORY_FILE)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"qa": [], "notice": [], "paper": []}, f, ensure_ascii=False, indent=2)

def load_history() -> dict:
    ensure_history_file()
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_history(data: dict):
    _ensure_parent(HISTORY_FILE)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_history(kind: str, title: str, preview: str, limit: int = 20):
    data = load_history()
    items = data.get(kind, [])
    items.insert(0, {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "title": title[:120],
        "preview": preview[:500],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    data[kind] = items[:limit]
    save_history(data)

def list_history(kind: str) -> list[dict]:
    return load_history().get(kind, [])
