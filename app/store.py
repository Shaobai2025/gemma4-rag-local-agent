import json
import os
from typing import Dict, List
from app.config import FILE_REGISTRY_FILE, SPARSE_INDEX_FILE

def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_registry() -> Dict[str, dict]:
    if not os.path.exists(FILE_REGISTRY_FILE):
        return {}
    with open(FILE_REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_registry(data: Dict[str, dict]):
    ensure_parent(FILE_REGISTRY_FILE)
    with open(FILE_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_sparse_rows() -> List[dict]:
    if not os.path.exists(SPARSE_INDEX_FILE):
        return []
    rows = []
    with open(SPARSE_INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_sparse_rows(rows: List[dict]):
    ensure_parent(SPARSE_INDEX_FILE)
    with open(SPARSE_INDEX_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
