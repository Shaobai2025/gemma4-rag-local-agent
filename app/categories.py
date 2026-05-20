import json
import os
from typing import List, Dict

from app.config import CATEGORIES_FILE, DEFAULT_CATEGORIES


def ensure_categories_file() -> None:
    os.makedirs(os.path.dirname(CATEGORIES_FILE), exist_ok=True)
    if not os.path.exists(CATEGORIES_FILE):
        with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CATEGORIES, f, ensure_ascii=False, indent=2)


def load_categories() -> List[Dict[str, str]]:
    ensure_categories_file()
    try:
        with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return DEFAULT_CATEGORIES.copy()


def save_categories(categories: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(CATEGORIES_FILE), exist_ok=True)
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)


def add_category(key: str, label: str) -> List[Dict[str, str]]:
    key = (key or "").strip()
    label = (label or "").strip()

    if not key:
        raise ValueError("类别 key 不能为空")
    if not label:
        raise ValueError("类别 label 不能为空")

    categories = load_categories()
    if any(c.get("key") == key for c in categories):
        raise ValueError(f"类别已存在: {key}")

    categories.append({"key": key, "label": label})
    save_categories(categories)
    return categories


def delete_category(key: str) -> List[Dict[str, str]]:
    key = (key or "").strip()
    if not key:
        raise ValueError("类别 key 不能为空")
    if key == "general":
        raise ValueError("不能删除默认类别 general")

    categories = load_categories()
    new_categories = [c for c in categories if c.get("key") != key]

    if len(new_categories) == len(categories):
        raise ValueError(f"类别不存在: {key}")

    save_categories(new_categories)
    return new_categories