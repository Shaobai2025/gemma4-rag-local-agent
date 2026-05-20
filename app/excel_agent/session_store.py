from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

SESSION_ROOT = Path("data/excel_sessions")


def ensure_session_root() -> None:
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)


def new_session_id() -> str:
    return f"excel_{uuid.uuid4().hex[:12]}"


def session_dir(session_id: str) -> Path:
    ensure_session_root()
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "_-")
    return SESSION_ROOT / safe


def create_session(upload_path: Path, original_filename: str) -> str:
    ensure_session_root()
    session_id = new_session_id()
    d = session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_filename).suffix.lower() or ".xlsx"
    raw_path = d / f"raw{suffix}"
    shutil.copyfile(upload_path, raw_path)
    save_json(session_id, "meta.json", {
        "session_id": session_id,
        "filename": original_filename,
        "raw_file": raw_path.name,
    })
    return session_id


def get_raw_file(session_id: str) -> Path:
    meta = load_json(session_id, "meta.json")
    p = session_dir(session_id) / meta["raw_file"]
    if not p.exists():
        raise FileNotFoundError(f"Excel原始文件不存在: {p}")
    return p


def save_json(session_id: str, name: str, data: Dict[str, Any]) -> None:
    d = session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(session_id: str, name: str) -> Dict[str, Any]:
    p = session_dir(session_id) / name
    if not p.exists():
        raise FileNotFoundError(f"会话文件不存在: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _sheet_key(sheet_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in sheet_name)[:80]


def save_dataframe(session_id: str, sheet_name: str, df: pd.DataFrame) -> None:
    d = session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    df.to_pickle(d / f"df_{_sheet_key(sheet_name)}.pkl")


def load_dataframe(session_id: str, sheet_name: str) -> pd.DataFrame:
    p = session_dir(session_id) / f"df_{_sheet_key(sheet_name)}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"DataFrame未解析，请先选择Sheet解析: {sheet_name}")
    return pd.read_pickle(p)


def save_text(session_id: str, name: str, text: str) -> None:
    d = session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text, encoding="utf-8")


def load_text(session_id: str, name: str) -> str:
    return (session_dir(session_id) / name).read_text(encoding="utf-8")



def cleanup_old_sessions(max_age_hours: int = 24) -> int:
    """
    清理超过 max_age_hours 的 Excel 分析会话目录。
    默认清理 24 小时前的数据，避免 data/excel_sessions 长期堆积。
    """
    import time
    ensure_session_root()
    now = time.time()
    cutoff = now - max_age_hours * 3600
    removed = 0

    for p in SESSION_ROOT.iterdir():
        if not p.is_dir():
            continue
        try:
            # 使用目录修改时间判断，会话分析/导出时目录会更新
            if p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
        except Exception:
            continue
    return removed
