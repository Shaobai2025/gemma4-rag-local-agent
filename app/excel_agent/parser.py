from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style, apply openpyxl's default",
    category=UserWarning,
    module="openpyxl.styles.stylesheet",
)

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


HEADER_KEYWORDS = [
    "姓名", "学号", "班级", "专业", "年级", "课程", "成绩", "总评", "绩点",
    "平均", "电话", "民族", "学院", "宿舍", "性别", "日期", "时间",
]


def list_sheets(path: Path) -> List[Dict[str, Any]]:
    xls = pd.ExcelFile(path)
    infos = []
    for sheet in xls.sheet_names:
        try:
            tmp = pd.read_excel(path, sheet_name=sheet, header=None, engine=None)
            infos.append({
                "name": sheet,
                "rows": int(tmp.shape[0]),
                "columns": int(tmp.shape[1]),
                "detected_header_row": detect_header_row(tmp),
            })
        except Exception:
            infos.append({"name": sheet, "rows": 0, "columns": 0, "detected_header_row": None})
    return infos


def _cell_text(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def detect_header_row(raw: pd.DataFrame, max_scan_rows: int = 30) -> int:
    if raw.empty:
        return 0

    max_rows = min(max_scan_rows, len(raw))
    best_idx = 0
    best_score = -10**9

    for i in range(max_rows):
        row = [_cell_text(v) for v in raw.iloc[i].tolist()]
        nonempty = [x for x in row if x]
        if not nonempty:
            continue

        unique_ratio = len(set(nonempty)) / max(len(nonempty), 1)
        keyword_hits = sum(1 for x in nonempty for k in HEADER_KEYWORDS if k in x)

        # 表头通常是文本为主，下一行开始数据更多。
        text_like = sum(1 for x in nonempty if not _looks_number(x))
        next_nonempty = 0
        next_numeric = 0
        if i + 1 < len(raw):
            next_row = [_cell_text(v) for v in raw.iloc[i + 1].tolist()]
            next_nonempty_vals = [x for x in next_row if x]
            next_nonempty = len(next_nonempty_vals)
            next_numeric = sum(1 for x in next_nonempty_vals if _looks_number(x))

        score = (
            len(nonempty) * 2
            + keyword_hits * 5
            + unique_ratio * 5
            + text_like * 1.2
            + min(next_nonempty, len(nonempty)) * 0.8
            + next_numeric * 0.8
            - i * 0.15
        )

        # 如果某一行都是“说明/备注”长文本，扣分。
        avg_len = sum(len(x) for x in nonempty) / max(len(nonempty), 1)
        if avg_len > 18:
            score -= 8

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)


def _looks_number(x: str) -> bool:
    x = x.replace(",", "").replace("%", "").strip()
    if not x:
        return False
    try:
        float(x)
        return True
    except Exception:
        return False


def normalize_columns(cols: List[Any]) -> List[str]:
    result = []
    counts: Dict[str, int] = {}
    for idx, c in enumerate(cols):
        name = _cell_text(c)
        if not name or name.lower().startswith("unnamed"):
            name = f"列{idx + 1}"
        name = re.sub(r"\s+", "", name)
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > 1:
            name = f"{name}_{counts[name]}"
        result.append(name)
    return result


def read_sheet_smart(path: Path, sheet_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine=None)
    header_row = detect_header_row(raw)

    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, engine=None)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = normalize_columns(list(df.columns))

    # 去掉“合计/备注/说明”类空壳行，不做强删除，保守处理。
    df = df.reset_index(drop=True)

    info = {
        "sheet_name": sheet_name,
        "detected_header_row": int(header_row),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }
    return df, info


def preview_rows(df: pd.DataFrame, n: int = 8) -> List[Dict[str, Any]]:
    safe = df.head(n).copy()
    safe = safe.where(pd.notna(safe), None)
    return safe.to_dict(orient="records")
