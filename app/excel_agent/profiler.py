from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


ID_KEYWORDS = ["学号", "编号", "身份证", "电话", "手机", "id", "ID", "账号", "考号"]
NAME_KEYWORDS = ["姓名", "名字", "学生姓名", "学生"]
CLASS_KEYWORDS = ["班级", "行政班", "自然班", "教学班", "专业班", "班号"]
MAJOR_KEYWORDS = ["专业", "学院", "年级"]

SCORE_HINT_KEYWORDS = ["成绩", "分数", "总评", "期末", "平时", "考试", "补考", "重修"]
ABSENCE_KEYWORDS = ["缺考", "旷考", "缓考", "未考", "未参加", "弃考", "缺测", "旷考", "缺席"]

NON_COURSE_KEYWORDS = [
    "序号", "编号", "序", "No", "NO", "no.", "No.", "行号", "序列", "index", "Index",
    "平均", "均分", "平均分",
    "总分", "合计", "总成绩", "总评成绩",
    "绩点", "gpa", "GPA", "平均学分绩点", "学分绩点",
    "排名", "名次", "位次",
    "挂科", "不及格", "通过率", "及格率",
    "学分", "总学分", "已修", "未修",
    "预警", "等级", "状态",
    "班级", "年级", "专业", "学院",
    "学号", "姓名", "性别", "民族", "宿舍", "电话", "手机", "身份证",
]

SUMMARY_SCORE_KEYWORDS = [
    "平均分", "总分", "合计", "绩点", "GPA", "gpa", "平均学分绩点", "学分绩点",
    "排名", "名次", "挂科门数", "不及格门数", "通过率", "及格率",
]

GPA_KEYWORDS = ["平均学分绩点", "学分绩点", "绩点", "GPA", "gpa"]


def infer_column_type(series: pd.Series, col_name: str) -> str:
    name = str(col_name).strip()
    non_null = series.dropna()
    if non_null.empty:
        return "empty"

    if any(k in name for k in ID_KEYWORDS):
        return "id"
    if any(k in name for k in NAME_KEYWORDS):
        return "name"

    if any(k in name for k in CLASS_KEYWORDS):
        return "class"
    if any(k in name for k in MAJOR_KEYWORDS):
        return "category"

    numeric = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = numeric.notna().mean()
    text_values = non_null.astype(str).str.strip()
    absence_ratio = text_values.apply(_is_absence_value).mean()

    if any(k in name for k in GPA_KEYWORDS) and numeric_ratio >= 0.65:
        return "gpa"

    if numeric_ratio >= 0.85:
        vals = numeric.dropna()
        if len(vals) > 0:
            min_v, max_v = float(vals.min()), float(vals.max())

            if max_v > 1000000 and len(str(int(abs(max_v)))) >= 8:
                return "id"

            if any(k in name for k in SUMMARY_SCORE_KEYWORDS):
                return "summary_numeric"

            if 0 <= min_v and max_v <= 100 and not is_non_course_column(name):
                return "score"

        return "numeric"

    # 允许课程列中出现“缺考/旷考/缓考”等文本，只要大部分仍是数值或缺考标记
    if numeric_ratio >= 0.55 and absence_ratio > 0 and not is_non_course_column(name):
        vals = numeric.dropna()
        if len(vals) > 0 and vals.min() >= 0 and vals.max() <= 100:
            return "score"

    # 日期识别：使用 format="mixed" 避免 pandas warning；旧版 pandas 不支持时降级
    try:
        dt = pd.to_datetime(non_null, errors="coerce", format="mixed")
    except TypeError:
        dt = pd.to_datetime(non_null, errors="coerce")
    if dt.notna().mean() >= 0.75:
        return "date"

    unique_count = non_null.astype(str).nunique()
    unique_ratio = unique_count / max(len(non_null), 1)
    if unique_ratio <= 0.25 or unique_count <= 30:
        return "category"

    return "text"


def _is_absence_value(v: object) -> bool:
    text = str(v).strip()
    if not text or text.lower() in ["nan", "none"]:
        return False
    return any(k in text for k in ABSENCE_KEYWORDS)


def is_non_course_column(col_name: str, excluded_course_columns: Optional[List[str]] = None) -> bool:
    name = str(col_name).strip()
    lowered = name.lower()
    excluded = [str(x).strip() for x in (excluded_course_columns or []) if str(x).strip()]
    if name in excluded or lowered in [x.lower() for x in excluded]:
        return True
    if any(k.lower() in lowered for k in NON_COURSE_KEYWORDS):
        return True
    return False


def is_probable_course_column(col_name: str, col_type: str, excluded_course_columns: Optional[List[str]] = None) -> bool:
    if col_type != "score":
        return False
    if is_non_course_column(col_name, excluded_course_columns):
        return False
    return True


def profile_dataframe(df: pd.DataFrame, sheet_name: str = "", excluded_course_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    columns: List[Dict[str, Any]] = []

    for col in df.columns:
        s = df[col]
        col_type = infer_column_type(s, str(col))
        if is_non_course_column(str(col), excluded_course_columns) and col_type == "score":
            col_type = "summary_numeric"

        item: Dict[str, Any] = {
            "name": str(col),
            "type": col_type,
            "missing": int(s.isna().sum()),
            "missing_rate": round(float(s.isna().mean()), 4),
            "unique": int(s.nunique(dropna=True)),
            "is_course_score": is_probable_course_column(str(col), col_type, excluded_course_columns),
        }

        if col_type in ["numeric", "score", "summary_numeric", "gpa"]:
            num = pd.to_numeric(s, errors="coerce")
            item.update({
                "min": _safe_float(num.min()),
                "max": _safe_float(num.max()),
                "mean": _safe_float(num.mean()),
                "median": _safe_float(num.median()),
                "std": _safe_float(num.std()),
            })
            if col_type == "score":
                text_values = s.dropna().astype(str).str.strip()
                absence_count = int(text_values.apply(_is_absence_value).sum())
                item["absence_count"] = absence_count
        elif col_type in ["category", "class", "text", "name", "id"]:
            vc = s.dropna().astype(str).value_counts().head(8)
            item["top_values"] = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
        elif col_type == "date":
            try:
                dt = pd.to_datetime(s, errors="coerce", format="mixed")
            except TypeError:
                dt = pd.to_datetime(s, errors="coerce")
            item["min_date"] = str(dt.min()) if dt.notna().any() else None
            item["max_date"] = str(dt.max()) if dt.notna().any() else None

        columns.append(item)

    class_column = detect_by_keywords(columns, CLASS_KEYWORDS, allowed_types=["class"])
    name_column = detect_name_column(columns)
    id_column = detect_by_keywords(columns, ID_KEYWORDS, allowed_types=["id"])

    course_score_columns = [c["name"] for c in columns if c.get("is_course_score")]
    score_columns = [c["name"] for c in columns if c["type"] == "score"]
    numeric_columns = [c["name"] for c in columns if c["type"] in ["numeric", "score", "summary_numeric", "gpa"]]
    gpa_columns = [c["name"] for c in columns if c["type"] == "gpa" or any(k in c["name"] for k in GPA_KEYWORDS)]

    detected = {
        "name_column": name_column,
        "id_column": id_column,
        "class_column": class_column,
        "score_columns": score_columns,
        "course_score_columns": course_score_columns,
        "numeric_columns": numeric_columns,
        "gpa_columns": gpa_columns,
        "category_columns": [c["name"] for c in columns if c["type"] in ["category", "class"]],
        "summary_numeric_columns": [c["name"] for c in columns if c["type"] == "summary_numeric"],
        "excluded_course_columns": [str(x).strip() for x in (excluded_course_columns or []) if str(x).strip()],
    }

    tasks = []
    if len(course_score_columns) >= 1 and name_column:
        tasks.append("成绩预警分析")
    if class_column and course_score_columns:
        tasks.append("班级成绩对比")
    if gpa_columns:
        tasks.append("平均学分绩点分析")
    if numeric_columns:
        tasks.append("描述统计")
        tasks.append("异常值检测")
    if len(numeric_columns) >= 2:
        tasks.append("相关性分析")

    return {
        "sheet_name": sheet_name,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": columns,
        "detected": detected,
        "detected_tasks": tasks,
    }


def detect_name_column(columns: List[Dict[str, Any]]) -> Optional[str]:
    for c in columns:
        name = c["name"]
        if any(k in name for k in NAME_KEYWORDS):
            return name
    for c in columns:
        if c["type"] == "name":
            return c["name"]
    return None


def detect_by_keywords(columns: List[Dict[str, Any]], keywords: List[str], allowed_types: Optional[List[str]] = None) -> Optional[str]:
    for c in columns:
        name = c["name"]
        if any(k in name for k in keywords):
            if allowed_types is None or c["type"] in allowed_types:
                return name
    return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        if pd.isna(v):
            return None
        return round(float(v), 4)
    except Exception:
        return None
