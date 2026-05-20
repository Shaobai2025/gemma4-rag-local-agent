import io
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd


SUMMARY_COL_KEYWORDS = [
    "总成绩",
    "平均成绩",
    "总学分",
    "学分绩点",
    "平均学分绩点",
    "平均绩点",
    "绩点",
]

INFO_COL_KEYWORDS = {
    "student_id": ["学号", "学生学号", "student_id", "id"],
    "student_name": ["姓名", "学生姓名", "name"],
    "class_name": ["班级", "行政班", "专业班级", "class"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _load_table(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, List[str], str]:
    ext = filename.lower().split(".")[-1]

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        df = _normalize_columns(df)
        return df, ["CSV"], "CSV"

    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet_names = excel_file.sheet_names
    selected_sheet = sheet_name if sheet_name in sheet_names else sheet_names[0]
    df = pd.read_excel(excel_file, sheet_name=selected_sheet)
    df = _normalize_columns(df)
    return df, sheet_names, selected_sheet


def _find_col(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for col in cols:
        lower_col = str(col).lower()
        for kw in keywords:
            if kw.lower() in lower_col:
                return col
    return None


def _find_info_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        key: _find_col(df, kws)
        for key, kws in INFO_COL_KEYWORDS.items()
    }


def _is_summary_col(col: str) -> bool:
    return any(kw in str(col) for kw in SUMMARY_COL_KEYWORDS)


def _is_info_col(col: str, info_cols: Dict[str, Optional[str]]) -> bool:
    return col in {v for v in info_cols.values() if v}


def _course_columns(df: pd.DataFrame, info_cols: Dict[str, Optional[str]]) -> List[str]:
    result = []
    for col in df.columns:
        if _is_info_col(col, info_cols):
            continue
        if _is_summary_col(col):
            continue
        if str(col).strip() in {"序号", "备注", "说明"}:
            continue
        result.append(col)
    return result


def _safe_num(v) -> Optional[float]:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _status_of_score(v) -> str:
    if pd.isna(v):
        return "empty"

    s = str(v).strip()
    if s == "":
        return "empty"

    lowered = s.lower()

    if "旷考" in s:
        return "absent"
    if "取消" in s:
        return "cancelled"
    if "缺考" in s:
        return "absent"
    if lowered in {"nan", "none", "null"}:
        return "empty"

    n = _safe_num(v)
    if n is None:
        return "other_text"

    if n < 60:
        return "fail"
    if n < 70:
        return "low"
    return "pass"


def _pick_summary_value(row: pd.Series, candidates: List[str]) -> Optional[float]:
    for name in candidates:
        if name in row.index:
            v = _safe_num(row[name])
            if v is not None:
                return v
    return None


def _risk_level(
    fail_count: int,
    low_count: int,
    absent_count: int,
    cancelled_count: int,
    avg_score: Optional[float],
    avg_gpa: Optional[float],
) -> str:
    if absent_count >= 1:
        return "高风险"
    if fail_count >= 3:
        return "高风险"
    if avg_score is not None and avg_score < 60:
        return "高风险"
    if avg_gpa is not None and avg_gpa < 1.0:
        return "高风险"

    if fail_count >= 1:
        return "中风险"
    if low_count >= 3:
        return "中风险"
    if avg_score is not None and avg_score < 70:
        return "中风险"
    if avg_gpa is not None and avg_gpa < 2.0:
        return "中风险"

    if low_count >= 1 or cancelled_count >= 1:
        return "低风险"
    return "正常"


def _risk_reason(
    fail_courses: List[str],
    low_courses: List[str],
    absent_courses: List[str],
    cancelled_courses: List[str],
    avg_score: Optional[float],
    avg_gpa: Optional[float],
) -> str:
    parts = []

    if absent_courses:
        parts.append(f"旷考课程: {', '.join(absent_courses[:5])}")
    if fail_courses:
        parts.append(f"不及格课程: {', '.join(fail_courses[:5])}")
    if low_courses:
        parts.append(f"低分课程: {', '.join(low_courses[:5])}")
    if cancelled_courses:
        parts.append(f"取消课程: {', '.join(cancelled_courses[:5])}")
    if avg_score is not None and avg_score < 70:
        parts.append(f"平均成绩偏低({avg_score:.2f})")
    if avg_gpa is not None and avg_gpa < 2.0:
        parts.append(f"平均学分绩点偏低({avg_gpa:.2f})")

    return "；".join(parts) if parts else "整体相对稳定"


def _student_row_analysis(
    row: pd.Series,
    info_cols: Dict[str, Optional[str]],
    course_cols: List[str],
) -> Dict[str, Any]:
    fail_courses = []
    low_courses = []
    absent_courses = []
    cancelled_courses = []
    pass_count = 0
    numeric_scores = []

    for col in course_cols:
        v = row.get(col, None)
        status = _status_of_score(v)

        if status == "fail":
            fail_courses.append(col)
            n = _safe_num(v)
            if n is not None:
                numeric_scores.append(n)
        elif status == "low":
            low_courses.append(col)
            n = _safe_num(v)
            if n is not None:
                numeric_scores.append(n)
        elif status == "pass":
            pass_count += 1
            n = _safe_num(v)
            if n is not None:
                numeric_scores.append(n)
        elif status == "absent":
            absent_courses.append(col)
        elif status == "cancelled":
            cancelled_courses.append(col)

    avg_score = _pick_summary_value(row, ["平均成绩", "总成绩", "平均分"])
    avg_gpa = _pick_summary_value(row, ["平均学分绩点", "平均绩点", "学分绩点"])
    total_credit = _pick_summary_value(row, ["总学分"])

    if avg_score is None and numeric_scores:
        avg_score = sum(numeric_scores) / len(numeric_scores)

    min_score = min(numeric_scores) if numeric_scores else None
    max_score = max(numeric_scores) if numeric_scores else None

    risk = _risk_level(
        fail_count=len(fail_courses),
        low_count=len(low_courses),
        absent_count=len(absent_courses),
        cancelled_count=len(cancelled_courses),
        avg_score=avg_score,
        avg_gpa=avg_gpa,
    )

    reason = _risk_reason(
        fail_courses=fail_courses,
        low_courses=low_courses,
        absent_courses=absent_courses,
        cancelled_courses=cancelled_courses,
        avg_score=avg_score,
        avg_gpa=avg_gpa,
    )

    return {
        "学号": str(row.get(info_cols["student_id"], "")) if info_cols["student_id"] else "",
        "姓名": str(row.get(info_cols["student_name"], "")) if info_cols["student_name"] else "",
        "班级": str(row.get(info_cols["class_name"], "")) if info_cols["class_name"] else "",
        "课程门数": len(course_cols),
        "已出分课程数": len(numeric_scores) + len(absent_courses) + len(cancelled_courses),
        "及格门数": pass_count,
        "不及格门数": len(fail_courses),
        "低分门数": len(low_courses),
        "旷考门数": len(absent_courses),
        "取消门数": len(cancelled_courses),
        "平均成绩": round(avg_score, 2) if avg_score is not None else None,
        "最低分": round(min_score, 2) if min_score is not None else None,
        "最高分": round(max_score, 2) if max_score is not None else None,
        "平均学分绩点": round(avg_gpa, 2) if avg_gpa is not None else None,
        "总学分": round(total_credit, 2) if total_credit is not None else None,
        "不及格课程": "、".join(fail_courses[:10]),
        "低分课程": "、".join(low_courses[:10]),
        "旷考课程": "、".join(absent_courses[:10]),
        "取消课程": "、".join(cancelled_courses[:10]),
        "风险等级": risk,
        "风险说明": reason,
    }


def _summary_markdown(
    selected_sheet: str,
    student_df: pd.DataFrame,
    course_cols: List[str],
) -> str:
    total_students = len(student_df)
    high_risk = int((student_df["风险等级"] == "高风险").sum()) if not student_df.empty else 0
    medium_risk = int((student_df["风险等级"] == "中风险").sum()) if not student_df.empty else 0
    low_risk = int((student_df["风险等级"] == "低风险").sum()) if not student_df.empty else 0

    fail_students = int((student_df["不及格门数"] > 0).sum()) if not student_df.empty else 0
    absent_students = int((student_df["旷考门数"] > 0).sum()) if not student_df.empty else 0
    avg_score_series = pd.to_numeric(student_df["平均成绩"], errors="coerce")
    overall_avg = avg_score_series.dropna().mean() if not avg_score_series.dropna().empty else None

    lines = [
        "# 成绩预警分析报告",
        "",
        "## 一、总体情况",
        f"- 工作表：{selected_sheet}",
        f"- 学生总人数：{total_students}",
        f"- 识别课程数：{len(course_cols)}",
        f"- 存在不及格学生数：{fail_students}",
        f"- 存在旷考学生数：{absent_students}",
        f"- 高风险学生数：{high_risk}",
        f"- 中风险学生数：{medium_risk}",
        f"- 低风险学生数：{low_risk}",
    ]

    if overall_avg is not None:
        lines.append(f"- 整体平均成绩：{overall_avg:.2f}")

    lines.extend([
        "",
        "## 二、风险判定说明",
        "- 高风险：存在旷考，或不及格门数较多，或平均成绩/平均学分绩点明显偏低。",
        "- 中风险：存在不及格，或低分课程较多，或平均成绩偏低。",
        "- 低风险：无不及格，但存在低分或取消课程等情况。",
        "",
        "## 三、辅导员建议",
        "- 优先对高风险学生开展一对一学业预警谈话。",
        "- 对存在旷考、不及格较多的学生，建议同步班主任重点跟进。",
        "- 对中风险学生可开展提醒、帮扶与阶段复盘。",
        "- 对低风险学生建议持续关注，防止问题累积。",
    ])

    return "\n".join(lines)


def analyze_grade_warning(
    file_bytes: bytes,
    filename: str,
    sheet_name: Optional[str] = None,
):
    df, sheet_names, selected_sheet = _load_table(file_bytes, filename, sheet_name)

    if df.empty:
        return {
            "summary_markdown": "该表为空，无法分析。",
            "sheet_names": sheet_names,
            "selected_sheet": selected_sheet,
            "high_risk": [],
            "medium_risk": [],
            "low_risk": [],
            "table_preview": "空表",
        }

    info_cols = _find_info_columns(df)
    if not info_cols["student_name"]:
        return {
            "summary_markdown": "未能识别姓名列，无法进行成绩预警分析。",
            "sheet_names": sheet_names,
            "selected_sheet": selected_sheet,
            "high_risk": [],
            "medium_risk": [],
            "low_risk": [],
            "table_preview": df.head(20).to_markdown(index=False),
        }

    course_cols = _course_columns(df, info_cols)
    if not course_cols:
        return {
            "summary_markdown": "未识别到课程成绩列，无法进行成绩预警分析。",
            "sheet_names": sheet_names,
            "selected_sheet": selected_sheet,
            "high_risk": [],
            "medium_risk": [],
            "low_risk": [],
            "table_preview": df.head(20).to_markdown(index=False),
        }

    rows = []
    for _, row in df.iterrows():
        if pd.isna(row.get(info_cols["student_name"])) or str(row.get(info_cols["student_name"])).strip() == "":
            continue
        rows.append(_student_row_analysis(row, info_cols, course_cols))

    student_df = pd.DataFrame(rows)

    if student_df.empty:
        return {
            "summary_markdown": "未能生成有效的学生预警结果。",
            "sheet_names": sheet_names,
            "selected_sheet": selected_sheet,
            "high_risk": [],
            "medium_risk": [],
            "low_risk": [],
            "table_preview": df.head(20).to_markdown(index=False),
        }

    high_risk_df = student_df[student_df["风险等级"] == "高风险"].sort_values(
        by=["旷考门数", "不及格门数", "平均成绩"],
        ascending=[False, False, True],
        na_position="last",
    )
    medium_risk_df = student_df[student_df["风险等级"] == "中风险"].sort_values(
        by=["不及格门数", "低分门数", "平均成绩"],
        ascending=[False, False, True],
        na_position="last",
    )
    low_risk_df = student_df[student_df["风险等级"] == "低风险"].sort_values(
        by=["低分门数", "平均成绩"],
        ascending=[False, True],
        na_position="last",
    )

    summary_markdown = _summary_markdown(selected_sheet, student_df, course_cols)
    preview_cols = [
        c for c in [
            "学号", "姓名", "班级", "不及格门数", "低分门数", "旷考门数",
            "平均成绩", "平均学分绩点", "风险等级", "风险说明"
        ] if c in student_df.columns
    ]
    table_preview = student_df[preview_cols].head(50).to_markdown(index=False)

    return {
        "summary_markdown": summary_markdown,
        "sheet_names": sheet_names,
        "selected_sheet": selected_sheet,
        "high_risk": high_risk_df.head(30).to_dict(orient="records"),
        "medium_risk": medium_risk_df.head(50).to_dict(orient="records"),
        "low_risk": low_risk_df.head(50).to_dict(orient="records"),
        "table_preview": table_preview,
    }
