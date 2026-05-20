from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.excel_agent.profiler import profile_dataframe


ABSENCE_KEYWORDS = ["缺考", "旷考", "缓考", "未考", "未参加", "弃考", "缺测", "缺席"]


def _to_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame()
    for c in cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def _is_absence_value(v: object) -> bool:
    text = str(v).strip()
    if not text or text.lower() in ["nan", "none"]:
        return False
    return any(k in text for k in ABSENCE_KEYWORDS)


def _absence_label(v: object) -> str:
    text = str(v).strip()
    for k in ABSENCE_KEYWORDS:
        if k in text:
            return k
    return ""


def _get_course_score_columns(profile: Dict[str, Any], excluded_course_columns: Optional[List[str]] = None) -> List[str]:
    detected = profile.get("detected", {})
    excluded = {str(x).strip().lower() for x in (excluded_course_columns or detected.get("excluded_course_columns") or []) if str(x).strip()}
    cols = detected.get("course_score_columns") or []
    if not cols:
        cols = detected.get("score_columns", [])
    return [c for c in cols if str(c).strip().lower() not in excluded]


def basic_statistics(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    numeric_cols = profile["detected"].get("numeric_columns", [])
    result: Dict[str, Any] = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().sort_values(ascending=False).head(20).items()},
        "numeric_summary": {},
    }

    if numeric_cols:
        num = _to_numeric_df(df, numeric_cols)
        if not num.empty:
            desc = num.describe(percentiles=[0.25, 0.5, 0.75]).T
            for col, row in desc.iterrows():
                result["numeric_summary"][col] = {
                    "count": _safe_float(row.get("count")),
                    "mean": _safe_float(row.get("mean")),
                    "std": _safe_float(row.get("std")),
                    "min": _safe_float(row.get("min")),
                    "q25": _safe_float(row.get("25%")),
                    "median": _safe_float(row.get("50%")),
                    "q75": _safe_float(row.get("75%")),
                    "max": _safe_float(row.get("max")),
                }
    return result


def grade_warning_analysis(df: pd.DataFrame, profile: Dict[str, Any], excluded_course_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    detected = profile["detected"]
    name_col = detected.get("name_column")
    id_col = detected.get("id_column")
    class_col = detected.get("class_column")
    score_cols = _get_course_score_columns(profile, excluded_course_columns)

    if not score_cols:
        return {
            "available": False,
            "reason": "未识别到明确课程成绩列。已排除序号、平均分、总分、绩点、排名、挂科门数、学号等非课程字段。",
        }

    work = df.copy()
    scores = _to_numeric_df(work, score_cols)

    # 缺考/旷考/缓考等非数字状态要单独识别，并按风险处理。
    absence_df = pd.DataFrame(index=work.index)
    absence_detail: List[List[str]] = []
    for c in score_cols:
        absence_df[c] = work[c].apply(_is_absence_value) if c in work.columns else False

    for i in range(len(work)):
        details = []
        for c in score_cols:
            if c in work.columns and bool(absence_df.loc[i, c]):
                label = _absence_label(work.loc[i, c])
                details.append(f"{c}（{label or '缺考/旷考'}）")
        absence_detail.append(details)

    avg = scores.mean(axis=1, skipna=True)
    min_score = scores.min(axis=1, skipna=True)
    fail_mask = scores < 60
    absence_count = absence_df.sum(axis=1)
    fail_count = fail_mask.sum(axis=1)
    risk_count = fail_count + absence_count

    fail_courses = []
    for idx, row in fail_mask.iterrows():
        courses = [c for c in score_cols if bool(row.get(c, False))]
        courses.extend(absence_detail[idx])
        fail_courses.append("、".join(courses))

    def level(fc: int, ac: int, av: float) -> str:
        risk = fc + ac
        if ac >= 2 or risk >= 3 or (pd.notna(av) and av < 60):
            return "一级预警"
        if ac == 1 or fc == 2 or (pd.notna(av) and 60 <= av < 65):
            return "二级预警"
        if fc == 1 or (pd.notna(av) and 65 <= av < 70):
            return "三级预警"
        return "正常"

    levels = [level(int(fc), int(ac), float(av) if pd.notna(av) else np.nan) for fc, ac, av in zip(fail_count, absence_count, avg)]

    rows = []
    for i in range(len(work)):
        row: Dict[str, Any] = {
            "序号": int(i + 1),
            "平均分": _safe_float(avg.iloc[i]),
            "最低分": _safe_float(min_score.iloc[i]),
            "挂科门数": int(fail_count.iloc[i]),
            "缺考旷考门数": int(absence_count.iloc[i]),
            "风险课程数": int(risk_count.iloc[i]),
            "不及格/缺考课程": fail_courses[i],
            "预警等级": levels[i],
        }
        if name_col and name_col in work.columns:
            row["姓名"] = _safe_str(work.iloc[i][name_col])
        if id_col and id_col in work.columns and id_col != name_col:
            row["学号"] = _safe_str(work.iloc[i][id_col])
        if class_col and class_col in work.columns:
            row["班级"] = _safe_str(work.iloc[i][class_col])
        rows.append(row)

    warning_rows = [r for r in rows if r["预警等级"] != "正常"]
    warning_rows = sorted(
        warning_rows,
        key=lambda x: (
            {"一级预警": 0, "二级预警": 1, "三级预警": 2}.get(x["预警等级"], 9),
            -x["风险课程数"],
            -x["缺考旷考门数"],
            x["平均分"] if x["平均分"] is not None else 999,
        ),
    )

    summary = {
        "available": True,
        "score_columns": score_cols,
        "course_score_columns": score_cols,
        "student_count": int(len(work)),
        "warning_count": int(len(warning_rows)),
        "level_counts": _value_counts(levels),
        "fail_course_counts": {c: int((scores[c] < 60).sum()) for c in score_cols},
        "absence_course_counts": {c: int(absence_df[c].sum()) for c in score_cols},
        "total_absence_count": int(absence_count.sum()),
        "top_warning_students": warning_rows[:80],
        "class_column_detected": bool(class_col),
    }

    if class_col and class_col in work.columns:
        tmp = pd.DataFrame(rows)
        if "班级" in tmp.columns:
            group = tmp.groupby("班级", dropna=False).agg(
                学生数=("序号", "count"),
                预警人数=("预警等级", lambda s: int((s != "正常").sum())),
                平均分=("平均分", "mean"),
                平均挂科门数=("挂科门数", "mean"),
                平均缺考旷考门数=("缺考旷考门数", "mean"),
            ).reset_index()
            group["预警率"] = group["预警人数"] / group["学生数"]
            summary["class_summary"] = [
                {
                    "班级": _safe_str(r["班级"]),
                    "学生数": int(r["学生数"]),
                    "预警人数": int(r["预警人数"]),
                    "预警率": round(float(r["预警率"]), 4),
                    "平均分": _safe_float(r["平均分"]),
                    "平均挂科门数": _safe_float(r["平均挂科门数"]),
                    "平均缺考旷考门数": _safe_float(r["平均缺考旷考门数"]),
                }
                for _, r in group.sort_values(["预警率", "预警人数"], ascending=False).iterrows()
            ]
    else:
        summary["class_summary"] = []
        summary["class_summary_note"] = "未识别到明确班级列，因此未进行班级对比统计。"

    return summary


def gpa_analysis(df: pd.DataFrame, profile: Dict[str, Any], question: str = "") -> Dict[str, Any]:
    """
    平均学分绩点/GPA 统计分析。

    升级点：
    1. 总体统计：均值、中位数、标准差、分位数、低绩点比例；
    2. 学生定位：输出低绩点学生名单，关联姓名、学号、班级；
    3. 自定义阈值：支持从用户问题中识别“绩点低于2.8/低于3.0”等阈值。
    """
    gpa_cols = profile.get("detected", {}).get("gpa_columns", [])
    if not gpa_cols:
        return {"available": False, "reason": "未识别到平均学分绩点/GPA字段。"}

    detected = profile.get("detected", {})
    name_col = detected.get("name_column")
    id_col = detected.get("id_column")
    class_col = detected.get("class_column")

    custom_threshold = _extract_gpa_threshold(question)
    # 默认低绩点阈值：2.5；如果用户问题中写明“低于2.8”等，则使用用户阈值。
    low_threshold = custom_threshold if custom_threshold is not None else 2.5
    severe_threshold = min(2.0, low_threshold)

    results = []
    for col in gpa_cols:
        if col not in df.columns:
            continue

        raw = df[col]
        s_all = pd.to_numeric(raw, errors="coerce")
        s = s_all.dropna()
        if s.empty:
            continue

        mean = float(s.mean())
        std = float(s.std()) if len(s) > 1 else 0.0
        cv = std / mean if mean else None
        q10, q25, q50, q75, q90 = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()

        low_2 = int((s < 2.0).sum())
        low_custom = int((s < low_threshold).sum())
        high_35 = int((s >= 3.5).sum())

        z_series = None
        z_risk_count = 0
        if std and std > 0:
            z_series = (s_all - mean) / std
            z_risk_count = int((z_series <= -1.5).sum(skipna=True))

        low_students = []
        valid_count = int(s.count())

        for idx, gpa_value in s_all.items():
            if pd.isna(gpa_value):
                continue

            z_value = None
            if z_series is not None and idx in z_series.index and pd.notna(z_series.loc[idx]):
                z_value = float(z_series.loc[idx])

            # 低绩点名单：低于阈值，或 Z-score <= -1.5
            if float(gpa_value) < low_threshold or (z_value is not None and z_value <= -1.5):
                row: Dict[str, Any] = {
                    "序号": int(idx) + 1 if isinstance(idx, int) else str(idx),
                    "绩点字段": col,
                    "平均学分绩点": _safe_float(gpa_value),
                    "Z分数": _safe_float(z_value),
                    "低绩点阈值": low_threshold,
                    "风险原因": _gpa_risk_reason(float(gpa_value), z_value, low_threshold),
                }
                if name_col and name_col in df.columns:
                    row["姓名"] = _safe_str(df.loc[idx, name_col])
                if id_col and id_col in df.columns and id_col != name_col:
                    row["学号"] = _safe_str(df.loc[idx, id_col])
                if class_col and class_col in df.columns:
                    row["班级"] = _safe_str(df.loc[idx, class_col])
                low_students.append(row)

        low_students = sorted(
            low_students,
            key=lambda r: (
                r.get("平均学分绩点") if r.get("平均学分绩点") is not None else 999,
                r.get("Z分数") if r.get("Z分数") is not None else 999,
            ),
        )

        results.append({
            "column": col,
            "count": valid_count,
            "mean": _safe_float(mean),
            "median": _safe_float(q50),
            "std": _safe_float(std),
            "cv": _safe_float(cv),
            "min": _safe_float(s.min()),
            "q10": _safe_float(q10),
            "q25": _safe_float(q25),
            "q75": _safe_float(q75),
            "q90": _safe_float(q90),
            "max": _safe_float(s.max()),
            "custom_low_threshold": low_threshold,
            "custom_threshold_from_question": custom_threshold,
            "lt_2_0_count": low_2,
            "lt_2_0_rate": _safe_float(low_2 / len(s)),
            "lt_custom_count": low_custom,
            "lt_custom_rate": _safe_float(low_custom / len(s)),
            "lt_2_5_count": int((s < 2.5).sum()),
            "lt_2_5_rate": _safe_float(int((s < 2.5).sum()) / len(s)),
            "gte_3_5_count": high_35,
            "gte_3_5_rate": _safe_float(high_35 / len(s)),
            "z_le_minus_1_5_count": z_risk_count,
            "low_gpa_students": low_students[:100],
            "interpretation": _interpret_gpa(mean, std, cv, low_custom / len(s), threshold=low_threshold),
        })

    return {
        "available": bool(results),
        "gpa_columns": gpa_cols,
        "threshold": low_threshold,
        "items": results,
    }


def _extract_gpa_threshold(question: str = "") -> Optional[float]:
    """
    从用户问题里识别 GPA/绩点阈值：
    - 绩点低于2.8
    - GPA<3.0
    - 平均学分绩点小于2.5
    - 低绩点阈值2.7
    """
    import re

    q = question or ""
    patterns = [
        r"(?:绩点|GPA|gpa|平均学分绩点).{0,8}(?:低于|小于|少于|<|≤|不高于)\s*([0-5](?:\.\d+)?)",
        r"(?:低绩点阈值|绩点阈值|GPA阈值).{0,4}([0-5](?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            try:
                v = float(m.group(1))
                if 0 <= v <= 5:
                    return v
            except Exception:
                pass
    return None


def _gpa_risk_reason(gpa_value: float, z_value: Optional[float], low_threshold: float) -> str:
    reasons = []
    if gpa_value < 2.0:
        reasons.append("绩点低于2.0，属于重点学业风险")
    elif gpa_value < low_threshold:
        reasons.append(f"绩点低于{low_threshold}，低于设定关注阈值")
    if z_value is not None and z_value <= -1.5:
        reasons.append("相对群体均值明显偏低（Z≤-1.5）")
    return "；".join(reasons) or "绩点偏低"

def _interpret_gpa(mean: float, std: float, cv: Optional[float], low_rate: float, threshold: float = 2.5) -> str:
    notes = []
    if mean < 2.0:
        notes.append("整体平均绩点偏低，群体性学业风险较高")
    elif mean < 2.5:
        notes.append("整体平均绩点偏低，需要关注低绩点学生")
    else:
        notes.append("整体平均绩点处于相对可接受区间")

    if cv is not None and cv >= 0.25:
        notes.append("绩点离散程度较高，学生分化明显")
    if low_rate >= 0.2:
        notes.append("低于设定阈值的学生比例较高，建议建立重点帮扶名单")
    return "；".join(notes) + "。"


def group_statistics(df: pd.DataFrame, profile: Dict[str, Any], excluded_course_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    class_col = profile["detected"].get("class_column")
    score_cols = _get_course_score_columns(profile, excluded_course_columns)

    if not class_col or class_col not in df.columns:
        return {"available": False, "reason": "未识别到明确班级列，因此不进行班级/分组统计。"}
    if not score_cols:
        return {"available": False, "reason": "未识别到明确课程成绩列。"}

    scores = _to_numeric_df(df, score_cols)
    absence_df = pd.DataFrame(index=df.index)
    for c in score_cols:
        absence_df[c] = df[c].apply(_is_absence_value) if c in df.columns else False

    avg = scores.mean(axis=1, skipna=True)
    tmp = pd.DataFrame({
        "group": df[class_col].astype(str),
        "average_score": avg,
        "fail_count": (scores < 60).sum(axis=1),
        "absence_count": absence_df.sum(axis=1),
    })
    g = tmp.groupby("group").agg(
        count=("group", "count"),
        average_score=("average_score", "mean"),
        fail_count_avg=("fail_count", "mean"),
        absence_count_avg=("absence_count", "mean"),
        fail_student_count=("fail_count", lambda s: int((s > 0).sum())),
        absence_student_count=("absence_count", lambda s: int((s > 0).sum())),
    ).reset_index()
    g["fail_rate"] = g["fail_student_count"] / g["count"]
    g["absence_rate"] = g["absence_student_count"] / g["count"]
    return {
        "available": True,
        "group_column": class_col,
        "groups": [
            {
                "group": str(r["group"]),
                "count": int(r["count"]),
                "average_score": _safe_float(r["average_score"]),
                "fail_count_avg": _safe_float(r["fail_count_avg"]),
                "absence_count_avg": _safe_float(r["absence_count_avg"]),
                "fail_student_count": int(r["fail_student_count"]),
                "absence_student_count": int(r["absence_student_count"]),
                "fail_rate": round(float(r["fail_rate"]), 4),
                "absence_rate": round(float(r["absence_rate"]), 4),
            }
            for _, r in g.sort_values(["fail_rate", "absence_rate"], ascending=False).iterrows()
        ],
    }


def correlation_analysis(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    numeric_cols = profile["detected"].get("numeric_columns", [])
    if len(numeric_cols) < 2:
        return {"available": False, "reason": "数值列少于2列，无法做相关性分析。"}

    num = _to_numeric_df(df, numeric_cols)
    corr = num.corr(numeric_only=True)
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append({"x": cols[i], "y": cols[j], "corr": round(float(v), 4)})
    pairs = sorted(pairs, key=lambda x: abs(x["corr"]), reverse=True)
    return {"available": True, "top_pairs": pairs[:20]}


def outlier_detection(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    numeric_cols = profile["detected"].get("numeric_columns", [])
    result = {}
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 8:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        vals = s[(s < low) | (s > high)]
        result[col] = {
            "low_threshold": _safe_float(low),
            "high_threshold": _safe_float(high),
            "outlier_count": int(len(vals)),
            "min_outlier": _safe_float(vals.min()) if len(vals) else None,
            "max_outlier": _safe_float(vals.max()) if len(vals) else None,
        }
    return {"available": bool(result), "columns": result}


def run_auto_analysis(df: pd.DataFrame, question: str = "", mode: str = "auto", excluded_course_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    profile = profile_dataframe(df, excluded_course_columns=excluded_course_columns)
    q = question or ""

    results: Dict[str, Any] = {"basic_statistics": basic_statistics(df, profile)}

    course_score_cols = _get_course_score_columns(profile, excluded_course_columns)
    need_grade = (
        mode == "grade_warning"
        or any(k in q for k in ["成绩", "预警", "挂科", "不及格", "班级", "学业", "缺考", "旷考"])
        or len(course_score_cols) >= 1
    )

    if need_grade:
        results["grade_warning"] = grade_warning_analysis(df, profile, excluded_course_columns)
        results["group_statistics"] = group_statistics(df, profile, excluded_course_columns)

    if profile.get("detected", {}).get("gpa_columns") or any(k in q for k in ["绩点", "GPA", "平均学分绩点"]):
        results["gpa_analysis"] = gpa_analysis(df, profile, question=q)

    if any(k in q for k in ["相关", "关系", "影响"]) or len(profile["detected"].get("numeric_columns", [])) >= 2:
        results["correlation"] = correlation_analysis(df, profile)

    results["outliers"] = outlier_detection(df, profile)

    return {
        "profile": profile,
        "analysis_type": "grade_warning" if need_grade else "general",
        "results": results,
    }


def _safe_float(v: Any):
    try:
        if pd.isna(v):
            return None
        return round(float(v), 4)
    except Exception:
        return None


def _safe_str(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v)


def _value_counts(values: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
