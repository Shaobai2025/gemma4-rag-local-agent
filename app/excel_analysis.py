import io
import re
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
from langchain_ollama import ChatOllama

from app.config import OLLAMA_BASE_URL, QA_MODEL
from app.prompts import EXCEL_SYSTEM_PROMPT, EXCEL_USER_PROMPT_TEMPLATE


def _get_llm() -> ChatOllama:
    return ChatOllama(
        model=QA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _load_excel_all_sheets(file_bytes: bytes, filename: str) -> Tuple[dict, List[str]]:
    ext = filename.lower().split(".")[-1]

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        df = _normalize_columns(df)
        return {"CSV": df}, ["CSV"]

    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet_names = excel_file.sheet_names
    sheets = {}

    for name in sheet_names:
        df = pd.read_excel(excel_file, sheet_name=name)
        df = _normalize_columns(df)
        sheets[name] = df

    return sheets, sheet_names


def _choose_sheet(sheets: dict, sheet_names: List[str], sheet_name: Optional[str]) -> str:
    if sheet_name and sheet_name in sheets:
        return sheet_name
    return sheet_names[0]


def _filter_columns(df: pd.DataFrame, columns: Optional[List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    if not columns:
        return df, list(df.columns)

    normalized = [str(c).strip() for c in columns if str(c).strip()]
    existing = [c for c in normalized if c in df.columns]

    if not existing:
        return df, list(df.columns)

    return df[existing].copy(), existing


def _safe_df_preview(df: pd.DataFrame, max_rows: int = 80, max_cols: int = 30) -> pd.DataFrame:
    preview_df = df.copy()

    if len(preview_df.columns) > max_cols:
        preview_df = preview_df.iloc[:, :max_cols]

    if len(preview_df) > max_rows:
        preview_df = preview_df.head(max_rows)

    return preview_df


def _classify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _build_missing_summary(df: pd.DataFrame) -> str:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        return "无缺失值。"

    lines = []
    total_rows = max(len(df), 1)
    for col, cnt in missing.head(20).items():
        ratio = cnt / total_rows * 100
        lines.append(f"- {col}: 缺失 {cnt} 个，占比 {ratio:.1f}%")
    return "\n".join(lines)


def _build_numeric_summary(df: pd.DataFrame, numeric_cols: List[str]) -> str:
    if not numeric_cols:
        return "无数值列。"

    lines = []
    for col in numeric_cols[:15]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        std_val = series.std() if len(series) > 1 else 0.0

        lines.append(
            f"- {col}: "
            f"均值={series.mean():.2f}, "
            f"中位数={series.median():.2f}, "
            f"最小值={series.min():.2f}, "
            f"最大值={series.max():.2f}, "
            f"标准差={std_val:.2f}, "
            f"非空数={series.count()}"
        )

    return "\n".join(lines) if lines else "数值列存在，但没有可用统计结果。"


def _build_categorical_summary(df: pd.DataFrame, categorical_cols: List[str]) -> str:
    if not categorical_cols:
        return "无分类列。"

    lines = []
    for col in categorical_cols[:12]:
        vc = df[col].astype(str).fillna("空值").value_counts(dropna=False)
        top_items = vc.head(8).to_dict()
        top_text = ", ".join([f"{k}: {v}" for k, v in top_items.items()])
        lines.append(f"- {col}: {top_text}")

    return "\n".join(lines) if lines else "分类列存在，但没有可用统计结果。"


def _find_name_like_column(df: pd.DataFrame) -> Optional[str]:
    keywords = ["姓名", "学生姓名", "名字", "name", "student"]
    for col in df.columns:
        c = str(col).lower()
        if any(k.lower() in c for k in keywords):
            return col
    return None


def _find_class_like_column(df: pd.DataFrame) -> Optional[str]:
    keywords = ["班级", "专业", "年级", "class", "major"]
    for col in df.columns:
        c = str(col).lower()
        if any(k.lower() in c for k in keywords):
            return col
    return None


def _detect_scene(df: pd.DataFrame, question: str) -> str:
    cols = " ".join(map(str, df.columns)).lower()
    q = question.lower()

    if any(k in cols or k in q for k in ["成绩", "分数", "绩点", "挂科", "score", "grade", "gpa"]):
        return "grade"
    if any(k in cols or k in q for k in ["请假", "缺勤", "考勤", "晚归", "归寝", "未签到", "attendance", "leave"]):
        return "attendance"
    if any(k in cols or k in q for k in ["违纪", "处分", "通报", "discipline"]):
        return "discipline"
    if any(k in cols or k in q for k in ["心理", "情绪", "预警", "mental", "stress"]):
        return "mental"
    if any(k in cols or k in q for k in ["就业", "去向", "签约", "升学", "employment"]):
        return "employment"
    return "general"


def _top_students_by_count(df: pd.DataFrame, count_col_name: str = "记录数", top_n: int = 10) -> str:
    name_col = _find_name_like_column(df)
    if not name_col:
        return "未识别到姓名列，无法输出重点学生名单。"

    vc = df[name_col].astype(str).value_counts().head(top_n)
    if vc.empty:
        return "未识别到重点学生。"

    return "\n".join([f"- {name}: {cnt}{count_col_name}" for name, cnt in vc.items()])


def _build_grade_analysis(df: pd.DataFrame) -> str:
    numeric_cols, _ = _classify_columns(df)
    if not numeric_cols:
        return "未识别到可用于成绩分析的数值列。"

    lines = ["【成绩分析】"]
    for col in numeric_cols[:8]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        fail_cnt = int((s < 60).sum())
        low_cnt = int((s < 70).sum())
        lines.append(
            f"- {col}: 平均分={s.mean():.2f}，最高分={s.max():.2f}，最低分={s.min():.2f}，"
            f"不及格人数(<60)={fail_cnt}，低分人数(<70)={low_cnt}"
        )

    return "\n".join(lines)


def _build_attendance_analysis(df: pd.DataFrame) -> str:
    lines = ["【考勤/请假/归寝分析】"]
    lines.append(f"- 总记录数: {len(df)}")

    name_focus = _top_students_by_count(df, count_col_name="次")
    lines.append("【高频出现学生】")
    lines.append(name_focus)

    class_col = _find_class_like_column(df)
    if class_col:
        vc = df[class_col].astype(str).value_counts().head(8)
        lines.append("【高频班级/专业】")
        lines.extend([f"- {k}: {v}次" for k, v in vc.items()])

    return "\n".join(lines)


def _build_discipline_analysis(df: pd.DataFrame) -> str:
    lines = ["【违纪分析】"]
    lines.append(f"- 总记录数: {len(df)}")
    lines.append("【重点学生】")
    lines.append(_top_students_by_count(df, count_col_name="次"))

    for col in df.columns[:10]:
        vc = df[col].astype(str).value_counts().head(5)
        if len(vc) > 1:
            lines.append(f"【{col} 高频情况】")
            lines.extend([f"- {k}: {v}" for k, v in vc.items()])
            break

    return "\n".join(lines)


def _build_mental_analysis(df: pd.DataFrame) -> str:
    lines = ["【心理/预警分析】"]
    lines.append(f"- 总记录数: {len(df)}")
    lines.append("【重点学生】")
    lines.append(_top_students_by_count(df, count_col_name="次"))
    return "\n".join(lines)


def _build_employment_analysis(df: pd.DataFrame) -> str:
    lines = ["【就业去向分析】"]
    lines.append(f"- 总人数/记录数: {len(df)}")

    for col in df.columns[:12]:
        vc = df[col].astype(str).value_counts().head(8)
        if len(vc) > 1:
            lines.append(f"【{col} 分布】")
            lines.extend([f"- {k}: {v}" for k, v in vc.items()])
            break

    return "\n".join(lines)


def _build_scene_analysis(df: pd.DataFrame, scene: str) -> str:
    if scene == "grade":
        return _build_grade_analysis(df)
    if scene == "attendance":
        return _build_attendance_analysis(df)
    if scene == "discipline":
        return _build_discipline_analysis(df)
    if scene == "mental":
        return _build_mental_analysis(df)
    if scene == "employment":
        return _build_employment_analysis(df)
    return "【通用分析】未识别到明确业务场景，以下将基于全表统计进行综合分析。"


def _build_shape_text(df: pd.DataFrame, sheet_name: str, numeric_cols: List[str], categorical_cols: List[str], scene: str) -> str:
    return (
        f"工作表: {sheet_name}\n"
        f"分析场景: {scene}\n"
        f"总行数: {len(df)}\n"
        f"总列数: {len(df.columns)}\n"
        f"数值列数: {len(numeric_cols)}\n"
        f"分类列数: {len(categorical_cols)}"
    )


def _build_columns_text(df: pd.DataFrame, analyzed_columns: List[str]) -> str:
    cols = analyzed_columns if analyzed_columns else list(df.columns)
    return ", ".join(map(str, cols))


def _build_preview_text(
    preview_markdown: str,
    missing_text: str,
    numeric_text: str,
    categorical_text: str,
    scene_text: str,
) -> str:
    return (
        "【场景专项分析】\n"
        f"{scene_text}\n\n"
        "【表格预览】\n"
        f"{preview_markdown}\n\n"
        "【缺失值统计】\n"
        f"{missing_text}\n\n"
        "【数值列统计】\n"
        f"{numeric_text}\n\n"
        "【分类列统计】\n"
        f"{categorical_text}"
    )


def analyze_spreadsheet(
    file_bytes: bytes,
    filename: str,
    question: str,
    sheet_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
):
    sheets, sheet_names = _load_excel_all_sheets(file_bytes, filename)
    selected_sheet = _choose_sheet(sheets, sheet_names, sheet_name)

    df = sheets[selected_sheet]

    if df.empty:
        preview_markdown = "该工作表为空。"
        return (
            "该工作表为空，无法进行有效分析。",
            preview_markdown,
            sheet_names,
            selected_sheet,
            [],
        )

    df, analyzed_columns = _filter_columns(df, columns)

    numeric_cols, categorical_cols = _classify_columns(df)
    scene = _detect_scene(df, question)

    preview_df = _safe_df_preview(df, max_rows=80, max_cols=30)
    preview_markdown = preview_df.to_markdown(index=False)

    missing_text = _build_missing_summary(df)
    numeric_text = _build_numeric_summary(df, numeric_cols)
    categorical_text = _build_categorical_summary(df, categorical_cols)
    scene_text = _build_scene_analysis(df, scene)

    shape_text = _build_shape_text(df, selected_sheet, numeric_cols, categorical_cols, scene)
    columns_text = _build_columns_text(df, analyzed_columns)
    preview_text = _build_preview_text(
        preview_markdown=preview_markdown,
        missing_text=missing_text,
        numeric_text=numeric_text,
        categorical_text=categorical_text,
        scene_text=scene_text,
    )

    llm = _get_llm()

    prompt = (
        f"{EXCEL_SYSTEM_PROMPT}\n\n"
        + EXCEL_USER_PROMPT_TEMPLATE.format(
            question=question,
            shape_text=shape_text,
            columns_text=columns_text,
            preview_text=preview_text,
        )
    )

    result = llm.invoke(prompt)
    answer = result.content if hasattr(result, "content") else str(result)

    return (
        answer.strip(),
        preview_markdown,
        sheet_names,
        selected_sheet,
        analyzed_columns,
    )