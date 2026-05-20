from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_ollama import ChatOllama

from app.config import OLLAMA_BASE_URL, QA_MODEL


PLANNER_SYSTEM_PROMPT = """
你是 Excel 智能分析 Agent 的规划器。

你只负责根据数据摘要和用户问题，生成 JSON 分析计划，不直接计算。

必须遵守：
1. 只能输出 JSON，不要输出 Markdown。
2. tool 只能从允许工具中选择。
3. 不要编造不存在的列名。
4. 如果是成绩表，优先使用 grade_warning、group_statistics、chart_group_bar。
5. 如果用户问“相关/关系/影响”，使用 correlation。
6. 如果用户问“异常/离群/极端”，使用 outlier_detection。
7. 如果用户要求导出预警名单，使用 export_warning_table。
8. 如果用户提到绩点/GPA/平均学分绩点，使用 gpa_analysis。
8. 分析计划要短，不要超过 6 个步骤。

允许工具：
- basic_statistics
- grade_warning
- group_statistics
- correlation
- outlier_detection
- gpa_analysis
- chart_group_bar
- chart_score_hist
- export_warning_table

输出 JSON 格式：
{
  "analysis_goal": "...",
  "reasoning": "...",
  "steps": [
    {"tool": "basic_statistics", "params": {}}
  ]
}
"""

PLANNER_USER_TEMPLATE = """
用户问题：
{question}

数据摘要：
{profile_summary}

请生成 JSON 分析计划。
"""


def get_planner_llm(model_name: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model_name or QA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=1200,
        num_ctx=8192,
    )


def make_profile_summary(profile: Dict[str, Any]) -> str:
    detected = profile.get("detected", {})
    columns = profile.get("columns", [])
    compact_cols = []
    for c in columns[:80]:
        item = {
            "name": c.get("name"),
            "type": c.get("type"),
            "missing": c.get("missing"),
            "unique": c.get("unique"),
        }
        if c.get("type") in ["numeric", "score"]:
            item.update({"min": c.get("min"), "max": c.get("max"), "mean": c.get("mean")})
        compact_cols.append(item)

    summary = {
        "sheet_name": profile.get("sheet_name"),
        "shape": profile.get("shape"),
        "detected": detected,
        "detected_tasks": profile.get("detected_tasks", []),
        "columns": compact_cols,
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def plan_analysis(profile: Dict[str, Any], question: str, model_name: str | None = None) -> Dict[str, Any]:
    prompt = (
        PLANNER_SYSTEM_PROMPT
        + "\n\n"
        + PLANNER_USER_TEMPLATE.format(
            question=question or "请对这个 Excel 进行智能分析",
            profile_summary=make_profile_summary(profile),
        )
    )

    try:
        llm = get_planner_llm(model_name)
        content = llm.invoke(prompt).content.strip()
        plan = _extract_json(content)
        return validate_or_fallback_plan(plan, profile, question)
    except Exception as e:
        plan = fallback_plan(profile, question)
        plan["planner_error"] = str(e)
        return plan


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError("未找到 JSON")
        return json.loads(m.group(0))


def validate_or_fallback_plan(plan: Dict[str, Any], profile: Dict[str, Any], question: str) -> Dict[str, Any]:
    allowed = {
        "basic_statistics",
        "grade_warning",
        "group_statistics",
        "correlation",
        "outlier_detection",
        "gpa_analysis",
        "chart_group_bar",
        "chart_score_hist",
        "export_warning_table",
    }

    steps = plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return fallback_plan(profile, question)

    valid_steps = []
    for step in steps[:6]:
        if not isinstance(step, dict):
            continue
        tool = step.get("tool")
        if tool not in allowed:
            continue
        params = step.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        valid_steps.append({"tool": tool, "params": params})

    if not valid_steps:
        return fallback_plan(profile, question)

    return {
        "analysis_goal": str(plan.get("analysis_goal") or "Excel智能分析"),
        "reasoning": str(plan.get("reasoning") or "根据用户问题和字段结构生成分析计划。"),
        "steps": valid_steps,
    }


def fallback_plan(profile: Dict[str, Any], question: str) -> Dict[str, Any]:
    q = question or ""
    detected = profile.get("detected", {})
    score_cols = detected.get("score_columns", [])
    class_col = detected.get("class_column")
    numeric_cols = detected.get("numeric_columns", [])

    steps: List[Dict[str, Any]] = [{"tool": "basic_statistics", "params": {}}]

    if score_cols and any(k in q for k in ["成绩", "预警", "挂科", "不及格", "班级", "学业", "学生", "缺考", "旷考"]):
        steps.append({"tool": "grade_warning", "params": {}})
        if class_col:
            steps.append({"tool": "group_statistics", "params": {}})
            steps.append({"tool": "chart_group_bar", "params": {"metric": "fail_rate"}})
        steps.append({"tool": "chart_score_hist", "params": {}})

    if detected.get("gpa_columns") or any(k in q for k in ["绩点", "GPA", "平均学分绩点"]):
        steps.append({"tool": "gpa_analysis", "params": {"question": q}})

    if len(numeric_cols) >= 2 and any(k in q for k in ["相关", "关系", "影响"]):
        steps.append({"tool": "correlation", "params": {}})

    if any(k in q for k in ["异常", "离群", "极端", "最高", "最低"]):
        steps.append({"tool": "outlier_detection", "params": {}})

    if "导出" in q and score_cols:
        steps.append({"tool": "export_warning_table", "params": {}})

    return {
        "analysis_goal": "自动降级分析计划",
        "reasoning": "Gemma4 规划不可用或结果不规范，已根据字段类型和问题关键词生成规则计划。",
        "steps": steps[:6],
    }
