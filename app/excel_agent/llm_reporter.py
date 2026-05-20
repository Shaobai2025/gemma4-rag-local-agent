from __future__ import annotations

import json
from typing import Any, Dict

from langchain_ollama import ChatOllama

from app.config import OLLAMA_BASE_URL, QA_MODEL
from app.excel_agent.reporter import generate_markdown_report


REPORT_SYSTEM_PROMPT = """
你是 Excel 智能分析 Agent 的报告解释器。

你根据 Python 工具的真实计算结果，生成自然语言分析报告。
严禁编造数据；只能解释工具结果中已有的数字和表格。

要求：
1. 先说明分析目标和执行步骤。
2. 对关键结果进行解释。
3. 如果是成绩预警，要突出重点学生、班级差异、挂科课程、缺考/旷考情况、帮扶建议。
4. 如果有平均学分绩点/GPA分析，要解释均值、中位数、标准差、低绩点比例和离散程度，并重点列明低绩点学生名单中的姓名、学号、班级，方便辅导员查找。
4. 如果工具结果不足，要明确说明不足和下一步建议。
5. 输出 Markdown。
"""

REPORT_USER_TEMPLATE = """
用户问题：
{question}

分析计划：
{plan}

数据摘要：
{profile}

Python工具结果：
{results}

请生成一份清晰、正式、可复制到工作报告中的 Markdown 分析报告。
"""


def get_report_llm(model_name: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model_name or QA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=3500,
        num_ctx=12288,
    )


def generate_llm_report(
    question: str,
    profile: Dict[str, Any],
    plan: Dict[str, Any],
    results: Dict[str, Any],
    model_name: str | None = None,
) -> str:
    compact_profile = _compact(profile, max_chars=5000)
    compact_results = _compact(results, max_chars=10000)
    compact_plan = _compact(plan, max_chars=3000)

    prompt = REPORT_SYSTEM_PROMPT + "\n\n" + REPORT_USER_TEMPLATE.format(
        question=question or "请对这个 Excel 进行智能分析",
        plan=compact_plan,
        profile=compact_profile,
        results=compact_results,
    )

    try:
        llm = get_report_llm(model_name)
        text = llm.invoke(prompt).content.strip()
        if text:
            return text
    except Exception:
        pass

    # 兜底：不用模型也能生成确定性报告
    analysis_type = "grade_warning" if "grade_warning" in results else "general"
    return generate_markdown_report(question, profile, results, analysis_type)


def _compact(obj: Any, max_chars: int = 8000) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...（内容已截断）"
    return text
