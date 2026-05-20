from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama

from app.config import OLLAMA_BASE_URL, QA_MODEL


REPORT_SYSTEM_PROMPT = """
你是 Excel 智能分析 Agent 的高级报告生成器。

你根据：
1. 用户原始需求；
2. Gemma4 生成的 Python 代码；
3. Python 沙箱执行得到的真实结果；
生成正式 Markdown 报告。

要求：
1. 严禁编造结果，只解释 execution_result 中已有数据。
2. 如果执行失败，说明失败原因，并建议用户换一种问法或检查字段。
3. 如果结果包含表格，提炼重点，不要无限展开。
4. 面向辅导员业务场景时，突出学生定位、风险识别、跟进建议。
5. 输出 Markdown。
"""

REPORT_USER_TEMPLATE = """
用户需求：
{question}

执行状态：
{ok}

Python代码：
{code}

标准输出：
{stdout}

执行结果：
{result_json}

错误信息：
{error}

请生成 Markdown 分析报告。
"""


def generate_python_agent_report(
    question: str,
    code: str,
    sandbox_result: Dict[str, Any],
    model_name: Optional[str] = None,
) -> str:
    llm = ChatOllama(
        model=model_name or QA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=3500,
        num_ctx=12288,
    )

    prompt = REPORT_SYSTEM_PROMPT + "\n\n" + REPORT_USER_TEMPLATE.format(
        question=question or "",
        ok=sandbox_result.get("ok"),
        code=code,
        stdout=sandbox_result.get("stdout", ""),
        result_json=json.dumps(sandbox_result.get("result", {}), ensure_ascii=False, indent=2, default=str)[:14000],
        error=sandbox_result.get("error") or "",
    )

    try:
        text = llm.invoke(prompt).content.strip()
        if text:
            return text
    except Exception as e:
        return f"# Excel Python Agent 分析报告\n\n报告生成失败：{e}"

    return "# Excel Python Agent 分析报告\n\n未生成有效报告。"
