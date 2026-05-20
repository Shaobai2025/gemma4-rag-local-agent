from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama

from app.config import OLLAMA_BASE_URL, QA_MODEL


CODEGEN_SYSTEM_PROMPT = """
你是 Excel 智能分析 Agent 的 Python 代码生成器。

你根据用户自然语言需求、数据摘要 profile，生成安全的 pandas 分析代码。

硬性要求：
1. 只输出 Python 代码，不要输出 Markdown，不要包裹 ```。
2. 代码只能使用 df、profile、pd、np、math。
3. 不允许 import os/sys/subprocess/requests/pathlib/shutil/pickle 等危险模块。
4. 不允许读写文件，不允许网络访问，不允许 open/eval/exec。
5. 最终必须把分析结果保存到 result 变量，且 result 必须是 dict。
6. 不要修改原始 df，必要时使用 work = df.copy()。
7. 如果需要输出表格，放入 result["tables"]，值可以是 DataFrame 或 records 列表。
8. 如果需要输出统计指标，放入 result["metrics"]。
9. 如果发现用户要求的字段不存在，要在 result["warnings"] 中说明。
10. 代码要尽量短、稳、可执行。
"""

CODEGEN_USER_TEMPLATE = """
用户需求：
{question}

数据摘要 profile：
{profile_json}

请生成安全 pandas Python 分析代码。
"""


def _compact_profile(profile: Dict[str, Any], max_cols: int = 100) -> Dict[str, Any]:
    compact_cols = []
    for c in profile.get("columns", [])[:max_cols]:
        compact_cols.append({
            "name": c.get("name"),
            "type": c.get("type"),
            "missing": c.get("missing"),
            "unique": c.get("unique"),
            "min": c.get("min"),
            "max": c.get("max"),
            "mean": c.get("mean"),
            "is_course_score": c.get("is_course_score"),
        })

    return {
        "sheet_name": profile.get("sheet_name"),
        "shape": profile.get("shape"),
        "detected": profile.get("detected"),
        "detected_tasks": profile.get("detected_tasks", []),
        "columns": compact_cols,
    }


def generate_python_code(question: str, profile: Dict[str, Any], model_name: Optional[str] = None) -> str:
    llm = ChatOllama(
        model=model_name or QA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=2200,
        num_ctx=12288,
    )

    profile_json = json.dumps(_compact_profile(profile), ensure_ascii=False, indent=2)
    prompt = CODEGEN_SYSTEM_PROMPT + "\n\n" + CODEGEN_USER_TEMPLATE.format(
        question=question or "请对这个 Excel 进行分析",
        profile_json=profile_json,
    )

    content = llm.invoke(prompt).content.strip()
    return _strip_code_fence(content)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text
