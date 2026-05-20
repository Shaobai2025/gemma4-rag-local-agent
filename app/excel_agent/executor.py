from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from app.excel_agent.charts import make_group_bar_chart, make_score_hist_chart
from app.excel_agent.tools import (
    basic_statistics,
    correlation_analysis,
    grade_warning_analysis,
    group_statistics,
    outlier_detection,
    gpa_analysis,
)


def execute_plan(session_id: str, df: pd.DataFrame, profile: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {"charts": [], "tables": {}}

    for step in plan.get("steps", []):
        tool = step.get("tool")
        params = step.get("params") or {}

        if tool == "basic_statistics":
            results["basic_statistics"] = basic_statistics(df, profile)

        elif tool == "grade_warning":
            results["grade_warning"] = grade_warning_analysis(df, profile)
            gw = results["grade_warning"]
            if gw.get("available"):
                artifacts["tables"]["warning_students"] = gw.get("top_warning_students", [])

        elif tool == "group_statistics":
            results["group_statistics"] = group_statistics(df, profile)

        elif tool == "correlation":
            results["correlation"] = correlation_analysis(df, profile)

        elif tool == "outlier_detection":
            results["outliers"] = outlier_detection(df, profile)

        elif tool == "gpa_analysis":
            results["gpa_analysis"] = gpa_analysis(df, profile, question=params.get("question", ""))

        elif tool == "chart_group_bar":
            if "group_statistics" not in results:
                results["group_statistics"] = group_statistics(df, profile)
            metric = params.get("metric", "fail_rate")
            chart = make_group_bar_chart(session_id, results["group_statistics"], metric=metric)
            if chart:
                artifacts["charts"].append(chart)

        elif tool == "chart_score_hist":
            score_cols = profile.get("detected", {}).get("score_columns", [])
            chart = make_score_hist_chart(session_id, df, score_cols)
            if chart:
                artifacts["charts"].append(chart)

        elif tool == "export_warning_table":
            if "grade_warning" not in results:
                results["grade_warning"] = grade_warning_analysis(df, profile)
            gw = results["grade_warning"]
            if gw.get("available"):
                artifacts["tables"]["warning_students"] = gw.get("top_warning_students", [])

    return {"results": results, "artifacts": artifacts}
