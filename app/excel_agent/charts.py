from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.excel_agent.session_store import session_dir


def charts_dir(session_id: str) -> Path:
    d = session_dir(session_id) / "charts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_group_bar_chart(session_id: str, group_stats: Dict[str, Any], metric: str = "fail_rate") -> str | None:
    if not group_stats.get("available"):
        return None

    groups = group_stats.get("groups", [])[:20]
    if not groups:
        return None

    labels = [str(x.get("group", "")) for x in groups]
    values = [float(x.get(metric, 0) or 0) for x in groups]

    fig = plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(f"Group comparison: {metric}")
    plt.tight_layout()

    path = charts_dir(session_id) / f"group_bar_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def make_score_hist_chart(session_id: str, df: pd.DataFrame, score_cols: List[str]) -> str | None:
    if not score_cols:
        return None

    scores = pd.DataFrame()
    for c in score_cols:
        scores[c] = pd.to_numeric(df[c], errors="coerce")
    avg = scores.mean(axis=1, skipna=True).dropna()

    if avg.empty:
        return None

    fig = plt.figure(figsize=(8, 5))
    plt.hist(avg, bins=20)
    plt.xlabel("Average score")
    plt.ylabel("Count")
    plt.title("Distribution of average score")
    plt.tight_layout()

    path = charts_dir(session_id) / "average_score_hist.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def public_chart_paths(paths: List[str]) -> List[str]:
    # 当前先返回本地路径，前端暂不直接展示图片；后续可以挂静态目录。
    return [p for p in paths if p]
