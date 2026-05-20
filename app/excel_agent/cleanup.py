from __future__ import annotations

import asyncio
import os

from app.excel_agent.session_store import cleanup_old_sessions


async def excel_cleanup_loop() -> None:
    """
    Excel Agent 会话定时清理循环。

    环境变量：
    - EXCEL_SESSION_MAX_AGE_HOURS：保留多少小时以内的 Excel 分析数据，默认 24 小时
    - EXCEL_CLEANUP_INTERVAL_MINUTES：每隔多少分钟检查一次，默认 60 分钟
    """
    max_age_hours = int(os.getenv("EXCEL_SESSION_MAX_AGE_HOURS", "24"))
    interval_minutes = int(os.getenv("EXCEL_CLEANUP_INTERVAL_MINUTES", "60"))

    # 防止配置过小导致频繁清理；至少 5 分钟检查一次
    interval_minutes = max(5, interval_minutes)

    print(
        f"[ExcelAgent] Excel 会话清理任务运行中："
        f"每 {interval_minutes} 分钟检查一次，清理 {max_age_hours} 小时前的数据"
    )

    while True:
        try:
            removed = cleanup_old_sessions(max_age_hours=max_age_hours)
            print(f"[ExcelAgent] 本轮清理完成，删除过期会话：{removed} 个")
        except Exception as e:
            print(f"[ExcelAgent] 清理任务异常：{e}")

        await asyncio.sleep(interval_minutes * 60)


def start_excel_cleanup_task() -> None:
    """
    在 FastAPI startup 中调用：

        start_excel_cleanup_task()

    作用：
    - 启动后台异步任务；
    - 不阻塞主服务；
    - 定时清理 data/excel_sessions 下的过期 Excel 会话目录。
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(excel_cleanup_loop())
        print("[ExcelAgent] Excel 会话定时清理任务已启动")
    except RuntimeError:
        print("[ExcelAgent] 未找到运行中的事件循环，清理任务未启动")
