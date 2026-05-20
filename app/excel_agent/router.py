from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.excel_agent.executor import execute_plan
from app.excel_agent.codegen import generate_python_code
from app.excel_agent.python_reporter import generate_python_agent_report
from app.excel_agent.sandbox import execute_analysis_code
from app.excel_agent.exporter import (
    export_excel_response,
    export_markdown_response,
    export_pdf_response,
    export_word_response,
)
from app.excel_agent.llm_reporter import generate_llm_report
from app.excel_agent.parser import list_sheets, preview_rows, read_sheet_smart
from app.excel_agent.planner import plan_analysis
from app.excel_agent.profiler import profile_dataframe
from app.excel_agent.reporter import generate_markdown_report
from app.excel_agent.schemas import (
    ExcelAnalyzeRequest,
    ExcelAnalyzeResponse,
    ExcelPlanResponse,
    ExcelProfileRequest,
    ExcelProfileResponse,
    ExcelUploadResponse,
    SheetInfo,
)
from app.excel_agent.session_store import (
    cleanup_old_sessions,
    create_session,
    get_raw_file,
    load_dataframe,
    load_json,
    save_dataframe,
    save_json,
    save_text,
)
from app.excel_agent.tools import run_auto_analysis

router = APIRouter()

ALLOWED_SUFFIXES = {".xlsx", ".xls", ".xlsm"}


@router.post("/upload", response_model=ExcelUploadResponse)
async def upload_excel(file: UploadFile = File(...)):
    # 上传时顺手清理一次旧会话；后台定时任务也会清理。
    try:
        cleanup_old_sessions(max_age_hours=24)
    except Exception:
        pass

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="仅支持 .xlsx / .xls / .xlsm 文件")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        session_id = create_session(tmp_path, file.filename or f"upload{suffix}")
        raw_file = get_raw_file(session_id)
        sheets_raw = list_sheets(raw_file)
        sheets = [SheetInfo(**x) for x in sheets_raw]

        active_sheet = sheets[0].name if sheets else None
        profile = None
        rows = []

        if active_sheet:
            df, info = read_sheet_smart(raw_file, active_sheet)
            save_dataframe(session_id, active_sheet, df)
            profile = profile_dataframe(df, active_sheet)
            profile["parse_info"] = info
            save_json(session_id, f"profile_{_safe_name(active_sheet)}.json", profile)
            rows = preview_rows(df)

        save_json(session_id, "sheets.json", {"sheets": sheets_raw, "active_sheet": active_sheet})
        save_json(session_id, "analysis_history.json", {"items": []})

        return ExcelUploadResponse(
            session_id=session_id,
            filename=file.filename or "",
            sheets=sheets,
            active_sheet=active_sheet,
            profile=profile,
            preview_rows=rows,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel解析失败：{e}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/profile", response_model=ExcelProfileResponse)
def profile_sheet(req: ExcelProfileRequest):
    try:
        raw_file = get_raw_file(req.session_id)
        df, info = read_sheet_smart(raw_file, req.sheet_name)
        save_dataframe(req.session_id, req.sheet_name, df)
        profile = profile_dataframe(df, req.sheet_name, excluded_course_columns=req.excluded_course_columns)
        profile["parse_info"] = info
        save_json(req.session_id, f"profile_{_safe_name(req.sheet_name)}.json", profile)

        sheets = load_json(req.session_id, "sheets.json")
        sheets["active_sheet"] = req.sheet_name
        save_json(req.session_id, "sheets.json", sheets)

        return ExcelProfileResponse(
            session_id=req.session_id,
            sheet_name=req.sheet_name,
            profile=profile,
            preview_rows=preview_rows(df),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sheet解析失败：{e}")


@router.post("/plan", response_model=ExcelPlanResponse)
def plan_excel_agent(req: ExcelAnalyzeRequest):
    sheet_name = _resolve_sheet(req.session_id, req.sheet_name)
    df, profile = _load_df_and_profile(req.session_id, sheet_name, excluded_course_columns=req.excluded_course_columns)
    plan = plan_analysis(profile, req.question, model_name=None)
    return ExcelPlanResponse(session_id=req.session_id, sheet_name=sheet_name, analysis_plan=plan)


@router.post("/analyze", response_model=ExcelAnalyzeResponse)
def analyze_excel_agent(req: ExcelAnalyzeRequest):
    try:
        sheet_name = _resolve_sheet(req.session_id, req.sheet_name)
        df, profile = _load_df_and_profile(req.session_id, sheet_name, excluded_course_columns=req.excluded_course_columns)

        # 兼容旧模式：如果 mode=deterministic，则直接用规则分析和确定性报告
        if req.mode == "deterministic":
            analysis = run_auto_analysis(df, req.question, req.mode, excluded_course_columns=req.excluded_course_columns)
            report = generate_markdown_report(
                question=req.question,
                profile=analysis["profile"],
                results=analysis["results"],
                analysis_type=analysis["analysis_type"],
            )
            plan = {"analysis_goal": "确定性分析", "reasoning": "使用规则工具直接执行。", "steps": []}
            artifacts = {"charts": [], "tables": {}}
            profile = analysis["profile"]
            results = analysis["results"]
            analysis_type = analysis["analysis_type"]
        else:
            plan = plan_analysis(profile, req.question, model_name=None)
            _inject_question_into_plan(plan, req.question)
            executed = execute_plan(req.session_id, df, profile, plan)
            results = executed["results"]
            artifacts = executed["artifacts"]
            analysis_type = "grade_warning" if "grade_warning" in results else "general"
            report = generate_llm_report(req.question, profile, plan, results)

        save_text(req.session_id, "last_report.md", report)
        save_json(req.session_id, "last_plan.json", plan)
        save_json(req.session_id, "last_results.json", results)
        save_json(req.session_id, "last_artifacts.json", artifacts)
        _append_history(req.session_id, req.question, plan, report)

        return ExcelAnalyzeResponse(
            session_id=req.session_id,
            sheet_name=sheet_name,
            analysis_type=analysis_type,
            profile=profile,
            results=results,
            report_markdown=report,
            charts=artifacts.get("charts", []),
            analysis_plan=plan,
            artifacts=artifacts,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel智能分析失败：{e}")


@router.post("/followup", response_model=ExcelAnalyzeResponse)
def followup_excel_agent(req: ExcelAnalyzeRequest):
    # 追问本质上复用同一 session 的 DataFrame 和历史上下文
    try:
        history = load_json(req.session_id, "analysis_history.json")
        recent = history.get("items", [])[-3:]
        prefix = ""
        if recent:
            prefix = "以下是最近分析历史，请结合上下文回答追问：\n"
            for i, item in enumerate(recent, 1):
                prefix += f"{i}. 用户问题：{item.get('question')}\n"
                prefix += f"   分析目标：{item.get('plan', {}).get('analysis_goal', '')}\n"
        req.question = prefix + "\n当前追问：" + (req.question or "")
    except Exception:
        pass
    return analyze_excel_agent(req)




@router.post("/python_agent_analyze")
def python_agent_analyze(req: ExcelAnalyzeRequest):
    """
    高级版：自然语言 -> Gemma4 生成 Python -> 安全沙箱执行 -> Gemma4 写报告。
    不替代原有 /analyze，而是作为高级分析入口。
    """
    try:
        sheet_name = _resolve_sheet(req.session_id, req.sheet_name)
        df, profile = _load_df_and_profile(
            req.session_id,
            sheet_name,
            excluded_course_columns=req.excluded_course_columns,
        )

        code = generate_python_code(req.question, profile)
        sandbox = execute_analysis_code(code=code, df=df, profile=profile)

        sandbox_dict = {
            "ok": sandbox.ok,
            "stdout": sandbox.stdout,
            "result": sandbox.result,
            "error": sandbox.error,
            "code": sandbox.code,
        }

        report = generate_python_agent_report(
            question=req.question,
            code=code,
            sandbox_result=sandbox_dict,
        )

        save_text(req.session_id, "last_report.md", report)
        save_json(req.session_id, "last_python_agent_result.json", sandbox_dict)

        return {
            "session_id": req.session_id,
            "sheet_name": sheet_name,
            "ok": sandbox.ok,
            "generated_code": code,
            "execution_result": sandbox_dict,
            "report_markdown": report,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel Python Agent 分析失败：{e}")


@router.get("/export/markdown/{session_id}")
def export_markdown(session_id: str):
    return export_markdown_response(session_id)


@router.get("/export/word/{session_id}")
def export_word(session_id: str):
    return export_word_response(session_id)


@router.get("/export/pdf/{session_id}")
def export_pdf(session_id: str):
    return export_pdf_response(session_id)


@router.get("/export/excel/{session_id}")
def export_excel(session_id: str):
    return export_excel_response(session_id)



@router.post("/cleanup")
def cleanup_excel_sessions(max_age_hours: int = 24):
    removed = cleanup_old_sessions(max_age_hours=max_age_hours)
    return {"removed": removed, "max_age_hours": max_age_hours}



def _inject_question_into_plan(plan: dict, question: str):
    for step in plan.get("steps", []):
        if step.get("tool") == "gpa_analysis":
            params = step.setdefault("params", {})
            params["question"] = question or ""
    return plan


def _resolve_sheet(session_id: str, sheet_name: str | None) -> str:
    if sheet_name:
        return sheet_name
    sheets = load_json(session_id, "sheets.json")
    active = sheets.get("active_sheet")
    if not active:
        raise HTTPException(status_code=400, detail="未指定Sheet")
    return active


def _load_df_and_profile(session_id: str, sheet_name: str, excluded_course_columns=None):
    excluded_course_columns = excluded_course_columns or []
    try:
        df = load_dataframe(session_id, sheet_name)
    except Exception:
        raw_file = get_raw_file(session_id)
        df, _ = read_sheet_smart(raw_file, sheet_name)
        save_dataframe(session_id, sheet_name, df)

    # 如果用户指定了排除课程列，就必须重新生成 profile，不能直接用旧缓存。
    cache_name = f"profile_{_safe_name(sheet_name)}.json"
    if excluded_course_columns:
        profile = profile_dataframe(df, sheet_name, excluded_course_columns=excluded_course_columns)
        save_json(session_id, cache_name, profile)
    else:
        try:
            profile = load_json(session_id, cache_name)
        except Exception:
            profile = profile_dataframe(df, sheet_name)
            save_json(session_id, cache_name, profile)

    return df, profile


def _append_history(session_id: str, question: str, plan, report: str):
    try:
        data = load_json(session_id, "analysis_history.json")
    except Exception:
        data = {"items": []}

    data.setdefault("items", []).append({
        "question": question,
        "plan": plan,
        "report_preview": report[:1000],
    })
    data["items"] = data["items"][-20:]
    save_json(session_id, "analysis_history.json", data)


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)[:80]
