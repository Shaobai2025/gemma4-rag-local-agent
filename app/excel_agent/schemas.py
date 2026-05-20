from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SheetInfo(BaseModel):
    name: str
    rows: int
    columns: int
    detected_header_row: Optional[int] = None


class ExcelUploadResponse(BaseModel):
    session_id: str
    filename: str
    sheets: List[SheetInfo]
    active_sheet: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    preview_rows: List[Dict[str, Any]] = []


class ExcelProfileRequest(BaseModel):
    session_id: str
    sheet_name: str
    excluded_course_columns: List[str] = []


class ExcelProfileResponse(BaseModel):
    session_id: str
    sheet_name: str
    profile: Dict[str, Any]
    preview_rows: List[Dict[str, Any]]


class ExcelAnalyzeRequest(BaseModel):
    session_id: str
    sheet_name: Optional[str] = None
    question: str = "请对这个Excel进行智能分析"
    mode: str = "auto"
    excluded_course_columns: List[str] = []


class ExcelPlanResponse(BaseModel):
    session_id: str
    sheet_name: str
    analysis_plan: Dict[str, Any]


class ExcelAnalyzeResponse(BaseModel):
    session_id: str
    sheet_name: str
    analysis_type: str
    profile: Dict[str, Any]
    results: Dict[str, Any]
    report_markdown: str
    charts: List[str] = []
    analysis_plan: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None
