from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class SourceItem(BaseModel):
    content_preview: str
    metadata: Dict[str, Any]


class ChatRequest(BaseModel):
    question: str
    category: Optional[str] = None
    model_choice: Optional[str] = "auto"


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class NoticeRequest(BaseModel):
    topic: str
    audience: str
    category: Optional[str] = None
    style: str = "formal"
    length: str = "medium"
    must_include: Optional[str] = None
    model_choice: Optional[str] = "auto"


class NoticeResponse(BaseModel):
    notice: str
    sources: List[SourceItem]


class PaperRequest(BaseModel):
    topic: str
    paper_type: str = "research"
    section: str = "full_draft"
    category: Optional[str] = None
    style: str = "academic"
    length: str = "medium"
    must_include: Optional[str] = None
    constraints: Optional[str] = None
    model_choice: Optional[str] = "auto"


class PaperResponse(BaseModel):
    paper: str
    sources: List[SourceItem]


class TalkRecordRequest(BaseModel):
    student_name: str
    student_id: Optional[str] = None
    class_name: Optional[str] = None
    talk_type: str = "academic"
    background: str
    main_issue: str
    student_statement: Optional[str] = None
    counselor_guidance: Optional[str] = None
    follow_up_action: Optional[str] = None
    category: Optional[str] = None
    model_choice: Optional[str] = "auto"


class TalkRecordResponse(BaseModel):
    record: str
    sources: List[SourceItem]


class StudentProfileRequest(BaseModel):
    student_name: str
    class_name: Optional[str] = None
    profile_mode: str = "full"
    known_info: str
    additional_notes: Optional[str] = None
    category: Optional[str] = None
    model_choice: Optional[str] = "auto"


class StudentProfileResponse(BaseModel):
    profile: str
    sources: List[SourceItem]


class ExamRequest(BaseModel):
    topic: str
    question_type: str = "mixed"
    count: int = 5
    difficulty: str = "medium"
    include_answer: bool = True
    include_explanation: bool = True
    extra_requirements: Optional[str] = None
    category: Optional[str] = None
    model_choice: Optional[str] = "auto"


class ExamResponse(BaseModel):
    exam_content: str
    sources: List[SourceItem]





class CaseSandboxRequest(BaseModel):
    case_title: str
    student_info: Optional[str] = None
    main_issues: str
    background: Optional[str] = None
    current_state: Optional[str] = None
    current_measures: Optional[str] = None
    scenario_a: Optional[str] = None
    scenario_b: Optional[str] = None
    scenario_c: Optional[str] = None
    period: str = "两周"
    focus: Optional[str] = None
    category: Optional[str] = None
    model_choice: Optional[str] = "auto"


class CaseConsultationRequest(BaseModel):
    case_title: str
    student_info: Optional[str] = None
    main_issues: str
    background: Optional[str] = None
    current_measures: Optional[str] = None
    consultation_focus: Optional[str] = None
    urgency: str = "medium"
    extra_requirements: Optional[str] = None
    category: Optional[str] = None
    model_choice: Optional[str] = "auto"



class CaseRoleConsultationRequest(BaseModel):
    case_title: str
    student_info: Optional[str] = None
    main_issues: str
    background: Optional[str] = None
    current_measures: Optional[str] = None
    consultation_focus: Optional[str] = None
    urgency: str = "medium"
    extra_requirements: Optional[str] = None
    role_key: str
    role_name: str
    category: Optional[str] = None
    model_choice: Optional[str] = "auto"


class CaseFinalConsultationRequest(BaseModel):
    case_title: str
    student_info: Optional[str] = None
    main_issues: str
    background: Optional[str] = None
    current_measures: Optional[str] = None
    consultation_focus: Optional[str] = None
    urgency: str = "medium"
    extra_requirements: Optional[str] = None
    role_opinions: List[Dict[str, str]]
    category: Optional[str] = None
    model_choice: Optional[str] = "auto"


class CaseConsultationResponse(BaseModel):
    consultation_report: str
    sources: List[SourceItem]


class SimulationStartRequest(BaseModel):
    student_type: str
    topic: str
    difficulty: str = "medium"
    student_profile: Optional[str] = None
    category: Optional[str] = None
    model_choice: Optional[str] = "auto"


class SimulationStartResponse(BaseModel):
    student_type: str
    topic: str
    difficulty: str
    student_profile: str
    opening_message: str
    dialogue_history: List[Dict[str, str]]
    sources: List[SourceItem]


class SimulationReplyRequest(BaseModel):
    student_type: str
    topic: str
    difficulty: str = "medium"
    student_profile: str
    dialogue_history: List[Dict[str, str]]
    counselor_message: str
    category: Optional[str] = None
    model_choice: Optional[str] = "auto"


class SimulationReplyResponse(BaseModel):
    student_reply: str
    dialogue_history: List[Dict[str, str]]
    sources: List[SourceItem]


class SimulationEvaluateRequest(BaseModel):
    student_type: str
    topic: str
    difficulty: str = "medium"
    student_profile: str
    dialogue_history: List[Dict[str, str]]
    category: Optional[str] = None
    model_choice: Optional[str] = "auto"


class SimulationEvaluateResponse(BaseModel):
    evaluation: str
    sources: List[SourceItem]


class UploadResponse(BaseModel):
    message: str
    filename: str
    total_chunks: int
    category: str
    updated: bool


class FileItem(BaseModel):
    filename: str
    size: int
    category: str
    chunk_count: int


class FileListResponse(BaseModel):
    files: List[FileItem]


class DeleteResponse(BaseModel):
    message: str
    filename: str
    removed_chunks: int


class FileDetailResponse(BaseModel):
    filename: str
    size: int
    category: str
    chunk_count: int
    file_hash: str
    chunk_ids_preview: List[str]


class CategoryCreateRequest(BaseModel):
    key: str
    label: str


class CategoryItem(BaseModel):
    key: str
    label: str


class CategoryListResponse(BaseModel):
    categories: List[CategoryItem]


class HistoryRecord(BaseModel):
    title: str
    preview: str
    created_at: str


class HistoryListResponse(BaseModel):
    items: List[HistoryRecord]


class ExcelAnalysisResponse(BaseModel):
    answer: str
    preview_markdown: str
    sheet_names: List[str]
    selected_sheet: str
    analyzed_columns: List[str]


class SystemStatusResponse(BaseModel):
    service_status: str
    llm_model: str
    embed_model: str
    qa_model: str
    notice_model: str
    paper_model: str
    vision_model: str
    kb_entries: int
    category_count: int
    file_count: int
