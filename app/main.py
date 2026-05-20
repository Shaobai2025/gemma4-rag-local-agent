import json
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from app.categories import (
    add_category,
    delete_category,
    ensure_categories_file,
    load_categories,
)
from app.config import (
    ALLOWED_ANALYSIS_EXTENSIONS,
    ALLOWED_EXTENSIONS,
    DOCS_DIR,
    EMBED_MODEL,
    QA_MODEL,
    NOTICE_MODEL,
    PAPER_MODEL,
    VISION_MODEL,
)
from app.excel_analysis import analyze_spreadsheet
from app.excel_warning_analysis import analyze_grade_warning
from app.excel_agent.router import router as excel_agent_router
from app.excel_agent.cleanup import start_excel_cleanup_task
from app.exporters import markdown_bytes, docx_bytes, excel_report_bytes
from app.history import add_history, ensure_history_file, list_history
from app.ingest import bootstrap_existing_docs, delete_file_and_indexes, upsert_file
from app.loaders import infer_category
from app.rag import (
    answer_question,
    generate_notice,
    generate_paper,
    generate_talk_record,
    generate_student_profile,
    generate_exam_questions,
    start_simulation,
    simulate_student_reply,
    evaluate_simulation,
    generate_case_consultation,
    stream_generate_notice,
    stream_generate_paper,
    stream_generate_talk_record,
    stream_generate_student_profile,
    stream_generate_exam_questions,
    stream_start_simulation,
    stream_simulate_student_reply,
    stream_evaluate_simulation,
    stream_generate_case_consultation,
    stream_generate_case_role_consultation,
    stream_generate_case_final_consultation,
    stream_answer_question,
)
from app.schemas import (
    CaseSandboxRequest,
    CaseConsultationRequest,
    CaseConsultationResponse,
    CaseRoleConsultationRequest,
    CaseFinalConsultationRequest,
    CategoryCreateRequest,
    CategoryItem,
    CategoryListResponse,
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    ExamRequest,
    ExamResponse,
    ExcelAnalysisResponse,
    FileDetailResponse,
    FileItem,
    FileListResponse,
    HistoryListResponse,
    HistoryRecord,
    NoticeRequest,
    NoticeResponse,
    PaperRequest,
    PaperResponse,
    SourceItem,
    SimulationStartRequest,
    SimulationStartResponse,
    SimulationReplyRequest,
    SimulationReplyResponse,
    SimulationEvaluateRequest,
    SimulationEvaluateResponse,
    StudentProfileRequest,
    StudentProfileResponse,
    SystemStatusResponse,
    TalkRecordRequest,
    TalkRecordResponse,
    UploadResponse,
)
from app.store import load_registry, read_sparse_rows
from app.vision import analyze_image_with_ollama
from app.case_graph import stream_case_graph_events
from app.sandbox_graph import stream_case_sandbox_events
from app.consensus_graph import stream_consensus_case_events

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Gemma4 Hybrid RAG API", version="7.2.0")

app.include_router(excel_agent_router, prefix="/excel_agent", tags=["excel_agent"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def ensure_docs_dir():
    os.makedirs(DOCS_DIR, exist_ok=True)


def get_ext(filename: str) -> str:
    return Path(filename).suffix.lower()


def _event_text_from_chunk(chunk):
    if hasattr(chunk, "content") and chunk.content:
        return chunk.content
    if isinstance(chunk, str):
        return chunk
    return ""


def _source_names_from_docs(docs):
    names = []
    seen = set()
    for d in docs:
        name = (d.metadata or {}).get("source_file", "未知文件")
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _streaming_response(stream_iter, docs, history_kind: str | None = None, history_title: str | None = None, done_extra: dict | None = None):
    source_names = _source_names_from_docs(docs)

    def event_generator():
        full_text_parts = []
        try:
            for chunk in stream_iter:
                text = _event_text_from_chunk(chunk)
                if text:
                    full_text_parts.append(text)
                    yield f"data: {json.dumps({'type': 'token', 'content': text}, ensure_ascii=True)}\n\n"

            final_text = "".join(full_text_parts)
            if history_kind and history_title:
                add_history(history_kind, history_title, final_text)

            yield f"data: {json.dumps({'type': 'sources', 'sources': source_names}, ensure_ascii=True)}\n\n"
            done_payload = {'type': 'done'}
            if done_extra:
                done_payload.update(done_extra)
            yield f"data: {json.dumps(done_payload, ensure_ascii=True)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=True)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.on_event("startup")
def startup_sync():
    start_excel_cleanup_task()
    ensure_docs_dir()
    ensure_categories_file()
    ensure_history_file()
    bootstrap_existing_docs()


@app.get("/")
def root():
    return RedirectResponse(url="/ask")


@app.get("/kb")
def kb_page():
    page = FRONTEND_DIR / "kb.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="frontend/kb.html 不存在")
    return FileResponse(str(page))


@app.get("/ask")
def ask_page():
    page = FRONTEND_DIR / "workspace.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="frontend/workspace.html 不存在")
    return FileResponse(str(page))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/system_status", response_model=SystemStatusResponse)
def system_status():
    registry = load_registry()
    categories = load_categories()
    rows = read_sparse_rows()

    return SystemStatusResponse(
        service_status="running",
        llm_model=QA_MODEL,
        embed_model=EMBED_MODEL,
        qa_model=QA_MODEL,
        notice_model=NOTICE_MODEL,
        paper_model=PAPER_MODEL,
        vision_model=VISION_MODEL,
        kb_entries=len(rows),
        category_count=len(categories),
        file_count=len(registry),
    )


@app.get("/categories", response_model=CategoryListResponse)
def list_categories():
    return CategoryListResponse(categories=[CategoryItem(**c) for c in load_categories()])


@app.post("/categories", response_model=CategoryListResponse)
def create_category(req: CategoryCreateRequest):
    return CategoryListResponse(
        categories=[CategoryItem(**c) for c in add_category(req.key, req.label)]
    )


@app.delete("/categories/{key}", response_model=CategoryListResponse)
def remove_category(key: str):
    registry = load_registry()
    if any(meta.get("category") == key for meta in registry.values()):
        raise HTTPException(status_code=400, detail="该类别仍被文件使用，无法删除")
    return CategoryListResponse(
        categories=[CategoryItem(**c) for c in delete_category(key)]
    )


@app.get("/history/{kind}", response_model=HistoryListResponse)
def get_history(kind: str):
    return HistoryListResponse(items=[HistoryRecord(**x) for x in list_history(kind)])


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), category: Optional[str] = Form(None)):
    ensure_docs_dir()

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = get_ext(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    safe_name = os.path.basename(file.filename)
    save_path = os.path.join(DOCS_DIR, safe_name)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    total_chunks, final_category, updated = upsert_file(
        save_path,
        safe_name,
        category or infer_category(safe_name),
    )

    return UploadResponse(
        message="文件已增量入库",
        filename=safe_name,
        total_chunks=total_chunks,
        category=final_category,
        updated=updated,
    )


@app.get("/files", response_model=FileListResponse)
def list_files():
    ensure_docs_dir()
    registry = load_registry()
    items = []

    for fname in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, fname)
        if not os.path.isfile(path):
            continue

        meta = registry.get(fname, {})
        items.append(
            FileItem(
                filename=fname,
                size=os.path.getsize(path),
                category=meta.get("category", infer_category(fname)),
                chunk_count=meta.get("chunk_count", 0),
            )
        )

    return FileListResponse(files=items)


@app.get("/files/{filename}/detail", response_model=FileDetailResponse)
def file_detail(filename: str):
    ensure_docs_dir()
    safe_name = os.path.basename(filename)
    file_path = os.path.join(DOCS_DIR, safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    meta = load_registry().get(safe_name, {})
    return FileDetailResponse(
        filename=safe_name,
        size=os.path.getsize(file_path),
        category=meta.get("category", infer_category(safe_name)),
        chunk_count=meta.get("chunk_count", 0),
        file_hash=meta.get("file_hash", ""),
        chunk_ids_preview=meta.get("chunk_ids", [])[:10],
    )


@app.get("/files/{filename}/download")
def download_file(filename: str):
    ensure_docs_dir()
    safe_name = os.path.basename(filename)
    file_path = os.path.join(DOCS_DIR, safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        file_path,
        filename=safe_name,
        media_type="application/octet-stream",
    )


@app.delete("/files/{filename}", response_model=DeleteResponse)
def delete_file(filename: str):
    ensure_docs_dir()
    safe_name = os.path.basename(filename)
    file_path = os.path.join(DOCS_DIR, safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    removed = delete_file_and_indexes(file_path, safe_name)
    return DeleteResponse(
        message="文件及对应索引已增量删除",
        filename=safe_name,
        removed_chunks=removed,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    answer, docs = answer_question(req.question, req.category, req.model_choice)
    add_history("qa", req.question, answer)

    sources = [
        SourceItem(
            content_preview=d.page_content[:220].replace("\n", " "),
            metadata=d.metadata,
        )
        for d in docs
    ]

    return ChatResponse(answer=answer, sources=sources)


@app.post("/chat_stream")
async def chat_stream(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    stream_iter, docs = stream_answer_question(
        req.question,
        req.category,
        req.model_choice,
    )

    source_names = []
    seen = set()
    for d in docs:
        name = (d.metadata or {}).get("source_file", "未知文件")
        if name not in seen:
            seen.add(name)
            source_names.append(name)

    def event_generator():
        full_text_parts = []

        try:
            for chunk in stream_iter:
                text = ""
                if hasattr(chunk, "content") and chunk.content:
                    text = chunk.content
                elif isinstance(chunk, str):
                    text = chunk

                if text:
                    full_text_parts.append(text)
                    yield f"data: {json.dumps({'type': 'token', 'content': text}, ensure_ascii=True)}\n\n"

            final_answer = "".join(full_text_parts)
            add_history("qa", req.question, final_answer)

            yield f"data: {json.dumps({'type': 'sources', 'sources': source_names}, ensure_ascii=True)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=True)}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=True)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/generate_notice", response_model=NoticeResponse)
def create_notice(req: NoticeRequest):
    if not req.topic.strip() or not req.audience.strip():
        raise HTTPException(status_code=400, detail="主题和通知对象不能为空")

    notice, docs = generate_notice(
        topic=req.topic,
        audience=req.audience,
        style=req.style,
        length=req.length,
        must_include=req.must_include,
        category=req.category,
        model_choice=req.model_choice,
    )
    add_history("notice", req.topic, notice)

    sources = [
        SourceItem(
            content_preview=d.page_content[:220].replace("\n", " "),
            metadata=d.metadata,
        )
        for d in docs
    ]

    return NoticeResponse(notice=notice, sources=sources)


@app.post("/generate_paper", response_model=PaperResponse)
def create_paper(req: PaperRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="论文主题不能为空")

    paper, docs = generate_paper(
        topic=req.topic,
        paper_type=req.paper_type,
        section=req.section,
        style=req.style,
        length=req.length,
        must_include=req.must_include,
        constraints=req.constraints,
        category=req.category,
        model_choice=req.model_choice,
    )
    add_history("paper", req.topic, paper)

    sources = [
        SourceItem(
            content_preview=d.page_content[:220].replace("\n", " "),
            metadata=d.metadata,
        )
        for d in docs
    ]

    return PaperResponse(paper=paper, sources=sources)


@app.post("/generate_talk_record", response_model=TalkRecordResponse)
def create_talk_record(req: TalkRecordRequest):
    if not req.student_name.strip():
        raise HTTPException(status_code=400, detail="学生姓名不能为空")
    if not req.background.strip():
        raise HTTPException(status_code=400, detail="谈话背景不能为空")
    if not req.main_issue.strip():
        raise HTTPException(status_code=400, detail="主要问题不能为空")

    record, docs = generate_talk_record(
        student_name=req.student_name,
        student_id=req.student_id,
        class_name=req.class_name,
        talk_type=req.talk_type,
        background=req.background,
        main_issue=req.main_issue,
        student_statement=req.student_statement,
        counselor_guidance=req.counselor_guidance,
        follow_up_action=req.follow_up_action,
        category=req.category,
        model_choice=req.model_choice,
    )

    add_history("talk_record", req.student_name, record)

    sources = [
        SourceItem(
            content_preview=d.page_content[:220].replace("\n", " "),
            metadata=d.metadata,
        )
        for d in docs
    ]
    return TalkRecordResponse(record=record, sources=sources)


@app.post("/generate_student_profile", response_model=StudentProfileResponse)
def create_student_profile(req: StudentProfileRequest):
    if not req.student_name.strip():
        raise HTTPException(status_code=400, detail="学生姓名不能为空")
    if not req.known_info.strip():
        raise HTTPException(status_code=400, detail="已知信息不能为空")

    profile, docs = generate_student_profile(
        student_name=req.student_name,
        class_name=req.class_name,
        profile_mode=req.profile_mode,
        known_info=req.known_info,
        additional_notes=req.additional_notes,
        category=req.category,
        model_choice=req.model_choice,
    )

    add_history("student_profile", req.student_name, profile)

    sources = [
        SourceItem(
            content_preview=d.page_content[:220].replace("\n", " "),
            metadata=d.metadata,
        )
        for d in docs
    ]
    return StudentProfileResponse(profile=profile, sources=sources)


@app.post("/generate_exam_questions", response_model=ExamResponse)
def create_exam_questions(req: ExamRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="出题主题不能为空")

    exam_content, docs = generate_exam_questions(
        topic=req.topic,
        question_type=req.question_type,
        count=req.count,
        difficulty=req.difficulty,
        include_answer=req.include_answer,
        include_explanation=req.include_explanation,
        extra_requirements=req.extra_requirements,
        category=req.category,
        model_choice=req.model_choice,
    )

    add_history("exam", req.topic, exam_content)

    sources = [
        SourceItem(
            content_preview=d.page_content[:220].replace("\n", " "),
            metadata=d.metadata,
        )
        for d in docs
    ]

    return ExamResponse(exam_content=exam_content, sources=sources)




@app.post("/case_consultation", response_model=CaseConsultationResponse)
def create_case_consultation(req: CaseConsultationRequest):
    if not req.case_title.strip():
        raise HTTPException(status_code=400, detail="个案标题不能为空")
    if not req.main_issues.strip():
        raise HTTPException(status_code=400, detail="核心问题不能为空")

    consultation_report, docs = generate_case_consultation(
        case_title=req.case_title,
        student_info=req.student_info,
        main_issues=req.main_issues,
        background=req.background,
        current_measures=req.current_measures,
        consultation_focus=req.consultation_focus,
        urgency=req.urgency,
        extra_requirements=req.extra_requirements,
        category=req.category,
        model_choice=req.model_choice,
    )

    add_history("case_consultation", req.case_title, consultation_report)

    sources = [
        SourceItem(
            content_preview=d.page_content[:220].replace("\n", " "),
            metadata=d.metadata,
        )
        for d in docs
    ]

    return CaseConsultationResponse(
        consultation_report=consultation_report,
        sources=sources,
    )


@app.post("/simulation/start", response_model=SimulationStartResponse)
def simulation_start(req: SimulationStartRequest):
    if not req.student_type.strip():
        raise HTTPException(status_code=400, detail="学生类型不能为空")
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="谈话主题不能为空")

    student_profile, opening_message, dialogue_history, docs = start_simulation(
        student_type=req.student_type,
        topic=req.topic,
        difficulty=req.difficulty,
        student_profile=req.student_profile,
        category=req.category,
        model_choice=req.model_choice,
    )

    sources = [
        SourceItem(
            content_preview=d.page_content[:220].replace("\n", " "),
            metadata=d.metadata,
        )
        for d in docs
    ]

    return SimulationStartResponse(
        student_type=req.student_type,
        topic=req.topic,
        difficulty=req.difficulty,
        student_profile=student_profile,
        opening_message=opening_message,
        dialogue_history=dialogue_history,
        sources=sources,
    )


@app.post("/simulation/reply", response_model=SimulationReplyResponse)
def simulation_reply(req: SimulationReplyRequest):
    if not req.counselor_message.strip():
        raise HTTPException(status_code=400, detail="辅导员输入不能为空")

    student_reply, dialogue_history, docs = simulate_student_reply(
        student_type=req.student_type,
        topic=req.topic,
        difficulty=req.difficulty,
        student_profile=req.student_profile,
        dialogue_history=req.dialogue_history,
        counselor_message=req.counselor_message,
        category=req.category,
        model_choice=req.model_choice,
    )

    sources = [
        SourceItem(
            content_preview=d.page_content[:220].replace("\n", " "),
            metadata=d.metadata,
        )
        for d in docs
    ]

    return SimulationReplyResponse(
        student_reply=student_reply,
        dialogue_history=dialogue_history,
        sources=sources,
    )


@app.post("/simulation/evaluate", response_model=SimulationEvaluateResponse)
def simulation_evaluate(req: SimulationEvaluateRequest):
    if not req.dialogue_history:
        raise HTTPException(status_code=400, detail="对话记录为空，无法评分")

    evaluation, docs = evaluate_simulation(
        student_type=req.student_type,
        topic=req.topic,
        difficulty=req.difficulty,
        student_profile=req.student_profile,
        dialogue_history=req.dialogue_history,
        category=req.category,
        model_choice=req.model_choice,
    )

    add_history("simulation", req.topic, evaluation)

    sources = [
        SourceItem(
            content_preview=d.page_content[:220].replace("\n", " "),
            metadata=d.metadata,
        )
        for d in docs
    ]

    return SimulationEvaluateResponse(evaluation=evaluation, sources=sources)



@app.post("/generate_notice_stream")
def create_notice_stream(req: NoticeRequest):
    if not req.topic.strip() or not req.audience.strip():
        raise HTTPException(status_code=400, detail="主题和通知对象不能为空")
    stream_iter, docs = stream_generate_notice(
        topic=req.topic,
        audience=req.audience,
        style=req.style,
        length=req.length,
        must_include=req.must_include,
        category=req.category,
        model_choice=req.model_choice,
    )
    return _streaming_response(stream_iter, docs, "notice", req.topic)


@app.post("/generate_paper_stream")
def create_paper_stream(req: PaperRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="论文主题不能为空")
    stream_iter, docs = stream_generate_paper(
        topic=req.topic,
        paper_type=req.paper_type,
        section=req.section,
        style=req.style,
        length=req.length,
        must_include=req.must_include,
        constraints=req.constraints,
        category=req.category,
        model_choice=req.model_choice,
    )
    return _streaming_response(stream_iter, docs, "paper", req.topic)


@app.post("/generate_talk_record_stream")
def create_talk_record_stream(req: TalkRecordRequest):
    if not req.student_name.strip():
        raise HTTPException(status_code=400, detail="学生姓名不能为空")
    if not req.background.strip():
        raise HTTPException(status_code=400, detail="谈话背景不能为空")
    if not req.main_issue.strip():
        raise HTTPException(status_code=400, detail="主要问题不能为空")
    stream_iter, docs = stream_generate_talk_record(
        student_name=req.student_name,
        student_id=req.student_id,
        class_name=req.class_name,
        talk_type=req.talk_type,
        background=req.background,
        main_issue=req.main_issue,
        student_statement=req.student_statement,
        counselor_guidance=req.counselor_guidance,
        follow_up_action=req.follow_up_action,
        category=req.category,
        model_choice=req.model_choice,
    )
    return _streaming_response(stream_iter, docs, "talk_record", req.student_name)


@app.post("/generate_student_profile_stream")
def create_student_profile_stream(req: StudentProfileRequest):
    if not req.student_name.strip():
        raise HTTPException(status_code=400, detail="学生姓名不能为空")
    if not req.known_info.strip():
        raise HTTPException(status_code=400, detail="已知信息不能为空")
    stream_iter, docs = stream_generate_student_profile(
        student_name=req.student_name,
        class_name=req.class_name,
        profile_mode=req.profile_mode,
        known_info=req.known_info,
        additional_notes=req.additional_notes,
        category=req.category,
        model_choice=req.model_choice,
    )
    return _streaming_response(stream_iter, docs, "student_profile", req.student_name)


@app.post("/generate_exam_questions_stream")
def create_exam_questions_stream(req: ExamRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="出题主题不能为空")
    stream_iter, docs = stream_generate_exam_questions(
        topic=req.topic,
        question_type=req.question_type,
        count=req.count,
        difficulty=req.difficulty,
        include_answer=req.include_answer,
        include_explanation=req.include_explanation,
        extra_requirements=req.extra_requirements,
        category=req.category,
        model_choice=req.model_choice,
    )
    return _streaming_response(stream_iter, docs, "exam", req.topic)


@app.post("/simulation/start_stream")
def simulation_start_stream(req: SimulationStartRequest):
    if not req.student_type.strip():
        raise HTTPException(status_code=400, detail="学生类型不能为空")
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="谈话主题不能为空")
    stream_iter, docs, student_profile = stream_start_simulation(
        student_type=req.student_type,
        topic=req.topic,
        difficulty=req.difficulty,
        student_profile=req.student_profile,
        category=req.category,
        model_choice=req.model_choice,
    )

    source_names = _source_names_from_docs(docs)

    def event_generator():
        full_text_parts = []
        try:
            for chunk in stream_iter:
                text = _event_text_from_chunk(chunk)
                if text:
                    full_text_parts.append(text)
                    yield f"data: {json.dumps({'type': 'token', 'content': text}, ensure_ascii=True)}\n\n"
            opening = "".join(full_text_parts)
            dialogue_history = [{"role": "student", "content": opening}]
            yield f"data: {json.dumps({'type': 'sources', 'sources': source_names}, ensure_ascii=True)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'student_profile': student_profile, 'dialogue_history': dialogue_history}, ensure_ascii=True)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=True)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@app.post("/simulation/reply_stream")
def simulation_reply_stream(req: SimulationReplyRequest):
    if not req.counselor_message.strip():
        raise HTTPException(status_code=400, detail="辅导员输入不能为空")
    stream_iter, docs, history_with_counselor = stream_simulate_student_reply(
        student_type=req.student_type,
        topic=req.topic,
        difficulty=req.difficulty,
        student_profile=req.student_profile,
        dialogue_history=req.dialogue_history,
        counselor_message=req.counselor_message,
        category=req.category,
        model_choice=req.model_choice,
    )
    source_names = _source_names_from_docs(docs)

    def event_generator():
        full_text_parts = []
        try:
            for chunk in stream_iter:
                text = _event_text_from_chunk(chunk)
                if text:
                    full_text_parts.append(text)
                    yield f"data: {json.dumps({'type': 'token', 'content': text}, ensure_ascii=True)}\n\n"
            student_reply = "".join(full_text_parts)
            dialogue_history = list(history_with_counselor)
            dialogue_history.append({"role": "student", "content": student_reply})
            yield f"data: {json.dumps({'type': 'sources', 'sources': source_names}, ensure_ascii=True)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'dialogue_history': dialogue_history}, ensure_ascii=True)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=True)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@app.post("/simulation/evaluate_stream")
def simulation_evaluate_stream(req: SimulationEvaluateRequest):
    if not req.dialogue_history:
        raise HTTPException(status_code=400, detail="对话记录为空，无法评分")
    stream_iter, docs = stream_evaluate_simulation(
        student_type=req.student_type,
        topic=req.topic,
        difficulty=req.difficulty,
        student_profile=req.student_profile,
        dialogue_history=req.dialogue_history,
        category=req.category,
        model_choice=req.model_choice,
    )
    return _streaming_response(stream_iter, docs, "simulation", req.topic)









@app.post("/consensus_case_graph_stream")
def create_consensus_case_graph_stream(req: CaseConsultationRequest):
    if not req.case_title.strip():
        raise HTTPException(status_code=400, detail="个案标题不能为空")
    if not req.main_issues.strip():
        raise HTTPException(status_code=400, detail="核心问题不能为空")

    initial_state = {
        "case_title": req.case_title,
        "student_info": req.student_info,
        "main_issues": req.main_issues,
        "background": req.background,
        "current_measures": req.current_measures,
        "consultation_focus": req.consultation_focus,
        "urgency": req.urgency,
        "extra_requirements": req.extra_requirements,
        "category": req.category,
        "model_choice": req.model_choice,
    }

    def event_generator():
        full_text_parts = []
        try:
            for payload in stream_consensus_case_events(initial_state):
                if payload.get("type") == "token":
                    full_text_parts.append(payload.get("content", ""))
                yield f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"

            final_text = "".join(full_text_parts)
            if final_text:
                add_history("consensus_case", req.case_title, final_text)

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=True)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/case_sandbox_graph_stream")
def create_case_sandbox_graph_stream(req: CaseSandboxRequest):
    if not req.case_title.strip():
        raise HTTPException(status_code=400, detail="个案标题不能为空")
    if not req.main_issues.strip():
        raise HTTPException(status_code=400, detail="核心问题不能为空")

    initial_state = {
        "case_title": req.case_title,
        "student_info": req.student_info,
        "main_issues": req.main_issues,
        "background": req.background,
        "current_state": req.current_state,
        "current_measures": req.current_measures,
        "scenario_a": req.scenario_a,
        "scenario_b": req.scenario_b,
        "scenario_c": req.scenario_c,
        "period": req.period,
        "focus": req.focus,
        "category": req.category,
        "model_choice": req.model_choice,
    }

    def event_generator():
        full_text_parts = []
        try:
            for payload in stream_case_sandbox_events(initial_state):
                if payload.get("type") == "token":
                    full_text_parts.append(payload.get("content", ""))
                yield f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"

            final_text = "".join(full_text_parts)
            if final_text:
                add_history("case_sandbox", req.case_title, final_text)

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=True)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/case_consultation_graph_stream")
def create_case_consultation_graph_stream(req: CaseConsultationRequest):
    if not req.case_title.strip():
        raise HTTPException(status_code=400, detail="个案标题不能为空")
    if not req.main_issues.strip():
        raise HTTPException(status_code=400, detail="核心问题不能为空")

    initial_state = {
        "case_title": req.case_title,
        "student_info": req.student_info,
        "main_issues": req.main_issues,
        "background": req.background,
        "current_measures": req.current_measures,
        "consultation_focus": req.consultation_focus,
        "urgency": req.urgency,
        "extra_requirements": req.extra_requirements,
        "category": req.category,
        "model_choice": req.model_choice,
    }

    def event_generator():
        full_text_parts = []
        try:
            for payload in stream_case_graph_events(initial_state):
                if payload.get("type") == "token":
                    full_text_parts.append(payload.get("content", ""))
                yield f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"

            final_text = "".join(full_text_parts)
            if final_text:
                add_history("case_consultation_graph", req.case_title, final_text)

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=True)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/case_consultation_role_stream")
def create_case_consultation_role_stream(req: CaseRoleConsultationRequest):
    if not req.case_title.strip():
        raise HTTPException(status_code=400, detail="个案标题不能为空")
    if not req.main_issues.strip():
        raise HTTPException(status_code=400, detail="核心问题不能为空")
    if not req.role_name.strip():
        raise HTTPException(status_code=400, detail="会商角色不能为空")

    stream_iter, docs = stream_generate_case_role_consultation(
        case_title=req.case_title,
        student_info=req.student_info,
        main_issues=req.main_issues,
        background=req.background,
        current_measures=req.current_measures,
        consultation_focus=req.consultation_focus,
        urgency=req.urgency,
        extra_requirements=req.extra_requirements,
        role_key=req.role_key,
        role_name=req.role_name,
        category=req.category,
        model_choice=req.model_choice,
    )

    return _streaming_response(
        stream_iter,
        docs,
        history_kind=None,
        history_title=None,
        done_extra={"role_key": req.role_key, "role_name": req.role_name},
    )


@app.post("/case_consultation_final_stream")
def create_case_consultation_final_stream(req: CaseFinalConsultationRequest):
    if not req.case_title.strip():
        raise HTTPException(status_code=400, detail="个案标题不能为空")
    if not req.main_issues.strip():
        raise HTTPException(status_code=400, detail="核心问题不能为空")
    if not req.role_opinions:
        raise HTTPException(status_code=400, detail="角色会商意见为空，无法汇总")

    stream_iter, docs = stream_generate_case_final_consultation(
        case_title=req.case_title,
        student_info=req.student_info,
        main_issues=req.main_issues,
        background=req.background,
        current_measures=req.current_measures,
        consultation_focus=req.consultation_focus,
        urgency=req.urgency,
        extra_requirements=req.extra_requirements,
        role_opinions=req.role_opinions,
        category=req.category,
        model_choice=req.model_choice,
    )

    return _streaming_response(
        stream_iter,
        docs,
        history_kind="case_consultation",
        history_title=req.case_title,
        done_extra={"stage": "final"},
    )


@app.post("/case_consultation_stream")
def create_case_consultation_stream(req: CaseConsultationRequest):
    if not req.case_title.strip():
        raise HTTPException(status_code=400, detail="个案标题不能为空")
    if not req.main_issues.strip():
        raise HTTPException(status_code=400, detail="核心问题不能为空")
    stream_iter, docs = stream_generate_case_consultation(
        case_title=req.case_title,
        student_info=req.student_info,
        main_issues=req.main_issues,
        background=req.background,
        current_measures=req.current_measures,
        consultation_focus=req.consultation_focus,
        urgency=req.urgency,
        extra_requirements=req.extra_requirements,
        category=req.category,
        model_choice=req.model_choice,
    )
    return _streaming_response(stream_iter, docs, "case_consultation", req.case_title)


@app.post("/analyze_excel", response_model=ExcelAnalysisResponse)
async def analyze_excel_api(
    file: UploadFile = File(...),
    question: str = Form(...),
    sheet_name: Optional[str] = Form(None),
    columns: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = get_ext(file.filename)
    if ext not in ALLOWED_ANALYSIS_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的数据文件类型: {ext}")

    contents = await file.read()
    column_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None

    answer, preview_markdown, sheet_names, selected_sheet, analyzed_columns = analyze_spreadsheet(
        contents,
        file.filename,
        question,
        sheet_name=sheet_name,
        columns=column_list,
    )

    add_history("qa", f"Excel分析: {question}", answer)

    return ExcelAnalysisResponse(
        answer=answer,
        preview_markdown=preview_markdown,
        sheet_names=sheet_names,
        selected_sheet=selected_sheet,
        analyzed_columns=analyzed_columns,
    )


@app.post("/analyze_grade_warning")
async def analyze_grade_warning_api(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = get_ext(file.filename)
    if ext not in ALLOWED_ANALYSIS_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的数据文件类型: {ext}")

    file_bytes = await file.read()

    result = analyze_grade_warning(
        file_bytes=file_bytes,
        filename=file.filename,
        sheet_name=sheet_name,
    )

    add_history("qa", f"成绩预警分析: {file.filename}", result.get("summary_markdown", ""))
    return result


@app.post("/analyze_image")
async def analyze_image(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="图片文件名不能为空")

    ext = get_ext(file.filename)
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail=f"不支持的图片类型: {ext}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="图片内容为空")

    answer = analyze_image_with_ollama(image_bytes, question)
    add_history("qa", f"图片分析: {question[:80]}", answer)

    return {
        "answer": answer,
        "filename": file.filename,
    }


@app.post("/export/notice/markdown")
def export_notice_markdown(title: str = Form(...), body: str = Form(...)):
    return Response(
        content=markdown_bytes(title, body),
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="notice.md"'},
    )


@app.post("/export/notice/docx")
def export_notice_docx(title: str = Form(...), body: str = Form(...)):
    return Response(
        content=docx_bytes(title, body),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="notice.docx"'},
    )


@app.post("/export/paper/markdown")
def export_paper_markdown(title: str = Form(...), body: str = Form(...)):
    return Response(
        content=markdown_bytes(title, body),
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="paper.md"'},
    )


@app.post("/export/paper/docx")
def export_paper_docx(title: str = Form(...), body: str = Form(...)):
    return Response(
        content=docx_bytes(title, body),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="paper.docx"'},
    )


@app.post("/export/excel/markdown")
def export_excel_markdown(title: str = Form(...), body: str = Form(...)):
    return Response(
        content=markdown_bytes(title, body),
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="analysis.md"'},
    )


@app.post("/export/excel/report")
def export_excel_report(
    answer: str = Form(...),
    preview_markdown: str = Form(...),
    selected_sheet: str = Form(...),
    analyzed_columns: str = Form(...),
):
    cols = [c.strip() for c in analyzed_columns.split(",") if c.strip()]

    return Response(
        content=excel_report_bytes(answer, preview_markdown, selected_sheet, cols),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="analysis_report.xlsx"'},
    )


@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {str(exc)}"},
    )
