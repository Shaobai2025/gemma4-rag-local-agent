from typing import List, Optional

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from app.config import OLLAMA_BASE_URL, QA_MODEL, NOTICE_MODEL, PAPER_MODEL
from app.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    NOTICE_SYSTEM_PROMPT,
    NOTICE_USER_PROMPT_TEMPLATE,
    PAPER_SYSTEM_PROMPT,
    PAPER_USER_PROMPT_TEMPLATE,
    TALK_RECORD_SYSTEM_PROMPT,
    TALK_RECORD_USER_PROMPT_TEMPLATE,
    STUDENT_PROFILE_SYSTEM_PROMPT,
    STUDENT_PROFILE_USER_PROMPT_TEMPLATE,
    EXAM_SYSTEM_PROMPT,
    EXAM_USER_PROMPT_TEMPLATE,
    SIMULATION_STUDENT_SYSTEM_PROMPT,
    SIMULATION_STUDENT_USER_PROMPT_TEMPLATE,
    SIMULATION_EVALUATION_SYSTEM_PROMPT,
    SIMULATION_EVALUATION_USER_PROMPT_TEMPLATE,
    CASE_CONSULTATION_SYSTEM_PROMPT,
    CASE_CONSULTATION_USER_PROMPT_TEMPLATE,
    CASE_ROLE_CONSULTATION_SYSTEM_PROMPT,
    CASE_ROLE_CONSULTATION_USER_PROMPT_TEMPLATE,
    CASE_FINAL_CONSULTATION_SYSTEM_PROMPT,
    CASE_FINAL_CONSULTATION_USER_PROMPT_TEMPLATE,
)
from app.retrieval import hybrid_retrieve


ALLOWED_MODELS = {"auto", "gemma4:e4b", "gemma4:26b"}


def resolve_model(choice: Optional[str], default_model: str) -> str:
    if not choice or choice not in ALLOWED_MODELS or choice == "auto":
        return default_model
    return choice


def build_context(docs: List[Document]) -> str:
    if not docs:
        return "未检索到可用知识库资料，请基于一般知识谨慎回答，并明确说明未命中知识库。"

    blocks: List[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        block = (
            f"[资料{i}]\n"
            f"来源文件: {meta.get('source_file', '未知文件')}\n"
            f"类别: {meta.get('category', 'general')}\n"
            f"页码: {meta.get('page', '未知页')}\n\n"
            f"内容:\n{doc.page_content}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def get_llm(model_name: str) -> ChatOllama:
    return ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, temperature=0.2)


def answer_question(question: str, category: Optional[str] = None, model_choice: Optional[str] = "auto"):
    docs = hybrid_retrieve(question, category)
    context = build_context(docs)
    model_name = resolve_model(model_choice, QA_MODEL)
    llm = get_llm(model_name)
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(question=question, context=context)}"
    result = llm.invoke(prompt)
    return result.content, docs


def stream_answer_question(question: str, category: Optional[str] = None, model_choice: Optional[str] = "auto"):
    docs = hybrid_retrieve(question, category)
    context = build_context(docs)
    model_name = resolve_model(model_choice, QA_MODEL)
    llm = get_llm(model_name)
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(question=question, context=context)}"
    return llm.stream(prompt), docs


def generate_notice(topic: str, audience: str, style: str = "formal", length: str = "medium",
                    must_include: str | None = None, category: Optional[str] = None,
                    model_choice: Optional[str] = "auto"):
    query = f"{topic} {audience} {must_include or ''}".strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)
    model_name = resolve_model(model_choice, NOTICE_MODEL)
    llm = get_llm(model_name)
    prompt = (
        f"{NOTICE_SYSTEM_PROMPT}\n\n"
        f"{NOTICE_USER_PROMPT_TEMPLATE.format(topic=topic, audience=audience, style=style, length=length, must_include=must_include or '无', context=context)}"
    )
    result = llm.invoke(prompt)
    return result.content, docs


def generate_paper(topic: str, paper_type: str = "research", section: str = "full_draft",
                   style: str = "academic", length: str = "medium",
                   must_include: str | None = None, constraints: str | None = None,
                   category: Optional[str] = None, model_choice: Optional[str] = "auto"):
    query = f"{topic} {paper_type} {section} {must_include or ''} {constraints or ''}".strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)
    model_name = resolve_model(model_choice, PAPER_MODEL)
    llm = get_llm(model_name)
    prompt = (
        f"{PAPER_SYSTEM_PROMPT}\n\n"
        f"{PAPER_USER_PROMPT_TEMPLATE.format(topic=topic, paper_type=paper_type, section=section, style=style, length=length, must_include=must_include or '无', constraints=constraints or '无', context=context)}"
    )
    result = llm.invoke(prompt)
    return result.content, docs


def generate_talk_record(
    student_name: str,
    student_id: str | None = None,
    class_name: str | None = None,
    talk_type: str = "academic",
    background: str = "",
    main_issue: str = "",
    student_statement: str | None = None,
    counselor_guidance: str | None = None,
    follow_up_action: str | None = None,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    query = f"{student_name} {class_name or ''} {main_issue} {background}".strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)

    model_name = resolve_model(model_choice, NOTICE_MODEL)
    llm = get_llm(model_name)

    prompt = (
        f"{TALK_RECORD_SYSTEM_PROMPT}\n\n"
        f"{TALK_RECORD_USER_PROMPT_TEMPLATE.format(
            student_name=student_name,
            student_id=student_id or '无',
            class_name=class_name or '无',
            talk_type=talk_type,
            background=background,
            main_issue=main_issue,
            student_statement=student_statement or '无',
            counselor_guidance=counselor_guidance or '无',
            follow_up_action=follow_up_action or '无',
            context=context
        )}"
    )

    result = llm.invoke(prompt)
    return result.content, docs


def generate_student_profile(
    student_name: str,
    class_name: str | None = None,
    profile_mode: str = "full",
    known_info: str = "",
    additional_notes: str | None = None,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    query = f"{student_name} {class_name or ''} {known_info}".strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)

    model_name = resolve_model(model_choice, QA_MODEL)
    llm = get_llm(model_name)

    prompt = (
        f"{STUDENT_PROFILE_SYSTEM_PROMPT}\n\n"
        f"{STUDENT_PROFILE_USER_PROMPT_TEMPLATE.format(
            student_name=student_name,
            class_name=class_name or '无',
            profile_mode=profile_mode,
            known_info=known_info,
            additional_notes=additional_notes or '无',
            context=context
        )}"
    )

    result = llm.invoke(prompt)
    return result.content, docs


def generate_exam_questions(
    topic: str,
    question_type: str = "mixed",
    count: int = 5,
    difficulty: str = "medium",
    include_answer: bool = True,
    include_explanation: bool = True,
    extra_requirements: str | None = None,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    query = f"{topic} {question_type} {extra_requirements or ''}".strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)

    model_name = resolve_model(model_choice, PAPER_MODEL)
    llm = get_llm(model_name)

    prompt = (
        f"{EXAM_SYSTEM_PROMPT}\n\n"
        f"{EXAM_USER_PROMPT_TEMPLATE.format(
            topic=topic,
            question_type=question_type,
            count=count,
            difficulty=difficulty,
            include_answer='是' if include_answer else '否',
            include_explanation='是' if include_explanation else '否',
            extra_requirements=extra_requirements or '无',
            context=context
        )}"
    )

    result = llm.invoke(prompt)
    return result.content, docs


STUDENT_TYPE_DESCRIPTIONS = {
    "unwilling": "不愿沟通型：话少、回避，常用“没什么”“不用管”“我自己知道”等回应。",
    "depressed": "情绪低落型：表达疲惫、消极、自责，对未来缺乏信心，需要辅导员敏锐识别情绪风险。",
    "resistant": "逆反抵触型：对提醒和管理要求有抵触，容易反问、否认、质疑。",
    "anxious": "迷茫焦虑型：对学业、就业或未来规划焦虑，想改变但缺乏方向。",
    "defensive": "违纪狡辩型：对违纪行为找理由，转移责任，强调“别人也这样”。",
    "silent_family": "家庭困难沉默型：因家庭经济或家庭关系压力沉默、压抑，不愿主动求助。",
    "employment_avoidant": "就业逃避型：回避就业压力，拖延简历、投递、面试或升学选择。",
}


def _dialogue_to_text(dialogue_history):
    if not dialogue_history:
        return "暂无对话。"
    lines = []
    for item in dialogue_history:
        role = item.get("role", "")
        content = item.get("content", "")
        if role == "student":
            lines.append(f"学生：{content}")
        elif role == "counselor":
            lines.append(f"辅导员：{content}")
        else:
            lines.append(f"{role}：{content}")
    return "\n".join(lines)


def start_simulation(
    student_type: str,
    topic: str,
    difficulty: str = "medium",
    student_profile: str | None = None,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    type_desc = STUDENT_TYPE_DESCRIPTIONS.get(student_type, student_type)
    query = f"{topic} {type_desc} 辅导员 谈话 模拟".strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)

    if not student_profile:
        student_profile = (
            f"该虚拟学生属于“{type_desc}”。谈话主题为“{topic}”。"
            f"训练难度为“{difficulty}”。请结合高校辅导员谈话场景进行模拟。"
        )

    model_name = resolve_model(model_choice, QA_MODEL)
    llm = get_llm(model_name)

    opening_history = f"参考资料：\n{context}\n\n当前为开场，请学生先说一句符合自身状态的开场白。"
    prompt = (
        f"{SIMULATION_STUDENT_SYSTEM_PROMPT}\n\n"
        f"{SIMULATION_STUDENT_USER_PROMPT_TEMPLATE.format(
            student_type=student_type,
            topic=topic,
            difficulty=difficulty,
            student_profile=student_profile,
            student_type_desc=type_desc,
            dialogue_history=opening_history,
            counselor_message='辅导员刚刚请你坐下，准备开始谈话。'
        )}"
    )

    result = llm.invoke(prompt)
    opening = result.content.strip()
    dialogue_history = [{"role": "student", "content": opening}]
    return student_profile, opening, dialogue_history, docs


def simulate_student_reply(
    student_type: str,
    topic: str,
    difficulty: str,
    student_profile: str,
    dialogue_history,
    counselor_message: str,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    type_desc = STUDENT_TYPE_DESCRIPTIONS.get(student_type, student_type)
    query = f"{topic} {counselor_message} {type_desc}".strip()
    docs = hybrid_retrieve(query, category)
    history_with_counselor = list(dialogue_history or [])
    history_with_counselor.append({"role": "counselor", "content": counselor_message})

    model_name = resolve_model(model_choice, QA_MODEL)
    llm = get_llm(model_name)

    prompt = (
        f"{SIMULATION_STUDENT_SYSTEM_PROMPT}\n\n"
        f"{SIMULATION_STUDENT_USER_PROMPT_TEMPLATE.format(
            student_type=student_type,
            topic=topic,
            difficulty=difficulty,
            student_profile=student_profile,
            student_type_desc=type_desc,
            dialogue_history=_dialogue_to_text(history_with_counselor),
            counselor_message=counselor_message
        )}"
    )

    result = llm.invoke(prompt)
    reply = result.content.strip()
    history_with_counselor.append({"role": "student", "content": reply})
    return reply, history_with_counselor, docs


def evaluate_simulation(
    student_type: str,
    topic: str,
    difficulty: str,
    student_profile: str,
    dialogue_history,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    type_desc = STUDENT_TYPE_DESCRIPTIONS.get(student_type, student_type)
    query = f"{topic} {type_desc} 辅导员 谈话 评分 改进建议".strip()
    docs = hybrid_retrieve(query, category)

    model_name = resolve_model(model_choice, PAPER_MODEL)
    llm = get_llm(model_name)

    prompt = (
        f"{SIMULATION_EVALUATION_SYSTEM_PROMPT}\n\n"
        f"{SIMULATION_EVALUATION_USER_PROMPT_TEMPLATE.format(
            student_type=type_desc,
            topic=topic,
            difficulty=difficulty,
            student_profile=student_profile,
            dialogue_history=_dialogue_to_text(dialogue_history or [])
        )}"
    )

    result = llm.invoke(prompt)
    return result.content.strip(), docs


def generate_case_consultation(
    case_title: str,
    main_issues: str,
    student_info: str | None = None,
    background: str | None = None,
    current_measures: str | None = None,
    consultation_focus: str | None = None,
    urgency: str = "medium",
    extra_requirements: str | None = None,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    query = " ".join([
        case_title or "",
        main_issues or "",
        student_info or "",
        background or "",
        consultation_focus or "",
        "个案 会商 辅导员 心理 资助 就业 合规 学院"
    ]).strip()

    docs = hybrid_retrieve(query, category)
    context = build_context(docs)

    model_name = resolve_model(model_choice, PAPER_MODEL)
    llm = get_llm(model_name)

    prompt = (
        f"{CASE_CONSULTATION_SYSTEM_PROMPT}\n\n"
        f"{CASE_CONSULTATION_USER_PROMPT_TEMPLATE.format(
            case_title=case_title,
            student_info=student_info or '未提供',
            main_issues=main_issues,
            background=background or '未提供',
            current_measures=current_measures or '未提供',
            consultation_focus=consultation_focus or '未提供',
            urgency=urgency,
            extra_requirements=extra_requirements or '无',
            context=context
        )}"
    )

    result = llm.invoke(prompt)
    return result.content.strip(), docs


# =========================
# Streaming generation helpers
# =========================

def _stream_from_prompt(prompt: str, model_name: str):
    llm = get_llm(model_name)
    return llm.stream(prompt)


def stream_generate_notice(topic: str, audience: str, style: str = "formal", length: str = "medium",
                           must_include: str | None = None, category: Optional[str] = None,
                           model_choice: Optional[str] = "auto"):
    query = f"{topic} {audience} {must_include or ''}".strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)
    model_name = resolve_model(model_choice, NOTICE_MODEL)
    prompt = (
        f"{NOTICE_SYSTEM_PROMPT}\n\n"
        f"{NOTICE_USER_PROMPT_TEMPLATE.format(topic=topic, audience=audience, style=style, length=length, must_include=must_include or '无', context=context)}"
    )
    return _stream_from_prompt(prompt, model_name), docs


def stream_generate_paper(topic: str, paper_type: str = "research", section: str = "full_draft",
                          style: str = "academic", length: str = "medium",
                          must_include: str | None = None, constraints: str | None = None,
                          category: Optional[str] = None, model_choice: Optional[str] = "auto"):
    query = f"{topic} {paper_type} {section} {must_include or ''} {constraints or ''}".strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)
    model_name = resolve_model(model_choice, PAPER_MODEL)
    prompt = (
        f"{PAPER_SYSTEM_PROMPT}\n\n"
        f"{PAPER_USER_PROMPT_TEMPLATE.format(topic=topic, paper_type=paper_type, section=section, style=style, length=length, must_include=must_include or '无', constraints=constraints or '无', context=context)}"
    )
    return _stream_from_prompt(prompt, model_name), docs


def stream_generate_talk_record(
    student_name: str,
    student_id: str | None = None,
    class_name: str | None = None,
    talk_type: str = "academic",
    background: str = "",
    main_issue: str = "",
    student_statement: str | None = None,
    counselor_guidance: str | None = None,
    follow_up_action: str | None = None,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    query = f"{student_name} {class_name or ''} {main_issue} {background}".strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)
    model_name = resolve_model(model_choice, NOTICE_MODEL)
    prompt = (
        f"{TALK_RECORD_SYSTEM_PROMPT}\n\n"
        f"{TALK_RECORD_USER_PROMPT_TEMPLATE.format(
            student_name=student_name,
            student_id=student_id or '无',
            class_name=class_name or '无',
            talk_type=talk_type,
            background=background,
            main_issue=main_issue,
            student_statement=student_statement or '无',
            counselor_guidance=counselor_guidance or '无',
            follow_up_action=follow_up_action or '无',
            context=context
        )}"
    )
    return _stream_from_prompt(prompt, model_name), docs


def stream_generate_student_profile(
    student_name: str,
    class_name: str | None = None,
    profile_mode: str = "full",
    known_info: str = "",
    additional_notes: str | None = None,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    query = f"{student_name} {class_name or ''} {known_info}".strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)
    model_name = resolve_model(model_choice, QA_MODEL)
    prompt = (
        f"{STUDENT_PROFILE_SYSTEM_PROMPT}\n\n"
        f"{STUDENT_PROFILE_USER_PROMPT_TEMPLATE.format(
            student_name=student_name,
            class_name=class_name or '无',
            profile_mode=profile_mode,
            known_info=known_info,
            additional_notes=additional_notes or '无',
            context=context
        )}"
    )
    return _stream_from_prompt(prompt, model_name), docs


def stream_generate_exam_questions(
    topic: str,
    question_type: str = "mixed",
    count: int = 5,
    difficulty: str = "medium",
    include_answer: bool = True,
    include_explanation: bool = True,
    extra_requirements: str | None = None,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    query = f"{topic} {question_type} {extra_requirements or ''}".strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)
    model_name = resolve_model(model_choice, PAPER_MODEL)
    prompt = (
        f"{EXAM_SYSTEM_PROMPT}\n\n"
        f"{EXAM_USER_PROMPT_TEMPLATE.format(
            topic=topic,
            question_type=question_type,
            count=count,
            difficulty=difficulty,
            include_answer='是' if include_answer else '否',
            include_explanation='是' if include_explanation else '否',
            extra_requirements=extra_requirements or '无',
            context=context
        )}"
    )
    return _stream_from_prompt(prompt, model_name), docs


def stream_start_simulation(
    student_type: str,
    topic: str,
    difficulty: str = "medium",
    student_profile: str | None = None,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    type_desc = STUDENT_TYPE_DESCRIPTIONS.get(student_type, student_type)
    query = f"{topic} {type_desc} 辅导员 谈话 模拟".strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)
    if not student_profile:
        student_profile = (
            f"该虚拟学生属于“{type_desc}”。谈话主题为“{topic}”。"
            f"训练难度为“{difficulty}”。请结合高校辅导员谈话场景进行模拟。"
        )
    model_name = resolve_model(model_choice, QA_MODEL)
    opening_history = f"参考资料：\n{context}\n\n当前为开场，请学生先说一句符合自身状态的开场白。"
    prompt = (
        f"{SIMULATION_STUDENT_SYSTEM_PROMPT}\n\n"
        f"{SIMULATION_STUDENT_USER_PROMPT_TEMPLATE.format(
            student_type=student_type,
            topic=topic,
            difficulty=difficulty,
            student_profile=student_profile,
            student_type_desc=type_desc,
            dialogue_history=opening_history,
            counselor_message='辅导员刚刚请你坐下，准备开始谈话。'
        )}"
    )
    return _stream_from_prompt(prompt, model_name), docs, student_profile


def stream_simulate_student_reply(
    student_type: str,
    topic: str,
    difficulty: str,
    student_profile: str,
    dialogue_history,
    counselor_message: str,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    type_desc = STUDENT_TYPE_DESCRIPTIONS.get(student_type, student_type)
    query = f"{topic} {counselor_message} {type_desc}".strip()
    docs = hybrid_retrieve(query, category)
    history_with_counselor = list(dialogue_history or [])
    history_with_counselor.append({"role": "counselor", "content": counselor_message})
    model_name = resolve_model(model_choice, QA_MODEL)
    prompt = (
        f"{SIMULATION_STUDENT_SYSTEM_PROMPT}\n\n"
        f"{SIMULATION_STUDENT_USER_PROMPT_TEMPLATE.format(
            student_type=student_type,
            topic=topic,
            difficulty=difficulty,
            student_profile=student_profile,
            student_type_desc=type_desc,
            dialogue_history=_dialogue_to_text(history_with_counselor),
            counselor_message=counselor_message
        )}"
    )
    return _stream_from_prompt(prompt, model_name), docs, history_with_counselor


def stream_evaluate_simulation(
    student_type: str,
    topic: str,
    difficulty: str,
    student_profile: str,
    dialogue_history,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    type_desc = STUDENT_TYPE_DESCRIPTIONS.get(student_type, student_type)
    query = f"{topic} {type_desc} 辅导员 谈话 评分 改进建议".strip()
    docs = hybrid_retrieve(query, category)
    model_name = resolve_model(model_choice, PAPER_MODEL)
    prompt = (
        f"{SIMULATION_EVALUATION_SYSTEM_PROMPT}\n\n"
        f"{SIMULATION_EVALUATION_USER_PROMPT_TEMPLATE.format(
            student_type=type_desc,
            topic=topic,
            difficulty=difficulty,
            student_profile=student_profile,
            dialogue_history=_dialogue_to_text(dialogue_history or [])
        )}"
    )
    return _stream_from_prompt(prompt, model_name), docs


def stream_generate_case_consultation(
    case_title: str,
    main_issues: str,
    student_info: str | None = None,
    background: str | None = None,
    current_measures: str | None = None,
    consultation_focus: str | None = None,
    urgency: str = "medium",
    extra_requirements: str | None = None,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    query = " ".join([
        case_title or "",
        main_issues or "",
        student_info or "",
        background or "",
        consultation_focus or "",
        "个案 会商 辅导员 心理 资助 就业 合规 学院"
    ]).strip()
    docs = hybrid_retrieve(query, category)
    context = build_context(docs)
    model_name = resolve_model(model_choice, PAPER_MODEL)
    prompt = (
        f"{CASE_CONSULTATION_SYSTEM_PROMPT}\n\n"
        f"{CASE_CONSULTATION_USER_PROMPT_TEMPLATE.format(
            case_title=case_title,
            student_info=student_info or '未提供',
            main_issues=main_issues,
            background=background or '未提供',
            current_measures=current_measures or '未提供',
            consultation_focus=consultation_focus or '未提供',
            urgency=urgency,
            extra_requirements=extra_requirements or '无',
            context=context
        )}"
    )
    return _stream_from_prompt(prompt, model_name), docs


def _case_context_docs(
    case_title: str,
    main_issues: str,
    student_info: str | None = None,
    background: str | None = None,
    consultation_focus: str | None = None,
    category: Optional[str] = None,
):
    query = " ".join([
        case_title or "",
        main_issues or "",
        student_info or "",
        background or "",
        consultation_focus or "",
        "个案 会商 辅导员 心理 资助 就业 合规 学院"
    ]).strip()
    docs = hybrid_retrieve(query, category)
    # 个案会商只取前几条，避免长上下文拖死 26B/27B
    return docs[:4]


def _case_context_text(docs: List[Document]) -> str:
    context = build_context(docs[:4])
    return context[:5000]


def stream_generate_case_role_consultation(
    case_title: str,
    main_issues: str,
    role_key: str,
    role_name: str,
    student_info: str | None = None,
    background: str | None = None,
    current_measures: str | None = None,
    consultation_focus: str | None = None,
    urgency: str = "medium",
    extra_requirements: str | None = None,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    docs = _case_context_docs(
        case_title=case_title,
        main_issues=main_issues,
        student_info=student_info,
        background=background,
        consultation_focus=consultation_focus,
        category=category,
    )
    context = _case_context_text(docs)

    # 分角色阶段默认用 QA_MODEL，更快更稳；用户明确选择模型时才切换
    model_name = resolve_model(model_choice, QA_MODEL)

    prompt = (
        f"{CASE_ROLE_CONSULTATION_SYSTEM_PROMPT}\n\n"
        f"{CASE_ROLE_CONSULTATION_USER_PROMPT_TEMPLATE.format(
            role_name=role_name,
            case_title=case_title,
            student_info=student_info or '未提供',
            main_issues=main_issues,
            background=background or '未提供',
            current_measures=current_measures or '未提供',
            consultation_focus=consultation_focus or '未提供',
            urgency=urgency,
            extra_requirements=extra_requirements or '无',
            context=context
        )}"
    )
    return _stream_from_prompt(prompt, model_name), docs


def stream_generate_case_final_consultation(
    case_title: str,
    main_issues: str,
    role_opinions,
    student_info: str | None = None,
    background: str | None = None,
    current_measures: str | None = None,
    consultation_focus: str | None = None,
    urgency: str = "medium",
    extra_requirements: str | None = None,
    category: Optional[str] = None,
    model_choice: Optional[str] = "auto",
):
    docs = _case_context_docs(
        case_title=case_title,
        main_issues=main_issues,
        student_info=student_info,
        background=background,
        consultation_focus=consultation_focus,
        category=category,
    )

    role_blocks = []
    for item in role_opinions or []:
        role_name = item.get("role_name", "未知角色")
        content = item.get("content", "")
        if content:
            role_blocks.append(f"## {role_name}\n{content}")
    role_opinions_text = "\n\n".join(role_blocks)[:12000]

    # 最终汇总可用 PAPER_MODEL，但 auto 默认还是 PAPER_MODEL；若卡顿，前端可选 e4B
    model_name = resolve_model(model_choice, PAPER_MODEL)

    prompt = (
        f"{CASE_FINAL_CONSULTATION_SYSTEM_PROMPT}\n\n"
        f"{CASE_FINAL_CONSULTATION_USER_PROMPT_TEMPLATE.format(
            case_title=case_title,
            student_info=student_info or '未提供',
            main_issues=main_issues,
            background=background or '未提供',
            current_measures=current_measures or '未提供',
            consultation_focus=consultation_focus or '未提供',
            urgency=urgency,
            extra_requirements=extra_requirements or '无',
            role_opinions=role_opinions_text or '未形成有效角色意见'
        )}"
    )
    return _stream_from_prompt(prompt, model_name), docs
