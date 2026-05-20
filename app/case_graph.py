from __future__ import annotations

from typing import Any, Dict, Generator, Iterable, List, Optional, TypedDict

from langchain_ollama import ChatOllama

from app.config import OLLAMA_BASE_URL, QA_MODEL, PAPER_MODEL
from app.prompts import (
    CASE_FINAL_CONSULTATION_SYSTEM_PROMPT,
    CASE_FINAL_CONSULTATION_USER_PROMPT_TEMPLATE,
)
from app.retrieval import hybrid_retrieve

try:
    from langgraph.graph import END, StateGraph
except Exception:
    END = None
    StateGraph = None


ROLE_SEQUENCE = [
    ("counselor", "辅导员视角"),
    ("head_teacher", "班主任视角"),
    ("psychology", "心理中心视角"),
    ("financial_aid", "资助专员视角"),
    ("career", "就业指导老师视角"),
    ("deputy_secretary", "学院副书记视角"),
    ("policy", "政策合规视角"),
]


ROLE_SHORT_SYSTEM_PROMPT = """
你是高校学生工作“多角色协同个案会商智能体”中的一个指定角色。

你只需要站在当前指定角色角度，围绕复杂学生个案给出简洁、专业、可执行的会商短意见。

重要要求：
1. 只输出当前角色的意见，不要替其他角色发言。
2. 角色意见必须短而准，不要写成长篇报告。
3. 不得替代专业心理诊断、医疗诊断、法律结论或学校正式处分决定。
4. 涉及心理危机、自伤风险、严重违纪、失联、突发事件时，要优先提示启动学校既有工作机制。
5. 每个小节必须完整，不要留下半句话。
6. 输出总长度建议控制在 500-900 字之间。
"""


ROLE_SHORT_USER_PROMPT_TEMPLATE = """
请以“{role_name}”身份参与以下学生个案会商，输出简洁会商短意见。

个案标题：{case_title}
学生基本信息：{student_info}
核心问题：{main_issues}
已知背景：{background}
当前已采取措施：{current_measures}
需要重点研判的问题：{consultation_focus}
紧急程度：{urgency}
补充要求：{extra_requirements}

参考资料：
{context}

请严格按以下结构输出：

## {role_name}会商短意见

### 1. 核心判断
用 2-3 句话说明该角色的判断，不要展开成报告。

### 2. 关键建议
列出 3 条以内最重要、最可执行的建议。

### 3. 风险提醒
列出 2 条以内必须注意的风险。

注意：
- 不要输出最终综合报告。
- 不要重复其他角色职责。
- 不要写空泛套话。
- 必须完整收尾。
"""


FINAL_LONG_SYSTEM_PROMPT = """
你是高校学生工作复杂个案会商的最终汇总专家。

你需要基于多个角色已经形成的“短意见”，生成一份完整、稳妥、可执行、可归档的个案会商最终报告。

要求：
1. 必须综合不同角色意见，不要简单重复。
2. 输出要适合辅导员、学院副书记、班主任、相关职能部门共同参考。
3. 坚持安全、合规、审慎原则。
4. 不得替代专业心理诊断、医疗诊断、法律结论或学校正式处分决定。
5. 涉及高风险情况，要明确上报、联动、记录、跟进建议。
6. 语言正式、清晰、可归档。
7. 最终报告要比角色短意见更完整、更有执行性。
"""


FINAL_LONG_USER_PROMPT_TEMPLATE = """
请基于以下个案信息和各角色会商短意见，生成最终汇总报告。

个案标题：{case_title}
学生基本信息：{student_info}
核心问题：{main_issues}
已知背景：{background}
当前已采取措施：{current_measures}
需要重点研判的问题：{consultation_focus}
紧急程度：{urgency}
补充要求：{extra_requirements}

各角色会商短意见：
{role_opinions}

请严格按照以下结构输出：

# 多角色协同个案会商最终报告

## 一、个案基本判断
概括该个案的主要矛盾、风险来源、工作难点。

## 二、不同角色会商意见摘要
分别概括辅导员、班主任、心理中心、资助专员、就业指导老师、学院副书记、政策合规视角的核心意见。

## 三、综合风险等级
给出综合风险等级：低 / 中 / 较高 / 高。
说明判定依据。

## 四、处置优先级
按“立即处理、短期处理、持续跟进”分层列出。

## 五、72小时内行动建议
列出具体动作、责任人、完成时限、注意事项。

## 六、两周跟进计划
按时间节点列出跟进安排。

## 七、是否建议上报学院
明确建议：建议上报 / 暂不建议上报但需备案 / 需立即上报。
说明理由。

## 八、谈话提纲
给出辅导员下一次谈话可直接使用的提纲。

## 九、家校沟通话术
给出谨慎、温和、合规的家校沟通话术。
注意保护学生隐私，避免扩大化。

## 十、记录归档模板
生成一份可复制使用的个案跟进记录模板。
"""


ROLE_NUM_PREDICT = 1800
FINAL_NUM_PREDICT = 5200
NUM_CTX = 12288


class CaseGraphState(TypedDict, total=False):
    case_title: str
    student_info: Optional[str]
    main_issues: str
    background: Optional[str]
    current_measures: Optional[str]
    consultation_focus: Optional[str]
    urgency: str
    extra_requirements: Optional[str]
    category: Optional[str]
    model_choice: Optional[str]
    context: str
    sources: List[str]
    role_opinions: List[Dict[str, str]]
    final_report: str
    risk_flags: Dict[str, bool]


def _resolve_model(choice: Optional[str], default_model: str) -> str:
    if not choice or choice == "auto":
        return default_model
    return choice


def _llm(model_name: str, *, num_predict: int = 1400, num_ctx: int = NUM_CTX) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=num_predict,
        num_ctx=num_ctx,
    )


def _chunk_text(chunk) -> str:
    if hasattr(chunk, "content") and chunk.content:
        return str(chunk.content)
    if isinstance(chunk, str):
        return chunk
    return ""


def _source_names_from_docs(docs) -> List[str]:
    names: List[str] = []
    seen = set()
    for d in docs or []:
        name = (d.metadata or {}).get("source_file", "未知文件")
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _build_context_from_docs(docs) -> str:
    if not docs:
        return "未检索到可用知识库资料。"

    blocks = []
    for i, doc in enumerate(docs[:4], start=1):
        meta = doc.metadata or {}
        blocks.append(
            f"[资料{i}]\n"
            f"来源文件: {meta.get('source_file', '未知文件')}\n"
            f"类别: {meta.get('category', 'general')}\n\n"
            f"内容:\n{doc.page_content}"
        )
    return "\n\n".join(blocks)[:5000]


def _state_payload(state: CaseGraphState) -> Dict[str, Any]:
    return {
        "case_title": state.get("case_title", ""),
        "student_info": state.get("student_info") or "未提供",
        "main_issues": state.get("main_issues", ""),
        "background": state.get("background") or "未提供",
        "current_measures": state.get("current_measures") or "未提供",
        "consultation_focus": state.get("consultation_focus") or "未提供",
        "urgency": state.get("urgency", "medium"),
        "extra_requirements": state.get("extra_requirements") or "无",
    }


def retrieve_context_node(state: CaseGraphState) -> CaseGraphState:
    query = " ".join([
        state.get("case_title", ""),
        state.get("main_issues", ""),
        state.get("student_info") or "",
        state.get("background") or "",
        state.get("consultation_focus") or "",
        "个案 会商 辅导员 心理 资助 就业 合规 学院",
    ]).strip()

    docs = hybrid_retrieve(query, state.get("category"))
    state["context"] = _build_context_from_docs(docs)
    state["sources"] = _source_names_from_docs(docs)
    state["role_opinions"] = []
    return state


def risk_scan_node(state: CaseGraphState) -> CaseGraphState:
    text = " ".join([
        state.get("main_issues", ""),
        state.get("background") or "",
        state.get("consultation_focus") or "",
        state.get("current_measures") or "",
    ])

    state["risk_flags"] = {
        "psychological": any(k in text for k in ["情绪低落", "抑郁", "焦虑", "自伤", "自杀", "轻生", "失眠", "危机"]),
        "financial": any(k in text for k in ["经济困难", "贫困", "资助", "欠费", "家庭困难"]),
        "academic": any(k in text for k in ["挂科", "不及格", "学业预警", "退学", "留级", "旷考"]),
        "discipline": any(k in text for k in ["违纪", "处分", "旷课", "打架", "饮酒", "违规"]),
        "employment": any(k in text for k in ["就业", "毕业", "简历", "面试", "考研", "升学", "逃避"]),
    }
    return state


def _role_prompt(state: CaseGraphState, role_name: str) -> str:
    payload = _state_payload(state)
    return (
        f"{ROLE_SHORT_SYSTEM_PROMPT}\n\n"
        f"{ROLE_SHORT_USER_PROMPT_TEMPLATE.format(
            role_name=role_name,
            case_title=payload['case_title'],
            student_info=payload['student_info'],
            main_issues=payload['main_issues'],
            background=payload['background'],
            current_measures=payload['current_measures'],
            consultation_focus=payload['consultation_focus'],
            urgency=payload['urgency'],
            extra_requirements=payload['extra_requirements'],
            context=state.get('context', '未检索到资料')
        )}"
    )


def _final_prompt(state: CaseGraphState) -> str:
    payload = _state_payload(state)

    role_blocks = []
    for item in state.get("role_opinions", []):
        role_blocks.append(f"## {item.get('role_name', '未知角色')}\n{item.get('content', '')}")
    role_opinions_text = "\n\n".join(role_blocks)[:10000]

    return (
        f"{FINAL_LONG_SYSTEM_PROMPT}\n\n"
        f"{FINAL_LONG_USER_PROMPT_TEMPLATE.format(
            case_title=payload['case_title'],
            student_info=payload['student_info'],
            main_issues=payload['main_issues'],
            background=payload['background'],
            current_measures=payload['current_measures'],
            consultation_focus=payload['consultation_focus'],
            urgency=payload['urgency'],
            extra_requirements=payload['extra_requirements'],
            role_opinions=role_opinions_text or '未形成有效角色意见'
        )}"
    )


def _stream_llm(prompt: str, model_name: str, *, num_predict: int, num_ctx: int = NUM_CTX) -> Iterable[str]:
    llm = _llm(model_name, num_predict=num_predict, num_ctx=num_ctx)
    for chunk in llm.stream(prompt):
        text = _chunk_text(chunk)
        if text:
            yield text


def _generate_role_opinion(state: CaseGraphState, role_key: str, role_name: str) -> str:
    """
    非流式节点版本，保留给 build_case_consultation_graph().invoke() 使用。
    前端真实流式使用 stream_case_graph_events()。
    """
    model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
    prompt = _role_prompt(state, role_name)
    return "".join(_stream_llm(prompt, model_name, num_predict=ROLE_NUM_PREDICT)).strip()


def _role_node(role_key: str, role_name: str):
    def node(state: CaseGraphState) -> CaseGraphState:
        content = _generate_role_opinion(state, role_key, role_name)
        state.setdefault("role_opinions", [])
        state["role_opinions"].append(
            {"role_key": role_key, "role_name": role_name, "content": content}
        )
        return state

    return node


def final_summary_node(state: CaseGraphState) -> CaseGraphState:
    model_name = _resolve_model(state.get("model_choice"), PAPER_MODEL)
    prompt = _final_prompt(state)
    state["final_report"] = "".join(
        _stream_llm(prompt, model_name, num_predict=FINAL_NUM_PREDICT)
    ).strip()
    return state


def build_case_consultation_graph():
    """
    标准 LangGraph 图：用于非流式 invoke，或者后续扩展 checkpoint / 条件分支。

    当前前端使用 stream_case_graph_events()，它按同样节点顺序执行，
    但每个 LLM 节点内部使用 llm.stream() 做真正 token 流式输出。
    """
    if StateGraph is None:
        raise RuntimeError("未安装 langgraph。请先执行：pip install -U langgraph")

    graph = StateGraph(CaseGraphState)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("risk_scan", risk_scan_node)

    for role_key, role_name in ROLE_SEQUENCE:
        graph.add_node(role_key, _role_node(role_key, role_name))

    graph.add_node("final_summary", final_summary_node)

    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "risk_scan")

    previous = "risk_scan"
    for role_key, _ in ROLE_SEQUENCE:
        graph.add_edge(previous, role_key)
        previous = role_key

    graph.add_edge(previous, "final_summary")
    graph.add_edge("final_summary", END)

    return graph.compile()


def run_case_graph(state: CaseGraphState) -> CaseGraphState:
    app = build_case_consultation_graph()
    return app.invoke(state)


def _emit_risk_scan_text(state: CaseGraphState) -> str:
    flags = state.get("risk_flags", {})
    flag_names = {
        "psychological": "心理情绪风险",
        "financial": "经济困难风险",
        "academic": "学业风险",
        "discipline": "纪律行为风险",
        "employment": "就业发展风险",
    }
    flag_lines = [
        f"- {flag_names.get(k, k)}：{'需关注' if v else '暂未明显触发'}"
        for k, v in flags.items()
    ]

    tips = []
    if flags.get("psychological"):
        tips.append("- 心理情绪风险已触发：后续会商需重点关注情绪支持、危机识别与心理中心联动。")
    if flags.get("financial"):
        tips.append("- 经济困难风险已触发：后续会商需关注资助路径、临时困难帮扶与隐私保护。")
    if flags.get("academic"):
        tips.append("- 学业风险已触发：后续会商需关注学业预警、课程帮扶和阶段复盘。")
    if flags.get("discipline"):
        tips.append("- 纪律行为风险已触发：后续会商需关注事实核查、教育引导和程序合规。")
    if flags.get("employment"):
        tips.append("- 就业发展风险已触发：后续会商需关注职业规划、求职行动和阶段目标。")

    return (
        "## 风险初筛\n"
        + "\n".join(flag_lines)
        + "\n\n"
        + ("### 初筛提示\n" + "\n".join(tips) + "\n\n" if tips else "")
    )


def stream_case_graph_events(initial_state: CaseGraphState) -> Generator[Dict[str, Any], None, None]:
    """
    稳定版 LangGraph 会商真流式输出：

    1. 检索资料；
    2. 风险初筛；
    3. 每个角色只生成“短意见”，避免半截；
    4. 最终报告再完整展开；
    5. 每个角色和最终报告都使用 llm.stream() 逐 token 输出。
    """
    state: CaseGraphState = dict(initial_state)

    yield {"type": "stage", "stage": "检索资料"}
    state = retrieve_context_node(state)
    yield {"type": "sources", "sources": state.get("sources", [])}

    yield {"type": "stage", "stage": "风险初筛"}
    state = risk_scan_node(state)
    yield {"type": "token", "content": _emit_risk_scan_text(state)}

    for role_key, role_name in ROLE_SEQUENCE:
        yield {"type": "stage", "stage": role_name}

        heading = f"\n\n---\n\n# {role_name}\n\n"
        yield {"type": "token", "content": heading}

        role_text_parts: List[str] = []
        model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
        prompt = _role_prompt(state, role_name)

        for token in _stream_llm(prompt, model_name, num_predict=ROLE_NUM_PREDICT):
            role_text_parts.append(token)
            yield {"type": "token", "content": token}

        role_text = "".join(role_text_parts).strip()
        state.setdefault("role_opinions", [])
        state["role_opinions"].append(
            {"role_key": role_key, "role_name": role_name, "content": role_text}
        )

    yield {"type": "stage", "stage": "最终汇总报告"}
    final_heading = "\n\n---\n\n# 最终汇总报告\n\n"
    yield {"type": "token", "content": final_heading}

    final_parts: List[str] = []
    model_name = _resolve_model(state.get("model_choice"), PAPER_MODEL)
    final_prompt = _final_prompt(state)

    for token in _stream_llm(final_prompt, model_name, num_predict=FINAL_NUM_PREDICT):
        final_parts.append(token)
        yield {"type": "token", "content": token}

    state["final_report"] = "".join(final_parts).strip()
    yield {"type": "done", "stage": "final"}
