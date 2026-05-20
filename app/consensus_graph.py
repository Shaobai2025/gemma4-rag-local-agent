from __future__ import annotations

from typing import Any, Dict, Generator, Iterable, List, Optional, TypedDict

from langchain_ollama import ChatOllama

from app.config import OLLAMA_BASE_URL, QA_MODEL, PAPER_MODEL
from app.retrieval import hybrid_retrieve

try:
    from langgraph.graph import END, StateGraph
except Exception:
    END = None
    StateGraph = None


CONSENSUS_AGENTS = [
    ("counselor", "辅导员 Agent", "关注执行可行性、谈话安排、日常跟进、台账留痕和学生配合度。"),
    ("psychology", "心理 Agent", "关注心理风险、情绪状态、危机识别、转介边界和非诊断性支持。"),
    ("head_teacher", "班主任 Agent", "关注学业行为、课堂表现、课程补救、任课教师反馈和学业预警。"),
    ("financial_aid", "资助 Agent", "关注经济困难核实、资助帮扶、临时困难补助、勤工助学和隐私保护。"),
    ("compliance", "合规 Agent", "关注制度依据、程序正当、上报备案、处分流程和材料留痕。"),
    ("ethics", "伦理 Agent", "关注隐私保护、标签化风险、过度干预、学生尊严和家校沟通边界。"),
    ("deputy_secretary", "学院副书记 Agent", "关注学院层面风险、资源协调、部门联动、上报判断和遗漏盲区。"),
]


class ConsensusState(TypedDict, total=False):
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
    initial_reviews: List[Dict[str, str]]
    cross_questions: str
    risk_rebuttal: str
    plan_revision: str
    consensus: str
    dissent: str
    final_recommendation: str


# 稳定配置：
# 中间节点精炼会商纪要式输出，兼顾完整判断与简洁；
# 最终报告详细展开，尤其“最终可执行方案”。
ROLE_NUM_PREDICT = 1600
STAGE_NUM_PREDICT = 2200
FINAL_NUM_PREDICT = 7500
NUM_CTX = 16384


INITIAL_SYSTEM_PROMPT = """
你是高校复杂学生个案“多智能体协商式会商”中的一个指定角色。

你的任务是进行角色初评：站在本角色职责边界内，对个案提出短而准的初步判断。

要求：
1. 只代表当前角色，不要替其他角色发言。
2. 只保留关键决策信息，不写长篇报告。
3. 必须体现角色专业边界。
4. 不得替代心理诊断、医学诊断、法律结论或学校正式处分决定。
5. 每个小节必须完整收尾，不要留下半句话。
6. 输出总长度建议控制在 400-700 字，必须完整收尾。最后一行必须写：【本角色初评结束】。
"""

INITIAL_USER_PROMPT = """
请以“{agent_name}”身份进行角色初评。

角色职责：{agent_focus}

个案标题：{case_title}
学生基本信息：{student_info}
核心问题：{main_issues}
已知背景：{background}
当前已采取措施：{current_measures}
重点研判问题：{consultation_focus}
紧急程度：{urgency}
补充要求：{extra_requirements}

参考资料：
{context}

请严格按以下结构输出，保持简洁完整：

## {agent_name} 初评

### 1. 关键判断
用 1-2 句话说明本角色最核心的判断。

### 2. 关键风险
最多 2 条，每条不超过 1 句话。

### 3. 必要动作
最多 2 条，必须可执行。
"""


CROSS_SYSTEM_PROMPT = """
你是多智能体协商式个案会商的“交叉质询节点”。

你的任务是输出“精炼会商纪要式”的关键质询，不要写成长篇报告，也不要只写短语。

硬性要求：
1. 每条必须是完整判断，保留必要依据和质询指向。
2. 每条控制在 40-80 字。
3. 关键质询最多 5 条，必须核实信息最多 3 条。
4. 不写背景铺垫，不做最终结论。
5. 最后一行必须写：【交叉质询结束】
"""

CROSS_USER_PROMPT = """
以下是复杂学生个案和各角色初评，请生成精炼交叉质询纪要。

个案标题：{case_title}
核心问题：{main_issues}
已知背景：{background}

角色初评：
{initial_reviews}

请严格按以下结构输出：

# 交叉质询

## 关键质询
1. A Agent 质询 B Agent：……
2. ……

## 必须核实
- ……
- ……
- ……

【交叉质询结束】
"""

REBUTTAL_SYSTEM_PROMPT = """
你是多智能体协商式个案会商的“风险反驳节点”。

你的任务是输出“精炼会商纪要式”的风险反驳，不要写成长篇报告，也不要只写口号。

硬性要求：
1. 每条必须是完整判断，保留必要依据和风险指向。
2. 每条控制在 40-80 字。
3. 不能忽视的风险最多 4 条，禁忌做法最多 3 条。
4. 不写背景铺垫，不做最终结论。
5. 最后一行必须写：【风险反驳结束】
"""

REBUTTAL_USER_PROMPT = """
请基于角色初评和交叉质询，生成精炼风险反驳纪要。

个案标题：{case_title}
核心问题：{main_issues}

角色初评：
{initial_reviews}

交叉质询：
{cross_questions}

请严格按以下结构输出：

# 风险反驳

## 不能忽视的风险
- ……
- ……
- ……

## 禁忌做法
- ……
- ……
- ……

【风险反驳结束】
"""

REVISION_SYSTEM_PROMPT = """
你是多智能体协商式个案会商的“方案修正节点”。

你的任务是输出“精炼会商纪要式”的修正方案，不写最终报告，但要比简单清单更具体。

硬性要求：
1. 每条必须是完整动作或完整原则，保留责任主体、时间节点或处置指向。
2. 每条控制在 40-90 字。
3. 修正原则最多 3 条，关键动作最多 5 条，人工判断问题最多 3 条。
4. 动作必须具体、可执行，不写空泛口号。
5. 最后一行必须写：【方案修正结束】
"""

REVISION_USER_PROMPT = """
请根据交叉质询和风险反驳，生成精炼方案修正纪要。

个案标题：{case_title}
核心问题：{main_issues}
当前已采取措施：{current_measures}

角色初评：
{initial_reviews}

交叉质询：
{cross_questions}

风险反驳：
{risk_rebuttal}

请严格按以下结构输出：

# 方案修正

## 修正原则
- ……
- ……
- ……

## 关键动作
- ……
- ……
- ……

## 人工判断
- ……
- ……

【方案修正结束】
"""

CONSENSUS_SYSTEM_PROMPT = """
你是多智能体协商式个案会商的“共识形成节点”。

你的任务是输出“精炼会商纪要式”的共识意见，既不要长篇展开，也不要只写短语。

硬性要求：
1. 每条必须是完整判断，说明共识内容和执行指向。
2. 每条控制在 40-80 字。
3. 共识意见最多 5 条，立即动作最多 3 条。
4. 不要强行消除真实分歧。
5. 最后一行必须写：【共识形成结束】
"""

CONSENSUS_USER_PROMPT = """
请提炼本次多智能体会商形成的精炼共识纪要。

个案标题：{case_title}
核心问题：{main_issues}

方案修正：
{plan_revision}

请严格按以下结构输出：

# 共识形成

## 共识意见
- ……
- ……
- ……

## 立即动作
- ……
- ……
- ……

【共识形成结束】
"""

DISSENT_SYSTEM_PROMPT = """
你是多智能体协商式个案会商的“保留分歧节点”。

你的任务是输出“精炼会商纪要式”的保留分歧，不要展开成长篇，但要说明分歧的实质。

硬性要求：
1. 每条必须是完整判断，说明分歧点和角色关注差异。
2. 每条控制在 40-90 字。
3. 主要分歧最多 3 条，临时原则最多 3 条。
4. 不要把所有分歧都抹平。
5. 最后一行必须写：【保留分歧结束】
"""

DISSENT_USER_PROMPT = """
请识别并保留本次会商中的主要分歧，输出精炼会商纪要。

个案标题：{case_title}
核心问题：{main_issues}

交叉质询：
{cross_questions}

风险反驳：
{risk_rebuttal}

共识意见：
{consensus}

请严格按以下结构输出：

# 保留分歧

## 主要分歧
- ……
- ……
- ……

## 临时原则
- ……
- ……
- ……

【保留分歧结束】
"""

FINAL_SYSTEM_PROMPT = """
你是高校学生工作“多智能体协商式个案会商”的最终报告生成节点。

你的任务是综合角色初评、交叉质询、风险反驳、方案修正、共识形成和保留分歧，生成一份完整、稳妥、可执行、可归档的最终报告。

核心要求：
1. 最终报告不是简单汇总，而是体现“分歧显性化与共识生成机制”。
2. 必须突出：共识意见、主要分歧、争议风险、需要人工核实的信息、最终可执行方案。
3. “最终可执行方案”必须详细，是本报告重点。
4. 不得替代心理诊断、医学诊断、法律结论或学校正式处分决定。
5. 涉及高风险时，要明确上报、联动、留痕和跟进机制。
6. 语言正式、清晰、可归档。
7. 必须完整收尾。
"""

FINAL_USER_PROMPT = """
请生成最终报告。

个案标题：{case_title}
学生基本信息：{student_info}
核心问题：{main_issues}
已知背景：{background}
当前已采取措施：{current_measures}
重点研判问题：{consultation_focus}
紧急程度：{urgency}
补充要求：{extra_requirements}

角色初评：
{initial_reviews}

交叉质询：
{cross_questions}

风险反驳：
{risk_rebuttal}

方案修正：
{plan_revision}

共识形成：
{consensus}

保留分歧：
{dissent}

请严格按照以下结构输出：

# 多智能体协商式个案会商报告

## 一、个案基本判断
概括主要矛盾、风险来源、工作难点。

## 二、共识意见
列出最终达成的共识。

## 三、主要分歧
列出仍需人工判断或进一步核实的分歧。

## 四、争议风险
列出可能引发争议、误判、越界或执行失败的风险。

## 五、需要人工核实的信息
列出必须由辅导员、班主任、学院或相关部门进一步核实的信息。

## 六、最终可执行方案
这是重点，请详细展开。

### 1. 责任分工
明确辅导员、班主任、心理中心、资助、学院、家校沟通等主体分别做什么。

### 2. 72小时内行动
列出具体动作、责任人、完成标准、注意事项。

### 3. 一周内行动
列出具体动作、责任人、完成标准、注意事项。

### 4. 两周内行动
列出具体动作、责任人、完成标准、注意事项。

### 5. 风险升级条件
明确什么情况下必须上报学院、联动心理中心、联系家长或启动应急机制。

### 6. 留痕归档要求
说明需要形成哪些记录、谁负责、何时完成。

## 七、谈话提纲
给出下一次谈话可直接使用的提纲。

## 八、家校沟通话术
给出谨慎、温和、合规的话术。

## 九、归档模板
生成可复制使用的归档模板。
"""


def _resolve_model(choice: Optional[str], default_model: str) -> str:
    if not choice or choice == "auto":
        return default_model
    return choice


def _llm(model_name: str, *, num_predict: int, num_ctx: int = NUM_CTX) -> ChatOllama:
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


def _stream_llm(prompt: str, model_name: str, *, num_predict: int) -> Iterable[str]:
    llm = _llm(model_name, num_predict=num_predict, num_ctx=NUM_CTX)
    for chunk in llm.stream(prompt):
        text = _chunk_text(chunk)
        if text:
            yield text


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


def _payload(state: ConsensusState) -> Dict[str, str]:
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


def _initial_reviews_text(state: ConsensusState) -> str:
    blocks = []
    for item in state.get("initial_reviews", []):
        blocks.append(f"## {item.get('agent_name', '未知Agent')}\n{item.get('content', '')}")
    return "\n\n".join(blocks)[:9000]


def retrieve_context_node(state: ConsensusState) -> ConsensusState:
    query = " ".join([
        state.get("case_title", ""),
        state.get("main_issues", ""),
        state.get("student_info") or "",
        state.get("background") or "",
        state.get("consultation_focus") or "",
        "个案 会商 多智能体 协商 质询 反驳 共识 辅导员 心理 合规 伦理",
    ]).strip()

    docs = hybrid_retrieve(query, state.get("category"))
    state["context"] = _build_context_from_docs(docs)
    state["sources"] = _source_names_from_docs(docs)
    state["initial_reviews"] = []
    return state


def initial_review_node(state: ConsensusState, agent_key: str, agent_name: str, agent_focus: str) -> ConsensusState:
    model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
    p = _payload(state)

    prompt = INITIAL_SYSTEM_PROMPT + "\n\n" + INITIAL_USER_PROMPT.format(
        agent_name=agent_name,
        agent_focus=agent_focus,
        case_title=p["case_title"],
        student_info=p["student_info"],
        main_issues=p["main_issues"],
        background=p["background"],
        current_measures=p["current_measures"],
        consultation_focus=p["consultation_focus"],
        urgency=p["urgency"],
        extra_requirements=p["extra_requirements"],
        context=state.get("context", "未检索到资料"),
    )
    content = "".join(_stream_llm(prompt, model_name, num_predict=ROLE_NUM_PREDICT)).strip()
    state.setdefault("initial_reviews", [])
    state["initial_reviews"].append({"agent_key": agent_key, "agent_name": agent_name, "content": content})
    return state


def cross_questions_node(state: ConsensusState) -> ConsensusState:
    model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
    p = _payload(state)
    prompt = CROSS_SYSTEM_PROMPT + "\n\n" + CROSS_USER_PROMPT.format(
        case_title=p["case_title"],
        main_issues=p["main_issues"],
        background=p["background"],
        initial_reviews=_initial_reviews_text(state),
    )
    state["cross_questions"] = "".join(_stream_llm(prompt, model_name, num_predict=STAGE_NUM_PREDICT)).strip()
    return state


def risk_rebuttal_node(state: ConsensusState) -> ConsensusState:
    model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
    p = _payload(state)
    prompt = REBUTTAL_SYSTEM_PROMPT + "\n\n" + REBUTTAL_USER_PROMPT.format(
        case_title=p["case_title"],
        main_issues=p["main_issues"],
        initial_reviews=_initial_reviews_text(state),
        cross_questions=state.get("cross_questions", ""),
    )
    state["risk_rebuttal"] = "".join(_stream_llm(prompt, model_name, num_predict=STAGE_NUM_PREDICT)).strip()
    return state


def plan_revision_node(state: ConsensusState) -> ConsensusState:
    model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
    p = _payload(state)
    prompt = REVISION_SYSTEM_PROMPT + "\n\n" + REVISION_USER_PROMPT.format(
        case_title=p["case_title"],
        main_issues=p["main_issues"],
        current_measures=p["current_measures"],
        initial_reviews=_initial_reviews_text(state),
        cross_questions=state.get("cross_questions", ""),
        risk_rebuttal=state.get("risk_rebuttal", ""),
    )
    state["plan_revision"] = "".join(_stream_llm(prompt, model_name, num_predict=STAGE_NUM_PREDICT)).strip()
    return state


def consensus_node(state: ConsensusState) -> ConsensusState:
    model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
    p = _payload(state)
    prompt = CONSENSUS_SYSTEM_PROMPT + "\n\n" + CONSENSUS_USER_PROMPT.format(
        case_title=p["case_title"],
        main_issues=p["main_issues"],
        plan_revision=state.get("plan_revision", ""),
    )
    state["consensus"] = "".join(_stream_llm(prompt, model_name, num_predict=STAGE_NUM_PREDICT)).strip()
    return state


def dissent_node(state: ConsensusState) -> ConsensusState:
    model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
    p = _payload(state)
    prompt = DISSENT_SYSTEM_PROMPT + "\n\n" + DISSENT_USER_PROMPT.format(
        case_title=p["case_title"],
        main_issues=p["main_issues"],
        cross_questions=state.get("cross_questions", ""),
        risk_rebuttal=state.get("risk_rebuttal", ""),
        consensus=state.get("consensus", ""),
    )
    state["dissent"] = "".join(_stream_llm(prompt, model_name, num_predict=STAGE_NUM_PREDICT)).strip()
    return state


def final_node(state: ConsensusState) -> ConsensusState:
    model_name = _resolve_model(state.get("model_choice"), PAPER_MODEL)
    p = _payload(state)
    prompt = FINAL_SYSTEM_PROMPT + "\n\n" + FINAL_USER_PROMPT.format(
        case_title=p["case_title"],
        student_info=p["student_info"],
        main_issues=p["main_issues"],
        background=p["background"],
        current_measures=p["current_measures"],
        consultation_focus=p["consultation_focus"],
        urgency=p["urgency"],
        extra_requirements=p["extra_requirements"],
        initial_reviews=_initial_reviews_text(state),
        cross_questions=state.get("cross_questions", ""),
        risk_rebuttal=state.get("risk_rebuttal", ""),
        plan_revision=state.get("plan_revision", ""),
        consensus=state.get("consensus", ""),
        dissent=state.get("dissent", ""),
    )
    state["final_recommendation"] = "".join(_stream_llm(prompt, model_name, num_predict=FINAL_NUM_PREDICT)).strip()
    return state


def build_consensus_case_graph():
    if StateGraph is None:
        raise RuntimeError("未安装 langgraph。请先执行：pip install -U langgraph")

    graph = StateGraph(ConsensusState)
    graph.add_node("retrieve_context", retrieve_context_node)

    previous = "retrieve_context"
    for agent_key, agent_name, agent_focus in CONSENSUS_AGENTS:
        node_name = f"initial_{agent_key}"
        graph.add_node(
            node_name,
            lambda s, k=agent_key, n=agent_name, f=agent_focus: initial_review_node(s, k, n, f),
        )
        graph.add_edge(previous, node_name)
        previous = node_name

    graph.add_node("cross_questions", cross_questions_node)
    graph.add_node("risk_rebuttal", risk_rebuttal_node)
    graph.add_node("plan_revision", plan_revision_node)
    graph.add_node("consensus", consensus_node)
    graph.add_node("dissent", dissent_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("retrieve_context")
    graph.add_edge(previous, "cross_questions")
    graph.add_edge("cross_questions", "risk_rebuttal")
    graph.add_edge("risk_rebuttal", "plan_revision")
    graph.add_edge("plan_revision", "consensus")
    graph.add_edge("consensus", "dissent")
    graph.add_edge("dissent", "final")
    graph.add_edge("final", END)

    return graph.compile()


def run_consensus_case_graph(state: ConsensusState) -> ConsensusState:
    app = build_consensus_case_graph()
    return app.invoke(state)


def stream_consensus_case_events(initial_state: ConsensusState) -> Generator[Dict[str, Any], None, None]:
    """
    稳定版“角色冲突与共识形成机制”。

    设计原则：
    - 中间节点采用精炼会商纪要式输出，保留必要依据和处置指向；
    - 最终报告详细展开，尤其“最终可执行方案”；
    - 全流程保持真正逐 token 流式输出。
    """
    state: ConsensusState = dict(initial_state)

    yield {"type": "stage", "stage": "检索资料"}
    state = retrieve_context_node(state)
    yield {"type": "sources", "sources": state.get("sources", [])}

    yield {"type": "token", "content": "# 多智能体协商式个案会商\n\n"}
    yield {
        "type": "token",
        "content": "> 机制：角色初评 → 交叉质询 → 风险反驳 → 方案修正 → 共识形成 → 保留分歧 → 最终建议。\n\n",
    }

    model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
    p = _payload(state)

    for agent_key, agent_name, agent_focus in CONSENSUS_AGENTS:
        yield {"type": "stage", "stage": f"角色初评：{agent_name}"}
        yield {"type": "token", "content": f"\n\n---\n\n# {agent_name} 初评\n\n"}

        prompt = INITIAL_SYSTEM_PROMPT + "\n\n" + INITIAL_USER_PROMPT.format(
            agent_name=agent_name,
            agent_focus=agent_focus,
            case_title=p["case_title"],
            student_info=p["student_info"],
            main_issues=p["main_issues"],
            background=p["background"],
            current_measures=p["current_measures"],
            consultation_focus=p["consultation_focus"],
            urgency=p["urgency"],
            extra_requirements=p["extra_requirements"],
            context=state.get("context", "未检索到资料"),
        )
        parts: List[str] = []
        for token in _stream_llm(prompt, model_name, num_predict=ROLE_NUM_PREDICT):
            parts.append(token)
            yield {"type": "token", "content": token}
        content = "".join(parts).strip()
        if not content:
            content = f"## {agent_name} 初评\n\n### 1. 关键判断\n本角色未获得有效模型输出，建议人工复核该角色意见。\n\n### 2. 关键风险\n- 该角色视角暂缺，可能影响会商完整性。\n\n### 3. 必要动作\n- 请辅导员人工补充该角色意见。\n\n【本角色初评结束】"
            yield {"type": "token", "content": "\n\n" + content + "\n"}
        state.setdefault("initial_reviews", [])
        state["initial_reviews"].append(
            {"agent_key": agent_key, "agent_name": agent_name, "content": content}
        )

    yield {"type": "stage", "stage": "交叉质询"}
    yield {"type": "token", "content": "\n\n---\n\n# 交叉质询\n\n"}
    cross_prompt = CROSS_SYSTEM_PROMPT + "\n\n" + CROSS_USER_PROMPT.format(
        case_title=p["case_title"],
        main_issues=p["main_issues"],
        background=p["background"],
        initial_reviews=_initial_reviews_text(state),
    )
    cross_parts: List[str] = []
    for token in _stream_llm(cross_prompt, model_name, num_predict=STAGE_NUM_PREDICT):
        cross_parts.append(token)
        yield {"type": "token", "content": token}
    state["cross_questions"] = "".join(cross_parts).strip()
    if not state["cross_questions"]:
        state["cross_questions"] = "## 一、关键质询\n- 本阶段未获得有效模型输出，建议人工复核角色之间的质询问题。\n\n## 二、需要核实的信息\n- 需核实个案事实、风险等级、当前处置依据。\n\n【交叉质询结束】"
        yield {"type": "token", "content": "\n\n" + state["cross_questions"] + "\n"}

    yield {"type": "stage", "stage": "风险反驳"}
    yield {"type": "token", "content": "\n\n---\n\n# 风险反驳\n\n"}
    rebuttal_prompt = REBUTTAL_SYSTEM_PROMPT + "\n\n" + REBUTTAL_USER_PROMPT.format(
        case_title=p["case_title"],
        main_issues=p["main_issues"],
        initial_reviews=_initial_reviews_text(state),
        cross_questions=state.get("cross_questions", ""),
    )
    rebuttal_parts: List[str] = []
    for token in _stream_llm(rebuttal_prompt, model_name, num_predict=STAGE_NUM_PREDICT):
        rebuttal_parts.append(token)
        yield {"type": "token", "content": token}
    state["risk_rebuttal"] = "".join(rebuttal_parts).strip()
    if not state["risk_rebuttal"]:
        state["risk_rebuttal"] = "## 一、不能忽视的风险\n- 本阶段未获得有效模型输出，建议人工复核心理、合规、伦理风险。\n\n## 二、禁忌做法\n- 避免在事实不清时扩大处置范围或直接标签化学生。\n\n【风险反驳结束】"
        yield {"type": "token", "content": "\n\n" + state["risk_rebuttal"] + "\n"}

    yield {"type": "stage", "stage": "方案修正"}
    yield {"type": "token", "content": "\n\n---\n\n# 方案修正\n\n"}
    revision_prompt = REVISION_SYSTEM_PROMPT + "\n\n" + REVISION_USER_PROMPT.format(
        case_title=p["case_title"],
        main_issues=p["main_issues"],
        current_measures=p["current_measures"],
        initial_reviews=_initial_reviews_text(state),
        cross_questions=state.get("cross_questions", ""),
        risk_rebuttal=state.get("risk_rebuttal", ""),
    )
    revision_parts: List[str] = []
    for token in _stream_llm(revision_prompt, model_name, num_predict=STAGE_NUM_PREDICT):
        revision_parts.append(token)
        yield {"type": "token", "content": token}
    state["plan_revision"] = "".join(revision_parts).strip()
    if not state["plan_revision"]:
        state["plan_revision"] = "## 一、修正原则\n- 先核实事实，再分级干预；先保护安全，再推进教育帮扶。\n\n## 二、修正后的关键动作\n- 由辅导员完成有效接触和风险核实。\n- 视情况联动班主任、心理中心、资助专员和学院。\n\n## 三、仍需人工判断的问题\n- 是否达到上报或家校沟通条件。\n\n【方案修正结束】"
        yield {"type": "token", "content": "\n\n" + state["plan_revision"] + "\n"}

    yield {"type": "stage", "stage": "共识形成"}
    yield {"type": "token", "content": "\n\n---\n\n# 共识形成\n\n"}
    consensus_prompt = CONSENSUS_SYSTEM_PROMPT + "\n\n" + CONSENSUS_USER_PROMPT.format(
        case_title=p["case_title"],
        main_issues=p["main_issues"],
        plan_revision=state.get("plan_revision", ""),
    )
    consensus_parts: List[str] = []
    for token in _stream_llm(consensus_prompt, model_name, num_predict=STAGE_NUM_PREDICT):
        consensus_parts.append(token)
        yield {"type": "token", "content": token}
    state["consensus"] = "".join(consensus_parts).strip()
    if not state["consensus"]:
        state["consensus"] = "## 一、共识意见\n- 本个案需要综合考虑学业、心理、经济、合规和伦理因素。\n- 72小时内应完成有效接触、事实核实和初步风险分级。\n\n## 二、立即可执行动作\n- 建立跟进台账，明确责任人和复盘时间。\n\n【共识形成结束】"
        yield {"type": "token", "content": "\n\n" + state["consensus"] + "\n"}

    yield {"type": "stage", "stage": "保留分歧"}
    yield {"type": "token", "content": "\n\n---\n\n# 保留分歧\n\n"}
    dissent_prompt = DISSENT_SYSTEM_PROMPT + "\n\n" + DISSENT_USER_PROMPT.format(
        case_title=p["case_title"],
        main_issues=p["main_issues"],
        cross_questions=state.get("cross_questions", ""),
        risk_rebuttal=state.get("risk_rebuttal", ""),
        consensus=state.get("consensus", ""),
    )
    dissent_parts: List[str] = []
    for token in _stream_llm(dissent_prompt, model_name, num_predict=STAGE_NUM_PREDICT):
        dissent_parts.append(token)
        yield {"type": "token", "content": token}
    state["dissent"] = "".join(dissent_parts).strip()
    if not state["dissent"]:
        state["dissent"] = "## 一、主要分歧\n- 是否立即上报、是否联系家长、是否启动心理中心联动，仍需结合事实核实后判断。\n\n## 二、分歧来源\n不同角色在安全优先、隐私保护、程序合规和执行效率之间存在侧重点差异。\n\n## 三、临时处理原则\n- 信息不足时先采取低侵扰、可留痕、可升级的稳妥措施。\n\n【保留分歧结束】"
        yield {"type": "token", "content": "\n\n" + state["dissent"] + "\n"}

    yield {"type": "stage", "stage": "最终建议"}
    yield {"type": "token", "content": "\n\n---\n\n# 最终建议报告\n\n"}
    final_model = _resolve_model(state.get("model_choice"), PAPER_MODEL)
    final_prompt = FINAL_SYSTEM_PROMPT + "\n\n" + FINAL_USER_PROMPT.format(
        case_title=p["case_title"],
        student_info=p["student_info"],
        main_issues=p["main_issues"],
        background=p["background"],
        current_measures=p["current_measures"],
        consultation_focus=p["consultation_focus"],
        urgency=p["urgency"],
        extra_requirements=p["extra_requirements"],
        initial_reviews=_initial_reviews_text(state),
        cross_questions=state.get("cross_questions", ""),
        risk_rebuttal=state.get("risk_rebuttal", ""),
        plan_revision=state.get("plan_revision", ""),
        consensus=state.get("consensus", ""),
        dissent=state.get("dissent", ""),
    )

    final_parts: List[str] = []
    for token in _stream_llm(final_prompt, final_model, num_predict=FINAL_NUM_PREDICT):
        final_parts.append(token)
        yield {"type": "token", "content": token}
    state["final_recommendation"] = "".join(final_parts).strip()

    yield {"type": "done", "stage": "final"}
