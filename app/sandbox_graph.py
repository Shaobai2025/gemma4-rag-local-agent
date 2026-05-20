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


class SandboxState(TypedDict, total=False):
    case_title: str
    student_info: Optional[str]
    main_issues: str
    background: Optional[str]
    current_state: Optional[str]
    current_measures: Optional[str]
    scenario_a: Optional[str]
    scenario_b: Optional[str]
    scenario_c: Optional[str]
    period: str
    focus: Optional[str]
    category: Optional[str]
    model_choice: Optional[str]
    context: str
    sources: List[str]
    risk_baseline: str
    path_a_result: str
    path_b_result: str
    path_c_result: str
    stress_result: str
    final_decision: str


# 保持你当前推荐配置：中间节点短输出，最终报告长输出。
ROLE_NUM_PREDICT = 1400
FINAL_NUM_PREDICT = 7500
NUM_CTX = 16384


BASELINE_SYSTEM_PROMPT = """
你是高校学生个案沙盘推演 Agent 的“风险基线节点”。

你的任务不是写报告，而是为后续推演提供简短、准确、可决策的风险基线。

要求：
1. 只输出关键判断，不展开成长篇分析。
2. 不得进行医学诊断、心理诊断或法律结论。
3. 涉及自伤、自杀、失联、严重违纪、突发事件时，必须提示启动学校既有机制。
4. 每个小节都要简短完整，不要留下半句话。
5. 输出总长度建议控制在 400-700 字。
"""

BASELINE_USER_PROMPT = """
请为以下学生个案建立“当前风险基线”。

个案标题：{case_title}
学生基本信息：{student_info}
核心问题：{main_issues}
已知背景：{background}
当前状态：{current_state}
当前已采取措施：{current_measures}
推演周期：{period}
重点关注：{focus}

参考资料：
{context}

请严格按以下结构输出，保持简洁：

## 当前风险基线

### 1. 关键判断
用 1-2 句话说明当前主要风险。

### 2. 主要风险
最多列 3 条，每条不超过 1 句话。

### 3. 当前风险等级
低 / 中 / 较高 / 高，并用 1 句话说明理由。

### 4. 底线动作
列出 1-2 个必须立即保留的动作。
"""


PATH_SYSTEM_PROMPT = """
你是高校学生个案沙盘推演 Agent 的“处置路径推演节点”。

你的任务不是写完整报告，而是对某一条处置路径进行短推演，保留最重要的决策信息。

要求：
1. 只输出关键判断、主要收益、主要风险、是否推荐。
2. 不要写长篇背景，不要重复个案材料。
3. 不要输出最终报告。
4. 涉及心理危机或安全风险时，要提醒及时上报与联动。
5. 输出总长度建议控制在 400-700 字。
"""

PATH_USER_PROMPT = """
请对以下学生个案的一个处置路径进行短推演。

个案标题：{case_title}
学生基本信息：{student_info}
核心问题：{main_issues}
已知背景：{background}
当前状态：{current_state}
当前已采取措施：{current_measures}

当前风险基线：
{risk_baseline}

推演路径名称：{path_name}
推演路径内容：{path_content}
推演周期：{period}
重点关注：{focus}

参考资料：
{context}

请严格按以下结构输出，保持简洁：

## {path_name}

### 1. 关键判断
用 1-2 句话判断该路径是否适合当前个案。

### 2. 主要收益
最多 2 条。

### 3. 主要风险
最多 2 条。

### 4. 是否推荐
推荐 / 谨慎推荐 / 不推荐，并用 1 句话说明理由。
"""


STRESS_SYSTEM_PROMPT = """
你是高校学生个案沙盘推演 Agent 的“压力测试节点”。

你的任务是短促识别方案中最容易失效的地方，不要写完整报告。

要求：
1. 只保留关键压力点、升级风险、预案和禁忌做法。
2. 重点关注危机升级、沟通失败、家校沟通失当、学生拒绝配合、政策程序风险。
3. 输出要具体，不要只说“加强关注”。
4. 输出总长度建议控制在 500-800 字。
"""

STRESS_USER_PROMPT = """
请对以下个案沙盘推演结果进行压力测试。

个案标题：{case_title}
核心问题：{main_issues}

当前风险基线：
{risk_baseline}

路径A结果：
{path_a_result}

路径B结果：
{path_b_result}

路径C结果：
{path_c_result}

请严格按以下结构输出，保持简洁：

## 压力测试

### 1. 最可能失效点
最多 2 条。

### 2. 最可能升级风险
最多 2 条。

### 3. 必须提前准备的预案
最多 3 条。

### 4. 禁忌做法
最多 2 条。
"""


FINAL_SYSTEM_PROMPT = """
你是高校学生个案沙盘推演 Agent 的“最终决策节点”。

你的任务是综合前面各节点的关键决策信息，生成一份完整、详细、可执行、可归档的学生个案沙盘推演报告。

要求：
1. 最终报告要详细展开，不能只复述中间节点。
2. 输出要适合辅导员和学院学生工作负责人使用。
3. 结论要清楚：推荐路径、备选路径、不能做什么、什么时候升级上报。
4. 不得替代医学诊断、心理诊断、法律结论或学校正式决定。
5. 涉及高风险时，要明确上报、联动、留痕和跟进机制。
6. 报告要有时间线、责任分工、观察指标和归档模板。
7. 必须完整收尾。
"""

FINAL_USER_PROMPT = """
请基于以下沙盘推演材料，生成最终详细报告。

个案标题：{case_title}
学生基本信息：{student_info}
核心问题：{main_issues}
推演周期：{period}
重点关注：{focus}

当前风险基线：
{risk_baseline}

路径A推演：
{path_a_result}

路径B推演：
{path_b_result}

路径C推演：
{path_c_result}

压力测试：
{stress_result}

请严格按以下结构输出，内容要详细、可执行：

# 学生个案沙盘推演报告

## 一、个案推演目标
说明本次推演要解决什么问题，以及为什么需要沙盘推演。

## 二、当前风险基线
概括当前风险等级、主要风险维度、关键触发因素和保护因素。

## 三、三种处置路径比较
用表格比较路径A、路径B、路径C的收益、风险、执行难度、适用条件。

## 四、推荐处置路径
明确推荐主路径和备选路径，并说明理由。
同时说明不建议采用的做法。

## 五、风险演化预判
分别说明72小时内、两周内、一个月内可能出现的变化。

## 六、关键观察指标
列出需要重点观察的学生表现、联系状态、学业行为、情绪信号、家庭反馈等。

## 七、行动时间线
按“今天/72小时内/一周内/两周内/一个月内”列出动作、责任人、完成标准。

## 八、升级上报条件
明确什么情况下必须上报学院、联动心理中心、联系家长或启动应急机制。

## 九、沟通话术建议
分别给出辅导员谈话话术、家校沟通话术、班主任协同话术。

## 十、归档与复盘模板
生成一份可复制使用的跟进记录与复盘模板。
"""


def _resolve_model(choice: Optional[str], default_model: str) -> str:
    if not choice or choice == "auto":
        return default_model
    return choice


def _llm(model_name: str, *, num_predict: int = 1600, num_ctx: int = NUM_CTX) -> ChatOllama:
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


def _payload(state: SandboxState) -> Dict[str, str]:
    return {
        "case_title": state.get("case_title", ""),
        "student_info": state.get("student_info") or "未提供",
        "main_issues": state.get("main_issues", ""),
        "background": state.get("background") or "未提供",
        "current_state": state.get("current_state") or "未提供",
        "current_measures": state.get("current_measures") or "未提供",
        "period": state.get("period", "两周"),
        "focus": state.get("focus") or "学业、心理、家庭、经济、纪律、就业等综合风险",
    }


def retrieve_context_node(state: SandboxState) -> SandboxState:
    query = " ".join([
        state.get("case_title", ""),
        state.get("main_issues", ""),
        state.get("student_info") or "",
        state.get("background") or "",
        state.get("current_state") or "",
        "学生 个案 沙盘 推演 辅导员 风险 干预 会商",
    ]).strip()

    docs = hybrid_retrieve(query, state.get("category"))
    state["context"] = _build_context_from_docs(docs)
    state["sources"] = _source_names_from_docs(docs)
    return state


def baseline_node(state: SandboxState) -> SandboxState:
    model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
    p = _payload(state)
    prompt = BASELINE_SYSTEM_PROMPT + "\n\n" + BASELINE_USER_PROMPT.format(
        case_title=p["case_title"],
        student_info=p["student_info"],
        main_issues=p["main_issues"],
        background=p["background"],
        current_state=p["current_state"],
        current_measures=p["current_measures"],
        period=p["period"],
        focus=p["focus"],
        context=state.get("context", "未检索到资料"),
    )
    state["risk_baseline"] = "".join(
        _stream_llm(prompt, model_name, num_predict=ROLE_NUM_PREDICT)
    ).strip()
    return state


def _default_paths(state: SandboxState) -> Dict[str, str]:
    return {
        "A": state.get("scenario_a") or "路径A：温和支持型。以辅导员持续谈话、学习帮扶、情绪支持和班主任协同为主，先建立信任，再逐步推动改变。",
        "B": state.get("scenario_b") or "路径B：多方联动型。辅导员牵头，班主任、心理中心、资助专员、家长或学院共同参与，形成较强支持与约束。",
        "C": state.get("scenario_c") or "路径C：底线管理型。在明确风险和规则的基础上，强化纪律边界、学业预警、书面记录、上报备案和阶段性硬约束。",
    }


def path_node(state: SandboxState, key: str, name: str) -> SandboxState:
    model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
    p = _payload(state)
    paths = _default_paths(state)

    prompt = PATH_SYSTEM_PROMPT + "\n\n" + PATH_USER_PROMPT.format(
        case_title=p["case_title"],
        student_info=p["student_info"],
        main_issues=p["main_issues"],
        background=p["background"],
        current_state=p["current_state"],
        current_measures=p["current_measures"],
        risk_baseline=state.get("risk_baseline", ""),
        path_name=name,
        path_content=paths[key],
        period=p["period"],
        focus=p["focus"],
        context=state.get("context", "未检索到资料"),
    )
    result = "".join(_stream_llm(prompt, model_name, num_predict=ROLE_NUM_PREDICT)).strip()
    state[f"path_{key.lower()}_result"] = result
    return state


def stress_node(state: SandboxState) -> SandboxState:
    model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
    p = _payload(state)
    prompt = STRESS_SYSTEM_PROMPT + "\n\n" + STRESS_USER_PROMPT.format(
        case_title=p["case_title"],
        main_issues=p["main_issues"],
        risk_baseline=state.get("risk_baseline", ""),
        path_a_result=state.get("path_a_result", ""),
        path_b_result=state.get("path_b_result", ""),
        path_c_result=state.get("path_c_result", ""),
    )
    state["stress_result"] = "".join(
        _stream_llm(prompt, model_name, num_predict=ROLE_NUM_PREDICT)
    ).strip()
    return state


def final_node(state: SandboxState) -> SandboxState:
    model_name = _resolve_model(state.get("model_choice"), PAPER_MODEL)
    p = _payload(state)
    prompt = FINAL_SYSTEM_PROMPT + "\n\n" + FINAL_USER_PROMPT.format(
        case_title=p["case_title"],
        student_info=p["student_info"],
        main_issues=p["main_issues"],
        period=p["period"],
        focus=p["focus"],
        risk_baseline=state.get("risk_baseline", ""),
        path_a_result=state.get("path_a_result", ""),
        path_b_result=state.get("path_b_result", ""),
        path_c_result=state.get("path_c_result", ""),
        stress_result=state.get("stress_result", ""),
    )
    state["final_decision"] = "".join(
        _stream_llm(prompt, model_name, num_predict=FINAL_NUM_PREDICT)
    ).strip()
    return state


def build_case_sandbox_graph():
    if StateGraph is None:
        raise RuntimeError("未安装 langgraph。请先执行：pip install -U langgraph")

    graph = StateGraph(SandboxState)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("baseline", baseline_node)
    graph.add_node("path_a", lambda s: path_node(s, "A", "路径A：温和支持型"))
    graph.add_node("path_b", lambda s: path_node(s, "B", "路径B：多方联动型"))
    graph.add_node("path_c", lambda s: path_node(s, "C", "路径C：底线管理型"))
    graph.add_node("stress", stress_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "baseline")
    graph.add_edge("baseline", "path_a")
    graph.add_edge("path_a", "path_b")
    graph.add_edge("path_b", "path_c")
    graph.add_edge("path_c", "stress")
    graph.add_edge("stress", "final")
    graph.add_edge("final", END)

    return graph.compile()


def run_case_sandbox_graph(state: SandboxState) -> SandboxState:
    app = build_case_sandbox_graph()
    return app.invoke(state)


def stream_case_sandbox_events(initial_state: SandboxState) -> Generator[Dict[str, Any], None, None]:
    """
    稳定版学生个案沙盘推演。

    设计原则：
    - 中间节点只输出关键决策信息，避免卡顿；
    - 最终决策报告详细展开；
    - 全流程保持真正逐 token 流式输出。
    """
    state: SandboxState = dict(initial_state)

    yield {"type": "stage", "stage": "检索资料"}
    state = retrieve_context_node(state)
    yield {"type": "sources", "sources": state.get("sources", [])}

    model_name = _resolve_model(state.get("model_choice"), QA_MODEL)
    p = _payload(state)

    yield {"type": "stage", "stage": "建立风险基线"}
    yield {"type": "token", "content": "# 学生个案沙盘推演\n\n---\n\n"}
    baseline_prompt = BASELINE_SYSTEM_PROMPT + "\n\n" + BASELINE_USER_PROMPT.format(
        case_title=p["case_title"],
        student_info=p["student_info"],
        main_issues=p["main_issues"],
        background=p["background"],
        current_state=p["current_state"],
        current_measures=p["current_measures"],
        period=p["period"],
        focus=p["focus"],
        context=state.get("context", "未检索到资料"),
    )
    baseline_parts: List[str] = []
    for token in _stream_llm(baseline_prompt, model_name, num_predict=ROLE_NUM_PREDICT):
        baseline_parts.append(token)
        yield {"type": "token", "content": token}
    state["risk_baseline"] = "".join(baseline_parts).strip()

    paths = _default_paths(state)
    for key, name in [
        ("A", "路径A：温和支持型"),
        ("B", "路径B：多方联动型"),
        ("C", "路径C：底线管理型"),
    ]:
        yield {"type": "stage", "stage": name}
        yield {"type": "token", "content": f"\n\n---\n\n# {name}\n\n"}

        path_prompt = PATH_SYSTEM_PROMPT + "\n\n" + PATH_USER_PROMPT.format(
            case_title=p["case_title"],
            student_info=p["student_info"],
            main_issues=p["main_issues"],
            background=p["background"],
            current_state=p["current_state"],
            current_measures=p["current_measures"],
            risk_baseline=state.get("risk_baseline", ""),
            path_name=name,
            path_content=paths[key],
            period=p["period"],
            focus=p["focus"],
            context=state.get("context", "未检索到资料"),
        )

        parts: List[str] = []
        for token in _stream_llm(path_prompt, model_name, num_predict=ROLE_NUM_PREDICT):
            parts.append(token)
            yield {"type": "token", "content": token}
        state[f"path_{key.lower()}_result"] = "".join(parts).strip()

    yield {"type": "stage", "stage": "压力测试"}
    yield {"type": "token", "content": "\n\n---\n\n# 压力测试\n\n"}
    stress_prompt = STRESS_SYSTEM_PROMPT + "\n\n" + STRESS_USER_PROMPT.format(
        case_title=p["case_title"],
        main_issues=p["main_issues"],
        risk_baseline=state.get("risk_baseline", ""),
        path_a_result=state.get("path_a_result", ""),
        path_b_result=state.get("path_b_result", ""),
        path_c_result=state.get("path_c_result", ""),
    )
    stress_parts: List[str] = []
    for token in _stream_llm(stress_prompt, model_name, num_predict=ROLE_NUM_PREDICT):
        stress_parts.append(token)
        yield {"type": "token", "content": token}
    state["stress_result"] = "".join(stress_parts).strip()

    yield {"type": "stage", "stage": "最终决策报告"}
    yield {"type": "token", "content": "\n\n---\n\n# 最终决策报告\n\n"}

    final_model = _resolve_model(state.get("model_choice"), PAPER_MODEL)
    final_prompt = FINAL_SYSTEM_PROMPT + "\n\n" + FINAL_USER_PROMPT.format(
        case_title=p["case_title"],
        student_info=p["student_info"],
        main_issues=p["main_issues"],
        period=p["period"],
        focus=p["focus"],
        risk_baseline=state.get("risk_baseline", ""),
        path_a_result=state.get("path_a_result", ""),
        path_b_result=state.get("path_b_result", ""),
        path_c_result=state.get("path_c_result", ""),
        stress_result=state.get("stress_result", ""),
    )
    final_parts: List[str] = []
    for token in _stream_llm(final_prompt, final_model, num_predict=FINAL_NUM_PREDICT):
        final_parts.append(token)
        yield {"type": "token", "content": token}
    state["final_decision"] = "".join(final_parts).strip()

    yield {"type": "done", "stage": "final"}
