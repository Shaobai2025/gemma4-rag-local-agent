from __future__ import annotations

from typing import Any, Dict, List


def generate_markdown_report(question: str, profile: Dict[str, Any], results: Dict[str, Any], analysis_type: str) -> str:
    lines: List[str] = []
    shape = profile.get("shape", {})
    detected = profile.get("detected", {})

    lines.append("# Excel 智能分析报告")
    lines.append("")
    lines.append(f"**分析问题：** {question or '自动综合分析'}")
    lines.append("")
    lines.append("## 一、数据概况")
    lines.append("")
    lines.append(f"- 数据规模：{shape.get('rows', 0)} 行 × {shape.get('columns', 0)} 列")
    lines.append(f"- 识别到成绩列：{len(detected.get('score_columns', []))} 个")
    if detected.get("name_column"):
        lines.append(f"- 姓名列：`{detected.get('name_column')}`")
    if detected.get("id_column"):
        lines.append(f"- 学号/编号列：`{detected.get('id_column')}`")
    if detected.get("class_column"):
        lines.append(f"- 班级/分组列：`{detected.get('class_column')}`")
    if detected.get("score_columns"):
        lines.append(f"- 成绩列：{', '.join('`'+c+'`' for c in detected.get('score_columns', [])[:20])}")
    lines.append("")

    basic = results.get("basic_statistics", {})
    lines.append("## 二、基础统计")
    lines.append("")
    lines.append(f"- 总缺失单元格：{basic.get('missing_total', 0)}")
    miss = basic.get("missing_by_column", {})
    if miss:
        top_miss = [(k, v) for k, v in miss.items() if v][:8]
        if top_miss:
            lines.append("- 缺失较多字段：")
            for k, v in top_miss:
                lines.append(f"  - {k}：{v} 个缺失")
    lines.append("")

    if results.get("grade_warning", {}).get("available"):
        _append_grade_warning(lines, results["grade_warning"])

    if results.get("group_statistics", {}).get("available"):
        _append_group_stats(lines, results["group_statistics"])

    if results.get("gpa_analysis", {}).get("available"):
        _append_gpa_analysis(lines, results["gpa_analysis"])

    if results.get("correlation", {}).get("available"):
        _append_correlation(lines, results["correlation"])

    if results.get("outliers", {}).get("available"):
        _append_outliers(lines, results["outliers"])

    lines.append("## 八、初步建议")
    lines.append("")
    if analysis_type == "grade_warning":
        lines.append("- 对一级、二级预警学生优先开展一对一谈话，重点核实学习困难、出勤情况、缺考/旷考原因、心理状态和家庭支持情况。")
        lines.append("- 对挂科人数较多的课程，建议联系任课教师或班主任了解共性原因，区分学生个体问题和课程整体难度问题。")
        lines.append("- 对班级预警率较高的群体，建议建立班级层面的学业帮扶台账，并设置两周一次的复盘节点。")
    else:
        lines.append("- 建议结合业务目标进一步指定分组字段、关键指标和筛选规则，以获得更精准分析。")
        lines.append("- 对异常值和缺失值较多的字段，建议先回到原始表核对，再开展正式判断。")

    return "\n".join(lines)


def _append_grade_warning(lines: List[str], gw: Dict[str, Any]) -> None:
    lines.append("## 三、成绩预警分析")
    lines.append("")
    lines.append(f"- 学生总数：{gw.get('student_count', 0)}")
    lines.append(f"- 预警人数：{gw.get('warning_count', 0)}")
    level_counts = gw.get("level_counts", {})
    if level_counts:
        lines.append("- 预警等级分布：")
        for lv in ["一级预警", "二级预警", "三级预警", "正常"]:
            if lv in level_counts:
                lines.append(f"  - {lv}：{level_counts[lv]} 人")

    fail_course_counts = gw.get("fail_course_counts", {})
    if fail_course_counts:
        sorted_courses = sorted(fail_course_counts.items(), key=lambda x: x[1], reverse=True)
        lines.append("- 不及格人数较多课程：")
        for course, count in sorted_courses[:10]:
            if count > 0:
                lines.append(f"  - {course}：{count} 人")

    absence_course_counts = gw.get("absence_course_counts", {})
    if absence_course_counts:
        sorted_absence = sorted(absence_course_counts.items(), key=lambda x: x[1], reverse=True)
        nonzero_absence = [(c, n) for c, n in sorted_absence if n > 0]
        if nonzero_absence:
            lines.append("- 缺考/旷考较多课程：")
            for course, count in nonzero_absence[:10]:
                lines.append(f"  - {course}：{count} 人")

    top = gw.get("top_warning_students", [])
    if top:
        lines.append("")
        lines.append("### 重点预警学生名单（前80人）")
        lines.append("")
        cols = ["姓名", "学号", "班级", "平均分", "最低分", "挂科门数", "缺考旷考门数", "不及格/缺考课程", "预警等级"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in top:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("")


def _append_group_stats(lines: List[str], group: Dict[str, Any]) -> None:
    lines.append("## 四、班级/分组对比")
    lines.append("")
    groups = group.get("groups", [])
    if not groups:
        lines.append("- 未生成有效分组统计。")
        lines.append("")
        return

    cols = ["group", "count", "average_score", "fail_student_count", "fail_rate", "fail_count_avg"]
    headers = ["分组", "人数", "平均分", "挂科学生数", "挂科率", "平均挂科门数"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in groups[:20]:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("")



def _append_gpa_analysis(lines: List[str], gpa: Dict[str, Any]) -> None:
    lines.append("## 五、平均学分绩点统计分析")
    lines.append("")
    items = gpa.get("items", [])
    if not items:
        lines.append("- 未生成有效绩点分析。")
        lines.append("")
        return

    threshold = gpa.get("threshold", 2.5)
    lines.append(f"- 当前低绩点关注阈值：**{threshold}**。如需自定义，可在问题中写“绩点低于2.8的学生”。")
    lines.append("")
    lines.append("| 字段 | 人数 | 均值 | 中位数 | 标准差 | 变异系数 | P25 | P75 | <2.0人数 | 低于阈值人数 | >=3.5人数 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in items:
        lines.append(
            f"| {r.get('column')} | {r.get('count')} | {r.get('mean')} | {r.get('median')} | "
            f"{r.get('std')} | {r.get('cv')} | {r.get('q25')} | {r.get('q75')} | "
            f"{r.get('lt_2_0_count')} | {r.get('lt_custom_count')} | {r.get('gte_3_5_count')} |"
        )
        if r.get("interpretation"):
            lines.append(f"- **{r.get('column')}解释：** {r.get('interpretation')}")

        students = r.get("low_gpa_students", [])
        if students:
            lines.append("")
            lines.append(f"### {r.get('column')}低绩点学生名单（前100人）")
            lines.append("")
            cols = ["姓名", "学号", "班级", "平均学分绩点", "Z分数", "风险原因"]
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for stu in students[:100]:
                lines.append("| " + " | ".join(str(stu.get(c, "")) for c in cols) + " |")
    lines.append("")

def _append_correlation(lines: List[str], corr: Dict[str, Any]) -> None:
    lines.append("## 六、相关性分析")
    lines.append("")
    pairs = corr.get("top_pairs", [])
    if not pairs:
        lines.append("- 未发现可展示的相关性结果。")
    else:
        lines.append("| 指标1 | 指标2 | 相关系数 |")
        lines.append("| --- | --- | --- |")
        for p in pairs[:10]:
            lines.append(f"| {p.get('x')} | {p.get('y')} | {p.get('corr')} |")
    lines.append("")


def _append_outliers(lines: List[str], outliers: Dict[str, Any]) -> None:
    lines.append("## 七、异常值检测")
    lines.append("")
    cols = outliers.get("columns", {})
    if not cols:
        lines.append("- 未检测到明显异常值。")
    else:
        lines.append("| 字段 | 异常数量 | 下界 | 上界 | 最小异常 | 最大异常 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for col, r in list(cols.items())[:20]:
            lines.append(f"| {col} | {r.get('outlier_count')} | {r.get('low_threshold')} | {r.get('high_threshold')} | {r.get('min_outlier')} | {r.get('max_outlier')} |")
    lines.append("")
