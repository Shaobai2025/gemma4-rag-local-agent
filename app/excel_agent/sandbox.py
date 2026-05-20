from __future__ import annotations

import ast
import io
import math
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


class UnsafeCodeError(Exception):
    pass


BANNED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "requests", "urllib", "http",
    "ftplib", "paramiko", "shutil", "pathlib", "glob", "pickle",
    "sqlite3", "ctypes", "multiprocessing", "threading", "asyncio",
}

BANNED_CALL_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "breakpoint", "help",
}

BANNED_ATTRS = {
    "to_pickle", "to_sql", "to_clipboard",
    "read_pickle", "read_sql", "read_sql_query", "read_sql_table",
}


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    result: Dict[str, Any]
    error: Optional[str] = None
    code: Optional[str] = None


class SafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in BANNED_IMPORTS:
                raise UnsafeCodeError(f"禁止导入模块：{alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        root = (node.module or "").split(".")[0]
        if root in BANNED_IMPORTS:
            raise UnsafeCodeError(f"禁止导入模块：{node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALL_NAMES:
            raise UnsafeCodeError(f"禁止调用函数：{node.func.id}")
        if isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_ATTRS:
            raise UnsafeCodeError(f"禁止调用方法：{node.func.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("__"):
            raise UnsafeCodeError(f"禁止访问魔术属性：{node.attr}")
        self.generic_visit(node)


def validate_code_safety(code: str) -> None:
    tree = ast.parse(code)
    SafetyVisitor().visit(tree)


def _is_bad_float(v: Any) -> bool:
    try:
        return isinstance(v, float) and (math.isnan(v) or math.isinf(v))
    except Exception:
        return False


def _safe_scalar(v: Any) -> Any:
    """
    把 pandas/numpy/Python 中 JSON 不兼容的值转成标准 JSON 可接受值。
    重点处理：NaN / NaT / inf / -inf。
    """
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except Exception:
        pass

    if _is_bad_float(v):
        return None

    if isinstance(v, (str, int, float, bool)) or v is None:
        return v

    return str(v)


def _safe_to_jsonable(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        # 注意：DataFrame.where(..., None) 对数值列可能仍保留 NaN。
        # 所以必须先 to_dict，再递归清洗每一个单元格。
        records = obj.head(200).to_dict(orient="records")
        return {
            "__type__": "dataframe",
            "columns": [str(c) for c in obj.columns],
            "rows": _safe_to_jsonable(records),
            "shape": [int(obj.shape[0]), int(obj.shape[1])],
        }

    if isinstance(obj, pd.Series):
        return {
            "__type__": "series",
            "name": str(obj.name),
            "values": _safe_to_jsonable(obj.head(200).to_dict()),
            "length": int(len(obj)),
        }

    if isinstance(obj, dict):
        return {str(k): _safe_to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_safe_to_jsonable(x) for x in obj[:500]]

    if isinstance(obj, tuple):
        return [_safe_to_jsonable(x) for x in obj[:500]]

    try:
        import numpy as np
        if isinstance(obj, (np.ndarray,)):
            return _safe_to_jsonable(obj.tolist())
    except Exception:
        pass

    return _safe_scalar(obj)


def execute_analysis_code(code: str, df: pd.DataFrame, profile: Dict[str, Any]) -> SandboxResult:
    """
    执行 Gemma4 生成的分析代码。

    约定：
    - 代码可以访问 df、profile、pd、np、math；
    - 必须把最终结果写入变量 result；
    - result 必须是 dict；
    - 禁止任意文件读写、网络访问、系统调用。
    """
    try:
        validate_code_safety(code)
    except Exception as e:
        return SandboxResult(ok=False, stdout="", result={}, error=f"安全检查失败：{e}", code=code)

    try:
        import numpy as np
    except Exception:
        np = None

    safe_globals: Dict[str, Any] = {
        "__builtins__": {
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "print": print,
        },
        "pd": pd,
        "np": np,
        "math": math,
        "df": df.copy(),
        "profile": profile,
        "result": {},
    }

    stdout = io.StringIO()

    try:
        with redirect_stdout(stdout):
            compiled = compile(code, "<excel_agent_generated_code>", "exec")
            exec(compiled, safe_globals, safe_globals)

        result = safe_globals.get("result", {})
        if not isinstance(result, dict):
            result = {"value": result}

        return SandboxResult(
            ok=True,
            stdout=stdout.getvalue(),
            result=_safe_to_jsonable(result),
            error=None,
            code=code,
        )
    except Exception:
        return SandboxResult(
            ok=False,
            stdout=stdout.getvalue(),
            result={},
            error=traceback.format_exc(limit=6),
            code=code,
        )
