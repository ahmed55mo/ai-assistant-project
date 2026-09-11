from __future__ import annotations

import ast
import operator
import re
from typing import Any

from backend.tools.base import ToolDefinition, ToolError


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculate(arguments: dict[str, Any]) -> dict[str, Any]:
    expression = arguments.get("expression")
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
        raise ToolError("Calculator expression must be a non-empty string under 200 characters.")
    normalized_expression = _normalize_percentage(expression)
    try:
        tree = ast.parse(normalized_expression, mode="eval")
        result = _evaluate(tree.body)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ToolError("The calculator could not evaluate that expression safely.") from exc
    if not isinstance(result, (int, float)) or isinstance(result, bool):
        raise ToolError("The calculator result is not a number.")
    return {"expression": expression, "result": result}


def _normalize_percentage(expression: str) -> str:
    match = re.fullmatch(
        r"\s*([-+]?\d+(?:\.\d+)?)\s*%\s*of\s*([-+]?\d+(?:\.\d+)?)\s*",
        expression,
        flags=re.IGNORECASE,
    )
    if match:
        percentage, amount = match.groups()
        return f"({percentage} / 100) * {amount}"
    return expression


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Only numeric arithmetic is supported.")


CALCULATOR_TOOL = ToolDefinition(
    name="calculator",
    description="Safely evaluate a numeric arithmetic expression.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "additionalProperties": False,
    },
    execute=calculate,
)
