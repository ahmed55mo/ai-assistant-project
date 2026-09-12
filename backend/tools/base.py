from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class ToolError(RuntimeError):
    """A safe, user-facing tool execution error."""


class ConfirmationRequired(ToolError):
    """Raised when a side-effecting tool needs explicit user confirmation."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any]], dict[str, Any]]
    read_only: bool = True
    requires_confirmation: bool = False
    normalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def as_groq_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition] | None = None) -> None:
        self._tools = {tool.name: tool for tool in (tools or [])}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"Unknown tool: {name}") from exc

    def names(self) -> list[str]:
        return list(self._tools)

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.as_groq_tool() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any], confirmed: bool = False) -> dict[str, Any]:
        tool = self.get(name)
        if tool.normalize is not None:
            arguments = tool.normalize(arguments)
        if tool.requires_confirmation and not confirmed:
            raise ConfirmationRequired(f"Confirmation required before running {name}.")
        return tool.execute(arguments)
