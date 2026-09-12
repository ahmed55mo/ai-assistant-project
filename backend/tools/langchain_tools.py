from __future__ import annotations

import logging
from typing import Any

from pydantic import ConfigDict, Field, create_model
from langchain_core.tools import StructuredTool

from backend.tools.base import ToolDefinition, ToolRegistry
from backend.tools.registry import build_registry

logger = logging.getLogger(__name__)


def _field_type(schema: dict[str, Any]) -> Any:
    schema_type = schema.get("type")
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "object":
        return dict[str, Any]
    if schema_type == "array":
        return list[Any]
    return str


def _args_model(tool: ToolDefinition, properties: dict[str, Any]) -> type:
    required = set(tool.parameters.get("required", []))
    fields: dict[str, tuple[Any, Any]] = {}
    for name, schema in properties.items():
        default = ... if name in required else None
        description = schema.get("description")
        fields[name] = (
            _field_type(schema) | None if default is None else _field_type(schema),
            Field(default=default, description=description),
        )
    return create_model(
        f"{tool.name.title().replace('_', '')}Input",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _public_definition(tool: ToolDefinition) -> tuple[dict[str, Any], Any]:
    properties = dict(tool.parameters.get("properties", {}))
    if tool.name == "send_email":
        properties["recipient"] = properties.pop(
            "to", {"type": "string", "description": "Recipient email address."}
        )
    return properties, _args_model(tool, properties)


def adapt_tool(tool: ToolDefinition, registry: ToolRegistry) -> StructuredTool:
    properties, args_schema = _public_definition(tool)

    def invoke(**arguments: Any) -> dict[str, Any]:
        arguments = {key: value for key, value in arguments.items() if value is not None}
        if tool.name == "send_email":
            arguments = {**arguments, "to": arguments.pop("recipient")}
        logger.debug(
            "LangChain tool adapter invoking tool=%s argument_keys=%s",
            tool.name,
            sorted(arguments),
        )
        return registry.execute(tool.name, arguments)

    parameters = {
        "type": "object",
        "properties": properties,
        "required": [
            "recipient" if name == "to" else name
            for name in tool.parameters.get("required", [])
        ],
        "additionalProperties": False,
    }
    return StructuredTool.from_function(
        func=invoke,
        name=tool.name,
        description=tool.description,
        args_schema=args_schema,
        infer_schema=False,
        metadata={"read_only": tool.read_only, "requires_confirmation": tool.requires_confirmation},
    )


def build_langchain_tools(registry: ToolRegistry | None = None) -> list[StructuredTool]:
    """Adapt every registered Phase 3 tool without changing its implementation."""
    source = registry or build_registry()
    return [adapt_tool(source.get(name), source) for name in source.names()]


__all__ = ["adapt_tool", "build_langchain_tools"]
