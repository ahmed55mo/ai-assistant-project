"""Compatibility exports for the LangChain tool adaptation layer."""

from backend.tools.langchain_tools import adapt_tool, build_langchain_tools

__all__ = ["adapt_tool", "build_langchain_tools"]
