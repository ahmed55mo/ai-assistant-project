from backend.services.calendar_service import calendar_provider
from backend.services.gmail_service import GmailService
from backend.services.notification_service import NotificationService
from backend.tools.calculator import CALCULATOR_TOOL
from backend.tools.calendar_tools import calendar_tools
from backend.tools.email_tools import gmail_tools
from backend.tools.notification_tools import notification_tools
from backend.tools.base import ToolRegistry
from backend.rag.retrieval import RetrievalService
from backend.rag.tool import build_knowledge_search_tool


def build_registry(
    retrieval_service: RetrievalService | None = None,
) -> ToolRegistry:
    registry = ToolRegistry([CALCULATOR_TOOL])
    for tool in gmail_tools(GmailService()):
        registry.register(tool)
    for tool in calendar_tools(calendar_provider()):
        registry.register(tool)
    for tool in notification_tools(NotificationService()):
        registry.register(tool)
    if retrieval_service is not None:
        registry.register(build_knowledge_search_tool(retrieval_service))
    return registry
