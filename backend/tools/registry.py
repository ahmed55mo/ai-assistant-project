from backend.services.calendar_service import calendar_provider
from backend.services.gmail_service import GmailService
from backend.services.notification_service import NotificationService
from backend.tools.calculator import CALCULATOR_TOOL
from backend.tools.calendar_tools import calendar_tools
from backend.tools.email_tools import gmail_tools
from backend.tools.notification_tools import notification_tools
from backend.tools.base import ToolRegistry


def build_registry() -> ToolRegistry:
    registry = ToolRegistry([CALCULATOR_TOOL])
    for tool in gmail_tools(GmailService()):
        registry.register(tool)
    for tool in calendar_tools(calendar_provider()):
        registry.register(tool)
    for tool in notification_tools(NotificationService()):
        registry.register(tool)
    return registry
