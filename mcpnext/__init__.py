from importlib import import_module as _imp
from typing import Dict as _Dict, Any as _Any

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    
    # Main API
    "MCPNext",
    "MCPNextConfig",

    # Core Components
    "GroundingAgent",
    "GroundingClient",
    "LLMClient",
    "BaseTool",
    "ToolResult",
    "BackendType",

    # Recording System
    "RecordingManager",
    "RecordingViewer",
]

# Map attribute → sub-module that provides it.
_attr_to_module: _Dict[str, str] = {
    # Main API
    "MCPNext": "mcpnext.tool_layer",
    "MCPNextConfig": "mcpnext.tool_layer",

    # Core Components
    "GroundingAgent": "mcpnext.agents",
    "GroundingClient": "mcpnext.grounding.core.grounding_client",
    "LLMClient": "mcpnext.llm",
    "BaseTool": "mcpnext.grounding.core.tool.base",
    "ToolResult": "mcpnext.grounding.core.types",
    "BackendType": "mcpnext.grounding.core.types",

    # Recording System
    "RecordingManager": "mcpnext.recording",
    "RecordingViewer": "mcpnext.recording.viewer",
}


def __getattr__(name: str) -> _Any:
    """Dynamically import sub-modules on first attribute access.

    This keeps the *initial* package import lightweight and avoids raising
    `ModuleNotFoundError` for optional / heavy dependencies until the
    corresponding functionality is explicitly used.
    """
    if name not in _attr_to_module:
        raise AttributeError(f"module 'mcpnext' has no attribute '{name}'")

    module_name = _attr_to_module[name]
    module = _imp(module_name)
    value = getattr(module, name)
    globals()[name] = value 
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_attr_to_module.keys()))