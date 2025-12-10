from .provider import GUIProvider
from .session import GUISession
from .transport.connector import GUIConnector

try:
    from .anthropic_client import AnthropicGUIClient
    from . import anthropic_utils
    _anthropic_available = True
except ImportError:
    _anthropic_available = False

__all__ = [
    # Core Provider and Session
    "GUIProvider",
    "GUISession",
    
    # Transport layer
    "GUIConnector"
]

# Add Anthropic modules to exports if available
if _anthropic_available:
    __all__.extend(["AnthropicGUIClient", "anthropic_utils"])