from typing import Dict, Optional, Any, List
try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseModel, Field, validator as field_validator
    PYDANTIC_V2 = False

from mcpnext.grounding.core.types import (
    SessionConfig, 
    SecurityPolicy,
    BackendType
)
from .constants import LOG_LEVELS


class ConfigMixin:
    """Mixin to add utility methods for config access"""
    
    def get_value(self, key: str, default=None):
        """
        Safely get config value, works with both dict and Pydantic models.
        
        Args:
            key: Configuration key
            default: Default value if key not found
        """
        if isinstance(self, dict):
            return self.get(key, default)
        else:
            return getattr(self, key, default)


class BackendConfig(BaseModel, ConfigMixin):
    """Base backend configuration"""
    enabled: bool = Field(True, description="Whether the backend is enabled")
    timeout: int = Field(30, ge=1, le=300, description="Timeout in seconds")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")


class ShellConfig(BackendConfig):
    """
    Shell backend configuration
    
    Attributes:
        enabled: Whether shell backend is enabled
        timeout: Default timeout for shell operations (seconds)
        max_retries: Maximum number of retry attempts for failed operations
        retry_interval: Wait time between retries (seconds)
        default_shell: Path to default shell executable
        working_dir: Default working directory for bash scripts
        env: Default environment variables for shell operations
        conda_env: Conda environment name to activate before execution (optional)
        default_port: Default port for shell server connection
    """
    retry_interval: float = Field(3.0, ge=0.1, le=60.0, description="Wait time between retries in seconds")
    default_shell: str = Field("/bin/bash", description="Default shell path")
    working_dir: Optional[str] = Field(None, description="Default working directory for bash scripts")
    env: Dict[str, str] = Field(default_factory=dict, description="Default environment variables")
    conda_env: Optional[str] = Field(None, description="Conda environment name to activate (e.g., 'myenv')")
    default_port: int = Field(5000, ge=1, le=65535, description="Default port for shell server")
    
    @field_validator('default_shell')
    @classmethod
    def validate_shell(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Shell path must be a non-empty string")
        return v
    
    @field_validator('working_dir')
    @classmethod
    def validate_working_dir(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError("Working directory must be a string")
        return v

class WebConfig(BackendConfig):
    """
    Web backend configuration - AI Deep Research
    
    Attributes:
        enabled: Whether web backend is enabled
        timeout: Default timeout for web operations (seconds)
        max_retries: Maximum number of retry attempts
    
    Note:
        All web-specific parameters (API key, base URL) are loaded from 
        environment variables or use default values in WebSession:
        - OPENROUTER_API_KEY: API key for deep research (required)
        - Deep research base URL defaults to "https://openrouter.ai/api/v1"
    """
    pass


class MCPConfig(BackendConfig):
    """MCP backend configuration"""
    sandbox: bool = Field(False, description="Whether to enable sandbox")
    auto_initialize: bool = Field(True, description="Whether to auto initialize")
    eager_sessions: bool = Field(False, description="Whether to eagerly create sessions for all servers on initialization")
    retry_interval: float = Field(2.0, ge=0.1, le=60.0, description="Wait time between retries in seconds")
    servers: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="MCP servers configuration, loaded from config_mcp.json")
    sse_read_timeout: float = Field(300.0, ge=1.0, le=3600.0, description="SSE read timeout in seconds for HTTP/Sandbox connectors")


class GUIConfig(BackendConfig):
    """
    GUI backend configuration"""
    retry_interval: float = Field(5.0, ge=0.1, le=60.0, description="Wait time between retries in seconds")
    driver_type: str = Field("pyautogui", description="GUI driver type")
    failsafe: bool = Field(False, description="Whether to enable pyautogui failsafe mode")
    screenshot_on_error: bool = Field(True, description="Whether to capture screenshot on error")
    pkgs_prefix: str = Field(
        "import pyautogui; import time; pyautogui.FAILSAFE = {failsafe}; {command}",
        description="Python command prefix for pyautogui setup"
    )


class ToolSearchConfig(BaseModel):
    """Tool search and ranking configuration"""
    embedding_model: str = Field(
        "BAAI/bge-small-en-v1.5",
        description="Embedding model name for semantic search"
    )
    max_tools: int = Field(
        20,
        ge=1,
        le=1000,
        description="Maximum number of tools to return from search"
    )
    search_mode: str = Field(
        "hybrid",
        description="Default search mode: semantic, keyword, or hybrid"
    )
    enable_llm_filter: bool = Field(
        True,
        description="Whether to use LLM for backend/server filtering"
    )
    llm_filter_threshold: int = Field(
        50,
        ge=1,
        le=1000,
        description="Only apply LLM filter when tool count exceeds this threshold"
    )
    enable_cache_persistence: bool = Field(
        False,
        description="Whether to persist embeddings to disk"
    )
    cache_dir: Optional[str] = Field(
        None,
        description="Directory for embedding cache. None means use default ~/.mcpnext/embedding_cache"
    )
    
    @field_validator('search_mode')
    @classmethod
    def validate_search_mode(cls, v):
        valid_modes = ['semantic', 'keyword', 'hybrid']
        if v.lower() not in valid_modes:
            raise ValueError(f"Search mode must be one of {valid_modes}, got: {v}")
        return v.lower()


class GroundingConfig(BaseModel):
    """
    Main configuration for Grounding module.
    
    Contains configuration for all grounding backends and grounding-level settings.
    Note: Local server connection uses defaults or environment variables (LOCAL_SERVER_URL).
    """
    # Backend configurations
    shell: ShellConfig = Field(default_factory=ShellConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    gui: GUIConfig = Field(default_factory=GUIConfig)
    system: BackendConfig = Field(default_factory=BackendConfig)
    
    # Grounding-level settings
    tool_search: ToolSearchConfig = Field(default_factory=ToolSearchConfig)
    
    enabled_backends: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of enabled backends, each item: {'name': str, 'provider_cls': str}"
    )
    
    session_defaults: SessionConfig = Field(
        default_factory=lambda: SessionConfig(
            session_name="",
            backend_type=BackendType.SHELL,
            timeout=30,
            auto_reconnect=True,
            health_check_interval=30
        )
    )
    
    tool_cache_ttl: int = Field(
        300,
        ge=1,
        le=3600,
        description="Tool cache time-to-live in seconds"
    )
    tool_cache_maxsize: int = Field(
        300,
        ge=1,
        le=10000,
        description="Maximum number of tool cache entries"
    )
    
    debug: bool = Field(False, description="Debug mode")
    log_level: str = Field("INFO", description="Log level")
    security_policies: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        if v.upper() not in LOG_LEVELS:
            raise ValueError(f"Log level must be one of {LOG_LEVELS}, got: {v}")
        return v.upper()
    
    def get_backend_config(self, backend_type: str) -> BackendConfig:
        """Get configuration for specified backend"""
        name = backend_type.lower()
        if not hasattr(self, name):
            from mcpnext.utils.logging import Logger
            logger = Logger.get_logger(__name__)
            logger.warning(f"Unknown backend type: {backend_type}")
            return BackendConfig()
        return getattr(self, name)
    
    def get_security_policy(self, backend_type: str) -> SecurityPolicy:
        global_policy = self.security_policies.get("global", {})
        backend_policy = self.security_policies.get("backend", {}).get(backend_type.lower(), {})
        merged_policy = {**global_policy, **backend_policy}
        return SecurityPolicy.from_dict(merged_policy)


__all__ = [
    "BackendConfig",
    "ShellConfig",
    "WebConfig",
    "MCPConfig",
    "GUIConfig",
    "ToolSearchConfig",
    "GroundingConfig",
]