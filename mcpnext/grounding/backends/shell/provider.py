from mcpnext.grounding.core.provider import Provider
from mcpnext.grounding.core.types import BackendType, SessionConfig
from .session import ShellSession
from .transport.connector import ShellConnector
from mcpnext.config import get_config
from mcpnext.config.utils import get_config_value
from mcpnext.utils.logging import Logger

logger = Logger.get_logger(__name__)


class ShellProvider(Provider[ShellSession]):
    
    DEFAULT_SID = BackendType.SHELL.value
    
    def __init__(self, config: dict | None = None):
        super().__init__(BackendType.SHELL, config)
        # Note: _setup_security_policy() is already called by parent class __init__

    def _setup_security_policy(self, config: dict | None = None):
        security_policy = get_config().get_security_policy(self.backend_type.value)
    
        if config:
            security_config = get_config_value(config, "security", None)
            if security_config:
                for key, value in security_config.items():
                    if hasattr(security_policy, key):
                        setattr(security_policy, key, value)
            
            sandbox_enabled = get_config_value(config, "sandbox_enabled", None)
            if sandbox_enabled is not None:
                security_policy.sandbox_enabled = sandbox_enabled
        
        logger.info(f"Shell security policy: allow_shell_commands={security_policy.allow_shell_commands}, "
                   f"blocked_commands={security_policy.blocked_commands}")
        
        self.security_manager.set_backend_policy(BackendType.SHELL, security_policy)

    async def initialize(self) -> None:
        if not self.is_initialized:
            await self.create_session(SessionConfig(
                session_name=self.DEFAULT_SID,
                backend_type=BackendType.SHELL,
                connection_params={}
            ))
            self.is_initialized = True

    async def create_session(self, session_config: SessionConfig) -> ShellSession:
        sid = self.DEFAULT_SID
        if sid in self._sessions:
            return self._sessions[sid]
        
        # Load shell backend configuration
        shell_config = get_config().get_backend_config("shell")
        
        # Create connector with config parameters
        connector = ShellConnector(
            vm_ip=get_config_value(session_config.connection_params, "vm_ip", "localhost"),
            port=get_config_value(session_config.connection_params, "port", shell_config.default_port),
            retry_times=shell_config.max_retries,
            retry_interval=shell_config.retry_interval,
            security_manager=self.security_manager
        )
        
        # Create session with config parameters
        session = ShellSession(
            connector=connector,
            session_id=sid,
            security_manager=self.security_manager,
            default_working_dir=shell_config.working_dir,
            default_env=shell_config.env,
            default_conda_env=shell_config.conda_env
        )
        
        await session.initialize()
        self._sessions[sid] = session
        return session

    async def close_session(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess:
            await sess.disconnect()