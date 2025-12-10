from __future__ import annotations

import asyncio
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcpnext.agents import GroundingAgent
from mcpnext.llm import LLMClient
from mcpnext.grounding.core.grounding_client import GroundingClient
from mcpnext.config import get_config, load_config
from mcpnext.recording import RecordingManager
from mcpnext.utils.logging import Logger

logger = Logger.get_logger(__name__)


@dataclass
class MCPNextConfig:
    # LLM Configuration
    llm_model: str = "anthropic/claude-sonnet-4-5"
    llm_enable_thinking: bool = False
    llm_timeout: float = 120.0
    llm_max_retries: int = 3
    llm_rate_limit_delay: float = 0.0
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Grounding Configuration
    grounding_config_path: Optional[str] = None
    grounding_max_iterations: int = 20
    grounding_system_prompt: Optional[str] = None
    
    # Backend Configuration
    backend_scope: Optional[List[str]] = None  # None = All backends ["shell", "gui", "mcp", "web", "system"]
    
    # Workspace Configuration
    workspace_dir: Optional[str] = None
    
    # Recording Configuration
    enable_recording: bool = False
    recording_backends: Optional[List[str]] = None
    recording_log_dir: str = "./logs/recordings"
    enable_screenshot: bool = True
    enable_video: bool = True
    
    # Logging Configuration
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.llm_model:
            raise ValueError("llm_model is required")
        
        logger.debug(f"MCPNextConfig initialized with model: {self.llm_model}")


class MCPNext:
    def __init__(self, config: Optional[MCPNextConfig] = None):
        self.config = config or MCPNextConfig()
        
        self._llm_client: Optional[LLMClient] = None
        self._grounding_client: Optional[GroundingClient] = None
        self._grounding_agent: Optional[GroundingAgent] = None
        self._recording_manager: Optional[RecordingManager] = None
        
        self._initialized = False
        self._running = False
        
        logger.debug("MCPNext instance created")
    
    async def initialize(self) -> None:
        if self._initialized:
            logger.warning("MCPNext already initialized")
            return
        
        logger.info("Initializing MCPNext...")
        
        try:
            self._llm_client = LLMClient(
                model=self.config.llm_model,
                enable_thinking=self.config.llm_enable_thinking,
                rate_limit_delay=self.config.llm_rate_limit_delay,
                max_retries=self.config.llm_max_retries,
                timeout=self.config.llm_timeout,
                **self.config.llm_kwargs
            )
            logger.info(f"✓ LLM Client: {self.config.llm_model}")
            
            if self.config.grounding_config_path:
                grounding_config = load_config(self.config.grounding_config_path)
            else:
                grounding_config = get_config()
            
            self._grounding_client = GroundingClient(config=grounding_config)
            await self._grounding_client.initialize_all_providers()
            
            backends = list(self._grounding_client.list_providers().keys())
            logger.info(f"✓ Grounding Client: {len(backends)} backends")
            logger.debug(f"  Available backends: {[b.value for b in backends]}")
            
            if self.config.enable_recording:
                self._recording_manager = RecordingManager(
                    enabled=True,
                    task_id="",
                    log_dir=self.config.recording_log_dir,
                    backends=self.config.recording_backends,
                    enable_screenshot=self.config.enable_screenshot,
                    enable_video=self.config.enable_video,
                    agent_name="MCPNext",
                )
                logger.info(f"✓ Recording enabled: {len(self._recording_manager.backends or [])} backends")
            
            backend_scope = self.config.backend_scope or ["gui", "shell", "mcp", "web", "system"]
            self._grounding_agent = GroundingAgent(
                name="MCPNext-GroundingAgent",
                backend_scope=backend_scope,
                llm_client=self._llm_client,
                grounding_client=self._grounding_client,
                recording_manager=self._recording_manager,
                system_prompt=self.config.grounding_system_prompt,
                max_iterations=self.config.grounding_max_iterations,
            )
            logger.info(f"✓ GroundingAgent: {', '.join(backend_scope)}")
            
            self._initialized = True
            logger.info("="*60)
            logger.info("MCPNext ready to use!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Failed to initialize MCPNext: {e}")
            await self.cleanup()
            raise
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        workspace_dir: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError(
                "MCPNext not initialized. "
                "Call await tool_layer.initialize() first or use async with."
            )
        
        if self._running:
            raise RuntimeError("MCPNext is already running a task.")
        
        logger.info("="*60)
        logger.info(f"Task: {task[:100]}...")
        logger.info("="*60)
        
        self._running = True
        start_time = asyncio.get_event_loop().time()
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        try:
            execution_context = context or {}
            execution_context["task_id"] = task_id
            execution_context["instruction"] = task
            
            if max_iterations is not None:
                execution_context["max_iterations"] = max_iterations
            
            if self._recording_manager:
                if self._recording_manager.recording_status:
                    await self._recording_manager.stop()
                    logger.debug("Stopped previous recording session")
                
                self._recording_manager.task_id = task_id
                await self._recording_manager.start()
                logger.info(f"Recording started: {task_id}")
            
            if workspace_dir:
                execution_context["workspace_dir"] = workspace_dir
                logger.info(f"Workspace: {workspace_dir}")
            elif self.config.workspace_dir:
                execution_context["workspace_dir"] = self.config.workspace_dir
                logger.info(f"Workspace: {self.config.workspace_dir}")
            elif self._recording_manager and self._recording_manager.trajectory_dir:
                execution_context["workspace_dir"] = self._recording_manager.trajectory_dir
                logger.info(f"Workspace: {execution_context['workspace_dir']}")
            else:
                import tempfile
                from pathlib import Path
                workspace = Path(tempfile.gettempdir()) / "mcpnext_workspace" / task_id
                workspace.mkdir(parents=True, exist_ok=True)
                execution_context["workspace_dir"] = str(workspace)
                logger.info(f"Workspace: {execution_context['workspace_dir']}")
            
            logger.info(f"Executing with GroundingAgent (max {max_iterations or self.config.grounding_max_iterations} iterations)...")
            
            result = await self._grounding_agent.process(execution_context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            final_result = {
                **result,
                "task_id": task_id,
                "execution_time": execution_time,
            }
            
            status = result.get('status', 'unknown')
            iterations = result.get('iterations', 0)
            tool_count = len(result.get('tool_executions', []))
            
            logger.info("="*60)
            if status == "success":
                logger.info(
                    f"Task completed successfully! "
                    f"({iterations} iterations, {tool_count} tool calls, {execution_time:.2f}s)"
                )
            elif status == "incomplete":
                logger.warning(
                    f"Task incomplete after {iterations} iterations. "
                    f"Consider increasing max_iterations."
                )
            else:
                logger.error(f"Task failed: {result.get('error', 'Unknown error')}")
            logger.info("="*60)
            
            return final_result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            tb = traceback.format_exc(limit=10)
            logger.error(f"Task execution failed: {e}", exc_info=True)
            
            return {
                "status": "error",
                "error": str(e),
                "traceback": tb,
                "response": f"Task execution error: {str(e)}",
                "execution_time": execution_time,
                "task_id": task_id,
                "iterations": 0,
                "tool_executions": [],
            }
        
        finally:
            if self._recording_manager and self._recording_manager.recording_status:
                try:
                    await self._recording_manager.stop()
                    logger.debug(f"Recording stopped: {task_id}")
                except Exception as e:
                    logger.warning(f"Failed to stop recording: {e}")
            
            self._running = False
    
    async def cleanup(self) -> None:
        """
        Close all sessions and release resources.
        Automatically called when using context manager.
        """
        logger.info("Cleaning up MCPNext resources...")
        
        try:
            if self._grounding_client:
                await self._grounding_client.close_all_sessions()
                logger.debug("All grounding sessions closed")
            
            if self._recording_manager and self._recording_manager.recording_status:
                try:
                    await self._recording_manager.stop()
                    logger.debug("Recording manager stopped")
                except Exception as e:
                    logger.warning(f"Failed to stop recording: {e}")
            
            self._initialized = False
            self._running = False
            
            logger.info("MCPNext cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
    
    def is_initialized(self) -> bool:
        return self._initialized
    
    def is_running(self) -> bool:
        return self._running
    
    def get_config(self) -> MCPNextConfig:
        return self.config
    
    def list_backends(self) -> List[str]:
        if not self._initialized:
            raise RuntimeError("MCPNext not initialized")
        return [backend.value for backend in self._grounding_client.list_providers().keys()]
    
    def list_sessions(self) -> List[str]:
        if not self._initialized:
            raise RuntimeError("MCPNext not initialized")
        return self._grounding_client.list_sessions()
    
    async def __aenter__(self):
        """Context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.cleanup()
        return False
    
    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        if self._running:
            status = "running"
        backends = ", ".join(self.config.backend_scope) if self.config.backend_scope else "all"
        return f"<MCPNext(status={status}, backends={backends}, model={self.config.llm_model})>"