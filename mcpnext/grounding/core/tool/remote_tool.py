"""
RemoteTool.
Wrapper around a connector that calls a remote tool.
"""
from mcpnext.utils.logging import Logger
from ..types import BackendType, ToolResult, ToolSchema, ToolStatus
from .base import BaseTool
from mcpnext.grounding.core.transport.connectors import BaseConnector

logger = Logger.get_logger(__name__)


class RemoteTool(BaseTool):
    backend_type = BackendType.NOT_SET

    def __init__(
        self,
        connector: BaseConnector,
        remote_name: str,
        schema: ToolSchema | None = None,
        *,
        verbose: bool = False,
        backend: BackendType = BackendType.NOT_SET,
    ):
        self._conn = connector
        self._remote_name = remote_name
        self.backend_type = backend
        super().__init__(schema=schema, verbose=verbose)

    async def _arun(self, **kwargs):
        raw = await self._conn.invoke(self._remote_name, kwargs)
        
        if isinstance(raw, dict):
            import json
            try:
                content = json.dumps(raw, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                content = str(raw)
        elif isinstance(raw, (list, tuple)):
            import json
            try:
                content = json.dumps(raw, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                content = str(raw)
        elif isinstance(raw, (int, float, bool)):
            content = str(raw)
        elif isinstance(raw, str):
            content = raw
        else:
            content = str(raw)
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            content=content,  
            metadata={"raw": raw}, 
        )