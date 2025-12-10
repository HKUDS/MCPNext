"""
Streamable HTTP connection management for MCP implementations.

This module provides a connection manager for streamable HTTP-based MCP connections
that ensures proper task isolation and resource cleanup.
"""

from datetime import timedelta
from typing import Any, Tuple
from contextlib import asynccontextmanager

from mcp.client.streamable_http import streamablehttp_client
from mcpnext.utils.logging import Logger
from mcpnext.grounding.core.transport.task_managers import (
    AsyncContextConnectionManager,
)

logger = Logger.get_logger(__name__)


def _make_shim():
    @asynccontextmanager
    async def _shim(**kw):
        async with streamablehttp_client(**kw) as (r, w, _sid_cb):
            yield (r, w)       
    return _shim


class StreamableHttpConnectionManager(
    AsyncContextConnectionManager[Tuple[Any, Any], ...]
):
    """
    MCP Streamable-HTTP connection manager based on the generic
    AsyncContextConnectionManager.  Extra session-id callback returned by the
    SDK is discarded by the shim above.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 5,
        read_timeout: float = 60 * 5,
    ):
        shim = _make_shim()              
        super().__init__(
            shim,
            url=url,
            headers=headers or {},
            timeout=timedelta(seconds=timeout),
            sse_read_timeout=timedelta(seconds=read_timeout),
        )
        self.url = url
        self.headers = headers or {}
        logger.debug("StreamableHttpConnectionManager init url=%s", url)