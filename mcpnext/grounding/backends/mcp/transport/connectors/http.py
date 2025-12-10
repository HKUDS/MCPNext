"""
HTTP connector for MCP implementations.

This module provides a connector for communicating with MCP implementations
through HTTP APIs with SSE or Streamable HTTP for transport.
"""

import httpx
from mcp import ClientSession

from mcpnext.utils.logging import Logger
from mcpnext.grounding.core.transport.task_managers.base import BaseConnectionManager
from mcpnext.grounding.backends.mcp.transport.task_managers import SseConnectionManager, StreamableHttpConnectionManager
from mcpnext.grounding.backends.mcp.transport.connectors.base import MCPBaseConnector

logger = Logger.get_logger(__name__)


class HttpConnector(MCPBaseConnector):
    """Connector for MCP implementations using HTTP transport with SSE or streamable HTTP.

    This connector uses HTTP/SSE or streamable HTTP to communicate with remote MCP implementations,
    using a connection manager to handle the proper lifecycle management.
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
    ):
        """Initialize a new HTTP connector.

        Args:
            base_url: The base URL of the MCP HTTP API.
            auth_token: Optional authentication token.
            headers: Optional additional headers.
            timeout: Timeout for HTTP operations in seconds.
            sse_read_timeout: Timeout for SSE read operations in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.headers = headers or {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        
        # Create a placeholder connection manager (will be set up later in connect())
        # We use a placeholder here because the actual transport type (SSE vs Streamable HTTP)
        # can only be determined at runtime through server negotiation as per MCP specification
        from mcpnext.grounding.core.transport.task_managers import PlaceholderConnectionManager
        connection_manager = PlaceholderConnectionManager()
        super().__init__(connection_manager)

    async def _before_connect(self) -> None:
        """Negotiate transport type and set up the appropriate connection manager.
        
        Tries streamable HTTP first (new transport), falls back to SSE (old transport).
        This implements backwards compatibility per MCP specification.
        """
        self.transport_type = None
        connection_manager = None
        streamable_error = None

        # First, try the new streamable HTTP transport
        try:
            logger.debug(f"Attempting streamable HTTP connection to: {self.base_url}")
            connection_manager = StreamableHttpConnectionManager(
                self.base_url, self.headers, self.timeout, self.sse_read_timeout
            )

            # Test the connection by starting it
            read_stream, write_stream = await connection_manager.start()

            # Verify it works by testing ClientSession initialization
            test_client = ClientSession(read_stream, write_stream, sampling_callback=None)
            await test_client.__aenter__()

            try:
                await test_client.initialize()
                # Success! Clean up test client, stop the connection, and keep the manager
                await test_client.__aexit__(None, None, None)
                await connection_manager.stop()
                
                self.transport_type = "streamable HTTP"
                self._connection_manager = connection_manager
                logger.debug("Streamable HTTP transport selected")
                return
            except Exception as init_error:
                # Clean up the test client
                await test_client.__aexit__(None, None, None)
                raise init_error

        except Exception as e:
            streamable_error = e
            logger.debug(f"Streamable HTTP failed: {e}")

            # Clean up the failed connection manager
            if connection_manager:
                try:
                    await connection_manager.stop()
                except Exception:
                    pass

        # Determine if we should try SSE fallback
        should_fallback = False
        if isinstance(streamable_error, httpx.HTTPStatusError):
            if streamable_error.response.status_code in [404, 405]:
                should_fallback = True
        elif streamable_error and ("405 Method Not Allowed" in str(streamable_error) or "404 Not Found" in str(streamable_error)):
            should_fallback = True
        else:
            # For other errors, still try fallback
            should_fallback = True

        if should_fallback:
            try:
                # Fall back to the old SSE transport
                logger.debug(f"Attempting SSE fallback connection to: {self.base_url}")
                connection_manager = SseConnectionManager(
                    self.base_url, self.headers, self.timeout, self.sse_read_timeout
                )

                # Note: Don't start it yet, let the parent's connect() do that
                self.transport_type = "SSE"
                self._connection_manager = connection_manager
                logger.debug("SSE transport selected")
                return

            except Exception as sse_error:
                logger.error(
                    f"Both transport methods failed. Streamable HTTP: {streamable_error}, SSE: {sse_error}"
                )
                raise sse_error
        else:
            raise streamable_error

    async def _after_connect(self) -> None:
        """Create ClientSession and log success."""
        await super()._after_connect()
        logger.debug(f"Successfully connected to MCP implementation via {self.transport_type}: {self.base_url}")

    @property
    def public_identifier(self) -> str:
        """Get the identifier for the connector."""
        return {"type": self.transport_type, "base_url": self.base_url}
