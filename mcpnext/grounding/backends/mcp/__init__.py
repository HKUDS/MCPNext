"""
MCP Backend for MCPNext Grounding.

This module provides the MCP (Model Context Protocol) backend implementation
for the grounding framework. It includes:

- MCPProvider: Manages multiple MCP server sessions
- MCPSession: Handles individual MCP server connections
- MCPClient: High-level client for MCP server configuration
- MCPInstallerManager: Manages automatic installation of MCP dependencies
"""

from .provider import MCPProvider
from .session import MCPSession
from .client import MCPClient
from .installer import (
    MCPInstallerManager,
    get_global_installer,
    set_global_installer,
    MCPDependencyError,
    MCPCommandNotFoundError,
    MCPInstallationCancelledError,
    MCPInstallationFailedError,
)

__all__ = [
    "MCPProvider",
    "MCPSession",
    "MCPClient",
    "MCPInstallerManager",
    "get_global_installer",
    "set_global_installer",
    "MCPDependencyError",
    "MCPCommandNotFoundError",
    "MCPInstallationCancelledError",
    "MCPInstallationFailedError",
]