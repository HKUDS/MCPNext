"""
Tool converter for MCP.

This module provides utilities to convert MCP tools to BaseTool instances.
"""

from typing import Any, Dict
from mcp.types import Tool as MCPTool

from mcpnext.grounding.core.tool import BaseTool, RemoteTool
from mcpnext.grounding.core.types import BackendType, ToolSchema
from mcpnext.grounding.core.transport.connectors import BaseConnector
from mcpnext.utils.logging import Logger

logger = Logger.get_logger(__name__)


def convert_mcp_tool_to_base_tool(
    mcp_tool: MCPTool, 
    connector: BaseConnector
) -> BaseTool:
    """
    Convert an MCP Tool to a BaseTool (RemoteTool) instance.
    
    This function extracts the tool schema from an MCP tool object and creates
    a RemoteTool that can be used within the grounding framework.
    
    Args:
        mcp_tool: MCP Tool object from the MCP SDK
        connector: Connector instance for communicating with the MCP server
        
    Returns:
        RemoteTool instance wrapping the MCP tool
    """
    # Extract tool metadata
    tool_name = mcp_tool.name
    tool_description = getattr(mcp_tool, 'description', None) or ""
    
    # Convert MCP input schema to our parameter schema format
    input_schema: Dict[str, Any] = {}
    if hasattr(mcp_tool, 'inputSchema') and mcp_tool.inputSchema:
        input_schema = mcp_tool.inputSchema
    
    # Create ToolSchema
    schema = ToolSchema(
        name=tool_name,
        description=tool_description,
        parameters=input_schema,
        backend_type=BackendType.MCP,
    )
    
    # Create and return RemoteTool (pass description for content extraction)
    remote_tool = RemoteTool(
        connector=connector,
        remote_name=tool_name,
        schema=schema,
        backend=BackendType.MCP,
        tool_description=tool_description,  # Pass description for classification
    )
    
    logger.debug(f"Converted MCP tool '{tool_name}' to RemoteTool")
    return remote_tool