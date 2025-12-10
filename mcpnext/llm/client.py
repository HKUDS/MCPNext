import litellm
import json
import asyncio
import time
from typing import List, Sequence, Union, Dict, Optional
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionToolParam

from mcpnext.grounding.core.types import ToolSchema, ToolResult, ToolStatus
from mcpnext.grounding.core.tool import BaseTool
from mcpnext.utils.logging import Logger

load_dotenv()
# litellm._turn_on_debug()

logger = Logger.get_logger(__name__)

def _schema_to_openai(schema: ToolSchema) -> ChatCompletionToolParam:
    """Convert ToolSchema to OpenAI ChatCompletion tool format"""
    function_def = {
        "name": schema.name,
        "description": schema.description or "",
    }
    
    # Only add parameters field if tool has parameters
    if schema.parameters:
        function_def["parameters"] = schema.parameters
    
    return { 
        "type": "function",
        "function": function_def
    }
    
def _prepare_tools_for_llmclient(
    tools: List[BaseTool] | None,
    fmt: str = "openai",
) -> Sequence[Union[ToolSchema, ChatCompletionToolParam]]:
    """Convert BaseTool list to LLMClient usable format
    
    Args:
        tools: BaseTool instance list (should be obtained from GroundingClient and bound to runtime_info)
                if None or empty list, return empty list
        fmt: output format, "openai" for OpenAI format
    """
    if not tools:
        return []
    
    if fmt == "openai":
        return [_schema_to_openai(tool.schema) for tool in tools]
    return [tool.schema for tool in tools]

def _tool_result_to_message(result: ToolResult, *, tool_call_id: str, tool_name: str) -> Dict:
    """Convert ToolResult to LLMClient usable message format

    Args:
        result: Tool execution result
        tool_call_id: OpenAI tool_call ID
        tool_name: Tool name
        
    Returns:
        OpenAI ChatCompletion tool message (text only)
    """
    if result.is_error:
        text_content = f"[ERROR] {result.error or 'unknown error'}"
    else:
        text_content = (
            result.content
            if isinstance(result.content, str)
            else json.dumps(result.content, ensure_ascii=False, default=str)
        )
    
    return {
        "role": "tool",
        "name": tool_name,
        "content": text_content,
        "tool_call_id": tool_call_id,
    }

async def _execute_tool_call(
    tool: BaseTool,
    openai_tool_call: Dict,
) -> ToolResult:
    """Execute LLMClient returned tool_call

    Args:
        tool: BaseTool instance (must be obtained from GroundingClient and bound to runtime_info)
        openai_tool_call: LLMClient usable tool_call object, contains id, type, function etc. fields
    """
    if not tool.is_bound:
        raise ValueError(
            f"Tool '{tool.schema.name}' is not bound to runtime_info. "
            f"Please ensure tools are obtained from GroundingClient.list_tools() "
            f"with bind_runtime_info=True"
        )
    
    func = openai_tool_call["function"]
    arguments = func.get("arguments", "{}")
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    
    # Filter out parameters that are not in the tool's schema
    if isinstance(arguments, dict) and tool.schema.parameters:
        # Get valid parameter names from tool schema (JSON Schema format)
        schema_params = tool.schema.parameters
        valid_params = set()
        
        if isinstance(schema_params, dict) and "properties" in schema_params:
            valid_params = set(schema_params["properties"].keys())
        
        # Check for invalid parameters
        invalid_params = []
        for param_name in list(arguments.keys()):
            if param_name == "skip_visual_analysis":
                invalid_params.append(param_name)
                continue
            
            # Check if parameter is in the tool's schema
            if valid_params and param_name not in valid_params:
                invalid_params.append(param_name)
        
        # Remove invalid parameters
        for param in invalid_params:
            arguments.pop(param)
            logger.debug(
                f"Removed parameter '{param}' from {tool.schema.name} "
                f"(not in tool schema)"
            )

    return await tool.invoke(
        parameters=arguments,
        keep_session=True
    )


class LLMClient:
    """LLMClient class for single round call"""
    def __init__(
        self, 
        model: str = "anthropic/claude-sonnet-4-5", 
        enable_thinking: bool = False,
        rate_limit_delay: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 120.0,
        **litellm_kwargs
    ):
        """
        Args:
            model: LLM model identifier
            enable_thinking: Whether to enable extended thinking mode
            rate_limit_delay: Minimum delay between API calls in seconds (0 = no delay)
            max_retries: Maximum number of retries on rate limit errors
            retry_delay: Initial delay between retries in seconds (exponential backoff)
            timeout: Request timeout in seconds (default: 120s)
            **litellm_kwargs: Additional litellm parameters
        """
        self.model = model
        self.enable_thinking = enable_thinking
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.litellm_kwargs = litellm_kwargs
        self._logger = Logger.get_logger(__name__)
        self._last_call_time = 0.0
    
    async def _rate_limit(self):
        """Apply rate limiting by adding delay between API calls"""
        if self.rate_limit_delay > 0:
            current_time = time.time()
            time_since_last_call = current_time - self._last_call_time
            
            if time_since_last_call < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last_call
                self._logger.debug(f"Rate limiting: waiting {sleep_time:.2f}s before next API call")
                await asyncio.sleep(sleep_time)
            
            self._last_call_time = time.time()
    
    async def _call_with_retry(self, **completion_kwargs):
        """Call LLM with backoff retry on rate limit errors
        
        Timeout and retry strategy:
        - Single call timeout: self.timeout (default 120s)
        - Rate limit retry delays: 60s, 90s, 120s
        - Total max time: timeout * max_retries + sum(retry_delays)
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Add timeout to the completion call
                response = await asyncio.wait_for(
                    litellm.acompletion(**completion_kwargs),
                    timeout=self.timeout
                )
                return response
            except asyncio.TimeoutError:
                self._logger.error(
                    f"LLM call timed out after {self.timeout}s (attempt {attempt + 1}/{self.max_retries})"
                )
                last_exception = TimeoutError(f"LLM call timed out after {self.timeout}s")
                if attempt < self.max_retries - 1:
                    # Retry on timeout with shorter delay
                    self._logger.info(f"Retrying after {self.retry_delay}s delay...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    raise last_exception
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if it's a retryable error
                is_rate_limit = any(
                    keyword in error_str 
                    for keyword in ['rate limit', 'rate_limit', 'too many requests', '429']
                )
                
                is_overloaded = any(
                    keyword in error_str
                    for keyword in ['overloaded', '500', '502', '503', '504', 'internal server error', 'service unavailable']
                )
                
                if attempt < self.max_retries - 1 and (is_rate_limit or is_overloaded):
                    # Determine backoff delay based on error type
                    if is_rate_limit:
                        # Use longer backoff for rate limits to cross rate limit windows
                        backoff_delay = 60 + (attempt * 30)  # 60s, 90s, 120s
                        error_type = "Rate limit"
                    else:  # is_overloaded
                        # Use exponential backoff for server errors
                        backoff_delay = min(5 * (2 ** attempt), 60)  # 5s, 10s, 20s, max 60s
                        error_type = "Server overload"
                    
                    self._logger.warning(
                        f"{error_type} error (attempt {attempt + 1}/{self.max_retries}), "
                        f"waiting {backoff_delay}s before retry..."
                    )
                    await asyncio.sleep(backoff_delay)
                    continue
                else:
                    # Not a retryable error, or max retries reached
                    if attempt >= self.max_retries - 1:
                        self._logger.error(f"Max retries ({self.max_retries}) reached, giving up")
                    raise
        
        raise last_exception
    
    async def complete(
        self,
        messages: List[Dict] | str, 
        tools: List[BaseTool] | None = None,
        execute_tools: bool = True,
        summary_prompt: Optional[str] = None,
        tool_result_callback: Optional[callable] = None,
        **kwargs
    ) -> Dict:
        """
        Single-round LLM call with optional tool execution.
        
        Args:
            messages: conversation history (List[Dict] for standard OpenAI format, or str for text format)
            tools: BaseTool instance list (must be obtained from GroundingClient and bound to runtime_info)
                if None or empty list, only perform conversation, no tools
            execute_tools: if LLM returns tool_calls, whether to automatically execute tools
            summary_prompt: Optional custom prompt for requesting iteration summary. 
                If provided, will request summary after tool execution.
                If None, no summary will be requested.
            tool_result_callback: Optional async callback to process tool results after execution.
                Signature: async def callback(result: ToolResult, tool_name: str, tool_call: Dict, backend: str) -> ToolResult
            **kwargs: additional parameters for litellm completion
        """
        # 1. Process messages
        if isinstance(messages, str):
            current_messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            current_messages = messages.copy()
        else:
            raise ValueError("messages must be List[Dict] or str")
        
        # 2. prepare base litellm completion kwargs
        completion_kwargs = {
            "model": kwargs.get("model", self.model),
            **self.litellm_kwargs,
        }
        
        # Add thinking/reasoning_effort only if explicitly enabled and not using tools
        enable_thinking = kwargs.get("enable_thinking", self.enable_thinking)
        
        # 3. if tools are provided, add them to the request
        llm_tools = None
        if tools:
            llm_tools = _prepare_tools_for_llmclient(tools, fmt="openai")
            if llm_tools:
                completion_kwargs["tools"] = llm_tools
                completion_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")
                # Disable thinking when using tools to avoid format conflicts
                enable_thinking = False
                self._logger.debug(f"Prepared {len(llm_tools)} tools for LLM")
            else:
                self._logger.warning("Tools provided but none could be prepared for LLM")
        
        # Add thinking parameters if enabled
        if enable_thinking:
            completion_kwargs["reasoning_effort"] = kwargs.get("reasoning_effort", "medium")
        
        # 4. Apply rate limiting
        await self._rate_limit()
        
        # 5. Call LLM with retry (single round)
        completion_kwargs["messages"] = current_messages
        response = await self._call_with_retry(**completion_kwargs)
        
        if not response.choices:
            raise ValueError("LLM response has no choices")
        
        response_message = response.choices[0].message
        
        # 6. Build assistant message
        assistant_message = {
            "role": "assistant",
            "content": response_message.content or "",
        }
        
        tool_calls = getattr(response_message, 'tool_calls', None)
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in tool_calls
            ]
        
        # Add assistant message to conversation
        current_messages.append(assistant_message)
        
        # 7. Execute tools if requested
        tool_results = []
        if execute_tools and tool_calls and tools:
            self._logger.info(f"Executing {len(tool_calls)} tool calls...")
            tool_map = {tool.schema.name: tool for tool in tools}
            
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                
                # Extract tool metadata and check visual analysis request
                tool_obj = tool_map.get(tool_name)
                backend = None
                server_name = None
                
                if tool_obj:
                    try:
                        # Prefer runtime_info if bound
                        if getattr(tool_obj, 'is_bound', False) and getattr(tool_obj, 'runtime_info', None):
                            backend = tool_obj.runtime_info.backend.value
                            server_name = tool_obj.runtime_info.server_name
                        else:
                            backend = tool_obj.backend_type.value if hasattr(tool_obj, 'backend_type') else None
                    except Exception:
                        pass
                
                # Log tool execution
                try:
                    if isinstance(tool_call.function.arguments, str):
                        safe_args_str = tool_call.function.arguments.strip() or "{}"
                        args = json.loads(safe_args_str)
                    else:
                        args = tool_call.function.arguments
                    
                    args_str = json.dumps(args, ensure_ascii=False)[:200]
                    self._logger.info(f"Calling {tool_name} with args: {args_str}")
                except:
                    pass
                
                if tool_name not in tool_map:
                    result = ToolResult(
                        status=ToolStatus.ERROR,
                        error=f"Tool '{tool_name}' not found"
                    )
                else:
                    try:
                        result = await _execute_tool_call(
                            tool=tool_map[tool_name],
                            openai_tool_call={
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        )

                        # Apply tool result callback if provided
                        if tool_result_callback and not result.is_error:
                            try:
                                result = await tool_result_callback(
                                    result=result,
                                    tool_name=tool_name,
                                    tool_call=tool_call,
                                    backend=backend
                                )
                            except Exception as e:
                                self._logger.warning(f"Tool result callback failed for {tool_name}: {e}")
                    except Exception as e:
                        result = ToolResult(
                            status=ToolStatus.ERROR,
                            error=str(e)
                        )
                
                tool_message = _tool_result_to_message(
                    result, 
                    tool_call_id=tool_call.id, 
                    tool_name=tool_name
                )
                current_messages.append(tool_message)
                
                # Store result
                tool_results.append({
                    "tool_call": tool_call,
                    "result": result,
                    "message": tool_message,
                    "backend": backend,
                    "server_name": server_name,
                })
            
            self._logger.info(f"Tool execution completed, {len(tool_results)} tools executed")
        
        # 8. Request summary if provided and tools were executed
        iteration_summary = None
        
        if summary_prompt and tool_results:
            self._logger.debug("Requesting iteration summary from LLM")
            summary_message = {
                "role": "system",
                "content": summary_prompt
            }
            current_messages.append(summary_message)
            
            # Apply rate limiting before summary call
            await self._rate_limit()
            
            # Call LLM to generate summary (without tools)
            summary_kwargs = {
                **self.litellm_kwargs,
                "model": self.model,
                "messages": current_messages,
                "tools": [], 
                "tool_choice": "none",
            }
            
            summary_response = await self._call_with_retry(**summary_kwargs)
            
            if summary_response.choices:
                summary_message = summary_response.choices[0].message
                iteration_summary = summary_message.content or ""
                
                # Add summary response to messages
                current_messages.append({
                    "role": "assistant",
                    "content": iteration_summary
                })
                
                self._logger.debug(f"Generated iteration summary: {iteration_summary[:100]}...")
                
        # 9. Return single-round result        
        return {
            "message": assistant_message,
            "tool_results": tool_results,
            "messages": current_messages,
            "has_tool_calls": bool(tool_calls),
            "iteration_summary": iteration_summary
        }
    
    @staticmethod
    def format_messages_to_text(messages: List[Dict]) -> str:
        """Format conversation history to readable text (for logging/debugging)"""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            formatted += f"[{role}]\n{content}\n\n"
        return formatted