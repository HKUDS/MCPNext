from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from anytool.agents.base import BaseAgent
from anytool.grounding.core.types import BackendType, ToolResult
from anytool.platform.screenshot import ScreenshotClient
from anytool.prompts import GroundingAgentPrompts
from anytool.utils.logging import Logger

if TYPE_CHECKING:
    from anytool.llm import LLMClient
    from anytool.grounding.core.grounding_client import GroundingClient
    from anytool.recording import RecordingManager

logger = Logger.get_logger(__name__)


class GroundingAgent(BaseAgent):
    def __init__(
        self,
        name: str = "GroundingAgent",
        backend_scope: Optional[List[str]] = None,
        llm_client: Optional[LLMClient] = None,
        grounding_client: Optional[GroundingClient] = None,
        recording_manager: Optional[RecordingManager] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 15,
        visual_analysis_timeout: float = 30.0,
    ) -> None:
        """
        Initialize the Grounding Agent.
        
        Args:
            name: Agent name
            backend_scope: List of backends this agent can access (None = all available)
            llm_client: LLM client for reasoning
            grounding_client: GroundingClient for tool execution
            recording_manager: RecordingManager for recording execution
            system_prompt: Custom system prompt
            max_iterations: Maximum LLM reasoning iterations for self-correction
            visual_analysis_timeout: Timeout for visual analysis LLM calls in seconds
        """
        super().__init__(
            name=name,
            backend_scope=backend_scope or ["gui", "shell", "mcp", "web", "system"],
            llm_client=llm_client,
            grounding_client=grounding_client,
            recording_manager=recording_manager
        )
       
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._max_iterations = max_iterations
        self._visual_analysis_timeout = visual_analysis_timeout
        
        logger.info(f"Grounding Agent initialized: {name}")
        logger.info(f"Backend scope: {self._backend_scope}")
        logger.info(f"Max iterations: {self._max_iterations}")
        logger.info(f"Visual analysis timeout: {self._visual_analysis_timeout}s")
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task execution request with multi-round iteration control.
        """
        instruction = context.get("instruction", "")
        if not instruction:
            logger.error("Grounding Agent: No instruction provided")
            return {"error": "No instruction provided", "status": "error"}
        
        logger.info(f"Grounding Agent: Processing instruction at step {self.step}")
        
        # Exist workspace files check
        workspace_info = await self._check_workspace_artifacts(context)
        if workspace_info["has_files"]:
            context["workspace_artifacts"] = workspace_info
            logger.info(f"Workspace has {len(workspace_info['files'])} existing files: {workspace_info['files']}")
        
        # Get available tools (auto-search with cap)
        tools = await self._get_available_tools(instruction)
        
        # Initialize iteration state
        max_iterations = context.get("max_iterations", self._max_iterations)
        current_iteration = 0
        all_tool_results = []
        
        # Build initial messages
        messages = self.construct_messages(context)
        
        try:
            while current_iteration < max_iterations:
                current_iteration += 1
                logger.info(f"Grounding Agent: Iteration {current_iteration}/{max_iterations}")
                
                # Build iteration summary prompt for LLMClient
                # This will be used by LLMClient to automatically generate summary after tool execution
                iteration_summary_prompt = GroundingAgentPrompts.iteration_summary(
                    instruction=instruction,
                    iteration=current_iteration,
                    max_iterations=max_iterations
                ) if context.get("auto_execute", True) else None
                
                # Call LLMClient for single round
                # LLM will decide whether to call tools or finish with <COMPLETE>
                # If tools are executed, LLMClient will automatically generate iteration summary
                llm_response = await self._llm_client.complete(
                    messages=messages,
                    tools=tools if context.get("auto_execute", True) else None,
                    execute_tools=context.get("auto_execute", True),
                    summary_prompt=iteration_summary_prompt, 
                    tool_result_callback=self._visual_analysis_callback
                )
                
                # Update messages with LLM response
                messages = llm_response["messages"]
                
                # Collect tool results
                tool_results_this_iteration = llm_response.get("tool_results", [])
                if tool_results_this_iteration:
                    all_tool_results.extend(tool_results_this_iteration)

                llm_summary = llm_response.get("iteration_summary")
                if llm_summary:
                    logger.info(f"Iteration {current_iteration} summary: {llm_summary[:150]}...")
                
                assistant_message = llm_response.get("message", {})
                assistant_content = assistant_message.get("content", "")
                is_complete = GroundingAgentPrompts.TASK_COMPLETE in assistant_content
                
                if is_complete:
                    # Task is complete - LLM generated completion token
                    logger.info(f"Task completed at iteration {current_iteration} (LLM generated {GroundingAgentPrompts.TASK_COMPLETE})")
                    break
                
                else:
                    # LLM didn't generate <COMPLETE>, continue to next iteration
                    if tool_results_this_iteration:
                        logger.debug(f"Task in progress, LLM called {len(tool_results_this_iteration)} tools")
                    else:
                        logger.debug(f"Task in progress, LLM did not generate <COMPLETE>")
                    
                    # Remove guidance from previous iteration feedback (keep summary content)
                    self._remove_previous_guidance(messages)
                    
                    # Build feedback message for next iteration (with guidance)
                    feedback_msg = self._build_iteration_feedback(
                        iteration=current_iteration,
                        llm_summary=llm_summary,
                        add_guidance=True
                    )
                    
                    if feedback_msg:
                        messages.append(feedback_msg)
                        logger.debug(f"Added iteration {current_iteration} feedback with guidance")
                    
                    # Continue to next iteration
                    continue
            
            # Build final result
            result = await self._build_final_result(
                instruction=instruction,
                messages=messages,
                all_tool_results=all_tool_results,
                iterations=current_iteration,
                max_iterations=max_iterations
            )
            
            # Record agent action to recording manager
            if self._recording_manager:
                await self._record_agent_execution(result, instruction)
            
            # Increment step
            self.increment_step()
            
            logger.info(f"Grounding Agent: Execution completed with status: {result.get('status')}")
            return result
            
        except Exception as e:
            logger.error(f"Grounding Agent: Execution failed: {e}")
            result = {
                "error": str(e),
                "status": "error",
                "instruction": instruction,
                "iteration": current_iteration
            }
            self.increment_step()
            return result
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for the grounding agent."""
        return GroundingAgentPrompts.SYSTEM_PROMPT

    def construct_messages(
        self,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": self._system_prompt}]
        
        # Get instruction from context
        instruction = context.get("instruction", "")
        if not instruction:
            raise ValueError("context must contain 'instruction' field")
        
        # If only instruction, return directly
        if len(context) == 1 and "instruction" in context:
            messages.append({"role": "user", "content": instruction})
            return messages
        
        # Add workspace directory
        workspace_dir = context.get("workspace_dir")
        if workspace_dir:
            messages.append({
                "role": "system",
                "content": GroundingAgentPrompts.workspace_directory(workspace_dir)
            })
        
        # Add workspace artifacts information
        workspace_artifacts = context.get("workspace_artifacts")
        if workspace_artifacts and workspace_artifacts.get("has_files"):
            files = workspace_artifacts.get("files", [])
            matching_files = workspace_artifacts.get("matching_files", [])
            recent_files = workspace_artifacts.get("recent_files", [])
            
            if matching_files:
                artifact_msg = GroundingAgentPrompts.workspace_matching_files(matching_files)
            elif len(recent_files) >= 2:
                artifact_msg = GroundingAgentPrompts.workspace_recent_files(
                    total_files=len(files),
                    recent_files=recent_files
                )
            else:
                artifact_msg = GroundingAgentPrompts.workspace_file_list(files)
            
            messages.append({
                "role": "system",
                "content": artifact_msg
            })
        
        # User instruction
        messages.append({"role": "user", "content": instruction})
        
        return messages

    async def _get_available_tools(self, task_description: Optional[str]) -> List:
        """
        Retrieve tools with auto-search + cap to control prompt bloat.
        Falls back to returning all tools if search fails.
        """
        grounding_client = self.grounding_client
        if not grounding_client:
            return []

        backends = [BackendType(name) for name in self._backend_scope]

        try:
            tools = await grounding_client.get_tools_with_auto_search(
                task_description=task_description,
                backend=backends,
                use_cache=True,
            )
            logger.info(
                f"GroundingAgent selected {len(tools)} tools (auto-search) from {len(backends)} backends"
            )
            return tools
        except Exception as e:
            logger.warning(f"Auto-search tools failed, falling back to full list: {e}")

        # Fallback: fetch all tools (previous behaviour)
        all_tools = []
        for backend_name in self._backend_scope:
            try:
                backend_type = BackendType(backend_name)
                tools = await grounding_client.list_tools(backend=backend_type)
                all_tools.extend(tools)
                logger.debug(f"Retrieved {len(tools)} tools from backend: {backend_name}")
            except Exception as e:
                logger.debug(f"Could not get tools from {backend_name}: {e}")

        logger.info(
            f"GroundingAgent fallback retrieved {len(all_tools)} tools from {len(self._backend_scope)} backends"
        )
        return all_tools

    async def _visual_analysis_callback(
        self,
        result: ToolResult,
        tool_name: str,
        tool_call: Dict,
        backend: str
    ) -> ToolResult:
        """
        Callback for LLMClient to handle visual analysis after tool execution.
        """
        # 1. Check if LLM requested to skip visual analysis
        skip_visual_analysis = False
        try:
            arguments = tool_call.function.arguments
            if isinstance(arguments, str):
                args = json.loads(arguments.strip() or "{}")
            else:
                args = arguments
            
            if isinstance(args, dict) and args.get("skip_visual_analysis"):
                skip_visual_analysis = True
                logger.info(f"Visual analysis skipped for {tool_name} (meta-parameter set by LLM)")
        except Exception as e:
            logger.debug(f"Could not parse tool arguments: {e}")
        
        # 2. If skip requested, return original result
        if skip_visual_analysis:
            return result
        
        # 3. Check if this backend needs visual analysis
        if backend != "gui":
            return result
        
        # 4. Check if tool has visual data
        metadata = getattr(result, 'metadata', None)
        has_screenshots = metadata and (metadata.get("screenshot") or metadata.get("screenshots"))
        
        # 5. If no visual data, try to capture a screenshot
        if not has_screenshots:
            try:
                logger.info(f"No visual data from {tool_name}, capturing screenshot...")
                screenshot_client = ScreenshotClient()
                screenshot_bytes = await screenshot_client.capture()
                
                if screenshot_bytes:
                    # Add screenshot to result metadata
                    if metadata is None:
                        result.metadata = {}
                        metadata = result.metadata
                    metadata["screenshot"] = screenshot_bytes
                    has_screenshots = True
                    logger.info(f"Screenshot captured for visual analysis")
                else:
                    logger.warning("Failed to capture screenshot")
            except Exception as e:
                logger.warning(f"Error capturing screenshot: {e}")
        
        # 6. If still no screenshots, return original result
        if not has_screenshots:
            logger.debug(f"No visual data available for {tool_name}")
            return result
        
        # 7. Perform visual analysis
        return await self._enhance_result_with_visual_context(result, tool_name)
    
    async def _enhance_result_with_visual_context(
        self,
        result: ToolResult,
        tool_name: str
    ) -> ToolResult:
        """
        Enhance tool result with visual analysis for grounding agent workflows.
        """
        import asyncio
        import base64
        import litellm
        
        try:
            metadata = getattr(result, 'metadata', None)
            if not metadata:
                return result
            
            # Collect all screenshots
            screenshots_bytes = []
            
            # Check for multiple screenshots first
            if metadata.get("screenshots"):
                screenshots_list = metadata["screenshots"]
                if isinstance(screenshots_list, list):
                    screenshots_bytes = [s for s in screenshots_list if s]
            # Fall back to single screenshot
            elif metadata.get("screenshot"):
                screenshots_bytes = [metadata["screenshot"]]
            
            if not screenshots_bytes:
                return result
            
            # Select key screenshots if there are too many
            selected_screenshots = self._select_key_screenshots(screenshots_bytes, max_count=3)
            
            # Convert to base64
            visual_b64_list = []
            for visual_data in selected_screenshots:
                if isinstance(visual_data, bytes):
                    visual_b64_list.append(base64.b64encode(visual_data).decode('utf-8'))
                else:
                    visual_b64_list.append(visual_data)  # Already base64
            
            # Build prompt based on number of screenshots
            num_screenshots = len(visual_b64_list)
            
            prompt = GroundingAgentPrompts.visual_analysis(
                tool_name=tool_name,
                num_screenshots=num_screenshots
            )

            # Build content with text prompt + all images
            content = [{"type": "text", "text": prompt}]
            for visual_b64 in visual_b64_list:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{visual_b64}"
                    }
                })

            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=self._llm_client.model if self._llm_client else "anthropic/claude-sonnet-4-5",
                    messages=[{
                        "role": "user",
                        "content": content
                    }],
                    timeout=self._visual_analysis_timeout
                ),
                timeout=self._visual_analysis_timeout + 5
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Inject visual analysis into content
            original_content = result.content or "(no text output)"
            enhanced_content = f"{original_content}\n\n**Visual content**: {analysis}"
            
            # Create enhanced result
            enhanced_result = ToolResult(
                status=result.status,
                content=enhanced_content,
                error=result.error,
                metadata={**metadata, "visual_analyzed": True, "visual_analysis": analysis},
                execution_time=result.execution_time
            )
            
            logger.info(f"Enhanced {tool_name} result with visual analysis ({num_screenshots} screenshot(s))")
            return enhanced_result
            
        except asyncio.TimeoutError:
            logger.warning(f"Visual analysis timed out for {tool_name}, returning original result")
            return result
        except Exception as e:
            logger.warning(f"Failed to analyze visual content for {tool_name}: {e}")
            return result
    
    def _select_key_screenshots(
        self, 
        screenshots: List[bytes], 
        max_count: int = 3
    ) -> List[bytes]:
        """
        Select key screenshots if there are too many.
        """
        if len(screenshots) <= max_count:
            return screenshots
        
        selected_indices = set()
        
        # Always include last (final state)
        selected_indices.add(len(screenshots) - 1)
        
        # If room, include first (initial state)
        if max_count >= 2:
            selected_indices.add(0)
        
        # Fill remaining slots with evenly spaced middle screenshots
        remaining_slots = max_count - len(selected_indices)
        if remaining_slots > 0:
            # Calculate spacing
            available_indices = [
                i for i in range(1, len(screenshots) - 1)
                if i not in selected_indices
            ]
            
            if available_indices:
                step = max(1, len(available_indices) // (remaining_slots + 1))
                for i in range(remaining_slots):
                    idx = min((i + 1) * step, len(available_indices) - 1)
                    if idx < len(available_indices):
                        selected_indices.add(available_indices[idx])
        
        # Return screenshots in original order
        selected = [screenshots[i] for i in sorted(selected_indices)]
        
        logger.debug(
            f"Selected {len(selected)} screenshots at indices {sorted(selected_indices)} "
            f"from total of {len(screenshots)}"
        )
        
        return selected

    def _get_workspace_path(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Get workspace directory path from context.
        """
        return context.get("workspace_dir")
    
    def _scan_workspace_files(
        self,
        workspace_path: str,
        recent_threshold: int = 600 # seconds
    ) -> Dict[str, Any]:
        """
        Scan workspace directory and collect file information.
        
        Args:
            workspace_path: Path to workspace directory
            recent_threshold: Threshold in seconds for recent files
            
        Returns:
            Dictionary with file information:
                - files: List of all filenames
                - file_details: Dict mapping filename to file info (size, modified, age_seconds)
                - recent_files: List of recently modified filenames
        """
        import os
        import time
        
        result = {
            "files": [],
            "file_details": {},
            "recent_files": []
        }
        
        if not workspace_path or not os.path.exists(workspace_path):
            return result
        
        try:
            current_time = time.time()
            
            for filename in os.listdir(workspace_path):
                filepath = os.path.join(workspace_path, filename)
                if os.path.isfile(filepath):
                    result["files"].append(filename)
                    
                    # Get file stats
                    stat = os.stat(filepath)
                    file_info = {
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "age_seconds": current_time - stat.st_mtime
                    }
                    result["file_details"][filename] = file_info
                    
                    # Track recently created/modified files
                    if file_info["age_seconds"] < recent_threshold:
                        result["recent_files"].append(filename)
            
            result["files"] = sorted(result["files"])
        
        except Exception as e:
            logger.debug(f"Error scanning workspace files: {e}")
        
        return result
    
    async def _check_workspace_artifacts(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check workspace directory for existing artifacts that might be relevant to the task.
        Enhanced to detect if task might already be completed.
        """
        import re
        
        workspace_info = {"has_files": False, "files": [], "file_details": {}, "recent_files": []}
        
        try:
            # Get workspace path
            workspace_path = self._get_workspace_path(context)
            
            # Scan workspace files
            scan_result = self._scan_workspace_files(workspace_path, recent_threshold=600)
            
            if scan_result["files"]:
                workspace_info["has_files"] = True
                workspace_info["files"] = scan_result["files"]
                workspace_info["file_details"] = scan_result["file_details"]
                workspace_info["recent_files"] = scan_result["recent_files"]
                
                logger.info(f"Grounding Agent: Found {len(scan_result['files'])} existing files in workspace "
                           f"({len(scan_result['recent_files'])} recent)")
                
                # Check if instruction mentions specific filenames
                instruction = context.get("instruction", "")
                if instruction:
                    # Look for potential file references in instruction
                    potential_outputs = []
                    # Match common file patterns: filename.ext, "filename", 'filename'
                    file_patterns = re.findall(r'["\']?([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)["\']?', instruction)
                    for pattern in file_patterns:
                        if pattern in scan_result["files"]:
                            potential_outputs.append(pattern)
                    
                    if potential_outputs:
                        workspace_info["matching_files"] = potential_outputs
                        logger.info(f"Grounding Agent: Found {len(potential_outputs)} files matching task: {potential_outputs}")
        
        except Exception as e:
            logger.debug(f"Could not check workspace artifacts: {e}")
        
        return workspace_info
    
    def _build_iteration_feedback(
        self,
        iteration: int,
        llm_summary: Optional[str] = None,
        add_guidance: bool = True
    ) -> Optional[Dict[str, str]]:
        """
        Build feedback message to add to next iteration.
        """
        if not llm_summary:
            return None
        
        feedback_content = GroundingAgentPrompts.iteration_feedback(
            iteration=iteration,
            llm_summary=llm_summary,
            add_guidance=add_guidance
        )
        
        return {
            "role": "system",
            "content": feedback_content
        }
    
    def _remove_previous_guidance(self, messages: List[Dict[str, Any]]) -> None:
        """
        Remove guidance section from previous iteration feedback messages.
        """
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                # Check if this is an iteration feedback message with guidance
                if "## Iteration" in content and "Summary" in content and "---" in content:
                    # Remove everything from "---" onwards (the guidance part)
                    summary_only = content.split("---")[0].strip()
                    msg["content"] = summary_only

    async def _generate_final_summary(
        self,
        instruction: str,
        messages: List[Dict],
        iterations: int
    ) -> str:
        """
        Generate final summary across all iterations for reporting to upper layer.
        """
        final_summary_prompt = {
            "role": "system",
            "content": GroundingAgentPrompts.final_summary(
                instruction=instruction,
                iterations=iterations
            )
        }
        
        summary_messages = messages.copy()
        summary_messages.append(final_summary_prompt)
        
        try:
            # Call LLMClient to generate final summary (without tools)
            summary_response = await self._llm_client.complete(
                messages=summary_messages,
                tools=[],
                execute_tools=False
            )
            
            final_summary = summary_response.get("message", {}).get("content", "")
            
            if final_summary:
                logger.info(f"Generated final summary: {final_summary[:200]}...")
                return final_summary
            else:
                logger.warning("LLM returned empty final summary")
                return f"Task completed after {iterations} iteration(s). Check execution history for details."
        
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            return f"Task completed after {iterations} iteration(s), but failed to generate summary: {str(e)}"
    

    async def _build_final_result(
        self,
        instruction: str,
        messages: List[Dict],
        all_tool_results: List[Dict],
        iterations: int,
        max_iterations: int
    ) -> Dict[str, Any]:
        """
        Build final execution result.
        
        Args:
            instruction: Original instruction
            messages: Complete conversation history (including all iteration summaries)
            all_tool_results: All tool execution results
            iterations: Number of iterations performed
            max_iterations: Maximum allowed iterations
        """
        is_complete = self._check_task_completion(messages)
        
        tool_executions = self._format_tool_executions(all_tool_results)
        
        result = {
            "instruction": instruction,
            "step": self.step,
            "iterations": iterations,
            "tool_executions": tool_executions,
            "messages": messages,
            "keep_session": True
        }
        
        if is_complete:
            logger.info("Task completed with <COMPLETE> marker, generating final summary...")
            final_summary = await self._generate_final_summary(
                instruction=instruction,
                messages=messages,
                iterations=iterations
            )
            result["response"] = final_summary
            result["status"] = "success"
        else:
            result["response"] = self._extract_last_assistant_message(messages)
            result["status"] = "incomplete"
            result["warning"] = (
                f"Task reached max iterations ({max_iterations}) without completion. "
                f"This may indicate the task needs more steps or clarification."
            )
        
        return result
    
    def _format_tool_executions(self, all_tool_results: List[Dict]) -> List[Dict]:
        executions = []
        for tr in all_tool_results:
            tool_result_obj = tr.get("result")
            
            status = "unknown"
            if hasattr(tool_result_obj, 'status'):
                status_obj = tool_result_obj.status
                status = getattr(status_obj, 'value', status_obj)
            
            executions.append({
                "tool_name": tr.get("tool_call", {}).get("function", {}).get("name", "unknown"),
                "backend": tr.get("backend"),
                "server_name": tr.get("server_name"),
                "status": status,
                "content": tool_result_obj.content if hasattr(tool_result_obj, 'content') else None,
                "error": tool_result_obj.error if hasattr(tool_result_obj, 'error') else None,
                "execution_time": tool_result_obj.execution_time if hasattr(tool_result_obj, 'execution_time') else None,
                "metadata": tool_result_obj.metadata if hasattr(tool_result_obj, 'metadata') else {},
            })
        return executions
    
    def _check_task_completion(self, messages: List[Dict]) -> bool:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                return GroundingAgentPrompts.TASK_COMPLETE in content
        return False
    
    def _extract_last_assistant_message(self, messages: List[Dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""
    
    async def _record_agent_execution(
        self,
        result: Dict[str, Any],
        instruction: str
    ) -> None:
        """
        Record agent execution to recording manager.
        
        Args:
            result: Execution result
            instruction: Original instruction
        """
        if not self._recording_manager:
            return
        
        # Extract tool execution summary
        tool_summary = []
        if result.get("tool_executions"):
            for exec_info in result["tool_executions"]:
                tool_summary.append({
                    "tool": exec_info.get("tool_name", "unknown"),
                    "backend": exec_info.get("backend", "unknown"),
                    "status": exec_info.get("status", "unknown"),
                })
        
        await self._recording_manager.record_agent_action(
            agent_name=self.name,
            action_type="execute",
            input_data={"instruction": instruction},
            reasoning={
                "response": result.get("response", "")[:500],
                "tools_selected": tool_summary,
            },
            output_data={
                "status": result.get("status", "unknown"),
                "iterations": result.get("iterations", 0),
                "num_tool_executions": len(result.get("tool_executions", [])),
            },
            metadata={
                "step": self.step,
                "instruction": instruction,
            }
        )
