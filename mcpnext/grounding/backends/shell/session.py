import re
from mcpnext.grounding.core.types import BackendType
from mcpnext.grounding.core.session import BaseSession
from mcpnext.grounding.backends.shell.transport.connector import ShellConnector
from mcpnext.grounding.core.tool import BaseTool
from mcpnext.grounding.core.security.policies import SecurityPolicyManager
from mcpnext.llm import LLMClient
from mcpnext.utils.logging import Logger

logger = Logger.get_logger(__name__)


class ShellSession(BaseSession):
    backend_type = BackendType.SHELL

    def __init__(
        self, 
        connector: ShellConnector, 
        *, 
        session_id: str, 
        security_manager: SecurityPolicyManager = None,
        default_working_dir: str = None,
        default_env: dict = None,
        default_conda_env: str = None
    ):
        super().__init__(connector=connector, session_id=session_id,
                         backend_type=BackendType.SHELL)
        self.security_manager = security_manager
        self.default_working_dir = default_working_dir
        self.default_env = default_env or {}
        self.default_conda_env = default_conda_env

    async def initialize(self):
        self.tools = [ShellAgentTool(
            self, 
            security_manager=self.security_manager,
            default_working_dir=self.default_working_dir,
            default_env=self.default_env,
            default_conda_env=self.default_conda_env
        )]
        return {"tools": [t.name for t in self.tools]}

class PythonScriptTool(BaseTool):
    _name = "_python_exec"
    _description = "Internal helper: run python code."

    def __init__(self, session: "ShellSession", default_working_dir: str = None, default_env: dict = None, default_conda_env: str = None):
        self._session = session
        self._default_working_dir = default_working_dir
        self._default_env = default_env or {}
        self._default_conda_env = default_conda_env
        super().__init__()

    async def _arun(self, code: str, timeout: int = 90, working_dir: str | None = None, env: dict | None = None, conda_env: str | None = None):
        # Use provided params, or fall back to session defaults
        effective_working_dir = working_dir or self._default_working_dir
        effective_env = {**self._default_env, **(env or {})}  # Merge default and provided env
        effective_conda_env = conda_env or self._default_conda_env
        return await self._session.connector.run_python_script(
            code, 
            timeout=timeout, 
            working_dir=effective_working_dir,
            env=effective_env if effective_env else None,
            conda_env=effective_conda_env
        )

class BashScriptTool(BaseTool):
    _name = "_bash_exec"
    _description = "Internal helper: run bash script."

    def __init__(self, session: "ShellSession", default_working_dir: str = None, default_env: dict = None, default_conda_env: str = None):
        self._session = session
        self._default_working_dir = default_working_dir
        self._default_env = default_env or {}
        self._default_conda_env = default_conda_env
        super().__init__()

    async def _arun(self, script: str, timeout: int = 30, working_dir: str | None = None, env: dict | None = None, conda_env: str | None = None):
        # Use provided params, or fall back to session defaults
        effective_working_dir = working_dir or self._default_working_dir
        effective_env = {**self._default_env, **(env or {})}  # Merge default and provided env
        effective_conda_env = conda_env or self._default_conda_env
        return await self._session.connector.run_bash_script(
            script, 
            timeout=timeout, 
            working_dir=effective_working_dir,
            env=effective_env if effective_env else None,
            conda_env=effective_conda_env
        )

class ShellAgentTool(BaseTool):
    _name = "shell_agent"
    _description = """Execute commands or scripts directly in the computer's terminal. 
This tool uses an internal agent that will write and run Python or Bash code to accomplish tasks or inspect the current system state. The internal agent will automatically retry and fix errors when possible.

Use this tool when you need to:
- Execute any terminal-based task that requires code
- Check the current environment (files, processes, system info)
- Run calculations or data processing
- Install packages or modify system settings

The tool will keep trying until the task succeeds or determines it cannot be completed."""
    
    backend_type = BackendType.SHELL
    _CODE_RGX = re.compile(
        r"```(?P<lang>python|py|bash|shell|sh)[^\n]*\n(?P<code>.*?)```",
        re.S | re.I,
    )

    def __init__(
        self, 
        session: "ShellSession", 
        client_password: str = "", 
        max_steps: int = 5,
        security_manager: SecurityPolicyManager = None,
        default_working_dir: str = None,
        default_env: dict = None,
        default_conda_env: str = None
    ):
        self._session = session
        self._llm = LLMClient()
        self.client_password = client_password
        self.max_steps = max_steps
        self._system_info = None
        self.security_manager = security_manager
        self._default_working_dir = default_working_dir
        self._default_env = default_env or {}
        self._default_conda_env = default_conda_env
        self._py_tool = PythonScriptTool(session, default_working_dir=default_working_dir, default_env=default_env, default_conda_env=default_conda_env)
        self._bash_tool = BashScriptTool(session, default_working_dir=default_working_dir, default_env=default_env, default_conda_env=default_conda_env)
        super().__init__()

    async def _get_system_info(self):
        """
        Get system information for shell agent.
        
        First tries to get comprehensive info from local server's /platform endpoint.
        Falls back to simple bash commands if that fails.
        
        Returns:
            Dict with at least 'platform' and 'username' keys
        """
        if self._system_info is None:
            try:
                # Try to get system info from server via HTTP API
                try:
                    from mcpnext.platform import SystemInfoClient
                    
                    # Get base_url from connector
                    base_url = self._session.connector.base_url
                    
                    # Create temporary client
                    async with SystemInfoClient(base_url=base_url, timeout=5) as client:
                        info = await client.get_system_info(use_cache=False)
                        
                        if info:
                            # Use comprehensive info from server
                            self._system_info = {
                                "platform": info.get("system", "Linux"),
                                "username": info.get("username", "user"),
                                "machine": info.get("machine"),
                                "release": info.get("release"),
                                "full_info": info  # Keep full info for reference
                            }
                            logger.debug(f"Got system info from server: {info.get('system')}")
                            return self._system_info
                
                except ImportError:
                    logger.debug("SystemInfoClient not available, using bash commands")
                
                # Fallback: use simple bash commands (original method)
                platform_result = await self._session.connector.run_bash_script("uname -s", timeout=5)
                username_result = await self._session.connector.run_bash_script("whoami", timeout=5)
                
                platform = self._extract_output(platform_result).strip()
                username = self._extract_output(username_result).strip()
                
                self._system_info = {
                    "platform": platform,
                    "username": username
                }
                logger.debug(f"Got system info from bash: {platform}")
            
            except Exception as e:
                logger.warning(f"Failed to get system info: {e}, using defaults")
                self._system_info = {"platform": "Linux", "username": "user"}
        
        return self._system_info

    async def _arun(self, task: str, timeout: int = 300):
        from mcpnext.grounding.core.types import ToolResult, ToolStatus
        
        sys_info = await self._get_system_info()
        conversation_history = []
        iteration = 0
        last_error = None
        
        # record the code history
        code_history = []
        
        # Build environment context
        env_context = []
        if self._default_working_dir:
            env_context.append(f"Working Directory: {self._default_working_dir}")
        if self._default_conda_env:
            env_context.append(f"Conda Environment: {self._default_conda_env}")
        if self._default_env:
            env_vars = ", ".join([f"{k}={v}" for k, v in list(self._default_env.items())[:3]])
            if len(self._default_env) > 3:
                env_vars += f", ... (+{len(self._default_env)-3} more)"
            env_context.append(f"Custom Environment Variables: {env_vars}")
        
        env_section = "\n".join([f"# {ctx}" for ctx in env_context]) if env_context else ""
        
        SHELL_AGENT_SYSTEM_PROMPT = f"""You are an expert system administrator and programmer focused on executing tasks efficiently.

# System: {sys_info["platform"]}, User: {sys_info["username"]}
{env_section}

# Your task: {task}

# IMPORTANT: You MUST provide exactly ONE code block in EVERY response
# Either ```bash or ```python - never respond without code

# Available actions:
1. Execute bash commands: ```bash <commands>```
2. Write Python code: ```python <code>```

# Rules:
- ALWAYS include a code block in your response
- Write EXACTLY ONE code block per response
- If you need to understand the current environment, start with bash commands like: pwd, ls, ps, df, etc.
- If you get errors, analyze and fix them in the next iteration
- For sudo: use 'echo {self.client_password} | sudo -S <command>'
- The environment (working directory, conda env) is managed automatically

# CRITICAL: Avoid quote escaping errors in bash:
- For complex string operations (JSON, multi-line text, special chars): ALWAYS use Python with heredoc
- Good: ```python <your code>```
- Bad: bash commands with nested quotes like: echo "$(cat 'file' | grep "pattern")"
- When reading/writing files with complex content: prefer Python over bash
- When processing JSON: ALWAYS use Python's json module, never bash string manipulation

# Before executing, check if task output already exists:
- Use 'ls -la <directory>' to check for existing files
- If files exist, read and verify them first before recreating
- Avoid redundant work - reuse existing valid outputs

# Task completion marking:
When you believe the task is COMPLETED, end your response with:
[TASK_COMPLETED: brief explanation of what was accomplished]

When you encounter an UNRECOVERABLE error that you cannot fix, end your response with:
[TASK_FAILED: brief explanation of why it cannot be completed]"""

        conversation_history.append({"role": "system", "content": SHELL_AGENT_SYSTEM_PROMPT})
        
        no_code_counter = 0
        final_message = ""
        
        while iteration < self.max_steps:
            iteration += 1
            
            logger.info(f"[ShellAgent] Step {iteration}/{self.max_steps}: Processing task")
            
            try:           
                messages_text = LLMClient.format_messages_to_text(conversation_history)
                response = await self._llm.complete(messages_text)

                assistant_content = response["message"]["content"]
                logger.debug(f"[ShellAgent] Step {iteration} LLM response: {assistant_content[:200]}...")

                # extract and execute the code, and track the code block
                code_info, execution_result = await self._execute_code_from_response(assistant_content)
                if code_info:
                    code_history.append(code_info)
                
                logger.info(f"[ShellAgent] Step {iteration} execution result: {execution_result[:100]}...")
                if execution_result == "ERROR: No valid code block found":
                    no_code_counter += 1
                    if no_code_counter >= 3:
                        final_message = f"Task failed after {iteration} steps: LLM failed to provide code blocks repeatedly"
                        return ToolResult(
                            status=ToolStatus.ERROR,
                            content=final_message,
                            metadata={"tool": self._name, "code_history": code_history}
                        )
                else:
                    no_code_counter = 0
                
                completion_status = self._check_task_status(assistant_content, execution_result, last_error)
                
                if completion_status["completed"]:
                    content_parts = [f"Task completed successfully after {iteration} steps"]
                    content_parts.append(f"\n{'='*60}")
                    content_parts.append(f"\nFinal Result:")
                    content_parts.append(execution_result)
                    
                    if len(code_history) > 1:
                        content_parts.append(f"\n{'='*60}")
                        content_parts.append(f"\nExecution Summary ({len(code_history)} steps):")
                        for i, code_info in enumerate(code_history, 1):
                            lang = code_info.get("language", "unknown")
                            output = code_info.get("output", "")
                            output_preview = output[:200].replace('\n', ' ')
                            if len(output) > 200:
                                output_preview += "..."
                            content_parts.append(f"\n  Step {i} [{lang}]: {output_preview}")
                    
                    content_parts.append(f"\n{'='*60}")
                    content_parts.append(f"\nSummary: {completion_status['reason']}")
                    
                    final_message = "\n".join(content_parts)
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        content=final_message,
                        metadata={"tool": self._name, "code_history": code_history}
                    )
                elif completion_status["failed"]:
                    final_message = f"Task failed after {iteration} steps: {completion_status['reason']}\nLast result: {execution_result}"
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        content=final_message,
                        metadata={"tool": self._name, "code_history": code_history}
                    )

                feedback = self._generate_feedback(execution_result, iteration, last_error)
                
                conversation_history.extend([
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": feedback}
                ])
                
                last_error = execution_result if "ERROR" in execution_result else None
                
            except Exception as e:
                final_message = f"Tool execution failed at step {iteration}: {str(e)}"
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content=final_message,
                    metadata={"tool": self._name, "code_history": code_history}
                )
        
        final_message = f"Reached maximum steps ({self.max_steps}). Task may be too complex or impossible."
        return ToolResult(
            status=ToolStatus.ERROR,
            content=final_message,
            metadata={"tool": self._name, "code_history": code_history}
        )

    async def _execute_code_from_response(self, response: str):
        """
        execute the code and track the code block
        
        Returns:
            Tuple[Optional[Dict], str]: (code_info, execution_result)
            - code_info: {"lang": "python/bash", "code": "...", "status": "success/error"}
            - execution_result: the execution result string
        """
        matches = list(self._CODE_RGX.finditer(response))
        if not matches:
            return None, "ERROR: No valid code block found"
        
        lang, code = matches[0]["lang"].lower(), matches[0]["code"].strip()
        
        # standardize the language name
        lang_normalized = "python" if lang in ["python", "py"] else "bash"
        
        code_info = {
            "lang": lang_normalized,
            "code": code,
        }

        # Security check is only done at the Connector layer to avoid duplicate prompts
        
        try:
            if lang in ["python", "py"]:
                helper = self._py_tool
                result = await helper._arun(code)
            elif lang in ["bash", "shell", "sh"]:
                helper = self._bash_tool
                result = await helper._arun(code)
            else:
                execution_result = f"ERROR: Unsupported language: {lang}"
                code_info["status"] = "error"
                return code_info, execution_result
            
            execution_result = self._extract_output(result)
            code_info["status"] = "success" if "ERROR" not in execution_result else "error"
            return code_info, execution_result
            
        except Exception as e:
            execution_result = f"EXECUTION ERROR: {str(e)}"
            code_info["status"] = "error"
            return code_info, execution_result

    def _generate_feedback(self, result: str, iteration: int, last_error: str) -> str:
        feedback = f"Step {iteration} result:\n{result}\n\n"
        
        if "ERROR" in result:
            if last_error and last_error == result:
                feedback += "Same error as previous step. Try a different approach.\n"
            else:
                feedback += "Error occurred. Analyze the error and fix it.\n"
        else:
            feedback += "Execution successful. Continue to next step if needed.\n"
        
        feedback += "\nWhat's your next action? (Remember: provide exactly ONE code block)"
        return feedback

    def _extract_output(self, result):
        if isinstance(result, dict):
            # Check for execution errors
            stderr = result.get("error") or result.get("stderr") or ""
            returncode = result.get("returncode", 0)
            stdout = result.get("content") or result.get("output") or result.get("stdout") or ""
            
            # If there's a non-zero return code or stderr with actual errors, report it
            if returncode != 0 or (stderr and len(stderr.strip()) > 0):
                error_msg = f"EXECUTION ERROR (exit code {returncode}):\n"
                if stderr:
                    error_msg += f"stderr: {stderr}\n"
                if stdout:
                    error_msg += f"stdout: {stdout}"
                return error_msg
            
            return stdout or str(result)
        return str(result)

    def _check_task_status(self, response: str, execution_result: str, last_error: str) -> dict:
        if "[TASK_COMPLETED:" in response:
            reason = response.split("[TASK_COMPLETED:")[1].split("]")[0].strip()
            return {"completed": True, "failed": False, "reason": reason}

        if "[TASK_FAILED:" in response:
            reason = response.split("[TASK_FAILED:")[1].split("]")[0].strip()
            return {"completed": False, "failed": True, "reason": reason}

        # Extended error pattern detection
        error_patterns = [
            "ERROR:",
            "EXECUTION ERROR:",
            "CommandNotFoundError",
            "Traceback (most recent call last)",
            "Exception:",
            "PermissionError",
            "FileNotFoundError",
            "SyntaxError:",
            "ImportError:",
            "ModuleNotFoundError",
            "No such file or directory",
            "command not found",
        ]
        
        has_error = any(pattern in execution_result for pattern in error_patterns)
        
        if has_error:
            if last_error and last_error == execution_result:
                return {"completed": False, "failed": True, "reason": "Same error repeated - unable to resolve"}
            return {"completed": False, "failed": False, "reason": "Execution error occurred"}

        return {"completed": False, "failed": False, "reason": "Task in progress"}