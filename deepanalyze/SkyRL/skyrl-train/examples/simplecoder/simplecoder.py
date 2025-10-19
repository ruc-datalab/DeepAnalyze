from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import subprocess
import tempfile
from typing import Optional, Dict, Any, List

from openai import OpenAI


@dataclass
class ExecutionResult:
    """Result of a command execution."""

    output: str
    error: Optional[str] = None
    return_code: int = 0


class Executor(ABC):

    @abstractmethod
    def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        """Execute a command and return the result.

        Args:
            command: The command to execute
            timeout: Timeout in seconds (default: 30)

        Returns:
            ExecutionResult containing the command output and status
        """
        pass


class GuixExecutor(Executor):
    """Guix-based executor that runs commands in a sandboxed Guix shell environment."""

    def __init__(self, working_dir: str, manifest_file: Optional[str] = None):
        """Initialize the Guix executor.

        Args:
            working_dir: Working directory of the execution
            manifest_file: Path to a Guix manifest file specifying packages
        """
        self.working_dir = working_dir
        self.manifest_file = manifest_file
        self.current_env = ""

    def execute(
        self,
        command: str,
        timeout: int = 30,
    ) -> ExecutionResult:
        """Execute a command in a sandboxed Guix shell."""

        guix_cmd = ["guix", "shell"]

        if self.manifest_file:
            guix_cmd.extend(["-m", self.manifest_file])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_env.sh", delete=False
        ) as env_file:
            env_file.write(self.current_env)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_script.sh", delete=False
        ) as script_file:
            script_file.write(f"source {env_file.name}\n")
            script_file.write("cd $PWD\n")
            script_file.write(f"{command}\n")
            script_file.write(f"export -p > {env_file.name}\n")

        # Add a very lightweight sandbox using https://github.com/containers/bubblewrap.
        # Originally we were using the guix shell --container sandbox for this, but there
        # are environments where that does not work (e.g. mounting the /proc filesystem
        # can fail in a GPU container). We might want to revisit this.
        guix_cmd.extend(
            # fmt: off
            [
                "--",
                "bwrap",
                "--ro-bind", "/bin", "/bin",
                "--ro-bind", "/gnu", "/gnu",
                "--proc", "/proc",
                "--dev", "/dev",
                "--tmpfs", "/tmp",
                "--new-session",
                "--ro-bind", script_file.name, script_file.name,
                "--bind", env_file.name, env_file.name,
                "--ro-bind", "/etc/resolv.conf", "/etc/resolv.conf",
                "--bind", self.working_dir, "/home/skyrl",
                "--setenv", "HOME", "/home/skyrl/",
                "sh",
                script_file.name,
            ]
            # fmt: on
        )

        try:
            result = subprocess.run(
                guix_cmd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )
        except Exception as e:
            return ExecutionResult(
                output="",
                error=f"Execution failed: {str(e)}",
                return_code=-1,
            )

        with open(env_file.name, "r") as f:
            self.current_env = f.read()

        return ExecutionResult(
            output=result.stdout or "",
            error=result.stderr if result.stderr else None,
            return_code=result.returncode,
        )


@dataclass
class ToolResult:
    """Result from executing a tool"""

    success: bool
    output: str
    error: Optional[str] = None


class Tool(ABC):
    """Base class for all tools"""

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass


class ShellCommandTool(Tool):
    """Tool for executing shell commands"""

    def __init__(self, executor: Executor):
        """Initialize the shell command tool with an executor.

        Args:
            executor: The executor to use for running commands.
        """
        self.executor = executor

    def name(self) -> str:
        return "execute_bash"

    def description(self) -> str:
        return "Execute a shell command and return the output"

    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                }
            },
            "required": ["command"],
        }

    def execute(self, command: str, timeout: int = 30) -> ToolResult:
        """Execute a shell command using the configured executor."""
        execution_result = self.executor.execute(command, timeout=timeout)

        return ToolResult(
            success=execution_result.return_code == 0,
            output=execution_result.output,
            error=execution_result.error,
        )


class SimpleCoder:

    def __init__(self, api_key: str, model: str, executor: Executor):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.executor = executor
        self.tools = {
            tool.name(): tool
            for tool in [
                ShellCommandTool(executor=self.executor),
            ]
        }
        self.conversation_history = []

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions for all tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters(),
                },
            }
            for tool in self.tools.values()
        ]

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool with given arguments"""
        if tool_name not in self.tools:
            return ToolResult(
                success=False, output="", error=f"Unknown tool: {tool_name}"
            )

        tool = self.tools[tool_name]
        return tool.execute(**arguments)

    def run(self, task: str, max_iterations: int = 30):

        self.conversation_history = [
            {
                "role": "system",
                "content": """You are a Software Engineering Agent. You can:
1. Execute shell commands using execute_shell
2. Read, write, or append to files using edit_file

Break down complex tasks into steps and use the appropriate tools to complete them.
Always check the results of your actions and adapt your approach if needed.""",
            },
            {"role": "user", "content": task},
        ]

        for i in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=self._get_tool_definitions(),
                tool_choice="auto",
            )
            assistant_message = response.choices[0].message
            self.conversation_history.append(assistant_message.model_dump())

            # Check if the assistant wants to use tools
            if assistant_message.tool_calls:
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    print(f"\n🔧 Executing {function_name} with args: {arguments}")

                    # Execute the tool
                    result = self._execute_tool(function_name, arguments)

                    # Add tool result to conversation
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            {
                                "success": result.success,
                                "output": result.output,
                                "error": result.error,
                            }
                        ),
                    }
                    self.conversation_history.append(tool_message)

                    print(
                        f"✅ Result: {result.output}..."
                        if result.success
                        else f"❌ Error: {result.error}"
                    )
            else:
                print(f"\n🤖 Agent: {assistant_message.content}")
                return


if __name__ == "__main__":
    import os
    import simplecoder

    manifest = os.path.abspath("manifest.scm")
    working_dir = os.path.abspath("test-repo")
    executor = simplecoder.GuixExecutor(working_dir, manifest)

    coder = simplecoder.SimpleCoder(os.environ["OPENAI_API_KEY"], "o4-mini", executor)
    task = """
    I'm running missing_colon.py as follows:

division(23, 0)
but I get the following error:

  File "/Users/fuchur/Documents/24/git_sync/swe-agent-test-repo/tests/./missing_colon.py", line 4
    def division(a: float, b: float) -> float
                                             ^
SyntaxError: invalid syntax
"""
    coder.run(task)
