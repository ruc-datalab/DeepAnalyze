from skyrl_gym.tools.core import tool, ToolGroup
import subprocess


class PythonCodeExecutorToolGroup(ToolGroup):
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        super().__init__(name="PythonCodeExecutorToolGroup")

    @tool
    def python(self, code: str) -> str:
        try:
            result = subprocess.run(
                ["python", "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                text=True,
            )
            if result.stderr:
                return f"Error executing Python code: {result.stderr.strip()}"
            return result.stdout.strip() if result.stdout else ""
        except subprocess.TimeoutExpired:
            return f"Error executing Python code: Execution timed out after {self.timeout} seconds."
