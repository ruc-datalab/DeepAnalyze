from skyagent.tools.base import BaseTool, register_tool
import requests
import os
import asyncio
import time
from typing import Union


@register_tool("code_interpreter")
class CodeInterpreter(BaseTool):
    name = "code_interpreter"
    description = "Executes code in a sandbox for supported languages. Supports Python, JavaScript, C++, and more."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The code to be executed. Should be valid in the specified language.",
            },
            "language": {
                "type": "string",
                "description": "Programming language of the code (e.g., python, javascript, cpp). Defaults to python.",
                "default": "python",
            },
        },
        "required": ["code"],
    }
    assert (
        "SANDBOX_FUSION_URL" in os.environ
    ), "Environment variable SANDBOX_FUSION_URL must be set."
    sandbox_url = os.getenv("SANDBOX_FUSION_URL")
    memory_limit_mb = 1024  # 1 GB memory limit

    # Adapted from https://arxiv.org/pdf/2502.14382.
    @staticmethod
    def _post_process_code(code: str) -> str:
        """
        Remove any markdown formatting from the code.
        This is important to ensure the code is executed correctly in the sandbox.
        """
        if not code:
            return code

        # Remove markdown code block delimiters
        # Handle cases like ```python, ```javascript, ```cpp, or just ```
        import re

        # Remove opening markdown code blocks (```language or just ```)
        code = re.sub(r"^```\w*\s*\n?", "", code.strip(), flags=re.MULTILINE)

        # Remove closing markdown code blocks
        code = re.sub(r"\n?```\s*$", "", code, flags=re.MULTILINE)

        # Remove any remaining ``` that might be in the middle
        code = code.replace("```", "")

        code = code.strip()

        # From https://github.com/volcengine/verl/blob/7fc3029a1ec407f6e56f1f1ff02a659071da3b1d/recipe/retool/retool.py#L41C9-L49C32
        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        # Dacheng: Add a " " handle to avoid adding print with meaningful code with indentation.
        # print("Original code:", code)
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print") and not lines[i].startswith(" "):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)
        # print("Modified code:", code)

        return code

    # From https://github.com/volcengine/verl/blob/main/verl/tools/sandbox_fusion_tools.py
    def _execute_code_retool(self, code, timeout=30, language="python"):
        from skyagent.tasks.verifiers.sandbox_fusion.utils import _process_single_case

        result_status, metadata = _process_single_case(
            0,
            None,
            None,
            self.sandbox_url + "/run_code",
            code,
            timeout,
            self.memory_limit_mb,
            language,
        )
        # we should always expect this since we don't have correct answer
        if metadata["run_status"] == "Finished":
            actual_output = metadata["stdout"] + metadata["stderr"]
            # logger.debug(f"actual_output from sandbox fusion: {actual_output}")
            return actual_output
        else:
            return "no stdout here"

    def call(self, params: dict, **kwargs) -> Union[str, list, dict]:
        """
        Executes the provided code in a sandbox environment with retry logic.

        Args:
            params (dict): Dictionary containing 'code' and optionally 'language'.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The output of the executed code or an error message.
        """
        # verify required parameters
        try:
            params = self._verify_json_format_args(params)
        except ValueError as e:
            return {"error": f"Invalid parameters: {str(e)}"}
        # extract code and language from params

        code = params.get("code")
        # Post-process code to remove markdown formatting
        # Dacheng: sometimes model write ```py instead of ```python, so we just extract the code part.
        code = self._post_process_code(code)

        language = params.get("language", "python")
        if not code:
            return {"error": "Code parameter is required."}

        # Retry logic for timeout and connection errors
        for attempt in range(5):
            output = self._execute_code_retool(code, 30, language)
            if "RLIMIT_NPROC" in output:
                # This indicates a timeout or resource limit error
                print(f"Attempt {attempt + 1}: Timeout or resource limit exceeded.")
                time.sleep(5)
                continue  # Retry the request
            return output

        return {"error": "Max retries exceeded"}


if __name__ == "__main__":
    # Example usage for testing
    tool = CodeInterpreter()
    test_params = {"code": "print('Hello, World!')", "language": "python"}
    import json

    test_params = json.dumps(test_params)  # Convert to JSON string if needed
    result = tool.call(test_params)
    print(
        "Test Result:", result
    )  # Should print: {'output': 'Hello, World!'} or an error message
