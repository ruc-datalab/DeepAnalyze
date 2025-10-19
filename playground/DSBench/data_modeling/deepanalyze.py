import re
import io
import traceback
import contextlib
import os
import requests
from pathlib import Path


class DeepAnalyzeVLLM:
    """
    DeepAnalyzeVLLM provides functionality to generate and execute code
    using a vLLM API with multi-round reasoning.
    """

    def __init__(
        self,
        model_name: str,
        api_url: str = "http://localhost:8000/v1/chat/completions",
        max_rounds: int = 30,
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.max_rounds = max_rounds

    def execute_code(self, code_str: str) -> str:
        """
        Executes Python code and captures stdout and stderr outputs.
        Returns the output or formatted error message.
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
                stderr_capture
            ):
                exec(code_str, {})
            output = stdout_capture.getvalue()
            if stderr_capture.getvalue():
                output += stderr_capture.getvalue()
            return output
        except Exception as exec_error:
            code_lines = code_str.splitlines()
            tb_lines = traceback.format_exc().splitlines()
            error_line = None

            # Attempt to extract line number from traceback
            for line in tb_lines:
                if 'File "<string>", line' in line:
                    try:
                        line_num = int(line.split(", line ")[1].split(",")[0])
                        error_line = line_num
                        break
                    except (IndexError, ValueError):
                        continue

            # Build formatted error message
            error_message = "Traceback (most recent call last):\n"
            if error_line and 1 <= error_line <= len(code_lines):
                error_message += f'  File "<string>", line {error_line}, in <module>\n'
                error_message += f"    {code_lines[error_line - 1].strip()}\n"
            error_message += f"{type(exec_error).__name__}: {str(exec_error)}"
            if stderr_capture.getvalue():
                error_message += f"\n{stderr_capture.getvalue()}"
            return f"[Error]:\n{error_message.strip()}"

    def generate(
        self,
        prompt: str,
        workspace: str,
        temperature: float = 0.5,
        max_tokens: int = 32768,
        top_p: float = None,
        top_k: int = None,
    ) -> dict:
        """
        Generates content using vLLM API and executes any <Code> blocks found.
        Returns a dictionary containing the full reasoning process.
        """
        original_cwd = os.getcwd()
        os.chdir(workspace)
        reasoning = ""
        messages = [{"role": "user", "content": prompt}]
        response_message = []

        try:
            for round_idx in range(self.max_rounds):
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "add_generation_prompt": False,
                    "stop": ["</Code>"],
                }
                if top_p is not None:
                    payload["top_p"] = top_p
                if top_k is not None:
                    payload["top_k"] = top_k

                # Call vLLM API
                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
                response_data = response.json()

                ans = response_data["choices"][0]["message"]["content"]
                if response_data["choices"][0].get("stop_reason") == "</Code>":
                    ans += "</Code>"

                response_message.append(ans)

                # Check for <Code> block
                code_match = re.search(r"<Code>(.*?)</Code>", ans, re.DOTALL)
                if not code_match or "<Answer>" in ans:
                    break

                code_content = code_match.group(1).strip()
                md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
                code_str = md_match.group(1).strip() if md_match else code_content

                # Execute code and append output
                exe_output = self.execute_code(code_str)
                response_message.append(f"<Execute>\n{exe_output}\n</Execute>")

                # Append messages for next round
                messages.append({"role": "assistant", "content": ans})
                messages.append({"role": "execute", "content": exe_output})

            reasoning = "\n".join(response_message)

        except Exception:
            reasoning = "\n".join(response_message)

        os.chdir(original_cwd)
        return {"reasoning": reasoning}
