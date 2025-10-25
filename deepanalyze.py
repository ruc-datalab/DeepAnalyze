import re
import io
import traceback
import contextlib
import os
import requests
from pathlib import Path
from typing import Optional


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


class DeepAnalyzeOpenRouter:
    """
    DeepAnalyzeOpenRouter provides functionality to generate and execute code
    using OpenRouter API (OpenAI-compatible) with multi-round reasoning.

    Supports 100+ models including Claude, GPT-4, Gemini, DeepSeek, etc.
    """

    def __init__(
        self,
        model_name: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        api_url: str = "https://openrouter.ai/api/v1/chat/completions",
        max_rounds: int = 30,
        site_url: Optional[str] = None,
        app_name: str = "DeepAnalyze",
    ):
        """
        Initialize DeepAnalyzeOpenRouter.

        Args:
            model_name: OpenRouter model identifier (e.g., "anthropic/claude-3.5-sonnet")
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            api_url: OpenRouter API endpoint
            max_rounds: Maximum reasoning rounds
            site_url: Your site URL (optional, for rankings on openrouter.ai)
            app_name: Your app name (optional, for rankings on openrouter.ai)
        """
        self.model_name = model_name
        self.api_url = api_url
        self.max_rounds = max_rounds
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.site_url = site_url or os.environ.get("SITE_URL", "https://github.com/ruc-datalab/DeepAnalyze")
        self.app_name = app_name

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )

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
        top_p: Optional[float] = None,
    ) -> dict:
        """
        Generates content using OpenRouter API and executes any <Code> blocks found.
        Returns a dictionary containing the full reasoning process.

        Args:
            prompt: Input prompt with instructions and data info
            workspace: Working directory for code execution
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Returns:
            Dictionary with "reasoning" key containing the full conversation
        """
        original_cwd = os.getcwd()
        os.chdir(workspace)
        reasoning = ""
        messages = [{"role": "user", "content": prompt}]
        response_message = []

        # Prepare headers for OpenRouter
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }

        try:
            for round_idx in range(self.max_rounds):
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": ["</Code>"],  # Stop at end of code block
                }
                if top_p is not None:
                    payload["top_p"] = top_p

                # Call OpenRouter API
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                response_data = response.json()

                ans = response_data["choices"][0]["message"]["content"]

                # Check if stopped at </Code>
                finish_reason = response_data["choices"][0].get("finish_reason")
                if finish_reason == "stop" and "<Code>" in ans and "</Code>" not in ans:
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
                messages.append({"role": "user", "content": f"<Execute>\n{exe_output}\n</Execute>"})

            reasoning = "\n".join(response_message)

        except Exception as e:
            reasoning = "\n".join(response_message)
            print(f"Error during generation: {e}")

        os.chdir(original_cwd)
        return {"reasoning": reasoning}


# Alias for backward compatibility
DeepAnalyze = DeepAnalyzeVLLM
