import re
import io
import traceback
import contextlib
import os
import google.generativeai as genai


class DeepAnalyzeGemini:
    """
    DeepAnalyzeGemini provides functionality to generate and execute code
    using the Gemini API with multi-round reasoning.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        max_rounds: int = 30,
    ):
        self.model_name = model_name
        self.max_rounds = max_rounds
        genai.configure(api_key=api_key)

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

            for line in tb_lines:
                if 'File "<string>", line' in line:
                    try:
                        line_num = int(line.split(", line ")[1].split(",")[0])
                        error_line = line_num
                        break
                    except (IndexError, ValueError):
                        continue

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
        Generates content using Gemini API and executes any <Code> blocks found.
        Returns a dictionary containing the full reasoning process.
        """
        original_cwd = os.getcwd()
        os.chdir(workspace)
        reasoning = ""
        messages = [{"role": "user", "parts": [prompt]}]
        response_message = []

        model = genai.GenerativeModel(self.model_name)

        try:
            for round_idx in range(self.max_rounds):
                response = model.generate_content(
                    messages,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        stop_sequences=["</Code>"],
                    ),
                )

                ans = response.text
                response_message.append(ans)

                code_match = re.search(r"<Code>(.*?)</Code>", ans, re.DOTALL)
                if not code_match or "<Answer>" in ans:
                    break

                code_content = code_match.group(1).strip()
                md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
                code_str = md_match.group(1).strip() if md_match else code_content

                exe_output = self.execute_code(code_str)
                response_message.append(f"<Execute>\n{exe_output}\n</Execute>")

                messages.append({"role": "model", "parts": [ans]})
                messages.append({"role": "user", "parts": [exe_output]})

            reasoning = "\n".join(response_message)

        except Exception:
            reasoning = "\n".join(response_message)

        os.chdir(original_cwd)
        return {"reasoning": reasoning}
