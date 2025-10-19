from skyrl_gym.tools.core import tool, ToolGroup
import subprocess
import re
import contextlib
import io
import sys
from pathlib import Path
import random
import traceback
import os
import json
import multiprocessing
import os
import signal
import io
import traceback
import threading


def _code_execution_target(result_queue, workspace, code):
    original_cwd = os.getcwd()
    os.chdir(workspace)
    try:
        if code is None:
            result_queue.put(
                "[Error]:\nNo code is provided, so the execution results is empty."
            )
            return

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        globals_dict = {}
        exec(
            "import matplotlib; matplotlib.use('Agg')", globals_dict
        )  # Non-interactive backend to prevent blocking

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            exec(code, globals_dict)
        output = stdout_capture.getvalue()
        if stderr_capture.getvalue():
            output += stderr_capture.getvalue()
        result_queue.put(output)
    except Exception as exec_error:
        code_lines = code.splitlines()
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
        error_message = f"Traceback (most recent call last):\n"
        if error_line is not None and 1 <= error_line <= len(code_lines):
            error_message += f'  File "<string>", line {error_line}, in <module>\n'
            error_message += f"    {code_lines[error_line-1].strip()}\n"
        error_message += f"{type(exec_error).__name__}: {str(exec_error)}"
        if stderr_capture.getvalue():
            error_message += f"\n{stderr_capture.getvalue()}"
        result_queue.put(f"[Error]:\n{error_message.strip()}")
    finally:
        os.chdir(original_cwd)


class PythonCodeExecutorToolGroup(ToolGroup):
    def __init__(self, workspace, timeout: float = 10.0):
        self.workspace = workspace
        self.timeout = timeout
        super().__init__(name="PythonCodeExecutorToolGroup")

    def execute_code(self, code: str) -> str:
        if code is None:
            return f"[Error]:\nNo code is provided, so the execution results is empty."
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:

            # Prepare globals with non-interactive Matplotlib to avoid plt.show() blocking
            globals_dict = {}
            exec(
                "import matplotlib; matplotlib.use('Agg')", globals_dict
            )  # Set backend before any pyplot import

            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
                stderr_capture
            ):
                exec(code, globals_dict)

            output = stdout_capture.getvalue()
            if stderr_capture.getvalue():
                output += stderr_capture.getvalue()
            return output
        except Exception as exec_error:
            code_lines = code.splitlines()
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
            error_message = f"Traceback (most recent call last):\n"
            if error_line is not None and 1 <= error_line <= len(code_lines):
                error_message += f'  File "<string>", line {error_line}, in <module>\n'
                error_message += f"    {code_lines[error_line-1].strip()}\n"
            error_message += f"{type(exec_error).__name__}: {str(exec_error)}"
            if stderr_capture.getvalue():
                error_message += f"\n{stderr_capture.getvalue()}"
            return f"[Error]:\n{error_message.strip()}"

    @tool
    def python(self, code: str) -> str:
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_code_execution_target, args=(result_queue, self.workspace, code)
        )
        process.start()
        process.join(timeout=self.timeout)
        if process.is_alive():
            os.kill(process.pid, signal.SIGTERM)  # Or SIGKILL for forceful termination
            process.join()
            result = f"[Error]\nPython code execution timed out after {self.timeout} seconds."
        else:
            result = (
                result_queue.get()
                if not result_queue.empty()
                else "[Error]\nUnknown execution error."
            )
        return result
