import os
import subprocess
import shlex
from tempfile import NamedTemporaryFile, TemporaryDirectory

from .utils import _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS

CLI_ARG_SIZE_LIMIT = 1024 * 3
MEMORY_LIMIT_KB = 10 * 1024 * 1024  # 10GB in KB

def wrap_command_with_ulimit(command):
    cmd_str = ' '.join(shlex.quote(c) for c in command)
    return ["bash", "-c", f"ulimit -v {MEMORY_LIMIT_KB}; exec {cmd_str}"]

def code_exec_local(
    code,
    stdin: str = None,
    timeout=30,
    pytest: str = None,
    solution: str = None,
    python_env: str = os.environ.get("CONDA_BIN_PATH", None),
):
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]

    if python_env is None:
        python_executable = "/usr/bin/python3"
    else:
        python_executable = os.path.join(python_env, "python3")

    if solution:
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with solution_file"
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(solution)

            command = [
                "timeout",
                str(timeout),
                python_executable,
                os.path.join(tmpdir, "test_solution.py"),
            ]
            command = wrap_command_with_ulimit(command)
            result = subprocess.run(
                command,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
    elif pytest:
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with pytest"
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)

            command = [
                "timeout",
                str(timeout),
                python_executable,
                "-m",
                "pytest",
                tmpdir,
            ]
            command = wrap_command_with_ulimit(command)
            result = subprocess.run(
                command,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
    else:
        if len(code) < CLI_ARG_SIZE_LIMIT:
            command = ["timeout", str(timeout), python_executable, "-c", code]
            command = wrap_command_with_ulimit(command)
            result = subprocess.run(
                command,
                input=(stdin.encode() if stdin else None),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
        else:
            with NamedTemporaryFile() as tmp:
                tmp.write(code.encode())
                tmp.flush()
                command = ["timeout", str(timeout), python_executable, tmp.name]
                command = wrap_command_with_ulimit(command)
                result = subprocess.run(
                    command,
                    input=(stdin.encode() if stdin else None),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=False,
                )

    stderr = result.stderr.decode().strip()
    stdout = result.stdout.decode()

    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"