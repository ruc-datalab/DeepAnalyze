import requests
from pydantic import BaseModel
from typing import Optional, Dict
from enum import Enum

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 30   # 30 seconds is the default timeout for the executor


def check_executor_alive(executor):
    try:
        return requests.get(executor + "/").status_code in [200, 404]
    except Exception:
        return False

# The below code is from the code_sandbox repo: https://github.com/bytedance/SandboxFusion/blob/main/sandbox/server/sandbox_api.py
# - RunStatus, CommandRunStatus, CommandRunResult, RunCodeResponse are from the code_sandbox repo

class RunStatus(str, Enum):
    # all command finished successfully
    Success = 'Success'
    # one of the process has non-zero return code
    Failed = 'Failed'
    # error on sandbox side
    SandboxError = 'SandboxError'


class CommandRunStatus(str, Enum):
    Finished = 'Finished'
    Error = 'Error'
    TimeLimitExceeded = 'TimeLimitExceeded'


class CommandRunResult(BaseModel):
    status: CommandRunStatus
    execution_time: Optional[float] = None
    return_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

class RunCodeResponse(BaseModel):
    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}