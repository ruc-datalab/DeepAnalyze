from __future__ import annotations

import subprocess
import threading
from pathlib import Path, PurePosixPath

from ..settings import settings


_DOCKER_LOCK = threading.Lock()
_CONTAINER_STARTED_BY_APP = False
_CONTAINER_CREATED_BY_APP = False


def _run_docker_command(
    args: list[str],
    *,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _container_exists(container_name: str) -> bool:
    completed = _run_docker_command(
        ["ps", "-a", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
        check=False,
    )
    return container_name in (completed.stdout or "").splitlines()


def _container_is_running(container_name: str) -> bool:
    completed = _run_docker_command(
        ["inspect", "-f", "{{.State.Running}}", container_name],
        check=False,
    )
    return (completed.returncode == 0) and (completed.stdout or "").strip().lower() == "true"


def _workspace_root() -> Path:
    root = Path(settings.workspace_base_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _keepalive_command() -> list[str]:
    return ["sh", "-c", "while true; do sleep 3600; done"]


def ensure_execution_backend_ready() -> None:
    global _CONTAINER_CREATED_BY_APP, _CONTAINER_STARTED_BY_APP

    if not settings.use_docker_execution:
        return

    container_name = settings.docker_container_name
    with _DOCKER_LOCK:
        if _container_is_running(container_name):
            return

        if _container_exists(container_name):
            _run_docker_command(["start", container_name])
            _CONTAINER_STARTED_BY_APP = True
            return

        workspace_root = _workspace_root()
        _run_docker_command(
            [
                "run",
                "-d",
                "--name",
                container_name,
                "-v",
                f"{workspace_root}:{settings.docker_workspace_dir}",
                "-w",
                settings.docker_workspace_dir,
                settings.docker_image,
                *_keepalive_command(),
            ]
        )
        _CONTAINER_STARTED_BY_APP = True
        _CONTAINER_CREATED_BY_APP = True


def shutdown_execution_backend() -> None:
    global _CONTAINER_CREATED_BY_APP, _CONTAINER_STARTED_BY_APP

    if not settings.use_docker_execution or not settings.docker_stop_on_shutdown:
        return

    container_name = settings.docker_container_name
    with _DOCKER_LOCK:
        if not _CONTAINER_STARTED_BY_APP:
            return

        if _container_is_running(container_name):
            _run_docker_command(["stop", container_name], check=False, timeout=20)
        if _CONTAINER_CREATED_BY_APP:
            _run_docker_command(["rm", "-f", container_name], check=False, timeout=20)

        _CONTAINER_STARTED_BY_APP = False
        _CONTAINER_CREATED_BY_APP = False


def _resolve_container_workdir(workspace_dir: str) -> str:
    workspace_root = _workspace_root()
    exec_dir = Path(workspace_dir).resolve()
    relative_dir = exec_dir.relative_to(workspace_root)
    if str(relative_dir) in {"", "."}:
        return settings.docker_workspace_dir
    return str(PurePosixPath(settings.docker_workspace_dir) / relative_dir.as_posix())


def execute_python_in_docker(
    script_path: str,
    workspace_dir: str,
    timeout_sec: int,
) -> str:
    ensure_execution_backend_ready()
    container_workdir = _resolve_container_workdir(workspace_dir)
    script_name = Path(script_path).name

    try:
        completed = _run_docker_command(
            [
                "exec",
                "-e",
                "MPLBACKEND=Agg",
                "-e",
                "QT_QPA_PLATFORM=offscreen",
                "-w",
                container_workdir,
                settings.docker_container_name,
                settings.docker_python_bin,
                script_name,
            ],
            timeout=timeout_sec,
        )
        return (completed.stdout or "") + (completed.stderr or "")
    except subprocess.TimeoutExpired:
        return f"[Timeout]: execution exceeded {timeout_sec} seconds"
    except Exception as exc:
        return f"[Error]: {exc}"
