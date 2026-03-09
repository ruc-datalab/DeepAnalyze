from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..settings import settings


MANAGED_LABEL_KEY = "deepanalyze.managed"
SESSION_LABEL_KEY = "deepanalyze.session"


@dataclass
class SessionContainerState:
    session_id: str
    container_name: str
    created_by_app: bool
    started_by_app: bool
    last_used_at: float


_DOCKER_LOCK = threading.Lock()
_SESSION_CONTAINERS: dict[str, SessionContainerState] = {}


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


def _workspace_root() -> Path:
    root = Path(settings.workspace_base_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _keepalive_command() -> list[str]:
    return ["sh", "-c", "while true; do sleep 3600; done"]


def _sanitize_session_id(session_id: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (session_id or "default").strip())
    normalized = normalized.strip(".-") or "default"
    return normalized[:48]


def _container_name_for_session(session_id: str) -> str:
    prefix = settings.docker_container_name.strip() or "deepanalyze-chat-exec"
    suffix = _sanitize_session_id(session_id)
    return f"{prefix}-{suffix}"[:120]


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


def _touch_container(session_id: str, container_name: str, *, created_by_app: bool, started_by_app: bool) -> None:
    state = _SESSION_CONTAINERS.get(session_id)
    now = time.time()
    if state is None:
        _SESSION_CONTAINERS[session_id] = SessionContainerState(
            session_id=session_id,
            container_name=container_name,
            created_by_app=created_by_app,
            started_by_app=started_by_app,
            last_used_at=now,
        )
        return
    state.last_used_at = now
    state.created_by_app = state.created_by_app or created_by_app
    state.started_by_app = state.started_by_app or started_by_app


def _remove_container(container_name: str, *, remove: bool) -> None:
    if _container_is_running(container_name):
        _run_docker_command(["stop", container_name], check=False, timeout=20)
    if remove:
        _run_docker_command(["rm", "-f", container_name], check=False, timeout=20)


def _cleanup_idle_session_containers(now: float | None = None) -> None:
    if not settings.use_docker_execution:
        return

    ttl = max(0, settings.docker_session_idle_ttl_sec)
    if ttl <= 0:
        return

    now = now or time.time()
    expired_sessions = [
        session_id
        for session_id, state in _SESSION_CONTAINERS.items()
        if now - state.last_used_at >= ttl
    ]
    for session_id in expired_sessions:
        state = _SESSION_CONTAINERS.pop(session_id, None)
        if state is None:
            continue
        _remove_container(state.container_name, remove=state.created_by_app)


def ensure_execution_backend_ready(session_id: str | None = None) -> None:
    if not settings.use_docker_execution or not session_id:
        return

    workspace_root = _workspace_root()
    container_name = _container_name_for_session(session_id)

    with _DOCKER_LOCK:
        _cleanup_idle_session_containers()

        if _container_is_running(container_name):
            _touch_container(
                session_id,
                container_name,
                created_by_app=False,
                started_by_app=False,
            )
            return

        if _container_exists(container_name):
            _run_docker_command(["start", container_name])
            _touch_container(
                session_id,
                container_name,
                created_by_app=False,
                started_by_app=True,
            )
            return

        _run_docker_command(
            [
                "run",
                "-d",
                "--name",
                container_name,
                "--label",
                f"{MANAGED_LABEL_KEY}=true",
                "--label",
                f"{SESSION_LABEL_KEY}={session_id}",
                "-v",
                f"{workspace_root}:{settings.docker_workspace_dir}",
                "-w",
                settings.docker_workspace_dir,
                settings.docker_image,
                *_keepalive_command(),
            ]
        )
        _touch_container(
            session_id,
            container_name,
            created_by_app=True,
            started_by_app=True,
        )


def shutdown_execution_backend() -> None:
    if not settings.use_docker_execution or not settings.docker_stop_on_shutdown:
        return

    with _DOCKER_LOCK:
        for session_id, state in list(_SESSION_CONTAINERS.items()):
            _remove_container(state.container_name, remove=state.created_by_app)
            _SESSION_CONTAINERS.pop(session_id, None)


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
    session_id: str,
) -> str:
    ensure_execution_backend_ready(session_id)
    container_name = _container_name_for_session(session_id)
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
                container_name,
                settings.docker_python_bin,
                script_name,
            ],
            timeout=timeout_sec,
        )
        with _DOCKER_LOCK:
            _touch_container(
                session_id,
                container_name,
                created_by_app=False,
                started_by_app=False,
            )
        return (completed.stdout or "") + (completed.stderr or "")
    except subprocess.TimeoutExpired:
        return f"[Timeout]: execution exceeded {timeout_sec} seconds"
    except Exception as exc:
        return f"[Error]: {exc}"
