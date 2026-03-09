from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .docker_executor import ensure_execution_backend_ready, execute_python_in_docker
from .workspace import build_download_url, register_generated_paths, uniquify_path
from ..settings import IMAGE_EXTENSIONS, settings


def execute_code_safe(
    code_str: str,
    workspace_dir: str,
    timeout_sec: int | None = None,
) -> str:
    timeout_sec = timeout_sec or settings.execution_timeout_sec
    exec_cwd = os.path.abspath(workspace_dir)
    os.makedirs(exec_cwd, exist_ok=True)
    tmp_path: str | None = None
    try:
        if settings.use_docker_execution:
            ensure_execution_backend_ready()
        fd, tmp_path = tempfile.mkstemp(suffix=".py", dir=exec_cwd)
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as file:
            file.write(code_str)

        if settings.use_docker_execution:
            return execute_python_in_docker(tmp_path, exec_cwd, timeout_sec)

        child_env = os.environ.copy()
        child_env.setdefault("MPLBACKEND", "Agg")
        child_env.setdefault("QT_QPA_PLATFORM", "offscreen")
        child_env.pop("DISPLAY", None)

        completed = subprocess.run(
            [sys.executable, tmp_path],
            cwd=exec_cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            env=child_env,
        )
        return (completed.stdout or "") + (completed.stderr or "")
    except subprocess.TimeoutExpired:
        return f"[Timeout]: execution exceeded {timeout_sec} seconds"
    except Exception as exc:
        return f"[Error]: {exc}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def snapshot_workspace_files(workspace_dir: str) -> dict[Path, tuple[int, int]]:
    try:
        return {
            path.resolve(): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in Path(workspace_dir).rglob("*")
            if path.is_file()
        }
    except Exception:
        return {}


def collect_artifact_paths(
    before_state: dict[Path, tuple[int, int]],
    after_state: dict[Path, tuple[int, int]],
    generated_dir: str,
    session_id: str,
) -> list[Path]:
    generated_root = Path(generated_dir).resolve()
    generated_root.mkdir(parents=True, exist_ok=True)
    workspace_root = generated_root.parent

    artifact_paths: list[Path] = []
    generated_index_updates: set[str] = set()
    added_paths = sorted([path for path in after_state if path not in before_state], key=str)
    modified_paths = sorted(
        [path for path in after_state if path in before_state and after_state[path] != before_state[path]],
        key=str,
    )

    for path in added_paths:
        try:
            if generated_root not in path.parents:
                dest_path = uniquify_path(generated_root / path.name)
                shutil.copy2(path, dest_path)
                artifact_paths.append(dest_path.resolve())
                generated_index_updates.add(path.resolve().relative_to(workspace_root).as_posix())
                generated_index_updates.add(dest_path.resolve().relative_to(workspace_root).as_posix())
            else:
                artifact_paths.append(path)
                generated_index_updates.add(path.resolve().relative_to(workspace_root).as_posix())
        except Exception:
            artifact_paths.append(path)

    for path in modified_paths:
        try:
            dest_name = f"{path.stem}_modified{path.suffix}"
            dest_path = uniquify_path(generated_root / dest_name)
            shutil.copy2(path, dest_path)
            artifact_paths.append(dest_path.resolve())
            generated_index_updates.add(path.resolve().relative_to(workspace_root).as_posix())
            generated_index_updates.add(dest_path.resolve().relative_to(workspace_root).as_posix())
        except Exception:
            continue

    if generated_index_updates:
        register_generated_paths(session_id, generated_index_updates)

    return artifact_paths


def build_file_block(
    artifact_paths: list[Path],
    workspace_dir: str,
    session_id: str,
) -> str:
    if not artifact_paths:
        return ""

    workspace_root = Path(workspace_dir).resolve()
    lines = ["<File>"]
    for path in artifact_paths:
        try:
            rel_path = path.relative_to(workspace_root).as_posix()
        except Exception:
            rel_path = path.name

        url = build_download_url(f"{session_id}/{rel_path}")
        name = path.name
        lines.append(f"- [{name}]({url})")
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            lines.append(f"![{name}]({url})")
    lines.append("</File>")
    return "\n" + "\n".join(lines) + "\n"
