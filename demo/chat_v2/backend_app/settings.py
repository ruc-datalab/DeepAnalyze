from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")


def _load_demo_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_load_demo_env()


CHINESE_MATPLOTLIB_BOOTSTRAP = """
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
"""


PREVIEWABLE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".pdf",
    ".txt",
    ".doc",
    ".docx",
    ".csv",
    ".xlsx",
}


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


@dataclass(frozen=True)
class Settings:
    api_base: str = os.getenv("DEEPANALYZE_API_BASE", "http://localhost:8000/v1")
    model_path: str = os.getenv("DEEPANALYZE_MODEL_PATH", "DeepAnalyze-8B")
    workspace_base_dir: str = os.getenv("DEEPANALYZE_WORKSPACE_BASE", "workspace")
    http_server_host: str = os.getenv("DEEPANALYZE_FILE_SERVER_HOST", "localhost")
    http_server_port: int = int(os.getenv("DEEPANALYZE_FILE_SERVER_PORT", "8100"))
    backend_host: str = os.getenv("DEEPANALYZE_BACKEND_HOST", "0.0.0.0")
    backend_port: int = int(os.getenv("DEEPANALYZE_BACKEND_PORT", "8200"))
    execution_mode: str = os.getenv("DEEPANALYZE_EXECUTION_MODE", "local")
    execution_timeout_sec: int = int(os.getenv("DEEPANALYZE_EXECUTION_TIMEOUT_SEC", "120"))
    docker_image: str = os.getenv("DEEPANALYZE_DOCKER_IMAGE", "python:3.11-slim")
    docker_container_name: str = os.getenv(
        "DEEPANALYZE_DOCKER_CONTAINER_NAME",
        "deepanalyze-chat-exec",
    )
    docker_session_idle_ttl_sec: int = int(
        os.getenv("DEEPANALYZE_DOCKER_SESSION_IDLE_TTL_SEC", "1800")
    )
    docker_workspace_dir: str = os.getenv("DEEPANALYZE_DOCKER_WORKSPACE_DIR", "/workspace")
    docker_python_bin: str = os.getenv("DEEPANALYZE_DOCKER_PYTHON_BIN", "python")
    docker_stop_on_shutdown: bool = _get_bool_env(
        "DEEPANALYZE_DOCKER_STOP_ON_SHUTDOWN",
        True,
    )
    pdf_cjk_mainfont: str = os.getenv("DEEPANALYZE_PDF_CJK_MAINFONT", "").strip()

    @property
    def file_server_base(self) -> str:
        return f"http://{self.http_server_host}:{self.http_server_port}"

    @property
    def use_docker_execution(self) -> bool:
        return self.execution_mode.strip().lower() == "docker"


settings = Settings()
