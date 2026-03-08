from __future__ import annotations

import http.server
import json
import re
import shutil
import socketserver
import tempfile
import threading
import zipfile
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote

import httpx
from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask

from ..settings import PREVIEWABLE_EXTENSIONS, settings


_FILE_SERVER_LOCK = threading.Lock()
_FILE_SERVER_STARTED = False


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def get_session_workspace(session_id: str) -> str:
    safe_session_id = (session_id or "default").strip() or "default"
    session_dir = Path(settings.workspace_base_dir) / safe_session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return str(session_dir)


def build_download_url(rel_path: str) -> str:
    try:
        encoded = quote(rel_path, safe="/")
    except Exception:
        encoded = rel_path
    return f"{settings.file_server_base}/{encoded}"


def start_http_server() -> None:
    Path(settings.workspace_base_dir).mkdir(parents=True, exist_ok=True)
    handler = partial(
        http.server.SimpleHTTPRequestHandler,
        directory=settings.workspace_base_dir,
    )
    with ReusableTCPServer(("", settings.http_server_port), handler) as httpd:
        print(
            f"HTTP Server serving {settings.workspace_base_dir} at port {settings.http_server_port}"
        )
        httpd.serve_forever()


def ensure_http_server_started() -> None:
    global _FILE_SERVER_STARTED
    with _FILE_SERVER_LOCK:
        if _FILE_SERVER_STARTED:
            return
        threading.Thread(target=start_http_server, daemon=True).start()
        _FILE_SERVER_STARTED = True


def collect_file_info(source: str | Path | Sequence[str | Path]) -> str:
    file_paths: list[Path] = []
    seen: set[Path] = set()

    if isinstance(source, (str, Path)):
        candidate = Path(source)
        if not candidate.exists():
            return ""
        if candidate.is_dir():
            file_paths = sorted(
                [path for path in candidate.iterdir() if path.is_file()],
                key=lambda path: path.name.lower(),
            )
        elif candidate.is_file():
            file_paths = [candidate]
    else:
        for item in source or []:
            candidate = Path(item)
            if not candidate.exists() or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            file_paths.append(candidate)
        file_paths.sort(key=lambda path: path.name.lower())

    parts: list[str] = []
    for index, file_path in enumerate(file_paths, start=1):
        size_str = f"{file_path.stat().st_size / 1024:.1f}KB"
        file_info = {"name": file_path.name, "size": size_str}
        parts.append(f"File {index}:\n{json.dumps(file_info, indent=4, ensure_ascii=False)}\n")
    return "\n".join(parts)


def get_file_icon(extension: str) -> str:
    ext = extension.lower()
    icons = {
        (".jpg", ".jpeg", ".png", ".gif", ".bmp"): "🖼️",
        (".pdf",): "📃",
        (".doc", ".docx"): "📌",
        (".txt",): "📝",
        (".md",): "📑",
        (".csv", ".xlsx"): "📳",
        (".json", ".sqlite"): "🗽",
        (".mp4", ".avi", ".mov"): "🎴",
        (".mp3", ".wav"): "🎍",
        (".zip", ".rar", ".tar"): "🗞️",
    }
    for extensions, icon in icons.items():
        if ext in extensions:
            return icon
    return "📧"


TABLE_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".parquet",
    ".sqlite",
    ".db",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".bmp",
}


def classify_file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TABLE_EXTENSIONS:
        return "table"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "other"


def uniquify_path(target: Path) -> Path:
    if not target.exists():
        return target

    parent = target.parent
    stem = target.stem
    suffix = target.suffix
    match = re.match(r"^(.*) \((\d+)\)$", stem)
    base = stem
    start = 1
    if match:
        base = match.group(1)
        try:
            start = int(match.group(2)) + 1
        except ValueError:
            start = 1

    index = start
    while True:
        candidate = parent / f"{base} ({index}){suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def resolve_workspace_root(session_id: str) -> Path:
    return Path(get_session_workspace(session_id)).resolve()


def resolve_workspace_path(session_id: str, relative_path: str = "") -> Path:
    workspace_root = resolve_workspace_root(session_id)
    target = (workspace_root / (relative_path or "")).resolve()
    if target != workspace_root and workspace_root not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    return target


def list_workspace_files(session_id: str) -> list[dict]:
    workspace_root = resolve_workspace_root(session_id)
    files: list[dict] = []
    all_files = [path for path in workspace_root.rglob("*") if path.is_file()]
    for file_path in sorted(all_files, key=lambda path: _rel_path(path, workspace_root).lower()):
        rel = _rel_path(file_path, workspace_root)
        rel_path = f"{session_id}/{rel}"
        files.append(
            {
                "name": file_path.name,
                "path": rel,
                "size": file_path.stat().st_size,
                "extension": file_path.suffix.lower(),
                "icon": get_file_icon(file_path.suffix),
                "category": classify_file_type(file_path),
                "is_generated": rel == "generated" or rel.startswith("generated/"),
                "download_url": build_download_url(rel_path),
                "preview_url": (
                    build_download_url(rel_path)
                    if file_path.suffix.lower() in PREVIEWABLE_EXTENSIONS
                    else None
                ),
            }
        )
    return files


def download_generated_bundle(session_id: str, category: str = "all") -> FileResponse:
    workspace_root = resolve_workspace_root(session_id)
    generated_root = workspace_root / "generated"
    if not generated_root.exists() or not generated_root.is_dir():
        raise HTTPException(status_code=404, detail="generated folder not found")

    normalized_category = (category or "all").strip().lower()
    if normalized_category not in {"all", "table", "image", "other"}:
        raise HTTPException(status_code=400, detail="invalid category")

    files = [path for path in generated_root.rglob("*") if path.is_file()]
    if normalized_category != "all":
        files = [
            path for path in files if classify_file_type(path) == normalized_category
        ]

    if not files:
        raise HTTPException(status_code=404, detail="no files matched the category")

    temp_file = tempfile.NamedTemporaryFile(
        prefix=f"deepanalyze_{normalized_category}_",
        suffix=".zip",
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, file_path.relative_to(generated_root))

    filename = f"generated_{normalized_category}.zip"
    return FileResponse(
        path=temp_path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(lambda: temp_path.unlink(missing_ok=True)),
    )


def _rel_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.name


def build_tree(path: Path, root: Path | None = None, session_id: str = "default") -> dict:
    root = root or path
    node: dict = {
        "name": path.name or "workspace",
        "path": _rel_path(path, root),
        "is_dir": path.is_dir(),
    }

    if path.is_dir():
        def sort_key(candidate: Path) -> tuple[bool, bool, str]:
            return (candidate.name == "generated", not candidate.is_dir(), candidate.name.lower())

        node["children"] = [
            build_tree(child, root, session_id)
            for child in sorted(path.iterdir(), key=sort_key)
            if not child.name.startswith(".")
        ]
        return node

    rel = _rel_path(path, root)
    node["size"] = path.stat().st_size
    node["extension"] = path.suffix.lower()
    node["icon"] = get_file_icon(path.suffix)
    node["download_url"] = build_download_url(f"{session_id}/{rel}")
    return node


def delete_workspace_file(session_id: str, relative_path: str) -> dict:
    target = resolve_workspace_path(session_id, relative_path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    if target.is_dir():
        raise HTTPException(status_code=400, detail="Folder deletion not allowed")
    target.unlink()
    return {"message": "deleted"}


def move_workspace_path(session_id: str, src: str, dst_dir: str = "") -> dict:
    source = resolve_workspace_path(session_id, src)
    if not source.exists():
        raise HTTPException(status_code=404, detail="Source not found")

    target_dir = resolve_workspace_path(session_id, dst_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = uniquify_path(target_dir / source.name)
    shutil.move(str(source), str(target))
    return {
        "message": "moved",
        "new_path": target.relative_to(resolve_workspace_root(session_id)).as_posix(),
    }


def delete_workspace_dir(session_id: str, relative_path: str, recursive: bool = True) -> dict:
    workspace_root = resolve_workspace_root(session_id)
    target = resolve_workspace_path(session_id, relative_path)
    if target == workspace_root:
        raise HTTPException(status_code=400, detail="Cannot delete workspace root")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")
    if recursive:
        shutil.rmtree(target)
    else:
        target.rmdir()
    return {"message": "deleted"}


async def proxy_external_file(url: str) -> Response:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            response = await client.get(url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Proxy fetch failed: {exc}") from exc

    return Response(
        content=response.content,
        media_type=response.headers.get("content-type", "application/octet-stream"),
        headers={"Access-Control-Allow-Origin": "*"},
        status_code=response.status_code,
    )


async def _save_uploads(
    workspace_root: Path,
    target_dir: Path,
    files: Iterable[UploadFile],
) -> list[dict]:
    saved: list[dict] = []
    for file in files:
        dst = uniquify_path(target_dir / (file.filename or "untitled"))
        content = await file.read()
        with open(dst, "wb") as buffer:
            buffer.write(content)
        saved.append(
            {
                "name": dst.name,
                "size": len(content),
                "path": dst.relative_to(workspace_root).as_posix(),
            }
        )
    return saved


async def upload_files_to_workspace(session_id: str, files: Iterable[UploadFile]) -> dict:
    workspace_root = resolve_workspace_root(session_id)
    saved = await _save_uploads(workspace_root, workspace_root, files)
    return {
        "message": f"Successfully uploaded {len(saved)} files",
        "files": saved,
    }


async def upload_files_to_dir(session_id: str, directory: str, files: Iterable[UploadFile]) -> dict:
    workspace_root = resolve_workspace_root(session_id)
    target_dir = resolve_workspace_path(session_id, directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved = await _save_uploads(workspace_root, target_dir, files)
    return {"message": f"uploaded {len(saved)}", "files": saved}


def clear_workspace(session_id: str) -> dict:
    workspace_root = resolve_workspace_root(session_id)
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    return {"message": "Workspace cleared successfully"}
