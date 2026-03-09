from __future__ import annotations

import http.server
import json
import re
import shutil
import sqlite3
import socketserver
import tempfile
import threading
import zipfile
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote

import httpx
import pandas as pd
from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask

from ..settings import PREVIEWABLE_EXTENSIONS, settings


_FILE_SERVER_LOCK = threading.Lock()
_FILE_SERVER_STARTED = False
GENERATED_INDEX_FILENAME = ".deepanalyze_generated.json"


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


def _generated_index_path(workspace_root: Path) -> Path:
    return workspace_root / GENERATED_INDEX_FILENAME


def _normalize_generated_rel_path(path: str) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def load_generated_index(session_id: str) -> set[str]:
    workspace_root = resolve_workspace_root(session_id)
    index_path = _generated_index_path(workspace_root)
    if not index_path.exists():
        return set()
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(payload, list):
        return set()
    normalized = {
        _normalize_generated_rel_path(str(item))
        for item in payload
        if str(item).strip()
    }
    existing: set[str] = set()
    for rel_path in normalized:
        candidate = workspace_root / rel_path
        if candidate.exists() and candidate.is_file():
            existing.add(rel_path)
    if existing != normalized:
        save_generated_index(session_id, existing)
    return existing


def save_generated_index(session_id: str, rel_paths: Iterable[str]) -> None:
    workspace_root = resolve_workspace_root(session_id)
    index_path = _generated_index_path(workspace_root)
    normalized = sorted(
        {
            _normalize_generated_rel_path(str(path))
            for path in rel_paths
            if _normalize_generated_rel_path(str(path))
        }
    )
    index_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def register_generated_paths(session_id: str, rel_paths: Iterable[str]) -> set[str]:
    generated = load_generated_index(session_id)
    generated.update(
        _normalize_generated_rel_path(str(path))
        for path in rel_paths
        if _normalize_generated_rel_path(str(path))
    )
    save_generated_index(session_id, generated)
    return generated


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
                [
                    path
                    for path in candidate.iterdir()
                    if path.is_file() and path.name != GENERATED_INDEX_FILENAME
                ],
                key=lambda path: path.name.lower(),
            )
        elif candidate.is_file():
            if candidate.name == GENERATED_INDEX_FILENAME:
                return ""
            file_paths = [candidate]
    else:
        for item in source or []:
            candidate = Path(item)
            if (
                not candidate.exists()
                or not candidate.is_file()
                or candidate.name == GENERATED_INDEX_FILENAME
            ):
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

TEXT_PREVIEW_EXTENSIONS = {
    ".txt",
    ".log",
    ".py",
    ".sql",
    ".json",
    ".yaml",
    ".yml",
}

MARKDOWN_PREVIEW_EXTENSIONS = {
    ".md",
    ".markdown",
}

SQLITE_PREVIEW_EXTENSIONS = {
    ".sqlite",
    ".db",
}

BLOCKED_UPLOAD_EXTENSIONS = {
    ".py",
}


def classify_file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TABLE_EXTENSIONS:
        return "table"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "other"


def _json_safe_value(value):
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    return text if len(text) <= 500 else f"{text[:500]}…"


def _build_dataframe_preview(
    dataframe: pd.DataFrame,
    *,
    title: str | None = None,
    kind: str = "table",
    max_rows: int = 100,
    max_cols: int = 30,
    extra: dict | None = None,
) -> dict:
    trimmed = dataframe.copy()
    total_rows = int(len(trimmed.index))
    total_cols = int(len(trimmed.columns))
    if total_cols > max_cols:
        trimmed = trimmed.iloc[:, :max_cols]
    row_truncated = total_rows > max_rows
    col_truncated = total_cols > max_cols
    trimmed = trimmed.head(max_rows).fillna("")
    rows = [
        [_json_safe_value(value) for value in row]
        for row in trimmed.astype(object).values.tolist()
    ]
    payload = {
        "kind": kind,
        "title": title,
        "columns": [str(column) for column in trimmed.columns.tolist()],
        "rows": rows,
        "row_count": total_rows,
        "column_count": total_cols,
        "truncated": row_truncated or col_truncated,
    }
    if extra:
        payload.update(extra)
    return payload


def _clamp_page(page: int, page_size: int) -> tuple[int, int]:
    safe_page_size = max(1, min(page_size, 200))
    safe_page = max(1, page)
    return safe_page, safe_page_size


def _build_paginated_preview(
    dataframe: pd.DataFrame,
    *,
    title: str | None = None,
    kind: str = "table",
    page: int = 1,
    page_size: int = 50,
    max_cols: int = 30,
    extra: dict | None = None,
) -> dict:
    safe_page, safe_page_size = _clamp_page(page, page_size)
    total_rows = int(len(dataframe.index))
    total_cols = int(len(dataframe.columns))
    total_pages = max(1, (total_rows + safe_page_size - 1) // safe_page_size)
    safe_page = min(safe_page, total_pages)

    trimmed = dataframe.copy()
    if total_cols > max_cols:
        trimmed = trimmed.iloc[:, :max_cols]

    start = (safe_page - 1) * safe_page_size
    end = start + safe_page_size
    page_df = trimmed.iloc[start:end].fillna("")
    rows = [
        [_json_safe_value(value) for value in row]
        for row in page_df.astype(object).values.tolist()
    ]
    payload = {
        "kind": kind,
        "title": title,
        "columns": [str(column) for column in page_df.columns.tolist()],
        "rows": rows,
        "row_count": total_rows,
        "column_count": total_cols,
        "page": safe_page,
        "page_size": safe_page_size,
        "total_pages": total_pages,
        "truncated": total_cols > max_cols,
    }
    if extra:
        payload.update(extra)
    return payload


def preview_workspace_file(
    session_id: str,
    relative_path: str,
    *,
    page: int = 1,
    page_size: int = 50,
    table_name: str = "",
    sheet_name: str = "",
) -> dict:
    file_path = resolve_workspace_path(session_id, relative_path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    ext = file_path.suffix.lower()

    if ext in TEXT_PREVIEW_EXTENSIONS:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return {
            "kind": "text",
            "title": file_path.name,
            "content": content[:50000],
            "truncated": len(content) > 50000,
        }

    if ext in MARKDOWN_PREVIEW_EXTENSIONS:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return {
            "kind": "markdown",
            "title": file_path.name,
            "content": content[:50000],
            "truncated": len(content) > 50000,
        }

    if ext in {".csv", ".tsv"}:
        separator = "\t" if ext == ".tsv" else ","
        dataframe = pd.read_csv(file_path, sep=separator)
        return _build_paginated_preview(
            dataframe,
            title=file_path.name,
            page=page,
            page_size=page_size,
        )

    if ext in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(file_path)
        active_sheet = sheet_name or workbook.sheet_names[0]
        if active_sheet not in workbook.sheet_names:
            raise HTTPException(status_code=404, detail="Sheet not found")
        dataframe = workbook.parse(sheet_name=active_sheet)
        return _build_paginated_preview(
            dataframe,
            title=file_path.name,
            page=page,
            page_size=page_size,
            extra={
                "sheet_name": active_sheet,
                "sheet_names": workbook.sheet_names,
            },
        )

    if ext in SQLITE_PREVIEW_EXTENSIONS:
        with sqlite3.connect(file_path) as connection:
            cursor = connection.cursor()
            table_names = [
                row[0]
                for row in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
            ]
            if not table_name:
                tables: list[dict] = []
                for current_table in table_names:
                    total_rows = cursor.execute(
                        f'SELECT COUNT(*) FROM "{current_table.replace(chr(34), chr(34) * 2)}"'
                    ).fetchone()[0]
                    columns = [
                        row[1]
                        for row in cursor.execute(
                            f'PRAGMA table_info("{current_table.replace(chr(34), chr(34) * 2)}")'
                        ).fetchall()
                    ]
                    tables.append(
                        {
                            "table_name": current_table,
                            "title": current_table,
                            "row_count": int(total_rows),
                            "column_count": len(columns),
                            "columns": columns,
                        }
                    )

                return {
                    "kind": "database",
                    "view": "tables",
                    "title": file_path.name,
                    "tables": tables,
                    "table_names": table_names,
                }

            if table_name not in table_names:
                raise HTTPException(status_code=404, detail="Table not found")

            safe_table_name = table_name.replace('"', '""')
            dataframe = pd.read_sql_query(
                f'SELECT * FROM "{safe_table_name}"',
                connection,
            )
            preview = _build_paginated_preview(
                dataframe,
                title=file_path.name,
                kind="database",
                page=page,
                page_size=page_size,
                extra={
                    "view": "table",
                    "table_name": table_name,
                    "table_names": table_names,
                },
            )
            preview["total_rows"] = preview["row_count"]
            return preview

    raise HTTPException(status_code=415, detail="Preview not supported")


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


def _is_internal_workspace_file(path: Path) -> bool:
    return path.name == GENERATED_INDEX_FILENAME


def _is_generated_workspace_path(rel_path: str, generated_index: set[str]) -> bool:
    normalized = _normalize_generated_rel_path(rel_path)
    return normalized in generated_index or normalized == "generated" or normalized.startswith("generated/")


def list_workspace_files(session_id: str) -> list[dict]:
    workspace_root = resolve_workspace_root(session_id)
    generated_index = load_generated_index(session_id)
    files: list[dict] = []
    all_files = [
        path
        for path in workspace_root.rglob("*")
        if path.is_file() and not _is_internal_workspace_file(path)
    ]
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
                "is_generated": _is_generated_workspace_path(rel, generated_index),
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


def build_tree(
    path: Path,
    root: Path | None = None,
    session_id: str = "default",
    generated_index: set[str] | None = None,
) -> dict:
    root = root or path
    generated_index = generated_index if generated_index is not None else load_generated_index(session_id)
    node: dict = {
        "name": path.name or "workspace",
        "path": _rel_path(path, root),
        "is_dir": path.is_dir(),
    }

    if path.is_dir():
        def sort_key(candidate: Path) -> tuple[bool, bool, str]:
            return (candidate.name == "generated", not candidate.is_dir(), candidate.name.lower())

        node["children"] = [
            build_tree(child, root, session_id, generated_index)
            for child in sorted(path.iterdir(), key=sort_key)
            if not child.name.startswith(".") and not _is_internal_workspace_file(child)
        ]
        node["is_generated"] = node["path"] == "generated" or node["path"].startswith("generated/")
        return node

    rel = _rel_path(path, root)
    node["size"] = path.stat().st_size
    node["extension"] = path.suffix.lower()
    node["icon"] = get_file_icon(path.suffix)
    node["is_generated"] = _is_generated_workspace_path(rel, generated_index)
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
) -> tuple[list[dict], list[str]]:
    saved: list[dict] = []
    rejected: list[str] = []
    for file in files:
        filename = file.filename or "untitled"
        ext = Path(filename).suffix.lower()
        if ext in BLOCKED_UPLOAD_EXTENSIONS:
            rejected.append(filename)
            continue
        dst = uniquify_path(target_dir / filename)
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
    return saved, rejected


async def upload_files_to_workspace(session_id: str, files: Iterable[UploadFile]) -> dict:
    workspace_root = resolve_workspace_root(session_id)
    saved, rejected = await _save_uploads(workspace_root, workspace_root, files)
    return {
        "message": f"Successfully uploaded {len(saved)} files",
        "files": saved,
        "rejected": rejected,
    }


async def upload_files_to_dir(session_id: str, directory: str, files: Iterable[UploadFile]) -> dict:
    workspace_root = resolve_workspace_root(session_id)
    target_dir = resolve_workspace_path(session_id, directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved, rejected = await _save_uploads(workspace_root, target_dir, files)
    return {"message": f"uploaded {len(saved)}", "files": saved, "rejected": rejected}


def clear_workspace(session_id: str) -> dict:
    workspace_root = resolve_workspace_root(session_id)
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    return {"message": "Workspace cleared successfully"}
