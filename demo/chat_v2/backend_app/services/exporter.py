from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from ..settings import settings
from .workspace import (
    build_download_url,
    get_session_workspace,
    register_generated_paths,
    uniquify_path,
)


def extract_sections_from_messages(messages: list[dict[str, Any]]) -> str:
    if not isinstance(messages, list):
        return ""

    parts: list[str] = []
    appendix: list[str] = []
    tag_pattern = r"<(Analyze|Understand|Code|Execute|File|Answer)>([\s\S]*?)</\1>"

    for message in messages:
        if (message or {}).get("role") != "assistant":
            continue

        content = str((message or {}).get("content") or "")
        step = 1
        for match in re.finditer(tag_pattern, content, re.DOTALL):
            tag, segment = match.groups()
            segment = segment.strip()
            if tag == "Answer":
                parts.append(f"{segment}\n")
            appendix.append(f"\n### Step {step}: {tag}\n\n{segment}\n")
            step += 1

    final_text = "".join(parts).strip()
    if appendix:
        final_text += (
            "\n\n\\newpage\n\n# Appendix: Detailed Process\n"
            + "".join(appendix).strip()
        )
    return final_text


def save_md(md_text: str, base_name: str, workspace_dir: str) -> Path:
    target_dir = Path(workspace_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    md_path = uniquify_path(target_dir / f"{base_name}.md")
    md_path.write_text(md_text, encoding="utf-8")
    return md_path


_MARKDOWN_LINK_RE = re.compile(r"^\s*-\s+\[([^\]]+)\]\(([^)]+)\)\s*$")
_MARKDOWN_IMAGE_RE = re.compile(r"^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
PDF_EXPORT_BASE_ARGS = [
    "--standalone",
    "--pdf-engine=xelatex",
    "-V",
    "geometry=top=2cm,bottom=2.1cm,left=2.1cm,right=2.1cm",
]
PDF_CJK_FONT_CANDIDATES = [
    "SimHei",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "SimSun",
    "PingFang SC",
]


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_CHAR_RE.search(text or ""))


def _build_pdf_export_args(cjk_font: str | None = None) -> list[str]:
    args = list(PDF_EXPORT_BASE_ARGS)
    if cjk_font:
        args.extend(["-V", f"CJKmainfont={cjk_font}"])
    return args


def _resolve_cjk_font_candidates() -> list[str]:
    candidates: list[str] = []
    configured_font = settings.pdf_cjk_mainfont.strip()
    if configured_font:
        candidates.append(configured_font)
    for font in PDF_CJK_FONT_CANDIDATES:
        if font not in candidates:
            candidates.append(font)
    return candidates


def _is_workspace_child(candidate: Path, workspace_root: Path) -> bool:
    try:
        resolved = candidate.resolve()
        workspace_resolved = workspace_root.resolve()
    except Exception:
        return False
    return resolved == workspace_resolved or workspace_resolved in resolved.parents


def _resolve_pdf_asset_path(raw_target: str, workspace_root: Path) -> Path | None:
    target = str(raw_target or "").strip()
    if not target:
        return None

    parsed = None
    if target.startswith("/workspace/download"):
        parsed = urlparse(f"http://local{target}")
    elif re.match(r"^https?://", target, re.IGNORECASE):
        parsed = urlparse(target)

    if parsed is not None:
        if parsed.path == "/workspace/download":
            relative_path = parse_qs(parsed.query).get("path", [""])[0]
            if relative_path:
                candidate = (workspace_root / unquote(relative_path)).resolve()
                if candidate.exists() and candidate.is_file() and _is_workspace_child(
                    candidate,
                    workspace_root,
                ):
                    return candidate
        else:
            parts = Path(unquote(parsed.path.lstrip("/"))).parts
            workspace_name = workspace_root.name
            if workspace_name in parts:
                workspace_index = parts.index(workspace_name)
                relative_parts = parts[workspace_index + 1 :]
                if relative_parts:
                    candidate = (workspace_root / Path(*relative_parts)).resolve()
                    if candidate.exists() and candidate.is_file() and _is_workspace_child(
                        candidate,
                        workspace_root,
                    ):
                        return candidate
        return None

    for candidate in (
        (workspace_root / target).resolve(),
        (workspace_root / "generated" / target).resolve(),
    ):
        if candidate.exists() and candidate.is_file() and _is_workspace_child(
            candidate,
            workspace_root,
        ):
            return candidate
    return None


def _build_pdf_image_block(file_name: str, image_path: Path) -> list[str]:
    image_target = image_path.resolve().as_posix()
    return [
        "",
        f"![{file_name}](<{image_target}>)",
        "",
        f"`{file_name}`",
        "",
    ]


def prepare_pdf_markdown(md_text: str, workspace_root: Path) -> str:
    lines = md_text.splitlines()
    rendered: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        link_match = _MARKDOWN_LINK_RE.match(line)
        image_match = _MARKDOWN_IMAGE_RE.match(next_line)

        if link_match and image_match:
            file_name = link_match.group(1).strip() or image_match.group(1).strip() or "image"
            image_path = _resolve_pdf_asset_path(image_match.group(2), workspace_root)
            if image_path is not None:
                rendered.extend(_build_pdf_image_block(file_name, image_path))
                index += 2
                continue

        single_image_match = _MARKDOWN_IMAGE_RE.match(line)
        if single_image_match:
            image_path = _resolve_pdf_asset_path(single_image_match.group(2), workspace_root)
            if image_path is not None:
                alt = single_image_match.group(1).strip() or image_path.name
                rendered.extend(_build_pdf_image_block(alt, image_path))
                index += 1
                continue

        rendered.append(line)
        index += 1

    return "\n".join(rendered).strip() + "\n"


def _build_pdf_export_result(
    *,
    path: Path | None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "path": path,
        "status": status,
        "error": error,
    }


def _classify_pdf_export_failure(exc: Exception) -> tuple[str, str]:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()

    if "pandoc" in lowered and (
        "not found" in lowered
        or "couldn't find" in lowered
        or "could not find" in lowered
        or "no pandoc was found" in lowered
    ):
        return "missing_dependency", message

    if any(
        token in lowered
        for token in (
            "xelatex not found",
            "could not find xelatex",
            "couldn't find xelatex",
            "latexmk not found",
            "pdflatex not found",
        )
    ):
        return "missing_compiler", message

    return "failed", message


def _resolve_pandoc_cache_dir() -> Path:
    configured = settings.pdf_pandoc_cache_dir.strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(settings.workspace_base_dir) / ".tools" / "pandoc").resolve()


def _find_pandoc_binary(search_root: Path) -> Path | None:
    if not search_root.exists() or not search_root.is_dir():
        return None

    direct_candidates = (
        search_root / "pandoc",
        search_root / "pandoc.exe",
        search_root / "bin" / "pandoc",
        search_root / "bin" / "pandoc.exe",
    )
    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    for candidate in search_root.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.name.lower() not in {"pandoc", "pandoc.exe"}:
            continue
        return candidate.resolve()
    return None


def _ensure_pandoc_available(pypandoc: Any) -> tuple[bool, str | None]:
    try:
        pypandoc.get_pandoc_version()
        return True, None
    except Exception as first_exc:
        first_error = str(first_exc).strip() or "pandoc is not available"

    if not settings.pdf_auto_download_pandoc:
        return False, first_error

    cache_dir = _resolve_pandoc_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_binary = _find_pandoc_binary(cache_dir)
    if cached_binary is not None:
        os.environ["PYPANDOC_PANDOC"] = str(cached_binary)
        try:
            pypandoc.get_pandoc_version()
            return True, None
        except Exception:
            pass

    try:
        pypandoc.download_pandoc(targetfolder=str(cache_dir))
    except Exception as download_exc:
        return (
            False,
            f"{first_error}; auto-download failed: {str(download_exc).strip() or download_exc.__class__.__name__}",
        )

    downloaded_binary = _find_pandoc_binary(cache_dir)
    if downloaded_binary is None:
        return (
            False,
            f"{first_error}; downloaded pandoc binary not found under {cache_dir}",
        )

    os.environ["PYPANDOC_PANDOC"] = str(downloaded_binary)
    try:
        pypandoc.get_pandoc_version()
        return True, None
    except Exception as verify_exc:
        return (
            False,
            f"{first_error}; downloaded pandoc is unusable: {str(verify_exc).strip() or verify_exc.__class__.__name__}",
        )


def save_pdf(md_text: str, base_name: str, workspace_dir: str) -> dict[str, Any]:
    try:
        import pypandoc
    except Exception as exc:
        return _build_pdf_export_result(
            path=None,
            status="missing_dependency",
            error=str(exc) or "pypandoc is not installed",
        )

    if shutil.which("xelatex") is None:
        return _build_pdf_export_result(
            path=None,
            status="missing_compiler",
            error="xelatex is not available in PATH",
        )

    pandoc_ready, pandoc_error = _ensure_pandoc_available(pypandoc)
    if not pandoc_ready:
        return _build_pdf_export_result(
            path=None,
            status="missing_dependency",
            error=pandoc_error or "pandoc is not available",
        )

    target_dir = Path(workspace_dir)
    workspace_root = target_dir.parent.parent.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = uniquify_path(target_dir / f"{base_name}.pdf")
    pdf_markdown = prepare_pdf_markdown(md_text, workspace_root)
    base_args = _build_pdf_export_args()
    has_cjk = _contains_cjk(pdf_markdown)

    def _convert(extra_args: list[str]) -> None:
        if pdf_path.exists():
            pdf_path.unlink(missing_ok=True)
        pypandoc.convert_text(
            pdf_markdown,
            "pdf",
            format="md",
            outputfile=str(pdf_path),
            extra_args=extra_args,
        )

    if not has_cjk:
        try:
            _convert(base_args)
            return _build_pdf_export_result(path=pdf_path, status="ok")
        except Exception as exc:
            status, error = _classify_pdf_export_failure(exc)
            return _build_pdf_export_result(path=None, status=status, error=error)

    last_exc: Exception | None = None
    for font in _resolve_cjk_font_candidates():
        try:
            _convert(_build_pdf_export_args(font))
            return _build_pdf_export_result(path=pdf_path, status="ok")
        except Exception as exc:
            last_exc = exc
            continue

    if last_exc is None:
        return _build_pdf_export_result(
            path=None,
            status="failed",
            error="PDF export failed for unknown reason",
        )

    status, error = _classify_pdf_export_failure(last_exc)
    tried_fonts = ", ".join(_resolve_cjk_font_candidates())
    return _build_pdf_export_result(
        path=None,
        status=status,
        error=f"{error}. Tried CJK fonts: {tried_fonts}",
    )


def _sanitize_filename_component(
    raw: str,
    *,
    fallback: str,
    max_length: int = 80,
) -> str:
    text = str(raw or "").strip()
    if not text:
        return fallback

    # Windows 禁止字符 + 控制字符（避免写文件时报错）
    text = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip(" ._")

    if not text:
        text = fallback

    if len(text) > max_length:
        text = text[:max_length].rstrip(" ._") or fallback

    return text


def _build_export_base_name(title: str, *, prefix: str, timestamp: str) -> str:
    safe_title = _sanitize_filename_component(title, fallback=prefix, max_length=80)
    return f"{safe_title}_{timestamp}"


def _to_file_meta(
    session_id: str,
    workspace_root: Path,
    file_path: Path | None,
) -> dict[str, Any] | None:
    if file_path is None:
        return None
    rel_path = file_path.relative_to(workspace_root).as_posix()
    return {
        "name": file_path.name,
        "path": rel_path,
        "download_url": build_download_url(f"{session_id}/{rel_path}"),
    }


def export_report_from_body(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    title = (body.get("title") or "").strip()
    session_id = body.get("session_id", "default")
    workspace_dir = get_session_workspace(session_id)
    workspace_root = Path(workspace_dir)

    md_text = extract_sections_from_messages(messages)
    if not md_text:
        md_text = "(No <Analyze>/<Understand>/<Code>/<Execute>/<Answer> sections found.)"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = _build_export_base_name(title, prefix="Report", timestamp=timestamp)

    export_dir = workspace_root / "generated" / "reports"
    export_dir.mkdir(parents=True, exist_ok=True)

    md_path = save_md(md_text, base_name, str(export_dir))
    pdf_result = save_pdf(md_text, base_name, str(export_dir))
    pdf_path = pdf_result["path"]
    register_generated_paths(
        session_id,
        [
            md_path.relative_to(workspace_root).as_posix(),
            *(
                [pdf_path.relative_to(workspace_root).as_posix()]
                if pdf_path is not None
                else []
            ),
        ],
    )

    md_meta = _to_file_meta(session_id, workspace_root, md_path)
    pdf_meta = _to_file_meta(session_id, workspace_root, pdf_path)

    return {
        "message": "exported",
        "md": md_path.name,
        "pdf": pdf_path.name if pdf_path else None,
        "files": {
            "md": md_meta,
            "pdf": pdf_meta,
        },
        "pdf_status": pdf_result["status"],
        "pdf_error": pdf_result["error"],
        "download_urls": {
            "md": md_meta["download_url"] if md_meta else None,
            "pdf": pdf_meta["download_url"] if pdf_meta else None,
        },
    }
