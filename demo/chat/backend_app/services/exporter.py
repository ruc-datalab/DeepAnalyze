from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

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


def save_pdf(md_text: str, base_name: str, workspace_dir: str) -> Path | None:
    try:
        import pypandoc
    except Exception:
        return None

    target_dir = Path(workspace_dir)
    workspace_root = target_dir.parent.parent.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = uniquify_path(target_dir / f"{base_name}.pdf")
    pdf_markdown = prepare_pdf_markdown(md_text, workspace_root)
    try:
        pypandoc.convert_text(
            pdf_markdown,
            "pdf",
            format="md",
            outputfile=str(pdf_path),
            extra_args=["--standalone", "--pdf-engine=xelatex"],
        )
        return pdf_path
    except Exception:
        return None


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
    pdf_path = save_pdf(md_text, base_name, str(export_dir))
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
        "download_urls": {
            "md": md_meta["download_url"] if md_meta else None,
            "pdf": pdf_meta["download_url"] if pdf_meta else None,
        },
    }
