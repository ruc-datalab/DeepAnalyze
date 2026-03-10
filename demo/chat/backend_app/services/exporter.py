from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

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


def build_history_markdown(
    messages: list[dict[str, Any]],
    *,
    title: str = "",
    exported_at: str = "",
) -> str:
    lines: list[str] = []
    heading = title.strip() or "Conversation History"
    lines.append(f"# {heading}")
    if exported_at:
        lines.append("")
        lines.append(f"- Exported at: {exported_at}")

    for index, message in enumerate(messages, start=1):
        role = str((message or {}).get("role") or "unknown").strip() or "unknown"
        content = str((message or {}).get("content") or "").strip()
        timestamp = str((message or {}).get("timestamp") or "").strip()
        attachments = (message or {}).get("attachments") or []

        lines.append("")
        lines.append(f"## {index}. {role.title()}")
        if timestamp:
            lines.append("")
            lines.append(f"- Timestamp: {timestamp}")
        if attachments:
            lines.append("")
            lines.append("- Attachments:")
            for attachment in attachments:
                name = str((attachment or {}).get("name") or "unnamed")
                lines.append(f"  - {name}")
        lines.append("")
        lines.append(content or "(empty)")

    return "\n".join(lines).strip() + "\n"


def save_md(md_text: str, base_name: str, workspace_dir: str) -> Path:
    target_dir = Path(workspace_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    md_path = uniquify_path(target_dir / f"{base_name}.md")
    md_path.write_text(md_text, encoding="utf-8")
    return md_path


def save_json(payload: dict[str, Any], base_name: str, workspace_dir: str) -> Path:
    target_dir = Path(workspace_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = uniquify_path(target_dir / f"{base_name}.json")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return json_path


def save_pdf(md_text: str, base_name: str, workspace_dir: str) -> Path | None:
    try:
        import pypandoc
    except Exception:
        return None

    target_dir = Path(workspace_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = uniquify_path(target_dir / f"{base_name}.pdf")
    try:
        pypandoc.convert_text(
            md_text,
            "pdf",
            format="md",
            outputfile=str(pdf_path),
            extra_args=["--standalone", "--pdf-engine=xelatex"],
        )
        return pdf_path
    except Exception:
        return None


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
    safe_title = re.sub(r"[^\w\-_.]+", "_", title) if title else "Report"
    base_name = f"{safe_title}_{timestamp}" if title else f"Report_{timestamp}"

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


def export_history_from_body(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    title = (body.get("title") or "").strip()
    session_id = body.get("session_id", "default")
    workspace_dir = get_session_workspace(session_id)
    workspace_root = Path(workspace_dir)

    exported_at = datetime.now().isoformat(timespec="seconds")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^\w\-_.]+", "_", title) if title else "History"
    base_name = f"{safe_title}_{timestamp}" if title else f"History_{timestamp}"

    export_dir = workspace_root / "generated" / "history"
    export_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "title": title or "Conversation History",
        "session_id": session_id,
        "exported_at": exported_at,
        "message_count": len(messages),
        "messages": messages,
    }
    md_text = build_history_markdown(
        messages,
        title=title or "Conversation History",
        exported_at=exported_at,
    )

    json_path = save_json(payload, base_name, str(export_dir))
    md_path = save_md(md_text, base_name, str(export_dir))
    register_generated_paths(
        session_id,
        [
            json_path.relative_to(workspace_root).as_posix(),
            md_path.relative_to(workspace_root).as_posix(),
        ],
    )

    json_meta = _to_file_meta(session_id, workspace_root, json_path)
    md_meta = _to_file_meta(session_id, workspace_root, md_path)

    return {
        "message": "exported",
        "json": json_path.name,
        "md": md_path.name,
        "files": {
            "json": json_meta,
            "md": md_meta,
        },
        "download_urls": {
            "json": json_meta["download_url"] if json_meta else None,
            "md": md_meta["download_url"] if md_meta else None,
        },
    }
