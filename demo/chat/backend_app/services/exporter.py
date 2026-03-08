from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .workspace import build_download_url, get_session_workspace, uniquify_path


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
    with open(md_path, "w", encoding="utf-8") as file:
        file.write(md_text)
    return md_path


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


def export_report_from_body(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    title = (body.get("title") or "").strip()
    session_id = body.get("session_id", "default")
    workspace_dir = get_session_workspace(session_id)

    md_text = extract_sections_from_messages(messages)
    if not md_text:
        md_text = "(No <Analyze>/<Understand>/<Code>/<Execute>/<Answer> sections found.)"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^\w\-_.]+", "_", title) if title else "Report"
    base_name = f"{safe_title}_{timestamp}" if title else f"Report_{timestamp}"

    export_dir = Path(workspace_dir) / "generated"
    export_dir.mkdir(parents=True, exist_ok=True)

    md_path = save_md(md_text, base_name, str(export_dir))
    pdf_path = save_pdf(md_text, base_name, str(export_dir))

    return {
        "message": "exported",
        "md": md_path.name,
        "pdf": pdf_path.name if pdf_path else None,
        "download_urls": {
            "md": build_download_url(f"{session_id}/generated/{md_path.name}"),
            "pdf": (
                build_download_url(f"{session_id}/generated/{pdf_path.name}")
                if pdf_path
                else None
            ),
        },
    }
