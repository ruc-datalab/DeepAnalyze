"""
Utility functions for DeepAnalyze API Server
Contains helper functions for file operations, workspace management, and more
"""

import os
import json
import re
import shutil
import sys
import traceback
import subprocess
import tempfile
import http.server
import socketserver
from pathlib import Path
from urllib.parse import quote
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from functools import partial

from config import WORKSPACE_BASE_DIR, HTTP_SERVER_PORT


def get_thread_workspace(thread_id: str) -> str:
    """Get workspace directory for a thread"""
    workspace_dir = os.path.join(WORKSPACE_BASE_DIR, thread_id)
    os.makedirs(workspace_dir, exist_ok=True)
    return workspace_dir


def build_download_url(thread_id: str, rel_path: str) -> str:
    """Build download URL for a file"""
    try:
        encoded = quote(f"{thread_id}/{rel_path}", safe="/")
    except Exception:
        encoded = f"{thread_id}/{rel_path}"
    return f"http://localhost:{HTTP_SERVER_PORT}/{encoded}"


def uniquify_path(target: Path) -> Path:
    """Return a unique path if target already exists"""
    return target


def _normalize_openai_message_content(raw_content: Any) -> str:
    """Normalize OpenAI-style message content into a plain string."""
    if isinstance(raw_content, list):
        parts: List[str] = []
        for item in raw_content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and "text" in item
            ):
                parts.append(item.get("text", {}).get("value", ""))
        return "".join(parts)
    return str(raw_content or "")


def extract_text_from_content(content: List[Dict[str, Any]]) -> str:
    """Extract plain text from message content items."""
    text_parts: List[str] = []
    for item in content or []:
        if isinstance(item, dict) and item.get("type") == "text":
            text_parts.append(item.get("text", {}).get("value", ""))
    return "".join(text_parts)


def collect_file_info(directory: str) -> str:
    """Collect file information from directory"""
    all_file_info_str = ""
    dir_path = Path(directory)
    if not dir_path.exists():
        return ""

    files = sorted([f for f in dir_path.iterdir() if f.is_file()])
    for idx, file_path in enumerate(files, start=1):
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024
        size_str = f"{size_kb:.1f}KB"
        file_info = {"name": file_path.name, "size": size_str}
        file_info_str = json.dumps(file_info, indent=4, ensure_ascii=False)
        all_file_info_str += f"File {idx}:\n{file_info_str}\n\n"
    return all_file_info_str


def collect_all_associated_files(thread_id: str, assistant, storage) -> List[str]:
    """
    Collect all file IDs from assistant, thread, and messages.
    Returns a list of unique file IDs.
    """
    all_file_ids = set()

    # Add assistant files
    all_file_ids.update(assistant.file_ids)

    # Add thread files
    thread = storage.get_thread(thread_id)
    if thread:
        all_file_ids.update(thread.file_ids)

    # Add message files
    messages = storage.list_messages(thread_id)
    for message in messages:
        all_file_ids.update(message.file_ids)

    # Add files from tool_resources analyze tool
    if thread and thread.tool_resources and "analyze" in thread.tool_resources:
        analyze_file_ids = thread.tool_resources["analyze"].get("file_ids", [])
        all_file_ids.update(analyze_file_ids)

    return list(all_file_ids)


def prepare_vllm_messages(
    messages: List[Dict[str, Any]],
    workspace_dir: str,
    assistant_instructions: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Convert incoming messages to vLLM format and inject DeepAnalyze template:
    - Always wrap user message with "# Instruction" heading
    - Optionally append workspace file info under "# Data"
    - If no user message exists, inject one with assistant instructions
    """
    vllm_messages: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else None
        raw_content = msg.get("content") if isinstance(msg, dict) else None
        content = _normalize_openai_message_content(raw_content)
        if role:
            vllm_messages.append({"role": role, "content": content})

    # Locate last user message
    last_user_idx: Optional[int] = None
    for idx in range(len(vllm_messages) - 1, -1, -1):
        if vllm_messages[idx].get("role") == "user":
            last_user_idx = idx
            break

    workspace_file_info = collect_file_info(workspace_dir)

    if last_user_idx is not None:
        user_content = str(vllm_messages[last_user_idx].get("content", "")).strip()
        instruction_parts: List[str] = []
        if assistant_instructions:
            instruction_parts.append(assistant_instructions.strip())
        if user_content:
            instruction_parts.append(user_content)
        instruction_body = "\n\n".join([p for p in instruction_parts if p])
        if workspace_file_info:
            vllm_messages[last_user_idx]["content"] = (
                f"# Instruction\n{instruction_body}\n\n# Data\n{workspace_file_info}"
            )
        else:
            vllm_messages[last_user_idx]["content"] = (
                f"# Instruction\n{instruction_body}" if instruction_body else "# Instruction"
            )
    elif assistant_instructions:
        # Inject initial instruction-only user message
        body = f"# Instruction\n{assistant_instructions.strip()}"
        if workspace_file_info:
            body = f"{body}\n\n# Data\n{workspace_file_info}"
        vllm_messages.insert(0, {"role": "user", "content": body})

    return vllm_messages


def execute_code_safe(
    code_str: str, workspace_dir: str, timeout_sec: int = 120
) -> str:
    """Execute Python code in a separate process with timeout"""
    exec_cwd = os.path.abspath(workspace_dir)
    os.makedirs(exec_cwd, exist_ok=True)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py", dir=exec_cwd)
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(code_str)

        child_env = os.environ.copy()
        child_env.setdefault("MPLBACKEND", "Agg")
        child_env.setdefault("QT_QPA_PLATFORM", "offscreen")
        child_env.pop("DISPLAY", None)

        completed = subprocess.run(
            [sys.executable, tmp_path],
            cwd=exec_cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=child_env,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return output
    except subprocess.TimeoutExpired:
        return f"[Timeout]: execution exceeded {timeout_sec} seconds"
    except Exception as e:
        return f"[Error]: {str(e)}"
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def extract_code_from_segment(segment: str) -> Optional[str]:
    """Extract python code between <Code>...</Code>, optionally fenced by ```python ... ```"""
    code_match = re.search(r"<Code>(.*?)</Code>", segment, re.DOTALL)
    if not code_match:
        return None
    code_content = code_match.group(1).strip()
    md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
    return (md_match.group(1).strip() if md_match else code_content)


def fix_tags_and_codeblock(s: str) -> str:
    """Fix unclosed tags and code blocks"""
    pattern = re.compile(
        r"<(Analyze|Understand|Code|Execute|Answer)>(.*?)(?:</\1>|(?=$))", re.DOTALL
    )
    matches = list(pattern.finditer(s))
    if not matches:
        return s

    last_match = matches[-1]
    tag_name = last_match.group(1)
    matched_text = last_match.group(0)

    if not matched_text.endswith(f"</{tag_name}>"):
        if tag_name == "Code":
            if "```" in s and s.count("```") % 2 != 0:
                s += "\n```"
        s += f"\n</{tag_name}>"

    return s


def extract_sections_from_history(messages: List[Dict[str, str]]) -> str:
    """Build report body and appendix from tagged assistant messages."""
    if not isinstance(messages, list):
        return ""

    parts: List[str] = []
    appendix: List[str] = []
    tag_pattern = re.compile(r"<(Analyze|Understand|Code|Execute|File|Answer)>([\s\S]*?)</\1>")

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if (msg.get("role") or "").lower() != "assistant":
            continue
        content = str(msg.get("content", ""))
        step = 1
        for match in tag_pattern.finditer(content):
            tag, segment = match.groups()
            segment = (segment or "").strip()
            if not segment:
                continue
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

    return final_text.strip()


def save_markdown_report(md_text: str, base_name: str, target_dir: Path) -> Path:
    """Persist markdown report under target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    md_path = uniquify_path(target_dir / f"{base_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    return md_path


def generate_markdown_report(
    thread_id: str,
    history_messages,
    final_reply: str,
    workspace_dir: str,
) -> Optional[Path]:
    """Create markdown report from conversation history."""
    history: List[Dict[str, str]] = []
    for msg in history_messages:
        history.append({
            "role": msg.role,
            "content": extract_text_from_content(msg.content),
        })
    history.append({"role": "assistant", "content": final_reply})

    md_text = extract_sections_from_history(history)
    if not md_text:
        md_text = "(No <Analyze>/<Understand>/<Code>/<Execute>/<File>/<Answer> sections found.)"

    export_dir = Path(workspace_dir) / "generated"
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"Report_{timestamp}"
    md_path = save_markdown_report(md_text, base_name, export_dir)
    return md_path


class WorkspaceTracker:
    """Track workspace file changes and collect artifacts into generated/ folder."""

    def __init__(self, workspace_dir: str, generated_dir: str):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.generated_dir = Path(generated_dir).resolve()
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.before_state = self._snapshot()

    def _snapshot(self) -> Dict[Path, Tuple[int, int]]:
        try:
            return {
                p.resolve(): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in self.workspace_dir.rglob("*")
                if p.is_file()
            }
        except Exception:
            return {}

    def diff_and_collect(self) -> List[Path]:
        """Compute added/modified files, copy into generated/, and return artifact paths."""
        try:
            after_state = {
                p.resolve(): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in self.workspace_dir.rglob("*")
                if p.is_file()
            }
        except Exception:
            after_state = {}

        added = [p for p in after_state.keys() if p not in self.before_state]
        modified = [
            p for p in after_state.keys()
            if p in self.before_state and after_state[p] != self.before_state[p]
        ]

        artifact_paths: List[Path] = []
        for p in added:
            try:
                if not str(p).startswith(str(self.generated_dir)):
                    dest = self.generated_dir / p.name
                    dest = uniquify_path(dest)
                    shutil.copy2(str(p), str(dest))
                    artifact_paths.append(dest.resolve())
                else:
                    artifact_paths.append(p)
            except Exception as e:
                print(f"Error moving file {p}: {e}")

        for p in modified:
            try:
                dest = self.generated_dir / f"{p.stem}_modified{p.suffix}"
                dest = uniquify_path(dest)
                shutil.copy2(str(p), str(dest))
                artifact_paths.append(dest.resolve())
            except Exception as e:
                print(f"Error copying modified file {p}: {e}")

        self.before_state = after_state
        return artifact_paths


def render_file_block(
    artifact_paths: List[Path],
    workspace_dir: str,
    thread_id: str,
    generated_files_sink: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the <File> markdown block and optionally collect generated file metadata."""
    if not artifact_paths:
        return ""

    lines = ["<File>"]
    for p in artifact_paths:
        try:
            rel = Path(p).resolve().relative_to(Path(workspace_dir).resolve()).as_posix()
        except Exception:
            rel = Path(p).name
        url = build_download_url(thread_id, rel)
        name = Path(p).name
        lines.append(f"- [{name}]({url})")
        #考虑去重
        if generated_files_sink is not None :
            if {"name": name, "url": url} not in generated_files_sink:
                generated_files_sink.append({"name": name, "url": url})
        if Path(p).suffix.lower() in [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".svg",
        ]:
            lines.append(f"![{name}]({url})")
    lines.append("</File>")
    return "\n" + "\n".join(lines) + "\n"


def generate_report_from_messages(
    original_messages: List[Dict[str, Any]],
    assistant_reply: str,
    workspace_dir: str,
    thread_id: str,
    generated_files_sink: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Generate markdown report from conversation history and return file block.

    Args:
        original_messages: Original message list from the API request
        assistant_reply: Complete assistant response text
        workspace_dir: Workspace directory path
        thread_id: Thread ID for building download URLs
        generated_files_sink: Optional list to append generated file metadata

    Returns:
        File block string with report link, or empty string on failure
    """
    # Build conversation history for report generation
    history_records: List[Dict[str, str]] = []
    for raw_msg in original_messages:
        role = raw_msg.get("role", "") if isinstance(raw_msg, dict) else ""
        raw_content = raw_msg.get("content", "") if isinstance(raw_msg, dict) else ""
        content_text = _normalize_openai_message_content(raw_content)
        history_records.append({"role": role, "content": content_text})

    history_records.append({"role": "assistant", "content": assistant_reply})

    try:
        md_text = extract_sections_from_history(history_records)
        if not md_text:
            md_text = (
                "(No <Analyze>/<Understand>/<Code>/<Execute>/<File>/<Answer> "
                "sections found.)"
            )

        export_dir = Path(workspace_dir) / "generated"
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"Conversation_Report_{timestamp}"
        report_path = save_markdown_report(md_text, base_name, export_dir)

        try:
            rel = report_path.resolve().relative_to(Path(workspace_dir).resolve())
            rel_path = rel.as_posix()
        except Exception:
            rel_path = report_path.name

        url = build_download_url(thread_id, rel_path)

        if generated_files_sink is not None:
            generated_files_sink.append({"name": report_path.name, "url": url})

        lines = ["<File>", f"- [{report_path.name}]({url})", "</File>"]
        return "\n" + "\n".join(lines) + "\n"

    except Exception as report_error:
        print(f"Report generation error: {report_error}")
        return ""


def start_http_server():
    """Start HTTP file server for workspace files"""
    os.makedirs(WORKSPACE_BASE_DIR, exist_ok=True)
    handler = partial(
        http.server.SimpleHTTPRequestHandler, directory=WORKSPACE_BASE_DIR
    )

    class SafeTCPServer(socketserver.TCPServer):
        """TCPServer with timeout to prevent blocking"""
        allow_reuse_address = True

        def serve_forever(self, poll_interval=0.5):
            """Override serve_forever with timeout"""
            self._BaseServer__shutdown_request = False
            while not self._BaseServer__shutdown_request:
                try:
                    self.handle_request()
                except (OSError, KeyboardInterrupt):
                    break
                except Exception as e:
                    print(f"HTTP server error: {e}")
                    continue

    with SafeTCPServer(("", HTTP_SERVER_PORT), handler) as httpd:
        httpd.timeout = 1  # Add timeout to prevent blocking
        print(f"HTTP Server serving {WORKSPACE_BASE_DIR} at port {HTTP_SERVER_PORT}")
        try:
            httpd.serve_forever(poll_interval=0.1)  # Use polling instead of blocking
        except KeyboardInterrupt:
            print("HTTP server shutting down...")
        except Exception as e:
            print(f"HTTP server error: {e}")