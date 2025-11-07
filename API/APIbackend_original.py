"""
OpenAI-compatible API Server for DeepAnalyze
Implements Assistants API with file management and code execution capabilities.
"""

import openai
import json
import os
import shutil
import re
import io
import contextlib
import traceback
import time
import uuid
from pathlib import Path
from urllib.parse import quote
import subprocess
import sys
import tempfile
import threading
import http.server
from functools import partial
import socketserver
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal, Tuple
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Body
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Environment setup
os.environ.setdefault("MPLBACKEND", "Agg")

# Configuration
API_BASE = "http://localhost:8000/v1"  # vllm api endpoint
MODEL_PATH = "DeepAnalyze-8B"
WORKSPACE_BASE_DIR = "workspace"
HTTP_SERVER_PORT = 8100
HTTP_SERVER_BASE = f"http://localhost:{HTTP_SERVER_PORT}"

# Initialize OpenAI client for vllm
vllm_client = openai.OpenAI(base_url=API_BASE, api_key="dummy")

# FastAPI app
app = FastAPI(title="DeepAnalyze OpenAI-Compatible API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Data Models (OpenAI Compatible)
# ============================================================================


class FileObject(BaseModel):
    """OpenAI File Object"""

    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str


class FileDeleteResponse(BaseModel):
    """OpenAI File Delete Response"""

    id: str
    object: Literal["file"] = "file"
    deleted: bool


class AssistantObject(BaseModel):
    """OpenAI Assistant Object"""

    id: str
    object: Literal["assistant"] = "assistant"
    created_at: int
    name: Optional[str] = None
    description: Optional[str] = None
    model: str
    instructions: Optional[str] = None
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    file_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ThreadObject(BaseModel):
    """OpenAI Thread Object"""

    id: str
    object: Literal["thread"] = "thread"
    created_at: int
    last_accessed_at: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    file_ids: List[str] = Field(default_factory=list)


class MessageObject(BaseModel):
    """OpenAI Message Object"""

    id: str
    object: Literal["thread.message"] = "thread.message"
    created_at: int
    thread_id: str
    role: Literal["user", "assistant"]
    content: List[Dict[str, Any]]
    file_ids: List[str] = Field(default_factory=list)
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunObject(BaseModel):
    """OpenAI Run Object"""

    id: str
    object: Literal["thread.run"] = "thread.run"
    created_at: int
    thread_id: str
    assistant_id: str
    status: Literal[
        "queued",
        "in_progress",
        "requires_action",
        "cancelling",
        "cancelled",
        "failed",
        "completed",
        "expired",
    ]
    required_action: Optional[Dict[str, Any]] = None
    last_error: Optional[Dict[str, Any]] = None
    expires_at: int
    started_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    failed_at: Optional[int] = None
    completed_at: Optional[int] = None
    model: str
    instructions: Optional[str] = None
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    file_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Storage Layer (In-Memory for simplicity, can be replaced with DB)
# ============================================================================


class Storage:
    """Simple in-memory storage for OpenAI objects"""

    def __init__(self):
        self.files: Dict[str, Dict[str, Any]] = {}
        self.assistants: Dict[str, Dict[str, Any]] = {}
        self.threads: Dict[str, Dict[str, Any]] = {}
        self.messages: Dict[str, List[Dict[str, Any]]] = {}  # thread_id -> messages
        self.runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_file(self, filename: str, filepath: str, purpose: str) -> FileObject:
        with self._lock:
            file_id = f"file-{uuid.uuid4().hex[:24]}"
            file_size = os.path.getsize(filepath)
            file_obj = {
                "id": file_id,
                "object": "file",
                "bytes": file_size,
                "created_at": int(time.time()),
                "filename": filename,
                "purpose": purpose,
                "filepath": filepath,
            }
            self.files[file_id] = file_obj
            return FileObject(**file_obj)

    def get_file(self, file_id: str) -> Optional[FileObject]:
        with self._lock:
            if file_id in self.files:
                return FileObject(**self.files[file_id])
            return None

    def delete_file(self, file_id: str) -> bool:
        with self._lock:
            if file_id in self.files:
                filepath = self.files[file_id].get("filepath")
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
                del self.files[file_id]
                return True
            return False

    def list_files(self, purpose: Optional[str] = None) -> List[FileObject]:
        with self._lock:
            files = list(self.files.values())
            if purpose:
                files = [f for f in files if f.get("purpose") == purpose]
            return [FileObject(**f) for f in files]

    def create_assistant(
        self,
        model: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        file_ids: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> AssistantObject:
        with self._lock:
            assistant_id = f"asst-{uuid.uuid4().hex[:24]}"
            assistant = {
                "id": assistant_id,
                "object": "assistant",
                "created_at": int(time.time()),
                "name": name,
                "description": description,
                "model": model,
                "instructions": instructions,
                "tools": tools or [],
                "file_ids": file_ids or [],
                "metadata": metadata or {},
            }
            self.assistants[assistant_id] = assistant
            return AssistantObject(**assistant)

    def get_assistant(self, assistant_id: str) -> Optional[AssistantObject]:
        with self._lock:
            if assistant_id in self.assistants:
                return AssistantObject(**self.assistants[assistant_id])
            return None

    def delete_assistant(self, assistant_id: str) -> bool:
        with self._lock:
            if assistant_id in self.assistants:
                del self.assistants[assistant_id]
                return True
            return False

    def list_assistants(self) -> List[AssistantObject]:
        with self._lock:
            return [AssistantObject(**a) for a in self.assistants.values()]

    def create_thread(self, metadata: Optional[Dict] = None, file_ids: Optional[List[str]] = None) -> ThreadObject:
        with self._lock:
            thread_id = f"thread-{uuid.uuid4().hex[:24]}"
            now = int(time.time())
            thread = {
                "id": thread_id,
                "object": "thread",
                "created_at": now,
                "last_accessed_at": now,
                "metadata": metadata or {},
                "file_ids": file_ids or [],
            }
            self.threads[thread_id] = thread
            self.messages[thread_id] = []
            # Create workspace for this thread
            workspace_dir = get_thread_workspace(thread_id)
            os.makedirs(workspace_dir, exist_ok=True)
            os.makedirs(os.path.join(workspace_dir, "generated"), exist_ok=True)

            # Copy files to thread workspace
            if file_ids:
                for fid in file_ids:
                    if fid in self.files:
                        file_data = self.files[fid]
                        src_path = file_data.get("filepath")
                        if src_path and os.path.exists(src_path):
                            dst_path = uniquify_path(Path(workspace_dir) / file_data["filename"])
                            shutil.copy2(src_path, dst_path)

            return ThreadObject(**thread)

    def get_thread(self, thread_id: str) -> Optional[ThreadObject]:
        with self._lock:
            if thread_id in self.threads:
                # Update last accessed time
                self.threads[thread_id]["last_accessed_at"] = int(time.time())
                return ThreadObject(**self.threads[thread_id])
            return None

    def delete_thread(self, thread_id: str) -> bool:
        with self._lock:
            if thread_id in self.threads:
                del self.threads[thread_id]
                if thread_id in self.messages:
                    del self.messages[thread_id]
                # Clean up workspace
                workspace_dir = get_thread_workspace(thread_id)
                if os.path.exists(workspace_dir):
                    shutil.rmtree(workspace_dir)
                return True
            return False

    def create_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        file_ids: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> MessageObject:
        with self._lock:
            if thread_id not in self.threads:
                raise ValueError(f"Thread {thread_id} not found")
            message_id = f"msg-{uuid.uuid4().hex[:24]}"
            message = {
                "id": message_id,
                "object": "thread.message",
                "created_at": int(time.time()),
                "thread_id": thread_id,
                "role": role,
                "content": [{"type": "text", "text": {"value": content}}],
                "file_ids": file_ids or [],
                "assistant_id": None,
                "run_id": None,
                "metadata": metadata or {},
            }
            self.messages[thread_id].append(message)
            return MessageObject(**message)

    def list_messages(self, thread_id: str) -> List[MessageObject]:
        with self._lock:
            if thread_id not in self.messages:
                return []
            return [MessageObject(**m) for m in self.messages[thread_id]]

    def create_run(
        self,
        thread_id: str,
        assistant_id: str,
        model: str,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> RunObject:
        with self._lock:
            if thread_id not in self.threads:
                raise ValueError(f"Thread {thread_id} not found")
            if assistant_id not in self.assistants:
                raise ValueError(f"Assistant {assistant_id} not found")

            run_id = f"run-{uuid.uuid4().hex[:24]}"
            now = int(time.time())
            run = {
                "id": run_id,
                "object": "thread.run",
                "created_at": now,
                "thread_id": thread_id,
                "assistant_id": assistant_id,
                "status": "queued",
                "required_action": None,
                "last_error": None,
                "expires_at": now + 600,  # 10 minutes
                "started_at": None,
                "cancelled_at": None,
                "failed_at": None,
                "completed_at": None,
                "model": model,
                "instructions": instructions,
                "tools": tools or [],
                "file_ids": [],
                "metadata": metadata or {},
            }
            self.runs[run_id] = run
            return RunObject(**run)

    def get_run(self, thread_id: str, run_id: str) -> Optional[RunObject]:
        with self._lock:
            if run_id in self.runs and self.runs[run_id]["thread_id"] == thread_id:
                return RunObject(**self.runs[run_id])
            return None

    def update_run_status(
        self, run_id: str, status: str, **kwargs
    ) -> Optional[RunObject]:
        with self._lock:
            if run_id in self.runs:
                self.runs[run_id]["status"] = status
                for key, value in kwargs.items():
                    if key in self.runs[run_id]:
                        self.runs[run_id][key] = value
                return RunObject(**self.runs[run_id])
            return None

    def cleanup_expired_threads(self, timeout_hours: float = 12) -> int:
        """Clean up threads that haven't been accessed for more than timeout_hours"""
        with self._lock:
            now = int(time.time())
            timeout_seconds = int(timeout_hours * 3600)
            expired_threads = []

            for thread_id, thread_data in self.threads.items():
                last_accessed = thread_data.get("last_accessed_at", thread_data.get("created_at", 0))
                if now - last_accessed > timeout_seconds:
                    expired_threads.append(thread_id)

        cleaned_count = 0
        for thread_id in expired_threads:
            try:
                # Delete thread and its workspace
                if self.delete_thread(thread_id):
                    cleaned_count += 1
                    print(f"Cleaned up expired thread: {thread_id}")
            except Exception as e:
                print(f"Error cleaning up thread {thread_id}: {e}")

        return cleaned_count


storage = Storage()


# ============================================================================
# Thread Cleanup Scheduler
# ============================================================================

def schedule_thread_cleanup():
    """Background thread to periodically clean up expired threads"""
    import time as time_module

    while True:
        try:
            cleaned_count = storage.cleanup_expired_threads(timeout_hours=12)
            if cleaned_count > 0:
                print(f"Thread cleanup completed: removed {cleaned_count} expired threads")
        except Exception as e:
            print(f"Thread cleanup error: {e}")

        # Sleep for 30 minutes
        time_module.sleep(30 * 60)


# Start cleanup scheduler in background thread
cleanup_thread = threading.Thread(target=schedule_thread_cleanup, daemon=True)
cleanup_thread.start()
print("Thread cleanup scheduler started (runs every 30 minutes, 12-hour timeout)")


# ============================================================================
# Utility Functions
# ============================================================================


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
    return f"{HTTP_SERVER_BASE}/{encoded}"


def uniquify_path(target: Path) -> Path:
    return target

# ------------------------------
# Message preparation helpers
# ------------------------------

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


# ------------------------------
# Workspace tracking and artifacts
# ------------------------------

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
        if generated_files_sink is not None:
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


def extract_code_from_segment(segment: str) -> Optional[str]:
    """Extract python code between <Code>...</Code>, optionally fenced by ```python ... ```"""
    code_match = re.search(r"<Code>(.*?)</Code>", segment, re.DOTALL)
    if not code_match:
        return None
    code_content = code_match.group(1).strip()
    md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
    return (md_match.group(1).strip() if md_match else code_content)


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


def collect_all_associated_files(thread_id: str, assistant: AssistantObject) -> List[str]:
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

    return list(all_file_ids)


def extract_text_from_content(content: List[Dict[str, Any]]) -> str:
    """Extract plain text from message content items."""
    text_parts: List[str] = []
    for item in content or []:
        if isinstance(item, dict) and item.get("type") == "text":
            text_parts.append(item.get("text", {}).get("value", ""))
    return "".join(text_parts)


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
    history_messages: List[MessageObject],
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


# ============================================================================
# HTTP File Server
# ============================================================================


def start_http_server():
    """Start HTTP file server for workspace files"""
    os.makedirs(WORKSPACE_BASE_DIR, exist_ok=True)
    handler = partial(
        http.server.SimpleHTTPRequestHandler, directory=WORKSPACE_BASE_DIR
    )
    with socketserver.TCPServer(("", HTTP_SERVER_PORT), handler) as httpd:
        print(f"HTTP Server serving {WORKSPACE_BASE_DIR} at port {HTTP_SERVER_PORT}")
        httpd.serve_forever()


# Start HTTP server in a separate thread
threading.Thread(target=start_http_server, daemon=True).start()


# ============================================================================
# OpenAI Files API
# ============================================================================


@app.post("/v1/files", response_model=FileObject)
async def create_file(
    file: UploadFile = File(...), purpose: str = Form("assistants")
):
    """Upload a file (OpenAI compatible)"""
    # Validate purpose
    valid_purposes = ["assistants", "fine-tune", "answers"]
    if purpose not in valid_purposes:
        raise HTTPException(
            status_code=400, detail=f"Invalid purpose. Must be one of {valid_purposes}"
        )

    # Save file to a persistent location
    file_storage_dir = os.path.join(WORKSPACE_BASE_DIR, "_files")
    os.makedirs(file_storage_dir, exist_ok=True)

    file_id = f"file-{uuid.uuid4().hex[:24]}"
    file_path = os.path.join(file_storage_dir, file_id)

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_obj = storage.create_file(file.filename, file_path, purpose)
        return file_obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/files", response_model=Dict[str, Any])
async def list_files(purpose: Optional[str] = Query(None)):
    """List files (OpenAI compatible)"""
    files = storage.list_files(purpose=purpose)
    return {"object": "list", "data": [f.dict() for f in files]}


@app.get("/v1/files/{file_id}", response_model=FileObject)
async def retrieve_file(file_id: str):
    """Retrieve file metadata (OpenAI compatible)"""
    file_obj = storage.get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found")
    return file_obj


@app.delete("/v1/files/{file_id}", response_model=FileDeleteResponse)
async def delete_file(file_id: str):
    """Delete a file (OpenAI compatible)"""
    success = storage.delete_file(file_id)
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    return FileDeleteResponse(id=file_id, object="file", deleted=True)


@app.get("/v1/files/{file_id}/content")
async def download_file(file_id: str):
    """Download file content (OpenAI compatible)"""
    file_obj = storage.get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found")

    filepath = storage.files[file_id].get("filepath")
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File content not found")

    with open(filepath, "rb") as f:
        content = f.read()

    return Response(content=content, media_type="application/octet-stream")


# ============================================================================
# OpenAI Assistants API
# ============================================================================


@app.post("/v1/assistants", response_model=AssistantObject)
async def create_assistant(
    model: str = Body(...),
    name: Optional[str] = Body(None),
    description: Optional[str] = Body(None),
    instructions: Optional[str] = Body(None),
    tools: Optional[List[Dict]] = Body(None),
    file_ids: Optional[List[str]] = Body(None),
    metadata: Optional[Dict] = Body(None),
):
    """Create an assistant (OpenAI compatible)"""
    # Validate tools - only allow code_interpreter for now
    if tools:
        for tool in tools:
            if tool.get("type") not in ["NotImplemented"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported tool type: {tool.get('type')}. Only 'code_interpreter' is supported.",
                )

    # Validate file_ids
    if file_ids:
        for fid in file_ids:
            if not storage.get_file(fid):
                raise HTTPException(status_code=400, detail=f"File {fid} not found")

    assistant = storage.create_assistant(
        model=model,
        name=name,
        description=description,
        instructions=instructions,
        tools=tools,
        file_ids=file_ids,
        metadata=metadata,
    )
    return assistant


@app.get("/v1/assistants", response_model=Dict[str, Any])
async def list_assistants():
    """List assistants (OpenAI compatible)"""
    assistants = storage.list_assistants()
    return {"object": "list", "data": [a.dict() for a in assistants]}


@app.get("/v1/assistants/{assistant_id}", response_model=AssistantObject)
async def retrieve_assistant(assistant_id: str):
    """Retrieve an assistant (OpenAI compatible)"""
    assistant = storage.get_assistant(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return assistant


@app.delete("/v1/assistants/{assistant_id}")
async def delete_assistant(assistant_id: str):
    """Delete an assistant (OpenAI compatible)"""
    success = storage.delete_assistant(assistant_id)
    if not success:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return {"id": assistant_id, "object": "assistant.deleted", "deleted": True}


# ============================================================================
# OpenAI Threads API
# ============================================================================


@app.post("/v1/threads", response_model=ThreadObject)
async def create_thread(
    metadata: Optional[Dict] = Body(None),
    file_ids: Optional[List[str]] = Body(None),
):
    """Create a thread (OpenAI compatible)"""
    # Validate file_ids
    if file_ids:
        for fid in file_ids:
            if not storage.get_file(fid):
                raise HTTPException(status_code=400, detail=f"File {fid} not found")

    thread = storage.create_thread(metadata=metadata or {}, file_ids=file_ids)
    return thread


@app.get("/v1/threads/{thread_id}", response_model=ThreadObject)
async def retrieve_thread(thread_id: str):
    """Retrieve a thread (OpenAI compatible)"""
    thread = storage.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@app.delete("/v1/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a thread (OpenAI compatible)"""
    success = storage.delete_thread(thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"id": thread_id, "object": "thread.deleted", "deleted": True}


# ============================================================================
# OpenAI Messages API
# ============================================================================


@app.post("/v1/threads/{thread_id}/messages", response_model=MessageObject)
async def create_message(
    thread_id: str,
    role: Literal["user", "assistant"] = Body(...),
    content: str = Body(...),
    file_ids: Optional[List[str]] = Body(None),
    metadata: Optional[Dict] = Body(None),
):
    """Create a message (OpenAI compatible)"""
    thread = storage.get_thread(thread_id)  # This will update last_accessed_at
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Validate file_ids
    if file_ids:
        for fid in file_ids:
            if not storage.get_file(fid):
                raise HTTPException(status_code=400, detail=f"File {fid} not found")

    # Copy files to thread workspace
    if file_ids:
        workspace_dir = get_thread_workspace(thread_id)
        for fid in file_ids:
            file_obj = storage.get_file(fid)
            if file_obj:
                src_path = storage.files[fid].get("filepath")
                if src_path and os.path.exists(src_path):
                    dst_path = uniquify_path(Path(workspace_dir) / file_obj.filename)
                    shutil.copy2(src_path, dst_path)

    message = storage.create_message(
        thread_id=thread_id,
        role=role,
        content=content,
        file_ids=file_ids,
        metadata=metadata,
    )
    return message


@app.get("/v1/threads/{thread_id}/messages", response_model=Dict[str, Any])
async def list_messages(thread_id: str):
    """List messages in a thread (OpenAI compatible)"""
    thread = storage.get_thread(thread_id)  # This will update last_accessed_at
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    messages = storage.list_messages(thread_id)
    return {"object": "list", "data": [m.dict() for m in reversed(messages)]}


# ============================================================================
# OpenAI Runs API (Core Logic)
# ============================================================================


def bot_stream_for_run(thread_id: str, assistant: AssistantObject):
    """
    Execute the DeepAnalyze model in a streaming fashion.
    This is the core logic that runs the analysis workflow.
    """
    workspace_dir = get_thread_workspace(thread_id)
    generated_dir = os.path.join(workspace_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)

    # Collect all associated files from assistant, thread, and messages
    all_file_ids = collect_all_associated_files(thread_id, assistant)

    # Copy all associated files to workspace if they don't exist already
    for fid in all_file_ids:
        file_obj = storage.get_file(fid)
        if file_obj:
            src_path = storage.files[fid].get("filepath")
            if src_path and os.path.exists(src_path):
                dst_path = Path(workspace_dir) / file_obj.filename
                # Only copy if file doesn't already exist in workspace
                if not dst_path.exists():
                    dst_path = uniquify_path(dst_path)
                    shutil.copy2(src_path, dst_path)

    # Get messages from thread
    messages_list = storage.list_messages(thread_id)

    # Convert to generic dict format for preparation
    conversation_messages = list(messages_list)
    if conversation_messages and conversation_messages[0].role == "assistant":
        conversation_messages = conversation_messages[1:]
    payload_messages: List[Dict[str, Any]] = [
        {"role": m.role, "content": extract_text_from_content(m.content)}
        for m in conversation_messages
    ]
    vllm_messages = prepare_vllm_messages(
        payload_messages, workspace_dir, assistant.instructions
    )

    # Workspace tracker for artifacts
    tracker = WorkspaceTracker(workspace_dir, generated_dir)

    assistant_reply = ""
    finished = False

    while not finished:
        # Call vllm
        response = vllm_client.chat.completions.create(
            model=assistant.model,
            messages=vllm_messages,
            temperature=0.4,
            stream=True,
            extra_body={
                "add_generation_prompt": False,
                "stop_token_ids": [151676, 151645],
                "max_new_tokens": 32768,
            },
        )

        cur_res = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                delta = chunk.choices[0].delta.content
                cur_res += delta
                assistant_reply += delta
                yield delta

            if "</Answer>" in cur_res:
                finished = True
                break
        
        if chunk.choices[0].finish_reason == "stop" and not finished:
            # Model stopped but didn't finish properly
            if not cur_res.endswith("</Code>"):
                cur_res += "</Code>"
                assistant_reply += "</Code>"
                yield "</Code>"

        # Handle code execution
        if "</Code>" in cur_res and not finished:
            vllm_messages.append({"role": "assistant", "content": cur_res})
            code_str = extract_code_from_segment(cur_res)
            if code_str:
                exe_output = execute_code_safe(code_str, workspace_dir)
                artifacts = tracker.diff_and_collect()
                exe_str = f"\n<Execute>\n```\n{exe_output}\n```\n</Execute>\n"
                file_block = render_file_block(artifacts, workspace_dir, thread_id)
                assistant_reply += exe_str + file_block
                yield exe_str + file_block
                vllm_messages.append({"role": "execute", "content": exe_output})

        

    final_message_text = fix_tags_and_codeblock(assistant_reply)
    report_block = ""
    try:
        report_path = generate_markdown_report(
            thread_id, messages_list, final_message_text, workspace_dir
        )
    except Exception as report_error:
        report_path = None
        print(f"Report generation error: {report_error}")

    if report_path:
        try:
            rel = report_path.resolve().relative_to(Path(workspace_dir).resolve())
            rel_path = rel.as_posix()
        except Exception:
            rel_path = report_path.name
        url = build_download_url(thread_id, rel_path)
        lines = ["<File>", f"- [{report_path.name}]({url})", "</File>"]
        report_block = "\n" + "\n".join(lines) + "\n"
        final_message_text += report_block
        yield report_block

    # Store final assistant message
    storage.create_message(
        thread_id=thread_id,
        role="assistant",
        content=final_message_text,
    )


@app.post("/v1/threads/{thread_id}/runs", response_model=RunObject)
async def create_run(
    thread_id: str,
    assistant_id: str = Body(...),
    model: Optional[str] = Body(None),
    instructions: Optional[str] = Body(None),
    tools: Optional[List[Dict]] = Body(None),
    metadata: Optional[Dict] = Body(None),
):
    """Create a run (OpenAI compatible)"""
    thread = storage.get_thread(thread_id)  # This will update last_accessed_at
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    assistant = storage.get_assistant(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # Validate tools
    if tools:
        for tool in tools:
            if tool.get("type") not in ["NotImplemented"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported tool type: {tool.get('type')}",
                )

    # Create run
    run = storage.create_run(
        thread_id=thread_id,
        assistant_id=assistant_id,
        model=model or assistant.model,
        instructions=instructions,
        tools=tools,
        metadata=metadata,
    )

    # Execute run in background
    def execute_run():
        try:
            storage.update_run_status(
                run.id, "in_progress", started_at=int(time.time())
            )

            # Execute the model
            full_response = ""
            for chunk in bot_stream_for_run(thread_id, assistant):
                full_response += chunk

            storage.update_run_status(
                run.id, "completed", completed_at=int(time.time())
            )
        except Exception as e:
            print(f"Run execution error: {e}")
            traceback.print_exc()
            storage.update_run_status(
                run.id,
                "failed",
                failed_at=int(time.time()),
                last_error={"code": "server_error", "message": str(e)},
            )

    threading.Thread(target=execute_run, daemon=True).start()

    return run


@app.get("/v1/threads/{thread_id}/runs/{run_id}", response_model=RunObject)
async def retrieve_run(thread_id: str, run_id: str):
    """Retrieve a run (OpenAI compatible)"""
    # Update thread access time
    thread = storage.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    run = storage.get_run(thread_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/v1/threads/{thread_id}/runs", response_model=Dict[str, Any])
async def list_runs(thread_id: str):
    """List runs for a thread (OpenAI compatible)"""
    thread = storage.get_thread(thread_id)  # This will update last_accessed_at
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    runs = [r for r in storage.runs.values() if r["thread_id"] == thread_id]
    return {"object": "list", "data": [RunObject(**r).dict() for r in runs]}


# ============================================================================
# Extended API: Get Generated Files
# ============================================================================


@app.get("/v1/threads/{thread_id}/files")
async def get_thread_files(thread_id: str):
    """
    Extended API: Get generated files for a thread.
    Returns list of files with download URLs.
    """
    thread = storage.get_thread(thread_id)  # This will update last_accessed_at
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    workspace_dir = get_thread_workspace(thread_id)
    generated_dir = Path(workspace_dir) / "generated"

    files = []
    if generated_dir.exists():
        for file_path in generated_dir.iterdir():
            if file_path.is_file():
                rel_path = f"generated/{file_path.name}"
                files.append(
                    {
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "url": build_download_url(thread_id, rel_path),
                        "created_at": int(file_path.stat().st_mtime),
                    }
                )

    return {"object": "list", "data": files}


# ============================================================================
# Extended API: Chat Completion with File Attachment (Temporary Conversation)
# ============================================================================


@app.post("/v1/chat/completions")
async def chat_completions(
    model: str = Body(...),
    messages: List[Dict[str, Any]] = Body(...),
    file_ids: Optional[List[str]] = Body(None),
    temperature: Optional[float] = Body(0.4),
    stream: Optional[bool] = Body(False),
    execute_code: Optional[bool] = Body(True),
):
    """
    Extended chat completion API with file attachment support.
    Creates a temporary conversation with associated files.
    
    Parameters:
    - model: Model name
    - messages: List of message objects with role and content
    - file_ids: Optional list of file IDs to attach to the conversation
    - temperature: Sampling temperature (default 0.4)
    - stream: Whether to stream the response (default False)
    - execute_code: Whether to enable code execution (default True)
    
    Returns:
    - Standard OpenAI chat completion response
    - Additional field 'generated_files' with list of generated file URLs (if execute_code=True)
    """
    # Create temporary thread
    temp_thread = storage.create_thread(metadata={"temporary": True})
    workspace_dir = get_thread_workspace(temp_thread.id)
    generated_dir = os.path.join(workspace_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)

    try:
        # Copy files to workspace
        if file_ids:
            for fid in file_ids:
                file_obj = storage.get_file(fid)
                if not file_obj:
                    raise HTTPException(status_code=400, detail=f"File {fid} not found")
                src_path = storage.files[fid].get("filepath")
                if src_path and os.path.exists(src_path):
                    dst_path = uniquify_path(Path(workspace_dir) / file_obj.filename)
                    shutil.copy2(src_path, dst_path)

        # Build messages with DeepAnalyze prompt template
        vllm_messages: List[Dict[str, Any]] = prepare_vllm_messages(
            messages, workspace_dir
        )

        # Track generated files
        generated_files = []

        # Stream response with code execution
        if stream and execute_code:
            def generate_stream_with_execution():
                assistant_reply = ""
                finished = False
                tracker = WorkspaceTracker(workspace_dir, generated_dir)
                
                while not finished:
                    response = vllm_client.chat.completions.create(
                        model=model,
                        messages=vllm_messages,
                        temperature=temperature,
                        stream=True,
                        extra_body={
                            "add_generation_prompt": False,
                            "stop_token_ids": [151676, 151645],
                            "max_new_tokens": 32768,
                        },
                    )

                    cur_res = ""
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            delta = chunk.choices[0].delta.content
                            cur_res += delta
                            assistant_reply += delta
                            
                            chunk_data = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"

                        if "</Answer>" in cur_res:
                            finished = True
                            break
                    
                    if chunk.choices[0].finish_reason == "stop" and not finished:
                        if not cur_res.endswith("</Code>"):
                            cur_res += "</Code>"
                            assistant_reply += "</Code>"

                    # Handle code execution
                    if "</Code>" in cur_res and not finished:
                        vllm_messages.append({"role": "assistant", "content": cur_res})

                        code_str = extract_code_from_segment(cur_res)
                        if code_str:
                            exe_output = execute_code_safe(code_str, workspace_dir)
                            artifacts = tracker.diff_and_collect()
                            exe_str = f"\n<Execute>\n```\n{exe_output}\n```\n</Execute>\n"
                            file_block = render_file_block(
                                artifacts, workspace_dir, temp_thread.id, generated_files
                            )
                            assistant_reply += exe_str + file_block
                            
                            # Stream execution result
                            for char in exe_str + file_block:
                                chunk_data = {
                                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": char},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"

                            vllm_messages.append({"role": "execute", "content": exe_output})

                    

                # Generate and stream report
                report_block = generate_report_from_messages(
                    messages, assistant_reply, workspace_dir, temp_thread.id, generated_files
                )
                if report_block:
                    for char in report_block:
                        chunk_data = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": char},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                # Send final chunk with generated files
                final_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                    "generated_files": generated_files,
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream_with_execution(), media_type="text/event-stream")
        
        elif stream:
            print("Streaming without code execution is not implemented.")
        else:
            # Non-streaming response processed with code execution workflow
            assistant_reply = ""
            finished = False
            generated_files = []
            tracker = WorkspaceTracker(workspace_dir, generated_dir)

            while not finished:
                response = vllm_client.chat.completions.create(
                    model=model,
                    messages=vllm_messages,
                    temperature=temperature,
                    stream=True,
                    extra_body={
                        "add_generation_prompt": False,
                        "stop_token_ids": [151676, 151645],
                        "max_new_tokens": 32768,
                    },
                )

                cur_res = ""
                last_finish_reason: Optional[str] = None
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        delta = chunk.choices[0].delta.content
                        cur_res += delta
                        assistant_reply += delta
                    last_finish_reason = chunk.choices[0].finish_reason
                    if "</Answer>" in cur_res:
                        finished = True
                        break

                if last_finish_reason == "stop" and not finished:
                    if not cur_res.endswith("</Code>"):
                        cur_res += "</Code>"
                        assistant_reply += "</Code>"

                if "</Answer>" in cur_res:
                    finished = True

                if "</Code>" in cur_res and not finished:
                    vllm_messages.append({"role": "assistant", "content": cur_res})
                    code_str = extract_code_from_segment(cur_res)
                    if code_str:
                        exe_output = execute_code_safe(code_str, workspace_dir)
                        artifacts = tracker.diff_and_collect()
                        exe_str = f"\n<Execute>\n```\n{exe_output}\n```\n</Execute>\n"
                        file_block = render_file_block(
                            artifacts, workspace_dir, temp_thread.id, generated_files
                        )
                        assistant_reply += exe_str + file_block
                        vllm_messages.append({"role": "execute", "content": exe_output})

                

            # Generate report
            report_block = generate_report_from_messages(
                messages, assistant_reply, workspace_dir, temp_thread.id, generated_files
            )
            assistant_reply += report_block

            result_content = assistant_reply
            result = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result_content,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

            if generated_files:
                result["generated_files"] = generated_files
            if file_ids:
                result["attached_files"] = file_ids

            return result
    finally:
        # Clean up temporary thread after some time (optional)
        pass


# ============================================================================
# Management API: Thread Cleanup
# ============================================================================


@app.post("/v1/admin/cleanup-threads")
async def manual_cleanup_threads(timeout_hours: int = Query(12, description="Timeout in hours for thread cleanup")):
    """
    Manual trigger for thread cleanup (Admin API)
    Clean up threads that haven't been accessed for more than timeout_hours
    """
    try:
        cleaned_count = storage.cleanup_expired_threads(timeout_hours=timeout_hours)
        return {
            "status": "success",
            "cleaned_threads": cleaned_count,
            "timeout_hours": timeout_hours,
            "timestamp": int(time.time())
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": int(time.time())
        }


@app.get("/v1/admin/threads-stats")
async def get_threads_stats():
    """
    Get statistics about threads (Admin API)
    """
    with storage._lock:
        total_threads = len(storage.threads)
        now = int(time.time())

        # Count threads by age categories
        recent_threads = 0  # < 1 hour
        old_threads = 0     # 1-12 hours
        expired_threads = 0 # > 12 hours

        for thread_data in storage.threads.values():
            last_accessed = thread_data.get("last_accessed_at", thread_data.get("created_at", 0))
            age_hours = (now - last_accessed) / 3600

            if age_hours < 1:
                recent_threads += 1
            elif age_hours <= 12:
                old_threads += 1
            else:
                expired_threads += 1

    return {
        "total_threads": total_threads,
        "recent_threads": recent_threads,  # < 1 hour
        "old_threads": old_threads,        # 1-12 hours
        "expired_threads": expired_threads, # > 12 hours
        "timeout_hours": 12,
        "timestamp": int(time.time())
    }


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": int(time.time())}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting DeepAnalyze OpenAI-Compatible API Server...")
    print(f"   - API Server: http://localhost:8200")
    print(f"   - File Server: http://localhost:{HTTP_SERVER_PORT}")
    print(f"   - Workspace: {WORKSPACE_BASE_DIR}")
    print("\n📖 API Endpoints:")
    print("   - Files API: /v1/files")
    print("   - Assistants API: /v1/assistants")
    print("   - Threads API: /v1/threads")
    print("   - Messages API: /v1/threads/{thread_id}/messages")
    print("   - Runs API: /v1/threads/{thread_id}/runs")
    print("   - Extended: /v1/threads/{thread_id}/files")
    print("   - Extended: /v1/chat/completions (with file_ids)")
    uvicorn.run(app, host="0.0.0.0", port=8200)
