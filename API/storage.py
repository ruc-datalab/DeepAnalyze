"""
Storage layer for DeepAnalyze API Server
Handles in-memory storage for OpenAI objects
"""

import os
import time
import uuid
import shutil
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

from models import (
    FileObject, AssistantObject, ThreadObject, MessageObject, RunObject
)
from utils import get_thread_workspace, uniquify_path


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
        """Create a file record"""
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
        """Get a file record"""
        with self._lock:
            if file_id in self.files:
                return FileObject(**self.files[file_id])
            return None

    def delete_file(self, file_id: str) -> bool:
        """Delete a file record"""
        with self._lock:
            if file_id in self.files:
                filepath = self.files[file_id].get("filepath")
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
                del self.files[file_id]
                return True
            return False

    def list_files(self, purpose: Optional[str] = None) -> List[FileObject]:
        """List files with optional purpose filter"""
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
        """Create an assistant record"""
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
        """Get an assistant record"""
        with self._lock:
            if assistant_id in self.assistants:
                return AssistantObject(**self.assistants[assistant_id])
            return None

    def delete_assistant(self, assistant_id: str) -> bool:
        """Delete an assistant record"""
        with self._lock:
            if assistant_id in self.assistants:
                del self.assistants[assistant_id]
                return True
            return False

    def list_assistants(self) -> List[AssistantObject]:
        """List all assistants"""
        with self._lock:
            return [AssistantObject(**a) for a in self.assistants.values()]

    def create_thread(
        self,
        metadata: Optional[Dict] = None,
        file_ids: Optional[List[str]] = None,
        tool_resources: Optional[Dict] = None
    ) -> ThreadObject:
        """Create a thread record"""
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
                "tool_resources": tool_resources,
            }
            self.threads[thread_id] = thread
            self.messages[thread_id] = []

            # Create workspace for this thread
            workspace_dir = get_thread_workspace(thread_id)
            os.makedirs(workspace_dir, exist_ok=True)
            os.makedirs(os.path.join(workspace_dir, "generated"), exist_ok=True)

            # Copy files to thread workspace
            all_file_ids = set(file_ids or [])

            # Add files from tool_resources analyze tool
            if tool_resources and "analyze" in tool_resources:
                analyze_file_ids = tool_resources["analyze"].get("file_ids", [])
                all_file_ids.update(analyze_file_ids)

            for fid in all_file_ids:
                if fid in self.files:
                    file_data = self.files[fid]
                    src_path = file_data.get("filepath")
                    if src_path and os.path.exists(src_path):
                        dst_path = uniquify_path(Path(workspace_dir) / file_data["filename"])
                        shutil.copy2(src_path, dst_path)

            return ThreadObject(**thread)

    def get_thread(self, thread_id: str) -> Optional[ThreadObject]:
        """Get a thread record"""
        with self._lock:
            if thread_id in self.threads:
                # Update last accessed time
                self.threads[thread_id]["last_accessed_at"] = int(time.time())
                return ThreadObject(**self.threads[thread_id])
            return None

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread record"""
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
        """Create a message record"""
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
        """List messages in a thread"""
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
        """Create a run record"""
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
        """Get a run record"""
        with self._lock:
            if run_id in self.runs and self.runs[run_id]["thread_id"] == thread_id:
                return RunObject(**self.runs[run_id])
            return None

    def update_run_status(
        self,
        run_id: str,
        status: str,
        **kwargs
    ) -> Optional[RunObject]:
        """Update run status"""
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


# Global storage instance
storage = Storage()